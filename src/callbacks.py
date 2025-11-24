import os
from datetime import datetime
from tensorflow import keras
from config import Config

class CallbacksManager:
    """Manage training callbacks"""
    
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or os.path.join(
            Config.LOG_DIR, 
            datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        
    def get_callbacks(self, fine_tuning=False):
        """Get list of callbacks for training"""
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
        
        suffix = "_fine_tuned" if fine_tuning else ""
        
        callbacks = [
            # TensorBoard
            keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch',
                profile_batch='10,20'
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=Config.MODEL_SAVE_PATH.replace('.h5', f'{suffix}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger
            keras.callbacks.CSVLogger(
                filename=f'training_log{suffix}.csv',
                append=True
            )
        ]
        
        return callbacks
    
    def print_tensorboard_command(self):
        """Print command to launch TensorBoard"""
        print(f"\n{'='*60}")
        print("To view training in TensorBoard, run:")
        print(f"tensorboard --logdir={Config.LOG_DIR}")
        print(f"{'='*60}\n")
