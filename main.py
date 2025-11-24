import os
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()


import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
from src.data_loader import DataLoader
from src.model import TransferLearningModel
from src.callbacks import CallbacksManager
from src.train import Trainer
from src.evaluate import Evaluator
from src.visualize import Visualizer
from config import Config

def main():
    """Main execution pipeline"""
    
    print("="*70)
    print(" CNN TRANSFER LEARNING PROJECT - IMAGE CLASSIFICATION")
    print("="*70)
    print(f"\nDataset: {Config.DATASET}")
    print(f"Image Size: {Config.IMG_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Fine-tune Epochs: {Config.FINE_TUNE_EPOCHS}")
    
    # Set random seed
    tf.random.set_seed(Config.SEED)
    
    # 1. Load Data
    print("\n[STEP 1/8] Loading Data...")
    data_loader = DataLoader(Config.DATASET)
    train_ds, val_ds, test_ds = data_loader.load_data()
    train_ds, val_ds, test_ds = data_loader.prepare_dataset(train_ds, val_ds, test_ds)
    
    # 2. Build Model
    print("\n[STEP 2/8] Building Model...")
    model_builder = TransferLearningModel(data_loader.num_classes)
    model = model_builder.build_model()
    model = model_builder.compile_model()
    print("\nModel Summary:")
    model_builder.get_model_summary()
    
    # 3. Setup Callbacks
    print("\n[STEP 3/8] Setting up Callbacks...")
    callbacks_manager = CallbacksManager()
    callbacks = callbacks_manager.get_callbacks()
    callbacks_manager.print_tensorboard_command()
    
    # 4. Initial Training
    print("\n[STEP 4/8] Training Model...")
    trainer = Trainer(model, callbacks)
    history = trainer.train(train_ds, val_ds, Config.EPOCHS)
    
    # 5. Fine-tuning
    print("\n[STEP 5/8] Fine-tuning Model...")
    model_builder.unfreeze_for_fine_tuning(num_layers=20)
    callbacks_fine = callbacks_manager.get_callbacks(fine_tuning=True)
    trainer_fine = Trainer(model, callbacks_fine)
    history_fine = trainer_fine.fine_tune(train_ds, val_ds, Config.FINE_TUNE_EPOCHS)
    
    # 6. Evaluation
    print("\n[STEP 6/8] Evaluating Model...")
    evaluator = Evaluator(model, data_loader.class_names)
    test_results = evaluator.evaluate(test_ds)
    evaluator.print_classification_report(test_ds)
    evaluator.plot_confusion_matrix(test_ds)
    
    # 7. Visualization
    print("\n[STEP 7/8] Creating Visualizations...")
    visualizer = Visualizer()
    visualizer.plot_training_history(history)
    visualizer.plot_training_history(history_fine, 
                                     f'{Config.PLOT_DIR}/fine_tuning_history.png')
    visualizer.plot_sample_predictions(model, test_ds, data_loader.class_names)
    
    # 8. Final Summary
    print("\n[STEP 8/8] Training Complete!")
    print("\n" + "="*70)
    print(" FINAL RESULTS")
    print("="*70)
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test Top-3 Accuracy: {test_results[2]:.4f}")
    print(f"Model saved to: {Config.MODEL_SAVE_PATH}")
    print(f"Plots saved to: {Config.PLOT_DIR}/")
    print("="*70)
    
    return model, history, history_fine, data_loader.class_names

if __name__ == "__main__":
    model, history, history_fine, class_names = main()