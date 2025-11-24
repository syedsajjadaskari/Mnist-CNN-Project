import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import Config
import os

class Visualizer:
    """Visualize training results"""
    
    @staticmethod
    def plot_training_history(history, save_path=None):
        """Plot training and validation metrics"""
        
        os.makedirs(Config.PLOT_DIR, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-3 Accuracy
        if 'top_3_accuracy' in history.history:
            axes[1, 0].plot(history.history['top_3_accuracy'], label='Train Top-3', linewidth=2)
            axes[1, 0].plot(history.history['val_top_3_accuracy'], label='Val Top-3', linewidth=2)
            axes[1, 0].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-3 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate (if available)
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], linewidth=2, color='red')
            axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = save_path or f'{Config.PLOT_DIR}/training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_sample_predictions(model, test_ds, class_names, num_samples=9):
        """Plot sample predictions"""
        
        os.makedirs(Config.PLOT_DIR, exist_ok=True)
        
        # Get a batch of images
        for images, labels in test_ds.take(1):
            predictions = model.predict(images[:num_samples], verbose=0)
            
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            axes = axes.ravel()
            
            for i in range(num_samples):
                axes[i].imshow(images[i])
                pred_class = class_names[predictions[i].argmax()]
                true_class = class_names[labels[i].numpy()]
                confidence = predictions[i].max()
                
                color = 'green' if pred_class == true_class else 'red'
                axes[i].set_title(
                    f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2%}',
                    color=color, fontsize=10
                )
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{Config.PLOT_DIR}/sample_predictions.png', dpi=300)
            print(f"Sample predictions saved to {Config.PLOT_DIR}/sample_predictions.png")
            plt.close()
            break
