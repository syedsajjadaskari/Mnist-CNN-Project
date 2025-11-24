import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from config import Config
import os

class Evaluator:
    """Evaluate model performance"""
    
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        
    def evaluate(self, test_ds):
        """Evaluate model on test set"""
        
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60 + "\n")
        
        results = self.model.evaluate(test_ds, verbose=1)
        
        print(f"\nTest Loss: {results[0]:.4f}")
        print(f"Test Accuracy: {results[1]:.4f}")
        print(f"Test Top-3 Accuracy: {results[2]:.4f}")
        
        return results
    
    def generate_predictions(self, test_ds):
        """Generate predictions for confusion matrix"""
        
        y_true = []
        y_pred = []
        
        for images, labels in test_ds:
            predictions = self.model.predict(images, verbose=0)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(labels.numpy())
        
        return np.array(y_true), np.array(y_pred)
    
    def plot_confusion_matrix(self, test_ds):
        """Plot confusion matrix"""
        
        os.makedirs(Config.PLOT_DIR, exist_ok=True)
        
        y_true, y_pred = self.generate_predictions(test_ds)
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{Config.PLOT_DIR}/confusion_matrix.png', dpi=300)
        plt.close()
        
        print(f"Confusion matrix saved to {Config.PLOT_DIR}/confusion_matrix.png")
    
    def print_classification_report(self, test_ds):
        """Print classification report"""
        
        y_true, y_pred = self.generate_predictions(test_ds)
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60 + "\n")
        
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            digits=4
        )
        print(report)
        
        return report
