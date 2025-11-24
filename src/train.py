import tensorflow as tf
from config import Config

class Trainer:
    """Handle model training"""
    
    def __init__(self, model, callbacks):
        self.model = model
        self.callbacks = callbacks
        self.history = None
        
    def train(self, train_ds, val_ds, epochs=Config.EPOCHS):
        """Train the model"""
        
        
        print("STARTING TRAINING")
        
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=self.callbacks,
            verbose=Config.VERBOSE
        )
        

        
        return self.history
    
    def fine_tune(self, train_ds, val_ds, epochs=Config.FINE_TUNE_EPOCHS):
        """Fine-tune the model"""
    
        
        history_fine = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=self.callbacks,
            verbose=Config.VERBOSE
        )
        
        
        print("FINE-TUNING COMPLETED")
        
        
        return history_fine
