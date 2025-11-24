import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from config import Config

class TransferLearningModel:
    """Build and compile transfer learning model"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = None
        self.base_model = None
        
    def build_model(self):
        """Build model using MobileNetV2"""
        
        print("\nBuilding model with MobileNetV2...")
        
        # Load pre-trained base model
        self.base_model = MobileNetV2(
            input_shape=Config.IMG_SHAPE,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        self.base_model.trainable = False
        
        # Build complete model
        inputs = keras.Input(shape=Config.IMG_SHAPE)
        x = self.base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(Config.DROPOUT_RATE)(x)
        x = layers.Dense(Config.DENSE_UNITS, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        print("Model built successfully!")
        return self.model
    
    def compile_model(self, learning_rate=Config.LEARNING_RATE):
        """Compile the model"""
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
            ]
        )
        
        print("Model compiled!")
        return self.model
    
    def unfreeze_for_fine_tuning(self, num_layers=20):
        """Unfreeze last layers for fine-tuning"""
        
        print(f"\nUnfreezing last {num_layers} layers for fine-tuning...")
        
        self.base_model.trainable = True
        
        # Freeze all layers except last num_layers
        for layer in self.base_model.layers[:-num_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.compile_model(learning_rate=Config.FINE_TUNE_LR)
        
        print("Model ready for fine-tuning!")
    
    def get_model_summary(self):
        """Print model summary"""
        return self.model.summary()