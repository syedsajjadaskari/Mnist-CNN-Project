"""MobileNet model wrapper"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetClassifierV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
import numpy as np

class MobileNetClassifier:
    def __init__(self):
        print("Loading model...")
        self.model = MobileNetV2(weights='imagenet')
        print("Model ready!")
    
    def predict(self, frames):
        if len(frames.shape) == 3:
            frames = np.expand_dims(frames, axis=0)
        
        preds = self.model.predict(frames, verbose=0)
        results = []
        for pred in preds:
            decoded = decode_predictions(np.expand_dims(pred, 0), top=5)[0]
            results.append([(name, float(prob)) for _, name, prob in decoded])
        return results