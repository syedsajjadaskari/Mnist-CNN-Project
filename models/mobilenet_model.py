from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.MobileNetV2 import decode_prediction
import numpy as np

class MobileNetClassfier:
    def __init__(self):
        print("Loding of the model.....")
        self.model = MobileNetV2(weight = 'imagenet')
        print("Model Ready!")

    def predict(self, frames):
        if len(frames.shape) == 3:
            frames = np.expand_dims(frames, axis=0)
        
        preds = self.model.predict(frames, verbose = 1)

        results = []
        for pred in preds:
            decode = decode_prediction(np.expand_dims(pred,0), top = 5)[0]
            results.append([(name, float(prob)) for _, name, prob in decode])
        return results
