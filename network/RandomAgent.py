import random
import numpy as np

class MyNetwork():
    def __init__(self, inputs=50, outputs=4, learning_rate=0.0005, dropout=0.3):
        self.outputs = outputs

    def predict(self, state):
        result = np.array([0, 0, 0, 0])
        result[random.randint(0, self.outputs - 1)] = 1
        return result

    def fit(self, state, target):
        pass

    def save_model(self):
        pass
