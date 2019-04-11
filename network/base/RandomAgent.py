import random
import numpy as np

class MyNetwork():
    def __init__(self, inputs=50, outputs=4, intermediary=0, learning_rate=0, dropout=0, name="random"):
        self.outputs = outputs
        self.inputs = inputs
        self.intermediary = intermediary
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.number_of_layers = 0
        self.name = name
        self.configuration_string = "Radom agent"


    def predict(self, state):
        result = np.array([0, 0, 0, 0])
        result[random.randint(0, self.outputs - 1)] = 1
        return result

    def fit(self, state, target):
        pass

    def save_model(self, path):
        pass
