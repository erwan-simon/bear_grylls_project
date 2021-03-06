import math
import game.Player as Player
import numpy as np

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import to_categorical

class MyNetwork():
    def __init__(self, inputs=50, outputs=4, intermediary=120, learning_rate=0.0005, dropout=0.5, name="keras"):
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.inputs = inputs
        self.intermediary = intermediary
        self.outputs = outputs
        self.model = self.network()
        self.name = name

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=intermediary, activation='relu', input_dim=self.inputs))
        model.add(Dropout(self.dropout))
        model.add(Dense(output_dim=intermediary, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(output_dim=intermediary, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(output_dim=self.outputs, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)
        if weights:
            model.load_weights(weights)
        return model

    def predict(self, state):
        prediction = self.model.predict(np.array([state]))
        return prediction[0]

    def fit(self, state, target):
        self.model.fit(np.array([state]), np.array([target]), epochs=1, verbose=0)

    def save_model(self, path):
        self.model.save_weights(path + '/weights.hdf5')
