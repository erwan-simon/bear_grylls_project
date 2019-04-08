import math
import game.Player as Player
import numpy as np

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import to_categorical

class MyNetwork():
    def __init__(self, inputs=50, outputs=4, learning_rate=0.0005, dropout=0.3):
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.inputs = inputs
        self.outputs = outputs
        self.model = self.network()

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=self.inputs))
        model.add(Dropout(self.dropout))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(output_dim=self.outputs, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)
        if weights:
            model.load_weights(weights)
        return model

    def predict(self, state):
        prediction = self.model.predict(state.reshape((1, self.inputs)))
        return prediction

    def fit(self, state, target):
        self.model.fit(state.reshape((1, self.inputs)), target, epochs=1, verbose=0)

    def save_model(self):
        self.model.save_weights('weights.hdf5')
