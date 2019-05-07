import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import math
from collections.abc import Iterable

class MyNetwork(nn.Module):
    def __init__(self, inputs, outputs, learning_rate=0.001, history_size=10, model=None):
        super(MyNetwork, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.history_size = history_size
        self.learning_rate = learning_rate
        self.lstm = nn.LSTM(self.inputs, self.outputs)
        if model is not None:
            self.load_state_dict(torch.load(model))
        self.drop_layer = nn.Dropout(0.5)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        self.softmax = nn.Softmax(dim=0)
        self.configuration_string = f"lstm with convolution and fc at the begining"

    def forward(self, states):
        inputs = np.zeros((len(states), 1, self.inputs))
        for state_index in range(len(states)):
            inputs_array = []
            inputs_array.extend(states[state_index][0])
            inputs_array.extend(states[state_index][1])

            inputs.itemset((state_index, 0), inputs_array)
        return self.lstm(torch.tensor(inputs, requires_grad=True).float())

    def predict(self, states):
        prediction = self(states)
        return prediction[0][-1].detach().numpy()[0]

    def fit(self, states, target):
        self.optimizer.zero_grad()
        output_train = self(states)
        loss = self.criterion(output_train[0][-1], torch.tensor(target, requires_grad=False).float())
        loss.backward()
        self.optimizer.step()

    def save_model(self, path, id):
        torch.save(self.model.state_dict(), f"{path}/agent_{id}.pth.tar")
