import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import math

class MyNetwork(nn.Module):
    def __init__(self, inputs, outputs, intermediary=4, learning_rate=0.0005, dropout=0.5, model=None):
        super(MyNetwork, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.intermediary = intermediary
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.fc1_1 = nn.Linear(42, self.intermediary)
        self.fc1_2 = nn.Linear(42, self.intermediary)
        self.fc3 = nn.Linear(self.intermediary * 2, self.outputs)
        if model is not None:
            self.load_state_dict(torch.load(model))
        self.fc1_1 = nn.Linear(self.inputs, self.intermediary)
        self.fc1_2 = nn.Linear(self.inputs, self.intermediary)
        self.drop_layer = nn.Dropout(self.dropout)
        self.criterion = nn.MSELoss()
        # self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=.9)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.softmax = nn.Softmax(dim=0)
        self.configuration_string = f"pytorch model with {self.inputs} inputs, {self.outputs} outputs, {self.intermediary} intermediary, {4} layers, dropout of {self.dropout}, Adam optimizer, MSELoss and softmax"


    def forward(self, x):
        x_1 = F.relu(self.fc1_1(x[0]))

        x_2 = F.relu(self.fc1_2(x[1]))

        x = self.softmax(self.fc3(torch.cat((x_1, x_2))))
        return x

    def predict(self, state):
        self.eval()
        prediction = self(torch.tensor(state, requires_grad=True).float())
        self.train()
        return prediction.detach().numpy()

    def fit(self, state, target):
        self.train()
        # self.fit(state.reshape((1, self.inputs)), target)
        self.optimizer.zero_grad()
        output_train = self(torch.tensor(state, requires_grad=True).float())
        loss = self.criterion(output_train, torch.tensor(target, requires_grad=True))
        loss.backward()
        self.optimizer.step()
        self.eval()

    def save_model(self, path, id):
        torch.save(self.state_dict(), f"{path}/agent_{id}.pth.tar")
