import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import math

class MyNetwork(nn.Module):
    def __init__(self, inputs, outputs, intermediary=120, learning_rate=0.0005, dropout=0.5, name="pytorch"):
        super(MyNetwork, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.fc1 = nn.Linear(inputs, intermediary)
        self.fc2 = nn.Linear(intermediary, intermediary)
        self.fc3 = nn.Linear(intermediary, intermediary)
        self.fc4 = nn.Linear(intermediary, intermediary)
        self.fc5 = nn.Linear(intermediary, intermediary)
        self.fc6 = nn.Linear(intermediary, outputs)
        self.drop_layer = nn.Dropout(self.dropout)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.softmax = nn.Softmax(dim=0)
        self.name = name

    def forward(self, x):
        x = self.drop_layer(F.relu(self.fc1(x)))
        x = self.drop_layer(F.relu(self.fc2(x)))
        x = self.drop_layer(F.relu(self.fc3(x)))
        x = self.drop_layer(F.relu(self.fc4(x)))
        x = self.drop_layer(F.relu(self.fc5(x)))
        x = self.softmax(self.fc6(x))
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

    def save_model(self):
        torch.save(self.model.state_dict(), f"agent.pth.tar")
