import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import math

class MyNetwork(nn.Module):
    def __init__(self, inputs, outputs, model=None):
        super(MyNetwork, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        intermediary = 300
        self.fc1 = nn.Linear(inputs, intermediary)
        self.fc2 = nn.Linear(intermediary, intermediary)
        self.fc3 = nn.Linear(intermediary, intermediary)
        self.fc4 = nn.Linear(intermediary, outputs)
        if model is not None:
            self.load_state_dict(torch.load(model))
        self.drop_layer = nn.Dropout(0.5)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.0005, momentum=.9)
        self.softmax = nn.Softmax(dim=0)
        self.configuration_string = f"pytorch model with {self.inputs} inputs, {self.outputs} outputs, {intermediary} intermediary, 4 layers, dropout of 0.5, SGD optimizer, MSELoss and softmax"


    def forward(self, x):
        x = self.drop_layer(F.relu(self.fc1(x)))
        x = self.drop_layer(F.relu(self.fc2(x)))
        x = self.drop_layer(F.relu(self.fc3(x)))
        x = self.softmax(self.fc4(x))
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

    def save_model(self, path):
        torch.save(self.state_dict(), f"{path}/agent_{self.name}.pth.tar")
