import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import math

class MyNetwork(nn.Module):
    def __init__(self, inputs, outputs, learning_rate=0.001, model=None):
        super(MyNetwork, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.fc1 = nn.Linear(self.inputs + 2 + self.outputs, self.outputs)
        # self.fc2 = nn.Linear(75, self.outputs)
        # self.fc2 = nn.Linear(150, self.outputs)
        if model is not None:
            self.load_state_dict(torch.load(model))
        self.drop_layer = nn.Dropout(0.5)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.softmax = nn.Softmax(dim=0)
        self.configuration_string = f"pytorch model with convolution"

    def forward(self, x):
        map = torch.tensor(x[0], requires_grad=True).float()
        food_number = np.zeros((self.outputs + 2))
        food_number = x[1]
        food_number = torch.tensor(food_number, requires_grad=True).float()
        map = F.max_pool2d(F.relu(self.conv1(map)), 2)
        # x = F.relu(self.fc1(x.reshape(self.inputs)))
        output = self.softmax(F.sigmoid((self.fc1(torch.cat((map.reshape(self.inputs), food_number))))))
        return output

    def predict(self, state):
        self.eval()
        prediction = self(state)
        # print(f"prediction = {prediction}")
        self.train()
        return prediction.detach().numpy()

    def fit(self, state, target):
        self.train()
        # print(f"state = {state} | target = {target}")
        # self.fit(state.reshape((1, self.inputs)), target)
        self.optimizer.zero_grad()
        output_train = self(state)
        loss = self.criterion(output_train, torch.tensor(target, requires_grad=True))
        loss.backward()
        # print(f"loss = {loss} | target = {target} | outputs = {output_train}")
        self.optimizer.step()
        self.eval()

    def save_model(self, path, id):
        torch.save(self.state_dict(), f"{path}/agent_{id}.pth.tar")

class MyNetwork2(MyNetwork):
    def __init__(self, inputs, outputs, learning_rate=0.001, model=None):
        super(MyNetwork2, self).__init__(inputs, outputs, learning_rate, model)
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.conv1 = nn.Conv2d(2, 4, 3)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.fc1 = nn.Linear(self.inputs + 1 + self.outputs, 75)
        self.fc2 = nn.Linear(75, 150)
        self.fc3 = nn.Linear(150, self.outputs)

    def forward(self, x):
        map = torch.tensor(x[0], requires_grad=True).float()
        food_number = np.zeros((self.outputs + 1))
        food_number = x[1]
        food_number = torch.tensor(food_number, requires_grad=True).float()
        map = F.max_pool2d(F.relu(self.conv1(map)), 2)
        map = F.max_pool2d(F.relu(self.conv2(map)), 2)
        output = F.relu(self.drop_layer(self.fc1(torch.cat((map.reshape(self.inputs), food_number)))))
        output = F.relu(self.drop_layer(self.fc2(output)))
        output = self.softmax(F.sigmoid(self.fc3(output)))
        return output
