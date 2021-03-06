import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import math
from collections.abc import Iterable

class MyNetwork(nn.Module):
    def __init__(self, inputs, outputs, history_size=10, model=None):
        super(MyNetwork, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.history_size = history_size
        intermediary = 300
        self.rnn_layer = nn.RNN(self.inputs, self.outputs)
        if model is not None:
            self.load_state_dict(torch.load(model))
        self.drop_layer = nn.Dropout(0.5)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.0005)
        self.softmax = nn.Softmax(dim=1)
        self.configuration_string = f"pytorch model with {self.inputs} inputs, {self.outputs} outputs, {intermediary} intermediary, 4 layers, dropout of 0.5, Adam optimizer, MSELoss and softmax"
        self.init_weights()

    def init_weights(self): # https://discuss.pytorch.org/t/initializing-rnn-gru-and-lstm-correctly/23605 
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, input):
        output, hidden = self.rnn_layer(input, torch.zeros((1, 1, self.outputs)))
        return F.softmax(output[0][-1], dim=0)


    def predict(self, states):
        inputs = np.zeros((len(states), 1, self.inputs))
        for state_index in range(len(states)):
            inputs_array = []
            inputs_array.extend(states[state_index][0])
            inputs_array.extend(states[state_index][1])
            # inputs_array.append(states[state_index][2])
            np.put(inputs, (state_index, 1), np.array(inputs_array))
        prediction = self(torch.tensor(inputs, requires_grad=True).float())
        # print(prediction[0][-1])
        # print(prediction)
        return prediction.detach().numpy()

    def fit(self, states, target):
        inputs = np.zeros((len(states), 1, self.inputs))
        for state_index in range(len(states)):
            inputs_array = []
            inputs_array.extend(states[state_index][0])
            inputs_array.extend(states[state_index][1])
            # inputs_array.append(states[state_index][2]) I do not give the reward to the network
            np.put(inputs, (state_index, 1), np.array(inputs_array))
        self.optimizer.zero_grad()
        output_train = self(torch.tensor(inputs, requires_grad=True).float())
        print(f"target = {target} | output_train = {output_train}")
        loss = self.criterion(output_train, torch.tensor(target, requires_grad=False).float())
        loss.backward()
        self.optimizer.step()

    def save_model(self, path, id):
        torch.save(self.state_dict(), f"{path}/agent_{id}.pth.tar")
