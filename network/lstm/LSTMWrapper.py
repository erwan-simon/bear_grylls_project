import numpy as np
import math
import random
from network.convolution.ConvolutionWrapper import ConvolutionWrapper

class LSTMWrapper(ConvolutionWrapper):
    def __init__(self, agent, history_size=10):
        super(LSTMWrapper, self).__init__(agent)
        self.history_size = history_size

    def request_action(self):
        #get old state
        self.state_old = self.get_state()
        reward_old = self.get_reward()
        self.total_moves += 1
        # print(f"state = {state_old}")
        #perform random actions based on agent.epsilon, or choose the action
        if random.randint(0, 500) > self.player.max_survival_time or random.randint(0, 10) == 0:
            self.last_move = random.randint(0, 3)
            self.random_moves += 1
            # print("random move")
        else:
            # predict action based on the old state
            states = []
            states_index = 0
            states.append(self.state_old)
            while states_index > -len(self.memory) and states_index > -self.history_size - 1 and self.memory[states_index][-1] != True:
                states.append(self.memory[states_index][0])
                states_index -= 1
            prediction = self.model.predict(states)
            self.last_move = np.argmax(prediction)
        #perform new move and get new state
        self.player.do_action(int(self.last_move))

    def replay_new(self):
        # print(f'random moves : {100 * float(self.random_moves) / self.total_moves}')
        self.random_moves = 0
        self.total_moves = 0
        # minibatch = [a for a in self.memory if a[2] != 0]
        minibatch = range(len(self.memory))
        if len(minibatch) > 1000:
            minibatch = random.sample(range(len(minibatch)), 1000)
        for index in minibatch:
            state, action, reward, next_state, done = self.memory[index]
            states = []
            states_index = 0
            while states_index + index >= 0 and states_index > -self.history_size - 1 and self.memory[states_index + index][-1] != True:
                states.append(self.memory[states_index + index][0])
                states_index -= 1
            if len(states) != 0:
                target = reward
                target_f = self.model.predict(states)
                target_f[action] = target
                self.model.fit(states, target_f)

    def train_short_memory(self):
        state, action, reward, next_state, done = self.memory[-1]
        states = []
        states_index = 0
        while states_index > -len(self.memory) and states_index > -self.history_size - 1 and self.memory[states_index][-1] != True:
            states.append(self.memory[states_index][0])
            states_index -= 1
        if len(states) != 0:
            target = reward
            target_f = self.model.predict(states)
            target_f[action] = target
            self.model.fit(states, target_f)
