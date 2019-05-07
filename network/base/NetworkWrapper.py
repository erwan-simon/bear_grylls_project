import numpy as np
import math
import random

class BaseWrapper():
    def __init__(self, agent):
        self.model = agent
        self.memory = []
        self.player = None
        self.game = None
        self.random_moves = 0
        self.total_moves = 0
        self.discount = 0.75

    def get_state(self):
        vision = self.player.take_a_look()
        state = [[], []]
        state[0] = np.zeros((3, len(vision) + 1))
        for square_index in range(len(vision)):
            state[0].itemset((0, square_index), 1 if vision[square_index].food else 0)
            state[0].itemset((1, square_index), 1 if vision[square_index].trap else 0)
            state[0].itemset((2, square_index), 1 if vision[square_index].stone else 0)
        state[1].append(self.player.food)
        state[1].append(self.player.stones)
        return state

    def get_reward(self):
        reward = 1 / self.player.agent.model.outputs
        if self.player.just_stone:
            reward = 1.5
        if self.player.just_eat:
            reward = 1
        if self.player.it_is_a_wall:
            reward = 0
        if self.player.dead:
            reward = 0
        return reward

    def remember(self, state, action, reward, next_state, dead):
        relative_reward = reward * self.discount
        index = -1
        actions = []
        while index > -len(self.memory) - 1 and self.memory[index][4] != True and abs(relative_reward) > 1 / self.player.agent.model.outputs and len(actions) < self.player.vision_distance + 1:
            if (self.memory[index][1] + 2) % self.model.outputs in actions:
                actions.remove((self.memory[index][1] + 2) % self.model.outputs)
            else:
                actions.append(self.memory[index][1])
            self.memory[index][2] = self.memory[index][2] + relative_reward
            relative_reward *= self.discount
            index -= 1
        self.memory.append([state, action, reward, next_state, dead])
        # print(f"{self.player.name} : {self.memory[-1]}")

    def request_action(self):
        epsilon = 50000 - len(self.memory)
        #get old state
        self.state_old = self.get_state()
        reward_old = self.get_reward()
        self.total_moves += 1
        # print(f"state = {state_old}")
        #perform random actions based on agent.epsilon, or choose the action
        if random.randint(0, 125000) < epsilon:
            self.last_move = random.randint(0, self.model.outputs - 1)
            self.random_moves += 1
            # print("random move")
        else:
            # predict action based on the old state
            prediction = self.model.predict(self.state_old)
            # print(f"prediction = {prediction}")
            # print(f"{self.player.name} : {self.state_old} {prediction}")
            self.last_move = np.argmax(prediction)
        #perform new move and get new state
        self.player.do_action(int(self.last_move))

    def after_effect(self):
        state_new = self.get_state()

        #set treward for the new state
        reward = self.get_reward()

        # store the new data into a long term memory
        self.remember(self.state_old, self.last_move, reward, state_new, self.player.dead)
        self.train_short_memory()

    def replay_new(self):
        # print(f'random moves : {100 * float(self.random_moves) / self.total_moves}')
        self.random_moves = 0
        self.total_moves = 0
        # minibatch = [a for a in self.memory if a[2] != 0]
        minibatch = []
        negative = []
        positive = []
        null = []
        minibatch_size = 20000
        if len(self.memory) >= minibatch_size:
            for m in self.memory:
                if m[2] == 1 / self.player.agent.model.outputs:
                    null.append(m)
                elif m[2] == 0:
                    negative.append(m)
                elif m[2] >= 1:
                    positive.append(m)
            minibatch += random.sample(positive, np.amin([int(minibatch_size / 3), len(positive)]))
            minibatch += random.sample(negative, np.amin([int(minibatch_size / 3), len(negative)]))
            minibatch += random.sample(null, np.amin([int(minibatch_size / 3), len(null)]))
            # print(f"{self.player.name} : positive = {len(positive)} | negative = {len(negative)} | null = {len(null)}")
        else:
            minibatch = self.memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            target_f = self.model.predict(state)
            target_f[action] = target
            self.model.fit(state, target_f)

    def train_short_memory(self):
        state, action, reward, next_state, done = self.memory[-1]
        target = reward
        target_f = self.model.predict(state)
        target_f[action] = target
        self.model.fit(state, target_f)

    def end_game(self, scores):
        self.save_agent()
        plot_seaborn()

class BaseWrapper2(BaseWrapper):
    pass
