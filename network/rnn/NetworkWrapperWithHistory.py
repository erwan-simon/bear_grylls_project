import numpy as np
import math
import random

class NetworkWrapper():
    def __init__(self, agent, history_size=10):
        self.history_size = history_size
        self.model = agent
        self.player = None
        self.game = None
        len_vision = 41 # hard codded, not pretty but anyway
        self.memory = [[[0 for i in range(len_vision)], [0 for i in range(self.model.outputs)], 0]]
        self.random_moves = 0
        self.total_moves = 0

    def get_state(self):
        vision = self.player.take_a_look()
        state = np.zeros(len(vision))
        for square_index in range(len(vision)):
            # state[square_index] = np.zeros(3)
            # np.put(state, [square_index, 0], 1 if vision[square_index].food else 0)
            state[square_index] = 1 if vision[square_index].food else 0
            #np.put(state, [square_index, 1], 1 if vision[square_index].stone else 0)
            #np.put(state, [square_index, 2], 1 if len(vision[square_index].players) > 0 else 0)
        # state[square_index] = self.player.food
        return state#.reshape(3 * len(vision))

    def get_reward(self):
        # self.reward = (math.sqrt(math.pow(self.game.board_width, 2) + math.pow(self.game.board_height, 2)) - self.player.get_distance_closest_food())
        reward = 0
        if self.player.dead:
            reward = -100
        elif self.player.just_eat:
            reward = 100
        """
        vision = self.player.take_a_look()
        food_in_vision = False
        for square in vision:
            if square.food:
                food_in_vision = True
                break
        if food_in_vision:
            reward += 1 * (math.sqrt(18) - self.player.get_distance_closest_food())
        """
        return reward

    def request_action(self):
        self.epsilon = self.game.game_number / 1.8 - self.game.game_index

        self.total_moves += 1
        #perform random actions based on agent.epsilon, or choose the action
        if random.randint(0, int(self.game.game_number * 1.3)) < self.epsilon:
            action = random.randint(0, 3)
            self.random_moves += 1
        else:
            # predict action based on the old state
            prediction = self.model.predict(self.memory)
            action = np.argmax(prediction)
        #perform new move and get new state
        self.player.do_action(int(action))
        state = self.get_state()

        #set treward for the new state
        reward = self.get_reward()

        # store the new data into a long term memory
        self.remember(state, action, reward)

        #train short memory base on the new action and state
        self.train_short_memory()


    def replay_new(self):
        # print(f'random moves : {100 * float(self.random_moves) / self.total_moves}')
        self.random_moves = 0
        self.total_moves = 0
        if len(self.memory) > 1000:
            minibatch_index = random.sample(range(1, len(self.memory)), 1000)
        else:
            minibatch_index = range(1, len(self.memory))
        for i in minibatch_index:
            # target is last action done
            # print(f"array = {self.memory[np.amax([i - self.history_size, 0]):i]} | i = {i} | left = {np.amax([i - self.history_size, 0])} | minibatch = {minibatch_index}")
            self.model.fit(self.memory[np.amax([i - self.history_size, 0]):i], np.multiply(self.memory[i][1], self.memory[i][2]))

    def train_short_memory(self):
        # target is last action done
        if self.memory[-1][2] != 0:
            self.model.fit(self.memory[-self.history_size:-1], np.multiply(self.memory[-1][1], self.memory[-1][2]))

    def remember(self, state, action, reward):
        # action is an int but we want it to be a one hot vector
        one_hot_action = [0 for i in range(self.model.outputs)]
        one_hot_action[action] = 1
        """
        if reward != 0 and len(self.memory) > 1:
            j = -1
            relative_reward = reward * (1 - 1 / self.history_size)
            while j != -1 * len(self.memory) or self.memory[j][2] != 0:
                self.memory[j][2] = relative_reward
                relative_reward *= (1 - 1 / self.history_size)
                j -= 1
        """
        self.memory.append([state, one_hot_action, reward])
        # print("*************************************************")
        # print(self.memory)

    def end_game(self, scores):
        self.save_agent()
        plot_seaborn()
