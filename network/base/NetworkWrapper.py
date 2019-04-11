import numpy as np
import math
import random

class NetworkWrapper():
    def __init__(self, agent, memory=[]):
        self.reward = 0
        self.model = agent
        self.memory = memory
        self.player = None
        self.game = None
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

    def set_reward(self):
        # self.reward = (math.sqrt(math.pow(self.game.board_width, 2) + math.pow(self.game.board_height, 2)) - self.player.get_distance_closest_food())
        self.reward = 0
        if self.player.just_eat:
            self.reward = 100
        vision = self.player.take_a_look()
        food_in_vision = False
        for square in vision:
            if square.food:
                food_in_vision = True
                break
        if food_in_vision:
            self.reward += 1 * (math.sqrt(18) - self.player.get_distance_closest_food())
        return self.reward

    def remember(self, state, reward_old, action, reward, next_state, done):
        self.memory.append((state, reward_old, action, reward, next_state, done))

    def request_action(self):
        self.epsilon = self.game.game_number / 1.8 - self.game.game_index

        #get old state
        state_old = self.get_state()
        reward_old = self.set_reward()
        self.total_moves += 1
        #perform random actions based on agent.epsilon, or choose the action
        if random.randint(0, int(self.game.game_number * 1.3)) < self.epsilon:
            final_move = random.randint(0, 3)
            self.random_moves += 1
        else:
            # predict action based on the old state
            prediction = self.model.predict(state_old)
            final_move = np.argmax(prediction)
        #perform new move and get new state
        self.player.do_action(int(final_move))
        state_new = self.get_state()

        #set treward for the new state
        reward = self.set_reward()

        #train short memory base on the new action and state
        self.train_short_memory(state_old, reward_old, final_move, reward, state_new, self.player.dead)

        # store the new data into a long term memory
        self.remember(state_old, reward_old, final_move, reward, state_new, self.player.dead)

    def replay_new(self):
        # print(f'random moves : {100 * float(self.random_moves) / self.total_moves}')
        self.random_moves = 0
        self.total_moves = 0
        if len(self.memory) > 1000:
            minibatch = random.sample(self.memory, 1000)
        else:
            minibatch = self.memory
        for state, reward_old, action, reward, next_state, done in minibatch:
            target = reward - reward_old
            target_f = self.model.predict(state)
            target_f[action] = target
            self.model.fit(state, target_f)

    def train_short_memory(self, state, reward_old, action, reward, next_state, done):
        target = reward - reward_old
        target_f = self.model.predict(state)
        target_f[action] = target
        self.model.fit(state, target_f)

    def end_game(self, scores):
        self.save_agent()
        plot_seaborn()
