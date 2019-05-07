from random import randint, choice
from itertools import product, islice
from collections import namedtuple
import math
import numpy as np

colors_combination = list(product([255, 0], repeat=3)) # make list of colors
colors_combination.remove((255, 255, 255)) # remove white
colors_combination.remove((0, 0, 0)) # remove black because it is color of stones
colors_combination.remove((255, 0, 0)) # remove red because it is the color of food

Actions = namedtuple("Actions", ["NORTH", "EAST", "SOUTH", "WEST", "STAY", "STEAL", "PICK", "DROP"])
actions = Actions(0, 1, 2, 3, 4, 5, 6, 7)

class Player(object):
    def __init__(self, id, game, network_wrapper, vision_distance=4, name="Player"):
        self.id = id
        self.x = randint(0, game.board_width - 1)
        self.y = randint(0, game.board_height - 1)
        game.board[self.y][self.x].players.append(self)
        self.food = 1
        self.vision_distance = vision_distance
        self.max_score_reached = 0
        self.stones = 0
        self.just_stone = False
        self.color = colors_combination[id]
        self.dead = False
        self.agent = network_wrapper
        self.agent.player = self
        self.agent.game = game
        self.game = game
        self.just_eat = False
        self.it_is_a_wall = False
        self.food_scores = []
        self.survival_scores = []
        self.name = name
        self.survival_time = 0
        self.max_survival_time = 0
        self.death_counter = []
        self.stones_scores = []
        self.game.logs_management(f"agent {self.id} named {self.name} configuration : {self.agent.model.configuration_string}")
        self.do_action(actions.NORTH)

    def eat(self):
        self.game.board[self.y][self.x].food = False
        self.game.spawn_food()
        self.food = math.ceil(self.food + 1) # reset consumption of food
        self.just_eat = True
        self.game.squares_with_food.remove(self.game.board[self.y][self.x])

    def drop(self, args=None):
        pass

    def pick(self, args=None):
        pass

    def steal(self):
        debug = f"{self.name} try to steal..."
        if len(self.game.board[self.y][self.x].players) != 0:
            debug += " and success !"
            for player in self.game.board[self.y][self.x].players:
                if player.stones > 0:
                    player.stones -= 1
                    self.stones += 1
                if player.food > 0:
                    player.food -= 1
                    self.eat()
        self.game.logs_management(debug)

    def update(self):
        # move and learn
        self.agent.request_action()

        self.just_eat = False
        self.just_stone = False
        self.survival_time += 1
        self.food_scores.append(self.food)
        self.survival_scores.append(self.survival_time)
        self.stones_scores.append(self.stones)
        # food vanishing
        self.food -= 1.0 / self.game.food_nutrition_value
        if self.food > self.max_score_reached:
            self.max_score_reached = self.food
        if self.survival_time > self.max_survival_time:
            self.max_survival_time = self.survival_time
        # death
        if self.food <= 0 or self.game.board[self.y][self.x].trap:
            # print(f"{self.name} ({self.id}) survived {self.survival_time} turns and died with a score of {self.score}.")
            self.dead = True
            self.survival_time = 0
            self.score = 0
            self.death_counter.append(len(self.survival_scores))
        # eat
        if self.game.board[self.y][self.x].food is True: #every players on the til eat
            self.eat()
        if self.game.board[self.y][self.x].stone is True: #every players on the til eat
            self.stones += 1
            self.game.board[self.y][self.x].stone = False
            self.game.squares_with_stone.remove(self.game.board[self.y][self.x])
            self.just_stone = True
        self.agent.after_effect()

    def get_distance_closest_food(self):
        result = math.sqrt(math.pow(self.game.board_width, 2) + math.pow(self.game.board_height, 2))
        for square in self.game.squares_with_food:
            distance = math.sqrt(math.pow(square.x - self.x, 2) + math.pow(square.y - self.y, 2))
            if distance <= result:
                result = distance
        return result

    def respawn(self):
        self.dead = False
        self.food = self.game.food_to_start
        self.stones = 0
        self.game.spawn_stones()
        self.game.remove_player_from_board(self)
        coo = list(product(range(0, self.game.board_width), range(0, self.game.board_height)))
        for trap in self.game.squares_with_trap:
            coo.remove((trap.y, trap.x))
        self.y, self.x = choice(coo)
        self.game.remove_player_from_board(self)
        self.game.board[self.y][self.x].players.append(self)
        self.agent.replay_new()
        self.survival_time = 0

    def do_action(self, action, args=None):
        self.it_is_a_wall = False
        if action is actions.NORTH:
            if self.game.board[(self.y + 1) % self.game.board_height][self.x].trap is True:
                self.it_is_a_wall = True
            else:
                self.y = (self.y + 1) % self.game.board_height
        elif action is actions.SOUTH:
            if self.game.board[(self.y - 1) % self.game.board_height][self.x].trap is True:
                self.it_is_a_wall = True
            else:
                self.y = (self.y - 1) % self.game.board_height
        elif action is actions.EAST:
            if self.game.board[self.y][(self.x + 1) % self.game.board_width].trap is True:
                self.it_is_a_wall = True
            else:
                self.x = (self.x + 1) % self.game.board_width
        elif action is actions.WEST:
            if self.game.board[self.y][(self.x - 1) % self.game.board_width].trap is True:
                self.it_is_a_wall = True
            else:
                self.x = (self.x - 1) % self.game.board_width
        elif action is actions.STAY:
            pass
        elif action is actions.STEAL:
            self.steal()
        elif action is actions.PICK:
            self.pick(args)
        elif action is actions.DROP:
            self.drop(args)
        self.last_action = action

    def take_a_look(self):
        result = []
        vision_distance = self.vision_distance
        if self.vision_distance == -1:
            return np.ravel(self.game.board)
        offset = 0
        for y in range(-vision_distance, vision_distance + 1):
            for x in range(-offset, offset + 1):
                result.append(self.game.board[(self.y + y) % self.game.board_height][(self.x + x) % self.game.board_width])
            offset += 1 if y < 0 else -1
        return result
