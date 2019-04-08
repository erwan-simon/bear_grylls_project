from random import randint
from itertools import product, islice
from collections import namedtuple
import math
import numpy as np
from network.NetworkWrapper import NetworkWrapper

FOOD_TO_START = 1 # food player will have at start
FOOD_NUTRITION_VALUE = 25 # number of update it takes before a food is consumed

colors_combination = list(product([255, 0], repeat=3)) # make list of colors
colors_combination.remove((255, 255, 255)) # remove white
colors_combination.remove((0, 0, 0)) # remove black
colors_combination.remove((255, 0, 0)) # remove red because it is the color of food

Directions = namedtuple("Directions", ["NORTH", "EAST", "SOUTH", "WEST"])
directions = Directions(0, 1, 2, 3)

class Player(object):

    def __init__(self, id, game, agent, name="Player"):
        self.id = id
        self.x = randint(0, game.board_width - 1)
        self.y = randint(0, game.board_height - 1)
        game.board[self.y][self.x].players.append(self)
        self.food = FOOD_TO_START
        self.color = colors_combination[id]
        self.dead = False
        self.agent = NetworkWrapper(game, self, agent)
        self.game = game
        self.just_eat = False
        self.score = 0
        self.scores = []
        self.name = name
        self.survival_time = 0
        self.do_action(directions.NORTH)

    def eat(self):
        before = self.food
        self.food += 1
        self.food = math.ceil(self.food) # reset consumption of food
        self.just_eat = True
        self.game.squares_with_food.remove(self.game.board[self.y][self.x])
        self.score += 1

    def update(self):
        self.just_eat = False
        self.survival_time += 1
        if self.dead:
            return

        # eat
        if self.game.board[self.y][self.x].food is True: #every players on the til eat
            self.eat()
            self.game.board[self.y][self.x].food = False
            self.game.spawn_food()

        # food vanishing
        self.food -= 1.0 / FOOD_NUTRITION_VALUE

        # death
        if self.food <= 0:
            self.dead = True
            print(f"{self.name} ({self.id}) survived {self.survival_time} turns and died with a score of {self.score}.")

        # move and learn
        self.agent.request_action()

    def get_distance_closest_food(self):
        result = math.sqrt(math.pow(self.game.board_width, 2) + math.pow(self.game.board_height, 2))
        for square in self.game.squares_with_food:
            distance = math.sqrt(math.pow(square.x - self.x, 2) + math.pow(square.y - self.y, 2))
            if distance <= result:
                result = distance
        return result


    def respawn(self):
        self.dead = False
        self.food = FOOD_TO_START
        self.x = randint(0, self.game.board_width - 1)
        self.y = randint(0, self.game.board_height - 1)
        self.game.remove_player_from_board(self)
        self.game.board[self.y][self.x].players.append(self)
        self.agent.replay_new()
        self.scores.append(self.score)
        self.score = 0
        self.survival_time = 0

    def do_action(self, action):
        if action is directions.NORTH:
            self.y = (self.y + 1) % self.game.board_height
        elif action is directions.SOUTH:
            self.y = (self.y - 1) % self.game.board_height
        elif action is directions.EAST:
            self.x = (self.x + 1) % self.game.board_width
        elif action is directions.WEST:
            self.x = (self.x - 1) % self.game.board_width

    def take_a_look(self):
        result = []
        vision_distance = 4
        offset = 0
        for y in range(-vision_distance, vision_distance + 1):
            for x in range(-offset, offset + 1):
                result.append(self.game.board[(self.y + y) % self.game.board_height][(self.x + x) % self.game.board_width])
            offset += 1 if y < 0 else -1
        return result
