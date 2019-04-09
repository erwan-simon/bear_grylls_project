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

Actions = namedtuple("Actions", ["NORTH", "EAST", "SOUTH", "WEST", "STEAL", "PICK", "DROP"])
actions = Actions(0, 1, 2, 3, 4, 5, 6)

class Player(object):

    def __init__(self, id, game, agent, name="Player"):
        self.id = id
        self.x = randint(0, game.board_width - 1)
        self.y = randint(0, game.board_height - 1)
        game.board[self.y][self.x].players.append(self)
        self.food = FOOD_TO_START
        self.stones = 0
        self.color = colors_combination[id]
        self.dead = False
        self.agent = NetworkWrapper(game, self, agent)
        self.game = game
        self.just_eat = False
        self.score = 0
        self.scores = []
        self.name = self.agent.name
        self.survival_time = 0
        self.death_counter = 0
        self.do_action(actions.NORTH)

    def eat(self):
        before = self.food
        self.food += 1
        self.food = math.ceil(self.food) # reset consumption of food
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
        print(debug)

    def update(self):
        self.just_eat = False
        self.survival_time += 1
        self.scores.append(self.food)
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
            # print(f"{self.name} ({self.id}) survived {self.survival_time} turns and died with a score of {self.score}.")
            self.dead = True
            self.score = 0
            self.death_counter += 1

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
        self.stones = 0
        self.x = randint(0, self.game.board_width - 1)
        self.y = randint(0, self.game.board_height - 1)
        self.game.remove_player_from_board(self)
        self.game.board[self.y][self.x].players.append(self)
        self.agent.replay_new()
        self.survival_time = 0

    def do_action(self, action, args=None):
        if action is actions.NORTH:
            self.y = (self.y + 1) % self.game.board_height
        elif action is actions.SOUTH:
            self.y = (self.y - 1) % self.game.board_height
        elif action is actions.EAST:
            self.x = (self.x + 1) % self.game.board_width
        elif action is actions.WEST:
            self.x = (self.x - 1) % self.game.board_width
        elif action is actions.STEAL:
            self.steal()
        elif action is actions.PICK:
            self.pick(args)
        elif action is actions.DROP:
            self.drop(args)

    def take_a_look(self):
        result = []
        vision_distance = 4
        offset = 0
        for y in range(-vision_distance, vision_distance + 1):
            for x in range(-offset, offset + 1):
                result.append(self.game.board[(self.y + y) % self.game.board_height][(self.x + x) % self.game.board_width])
            offset += 1 if y < 0 else -1
        return result
