from game.Lib import Lib
from random import randint
from game.Square import Square
from game.Player import Player

from network.RandomAgent import MyNetwork as random
from network.KerasAgent import MyNetwork as keras
from network.PytorchAgent import MyNetwork as pytorch

import matplotlib.pyplot as plt
import numpy as np

class Game:
    def __init__(self, board_width=50, board_height=50, display_option=True, general_turn_latency=0, highlight_turn_latency=300, number_of_games=10, food_scarcity=0.005):
        self.board = []
        self.display_option = display_option
        self.general_turn_latency = general_turn_latency
        self.highlight_turn_latency = highlight_turn_latency
        self.food_scarcity = food_scarcity
        self.board_width = board_width
        self.board_height = board_height
        self.game_best_score = 0
        self.game_index = 0
        self.game_number = number_of_games
        for y in range(board_height):
            self.board.append([])
            for x in range(board_width):
                self.board[-1].append(Square(y, x))
        self.players = []
        self.players.append(Player(len(self.players), self, keras, "Keras"))
        self.players.append(Player(len(self.players), self, pytorch, "Pytorch"))
        self.players.append(Player(len(self.players), self, random, "Random"))

        self.food_on_board = int(len(self.players) * board_width * board_height * food_scarcity)

        if (self.display_option):
            self.lib = Lib(self)
        self.squares_with_food = []
        for f in range(self.food_on_board):
            self.spawn_food()
        print(f"Starting {self.game_index}th game")

    def spawn_food(self):
        nb_alive = 0
        for player in self.players:
            if player.dead is False:
                nb_alive += 1
        self.food_on_board = int(nb_alive * self.board_width * self.board_height * self.food_scarcity)
        while len(self.squares_with_food) < self.food_on_board:
            while 1:
                random_x = randint(0, self.board_width - 1)
                random_y = randint(0, self.board_height - 1)
                if self.board[random_y][random_x].food is False:
                    break
            self.board[random_y][random_x].food = True
            self.squares_with_food.append(self.board[random_y][random_x])

    def restart(self):
        self.game_index += 1
        for player in self.players:
            player.respawn()
        present_food = 0
        for y in range(len(self.board)):
            for x in range(len(self.board)):
                if self.board[y][x].food:
                    present_food += 1
        if self.game_index >= self.game_number:
            print(f"{self.game_number} games played. Exiting...")
            self.plot_scores()
            exit(0)
        for f in range(present_food, self.food_on_board):
            self.spawn_food()
        print(f"Starting {self.game_index}/{self.game_number}")

    def remove_player_from_board(self, player):
        for y in range(len(self.board)):
            for x in range(len(self.board)):
                if player in self.board[y][x].players:
                    self.board[y][x].players.remove(player)
                    return;

    def update(self):
        # players
        for player in self.players:
            player.update()

        # display
        if (self.display_option):
            if hasattr(self, 'lib') is False:
                self.lib = Lib()
            self.lib.update()
        elif hasattr(self, 'lib') is True:
            self.lib = None

        # restart game if needed
        all_dead = True
        for player in self.players:
            if player.dead is False:
                all_dead = False
                return
        if all_dead is True:
            self.restart()

    def plot_scores(self):
        scores = []
        for player in self.players:
            color = (player.color[0] / 255, player.color[1] / 255, player.color[2] / 255)
            plt.plot(range(0, self.game_number), player.scores, label=player.name, color=color)
        plt.ylabel("score")
        plt.xlabel("game index")
        plt.legend(title="score evolution")
        ax = plt.gca()
        ax.set_facecolor((0.65, 0.65, 0.65))
        plt.show()
