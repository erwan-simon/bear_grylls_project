from game.Lib import Lib
from random import randint
from game.Square import Square
from game.Player import Player

from network.RandomAgent import MyNetwork as random
from network.KerasAgent import MyNetwork as keras
from network.PytorchAgent import MyNetwork as pytorch
from network.PytorchAgent2 import MyNetwork as pytorch2

import matplotlib.pyplot as plt
import numpy as np
import os

class Game:
    def __init__(self, id=0,
                 board_width=50,
                 board_height=50,
                 display_option=True,
                 general_turn_latency=0,
                 highlight_turn_latency=300,
                 number_of_games=150,
                 max_score=1000,
                 food_offset=6,
                 max_number_of_stones=12,
                 save_logs=True,
                 verbose=True,
                 display_plot=False):
        # settings
        self.id = id
        self.verbose = verbose
        self.display_plot = display_plot
        self.save_logs = save_logs
        self.logs = []

        # board settings
        self.board = []
        self.food_offset = food_offset
        self.current_food_offset = food_offset
        self.max_number_of_stones = max_number_of_stones
        self.board_width = board_width
        self.board_height = board_height
        for y in range(board_height):
            self.board.append([])
            for x in range(board_width):
                self.board[-1].append(Square(y, x))

        # players
        self.players = []
        self.players.append(Player(len(self.players), self, pytorch2(inputs=41, outputs=4, intermediary=50, name="50 - 6 layers")))
        self.players.append(Player(len(self.players), self, pytorch(inputs=41, outputs=4, intermediary=300, name="300 - 4 layers")))
        self.players.append(Player(len(self.players), self, random()))

        # food and stones initialisation
        self.squares_with_food = []
        self.squares_with_stone = []
        self.spawn_food()
        self.spawn_stones()

        # others
        self.game_best_score = 0
        self.game_index = 0
        self.max_score = max_score
        self.game_number = number_of_games

        # lib settings
        self.display_option = display_option
        self.general_turn_latency = general_turn_latency
        self.highlight_turn_latency = highlight_turn_latency
        if (self.display_option):
            self.lib = Lib(self)

    def where_to_spawn_food_2(self, i, j):
        for y in range(self.current_food_offset):
            for x in range(self.current_food_offset):
                if j + y >= self.board_height or i + x >= self.board_width or self.board[j + y][i + x].food == True:
                    return -1, -1
        return randint(i, int(i + self.current_food_offset - 1)), randint(j, int(j + self.current_food_offset - 1))

    def where_to_spawn_food(self):
        result = []
        for y in range(0, self.board_height, self.current_food_offset):
            for x in range(0, self.board_width, self.current_food_offset):
                res_x, res_y = self.where_to_spawn_food_2(x, y)
                if res_x != -1 and res_y != -1:
                    result.append([res_x, res_y])
        return result
        # print(len(self.squares_with_food))

    def spawn_food(self):
        result = self.where_to_spawn_food()
        if len(result) != 0:
            for coo in result:
                self.board[coo[1]][coo[0]].food = True
                self.squares_with_food.append(self.board[coo[1]][coo[0]])

    def spawn_stones(self):
        current_stones_count = len(self.squares_with_stone)
        for player in self.players:
            current_stones_count += player.stones
        while current_stones_count < self.max_number_of_stones:
            while 1:
                random_x = randint(0, self.board_width - 1)
                random_y = randint(0, self.board_height - 1)
                if self.board[random_y][random_x].stone is False:
                    self.board[random_y][random_x].stone = True
                    self.squares_with_stone.append(self.board[random_y][random_x])
                    current_stones_count += 1
                    break

    def restart(self):
        self.game_index += 1
        for player in self.players:
            player.respawn()

        if self.game_index >= self.game_number:
            self.logs_management(f"{self.game_number} games played. Exiting...")
            self.plot_scores()
            for player in self.players:
                self.logs_management(f"mean score of {player.name}: {sum(player.scores) / len(player.scores)}")
            exit(0)
        self.spawn_food()
        self.spawn_stones()
        self.logs_management(f"Starting {self.game_index}/{self.game_number}")

    def remove_player_from_board(self, player):
        for y in range(len(self.board)):
            for x in range(len(self.board)):
                if player in self.board[y][x].players:
                    self.board[y][x].players.remove(player)
                    return;

    def end_game(self):
        for player in self.players:
            player.respawn()
            self.logs_management(f"current score of {player.name}: {player.score} | mean score: {sum(player.scores) / len(player.scores)} for {player.death_counter} death")
        if self.save_logs:
            directory = f"./logs/Game {self.id}"
            if not os.path.isdir(directory):
                os.makedirs(directory)
            f = open(f"./logs/Game {self.id}/logs.txt", "w+")
            for log in self.logs:
                f.write(log)
            f.close()
            print(f"Saved logs in {directory}.")
        self.plot_scores()
        exit(0)

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
        for player in self.players:
            if player.dead is True:
                player.respawn()
            if player.food >= self.max_score:
                self.logs_management(f"{player.name} won this game with {player.score} food harvested in {player.survival_time} turns !")
                self.end_game()

    def logs_management(self, logs):
        if self.verbose:
            print(logs)
        self.logs.append(logs)

    def plot_scores(self):
        scores = []
        for player in self.players:
            color = (player.color[0] / 255, player.color[1] / 255, player.color[2] / 255)
            plt.plot(range(0, len(player.scores)), player.scores, label=player.name, color=color)
        plt.ylabel("score")
        plt.xlabel("death index")
        plt.legend(title=f"score evolution in game {self.id}")
        ax = plt.gca()
        ax.set_facecolor((0.65, 0.65, 0.65))
        if self.display_plot:
            plt.show()
        if self.save_logs:
            plt.savefig(f'./logs/Game {self.id}/figure.png')
