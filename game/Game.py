from game.Lib import Lib
from random import randint, choice
from itertools import product
from game.Square import Square
from game.Player import Player
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from game.getch import _Getch

getch = _Getch()

from collections import namedtuple

class Game:
    def __init__(self, id=0,
                 board_width=50,
                 board_height=50,
                 display_option=True,
                 general_turn_latency=0,
                 highlight_turn_latency=300,
                 number_of_games=150,
                 max_score=1000,
                 food_to_start=1,
                 food_nutrition_value=25,
                 food_offset=6,
                 max_number_of_stones=12,
                 save_logs=True,
                 save_models=True,
                 verbose=True,
                 display_plot=False,
                 players=[],
                 max_turns=10000,
                 debug=False,
                 trap_percentage=0):
        # settings
        self.id = id
        self.debug = debug
        self.verbose = verbose
        self.display_plot = display_plot
        self.save_logs = save_logs
        self.save_models = save_models
        self.max_turns = max_turns
        self.logs = []
        if self.save_logs or self.save_models:
            directory = f"./logs/Game_{self.id}"
            if not os.path.isdir(directory):
                os.makedirs(directory)

        # board settings
        self.board = []
        self.trap_percentage = trap_percentage
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
        self.food_nutrition_value = food_nutrition_value
        self.food_to_start = food_to_start
        self.players = players

        # food and stones initialisation
        self.squares_with_food = []
        self.squares_with_stone = []
        self.squares_with_trap = []
        self.spawn_trap()
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
        self.turn_latency = general_turn_latency
        if (self.display_option):
            self.lib = Lib(self)
        self.logs_management(f"Game {self.id} settings : food offset = {self.food_offset}, food nutrition value = {self.food_nutrition_value}, food to start = {self.food_to_start}, max number of stones = {self.max_number_of_stones}, board width = {self.board_width}, board height = {self.board_height}")

    """
    # this is a way to make sure that food spawns regularly on the board
    # problem is that I think it could be seen as a sort of bias given that AI
    # could develop a sort of "lazy behaviour" because it knows that food will
    # spawn around after it ate one.
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
    """

    def spawn_trap(self):
        while len(self.squares_with_trap) != int(self.board_width * self.board_height * self.trap_percentage):
            if len(self.squares_with_trap) < int(self.board_width * self.board_height * self.trap_percentage):
                coo = list(product(range(0, self.board_width), range(0, self.board_height)))
                for trap in (self.squares_with_trap):
                    coo.remove((trap.y, trap.x))
                random_y, random_x = choice(coo)
                self.board[random_y][random_x].trap = True
                self.squares_with_trap.append(self.board[random_y][random_x])
            else:
                index = randint(0, len(self.squares_with_trap) - 1)
                self.board[self.squares_with_trap[index].y][self.squares_with_trap[index].x].trap = False
                self.squares_with_trap.remove(self.squares_with_trap[index])

    def spawn_food(self):
        while len(self.squares_with_food) != self.food_offset:
            if len(self.squares_with_food) < self.food_offset:
                coo = list(product(range(0, self.board_width), range(0, self.board_height)))
                for trap in (self.squares_with_trap + self.squares_with_food):
                    coo.remove((trap.y, trap.x))
                random_y, random_x = choice(coo)
                self.board[random_y][random_x].food = True
                self.squares_with_food.append(self.board[random_y][random_x])
            else:
                index = randint(0, len(self.squares_with_food) - 1)
                self.board[self.squares_with_food[index].y][self.squares_with_food[index].x].food = False
                self.squares_with_food.remove(self.squares_with_food[index])

    def spawn_stones(self):
        current_stones_count = len(self.squares_with_stone)
        for player in self.players:
            current_stones_count += player.stones
        while current_stones_count < self.max_number_of_stones:
            coo = list(product(range(0, self.board_width), range(0, self.board_height)))
            for square in (self.squares_with_trap + self.squares_with_stone):
                coo.remove((square.y, square.x))
            random_y, random_x = choice(coo)
            self.board[random_y][random_x].stone = True
            self.squares_with_stone.append(self.board[random_y][random_x])
            current_stones_count += 1

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
            self.logs_management(f"current score of {player.name}: {player.food_scores[-1]} | mean score: {sum(player.food_scores) / len(player.food_scores)} for {len(player.death_counter)} death")
            if self.save_models:
                player.agent.model.save_model(f"./logs/Game_{self.id}", player.id)
        if self.save_logs:
            f = open(f"./logs/Game_{self.id}/logs.txt", "w+")
            for log in self.logs:
                f.write(log)
            f.close()
            directory = f"./logs/Game_{self.id}"
            print(f"Saved logs in {directory}.")
            """
            f2 = open(f"./logs/Game_{self.id}/memory.txt", "w+")
            for player in self.players:
                for m in player.agent.memory:
                    f2.write(m)
            """
        self.plot_scores()

    def manage_keys(self):
        if self.lib.get_key() == 'f':
            print("Type the new max food on board:")
            try:
                new_food_offset = int(input())
            except:
                print("Please type a valid number [0-...]")
                return True
            if new_food_offset < 0:
                print("Please type a valid number [0-...]")
            else:
                self.food_offset = new_food_offset
                self.spawn_food()
        elif self.lib.get_key() == 's':
            print("Type the new turn latency:")
            try:
                new_turn_latency = int(input())
            except:
                print("Please type a valid number [0-...]")
                return True
            if new_turn_latency < 0:
                print("Please type a valid number [0-...]")
            else:
                self.turn_latency = new_turn_latency
        elif self.lib.get_key() == 'q':
            print("Are you sure you want to end game ? [y/n]")
            if input() != "y":
                return True
            self.end_game()
            return False
        elif self.lib.get_key() == 'd':
            print("Debug mode activated. You now need to press a key on terminal window to go to next turn (press d on terminal window to disable)")
            self.debug = True
        elif self.lib.get_key() == 'p':
            self.plot_scores()
        return True

    def update(self):
        """
        if len(self.players[0].scores) % 1000 == 0: # every 1000 turns, change situation
            self.current_food_offset = 10 if self.current_food_offset == 5 else 5
        """
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

        if self.max_turns != -1 and len(self.players[0].scores) >= self.max_turns:
            self.logs_management(f"Max number of turn reached ({self.max_turns} turns). Exiting...")
            self.end_game()


        if self.debug:
            key = getch()
            if key == 'd':
                self.debug = False

        # restart game if needed
        for player in self.players:
            if player.dead is True:
                player.respawn()
            if self.max_score != -1 and player.food >= self.max_score:
                self.logs_management(f"{player.name} won game {self.id} with {player.food} food harvested in {player.survival_time} turns !")
                self.end_game()
                return False
        return self.manage_keys()

    def logs_management(self, logs):
        if self.verbose:
            print(logs)
        self.logs.append(logs)

    def plot_scores(self):
        food_array = []
        total_food = []
        survival_array = []
        total_survival = []
        for player in self.players:
            food_array.append([])
            total_food.append(0)
            survival_array.append([])
            total_survival.append(0)
        for i in range(len(self.players[0].food_scores)): # number of turn in game
            for player_index in range(len(self.players)):
                total_food[player_index] += self.players[player_index].food_scores[i]
                food_array[player_index].append(total_food[player_index] / (i + 1))
                total_survival[player_index] += self.players[player_index].survival_scores[i]
                survival_array[player_index].append(total_survival[player_index] / (i + 1))
        for player in self.players:
            color = (player.color[0] / 255, player.color[1] / 255, player.color[2] / 255)
            plt.plot(range(0, len(player.food_scores)), food_array[self.players.index(player)], label=player.name, color=color)
        plt.ylabel("mean food amount")
        plt.xlabel("turn index")
        plt.legend(title=f"food evolution in game {self.id}")
        ax = plt.gca()
        fig1 = plt.gcf()
        ax.set_facecolor((0.65, 0.65, 0.65))
        if self.display_plot:
            plt.show()
        if self.save_logs:
            # plt.figure(figsize=(655,655))
            fig1.savefig(f'./logs/Game_{self.id}/food_figure.png')

        plt.clf()
        for player in self.players:
            color = (player.color[0] / 255, player.color[1] / 255, player.color[2] / 255)
            plt.plot(range(0, len(player.survival_scores)), survival_array[self.players.index(player)], label=player.name, color=color)
        plt.ylabel("mean survival time")
        plt.xlabel("turn index")
        plt.legend(title=f"mean survival time evolution in game {self.id}")
        ax = plt.gca()
        ax.set_facecolor((0.65, 0.65, 0.65))
        fig2 = plt.gcf()
        if self.display_plot:
            plt.show()
        if self.save_logs:
            # plt.figure(figsize=(655,655))
            fig2.savefig(f'./logs/Game_{self.id}/survival_figure.png')

        plt.clf()
        for player in self.players:
            color = (player.color[0] / 255, player.color[1] / 255, player.color[2] / 255)
            plt.plot(range(0, len(player.stones_scores)), player.stones_scores, label=player.name, color=color)
        plt.ylabel("stone number")
        plt.xlabel("turn index")
        plt.legend(title=f"stone possessing evolution in game {self.id}")
        ax = plt.gca()
        ax.set_facecolor((0.65, 0.65, 0.65))
        fig3 = plt.gcf()
        if self.display_plot:
            plt.show()
        if self.save_logs:
            # plt.figure(figsize=(655,655))
            fig3.savefig(f'./logs/Game_{self.id}/stone_figure.png')
