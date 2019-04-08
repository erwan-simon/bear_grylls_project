import sys
import termios
import tty
from game.Lib import Lib
from random import randint
from game.Square import Square
from game.Player import Player

# Set options to activate or deactivate the game view, and its speed (50 is good to see what is happening)
DISPLAY_OPTION = True
TURN_LATENCY = 0
NUMBER_OF_PLAYERS = 1
NUMBER_OF_GAMES = 600
FOOD_ON_BOARD = 5

class Game:
    def __init__(self, game_width=20, game_height=20):
        self.board = []
        self.display = DISPLAY_OPTION
        self.turn_latency = TURN_LATENCY
        self.board_width = game_width
        self.board_height = game_height
        self.game_best_score = 0
        self.game_index = 1
        self.game_number = NUMBER_OF_GAMES
        for y in range(game_height):
            self.board.append([])
            for x in range(game_width):
                self.board[-1].append(Square(y, x))
        self.players = []
        for i in range(NUMBER_OF_PLAYERS):
            self.players.append(Player(len(self.players), self))
        if (DISPLAY_OPTION):
            self.lib = Lib(self)
        self.squares_with_food = []
        for f in range(FOOD_ON_BOARD):
            self.spawn_food()
        print(f"Starting {self.game_index}th game")

    def spawn_food(self):
        while 1:
            random_x = randint(0, self.board_width - 1)
            random_y = randint(0, self.board_height - 1)
            if self.board[random_y][random_x].food is False:
                break
        self.board[random_y][random_x].food = True
        self.squares_with_food.append(self.board[random_y][random_x])

    def restart(self):
        self.game_index += 1
        best_score = 0
        best_player = None
        for player in self.players:
            if player.score >= best_score:
                best_score = player.score
                best_player = player
            player.respawn()
        if self.game_best_score < best_score:
            self.game_best_score = best_score
        if self.game_index >= NUMBER_OF_GAMES:
            print(f"{self.game_number} games played. Exiting...")
            if best_player is not None:
                best_player.agent.save_agent()
            exit(0)
        present_food = 0
        for y in range(len(self.board)):
            for x in range(len(self.board)):
                if self.board[y][x].food:
                    present_food += 1
        for f in range(present_food, FOOD_ON_BOARD):
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
        if (self.display):
            if hasattr(self, 'lib') is False:
                self.lib = Lib()
            self.lib.update()
        elif hasattr(self, 'lib') is True:
            self.lib = None

        # restart game if needed
        all_dead = True
        for player in self.players:
            all_dead = player.dead
        if all_dead is True:
            self.restart()
