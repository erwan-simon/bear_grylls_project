#!/usr/bin/env python3

from game.Game import Game

if __name__ == '__main__':
    game = Game(board_width=50, board_height=50, display_option=False, general_turn_latency=0, highlight_turn_latency=300, number_of_games=3, food_scarcity=0.005)
    while (1):
        game.update()
