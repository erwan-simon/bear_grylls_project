#!/usr/bin/env python3

import sys
from game.Game import Game

if __name__ == '__main__':
    game = Game(board_width=50,
    board_height=50,
    display_option=False,
    general_turn_latency=0,
    highlight_turn_latency=300,
    number_of_games=150,
    food_offset=8,
    max_number_of_stones=12,
    max_score=1000)
    while (1):
        game.update()
