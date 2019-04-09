#!/usr/bin/env python3

import sys
import os
from game.Game import Game

def find_id():
    list_dir = os.listdir("./logs/")
    id = 0
    while 1:
        same = False
        for dir in list_dir:
            if int(dir[dir.index(' '):]) == id:
                id += 1
                same = True
                break
        if same is False:
            return id

if __name__ == '__main__':
    game = Game(id=find_id(),
                board_width=50,
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
