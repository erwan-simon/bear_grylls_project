#!/usr/bin/env python3

import sys
import os

from game.Game import Game
from game.Player import Player

from network.base.NetworkWrapper import NetworkWrapper
from network.rnn.NetworkWrapperWithHistory import NetworkWrapper as NetworkWrapperWithHistory

from network.base.RandomAgent import MyNetwork as random
from network.base.BasePytorch import MyNetwork as pytorch
from network.base.BasePytorch2 import MyNetwork as pytorch2
from network.rnn.RNNPytorch import MyNetwork as rnn

def find_id():
    list_dir = os.listdir("./logs/")
    id = 0
    while 1:
        same = False
        for dir in list_dir:
            if int(dir[dir.index('_') + 1:]) == id:
                id += 1
                same = True
                break
        if same is False:
            return id

def run_game(game):
    while (game.update()):
        pass

if __name__ == '__main__':
    players = []
    game = Game(id=find_id(),
                board_width=50,
                board_height=50,
                display_option=True,
                general_turn_latency=0,
                highlight_turn_latency=300,
                number_of_games=150,
                food_offset=50,
                max_number_of_stones=12,
                max_score=250,
                players=players,
                food_to_start=10,
                food_nutrition_value=25,
                save_logs=True,
                save_models=True,
                display_plot=True,
                max_turns=-1,
                debug=False)
    common_memory = []
    # players.append(Player(len(players), game, NetworkWrapper(pytorch(inputs=41, outputs=4)), name="base"))#, model="./logs/Game_1/agent_dropout 0.5.pth.tar")))
    players.append(Player(len(players), game, NetworkWrapper(random(inputs=41, outputs=4)), name="random"))
    players.append(Player(len(players), game, NetworkWrapperWithHistory(rnn(inputs=41, outputs=4), history_size=30), name="rnn"))#, model="./logs/Game_1/agent_dropout 0.5.pth.tar")))
    run_game(game)
