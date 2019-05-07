#!/usr/bin/env python3

import sys
import os

from game.Game import Game
from game.Player import Player

from network.base.NetworkWrapper import BaseWrapper
from network.convolution.ConvolutionWrapper import ConvolutionWrapper
from network.lstm.LSTMWrapper import LSTMWrapper

from network.base.RandomAgent import MyNetwork as random
from network.base.BasePytorch import MyNetwork as pytorch
from network.convolution.ConvolutionAgent import MyNetwork as convolution
from network.base.BasePytorch2 import MyNetwork as pytorch2
from network.lstm.LSTMPytorch import MyNetwork as lstm

from network.convolution.ConvolutionWrapper import ConvolutionWrapper2
from network.convolution.ConvolutionAgent import MyNetwork2 as convolution2

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
                food_offset=150,
                max_number_of_stones=12,
                max_score=-1,
                players=players,
                food_to_start=1,
                food_nutrition_value=25,
                save_logs=True,
                save_models=True,
                display_plot=True,
                max_turns=-1,
                debug=False,
                trap_percentage=0.2)
    # inputs = vision_distance^2 + vision_distance * 2 + 2
    players.append(Player(len(players), game, BaseWrapper(random(inputs=41, outputs=4)), name="random"))
    players.append(Player(len(players), game, ConvolutionWrapper(convolution(inputs=196, outputs=4, learning_rate=0.001)), vision_distance=7, name="0.001"))
    players.append(Player(len(players), game, ConvolutionWrapper(convolution(inputs=196, outputs=4, learning_rate=0.0001)), vision_distance=7, name="0.0001"))
    run_game(game)
