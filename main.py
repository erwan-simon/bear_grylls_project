#!/usr/bin/env python3

from game.Game import Game

if __name__ == '__main__':
    game = Game(25, 25)
    while (1):
        game.update()
