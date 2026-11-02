from Classes import *
import pygame
import time
import numpy as np
import sys
import argparse

"""credits: http://ai.berkeley.edu/reinforcement.html; https://github.com/gauravmittal1995/Pyman"""
"""gridName: miniGrid, xsmallGrid, smallGrid, mediumGrid, largeGrid"""


def getParams():
    parser = argparse.ArgumentParser()

    # hyper-params:
    parser.add_argument("-epsilon", action="store", type=float, default=1.0)
    parser.add_argument("-gamma", action="store", type=float, default=0.99)
    parser.add_argument("-lr", action="store", type=float, default=0.2)
    # train/test params
    parser.add_argument("-trainEpi", action="store", type=int, default=100000)
    parser.add_argument("-trainDelay", action="store", type=int, default=55)
    parser.add_argument("-testEpi", action="store", type=int, default=100)
    parser.add_argument("-testDelay", action="store", type=int, default=100)
    parser.add_argument("-phase", choices=("train", "test"), default="train")
    parser.add_argument("-checkpoint", type=str, default="pacman_checkpoint.pkl")
    # game map
    parser.add_argument("-grid", action="store", type=str, default="mediumGrid")

    params = parser.parse_args()
    return params


if __name__ == "__main__" or __name__ == "Pacman":
    pygame.init()

    params = getParams()

    game = Game(
        epsilon=params.epsilon,
        gamma=params.gamma,
        lr=params.lr,
        trainEpi=params.trainEpi,
        trainDelay=params.trainDelay,
        testEpi=params.testEpi,
        testDelay=params.testDelay,
        gridName=params.grid,
    )

    pacman = Pacman(game)
    # ghosts = [Ghost(game, "GREEN")]
    ghosts = [Ghost(game, "GREEN"), Ghost(game, "RED")]
    # ghosts = [Ghost(game, "GREEN"), Ghost(game, "RED"), Ghost(game, "PINK")]

    run = Run(game, pacman, ghosts)
    run.flow(phase=params.phase, checkpointPath=params.checkpoint)

    # a = run.MCST_Train(mode=True)

    # print(a)
