import pygame
from pygame.locals import *
import math
import os
import pickle
from copy import deepcopy
import time
import numpy as np
from random import randint
from collections import Counter, deque
import matplotlib.pyplot as plt

PAC_YELLOW = (255, 220, 111)


def draw_pacman(surf, cx, cy, r, direction_deg, mouth_deg, glow):
    if glow:
        surf.blit(glow, (cx - glow.get_width() // 2, cy - glow.get_height() // 2))
    pts = [(cx, cy)]
    start, end = direction_deg + mouth_deg, direction_deg + 360 - mouth_deg
    n = 26
    for i in range(n + 1):
        a = math.radians(start + (end - start) * i / n)
        pts.append((round(cx + r * math.cos(a)), round(cy - r * math.sin(a))))
    pygame.draw.polygon(surf, PAC_YELLOW, pts)


def draw_ghost(surf, cx, cy, r, color, look, t, glow=None):
    if glow:
        surf.blit(glow, (cx - glow.get_width() // 2, cy - glow.get_height() // 2))
    pts = []
    for i in range(21):
        a = math.radians(180 - 180 * i / 20)
        pts.append((round(cx + r * math.cos(a)), round(cy - r * math.sin(a))))
    base, teeth = cy + r * 0.92, 4
    wob = math.sin(t * 9) * r * 0.06
    for i in range(teeth * 2 + 1):
        x = cx + r - (2 * r) * i / (teeth * 2)
        up = i % 2 == 1
        pts.append(
            (round(x), round(base - (r * 0.32 if up else 0) + (wob if up else -wob)))
        )
    pygame.draw.polygon(surf, color, pts)
    for side in (-1, 1):
        ex, ey = cx + side * r * 0.36, cy - r * 0.1
        pygame.draw.circle(
            surf, (255, 255, 255), (round(ex), round(ey)), round(r * 0.3)
        )
        pygame.draw.circle(
            surf,
            (30, 30, 110),
            (round(ex + look[0] * r * 0.12), round(ey + look[1] * r * 0.12)),
            round(r * 0.15),
        )


"""
QL: 
(state, action, reward, nextstate)

"""


class Game:

    def __init__(
        self,
        epsilon,
        gamma,
        lr,
        trainEpi,
        trainDelay,
        testEpi,
        testDelay,
        gridName,
    ):
        # e-greedy, discount factor, learning rate
        self.epsilon, self.gamma, self.lr = epsilon, gamma, lr
        # training and testing episodes
        self.trainEpi, self.testEpi = trainEpi, testEpi
        self.trainDelay = trainDelay
        self.testDelay = testDelay

        # game knowledge
        self.grid, self.gridName = [], gridName
        self.oldScore = 0
        self.score = 0
        self.episode = 1
        self.foodGrid = []
        self.walls = []
        self.paths = []
        self.won = self.lost = False

        self.maxSteps = 500
        self.currentSteps = 0

        # display settings
        self.BLACK = (12, 9, 20)
        self.BACKGROUND = (12, 9, 20)
        self.SURFACE = (25, 19, 39)
        self.SURFACE_LIGHT = (35, 27, 53)
        self.WALL = (104, 70, 176)
        self.WALL_HIGHLIGHT = (169, 127, 238)
        self.TEXT = (246, 240, 255)
        self.MUTED = (158, 145, 180)
        self.YELLOW = (255, 220, 111)
        self.GREY = (24, 18, 38)
        self.RED = (255, 112, 145)
        self.BLUE = self.WALL
        self.WHITE = (239, 223, 255)
        self.GREEN = (112, 230, 177)
        self.PINK = (255, 163, 215)
        self.font = pygame.font.SysFont("segoeui", 25, bold=True)
        self.headingFont = pygame.font.SysFont("segoeui", 30, bold=True)
        self.valueFont = pygame.font.SysFont("segoeui", 27, bold=True)
        self.smallFont = pygame.font.SysFont("segoeui", 16)
        self.labelFont = pygame.font.SysFont("segoeui", 13, bold=True)
        # the game board
        self.size = [940, 540]
        self.width = 22
        self.height = 22
        self.margin = 1
        self.create()
        self.boardWidth = len(self.grid[0]) * (self.width + self.margin)
        self.boardHeight = len(self.grid) * (self.height + self.margin)
        self.boardX = 28 + (520 - self.boardWidth) // 2
        self.boardY = 116 + (330 - self.boardHeight) // 2
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("PACMAN  |  Purple Maze")

    def create(self):
        miniGrid = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
        xsmallGrid = [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ]
        smallGrid = [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]
        mediumGrid = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 2, 2, 0, 2, 2, 0, 2, 1],
            [1, 2, 1, 2, 1, 0, 1, 1, 0, 1],
            [1, 2, 1, 2, 1, 2, 0, 2, 2, 1],
            [1, 0, 1, 2, 1, 1, 1, 1, 2, 1],
            [1, 0, 2, 0, 2, 1, 2, 2, 0, 1],
            [1, 2, 1, 0, 2, 0, 2, 1, 2, 1],
            [1, 0, 1, 2, 1, 1, 2, 1, 2, 1],
            [1, 0, 0, 2, 2, 2, 2, 2, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        largeGrid = [
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                1,
            ],
            [
                1,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                1,
                1,
                1,
                0,
                1,
            ],
            [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
            ],
            [
                1,
                0,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                2,
                1,
                0,
                0,
                0,
                1,
                1,
                0,
                1,
            ],
            [
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                1,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                0,
                1,
            ],
            [
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            [
                1,
                0,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                0,
                1,
            ],
            [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        ]
        gridList = {
            "miniGrid": miniGrid,
            "xsmallGrid": xsmallGrid,
            "smallGrid": smallGrid,
            "mediumGrid": mediumGrid,
            "largeGrid": largeGrid,
        }
        self.grid = gridList[self.gridName]

        # grid[y, x]
        yA = len(self.grid)
        xA = len(self.grid[0])
        self.foodGrid = np.zeros((yA, xA))
        for y in range(yA):
            for x in range(xA):
                if self.grid[y][x] == 1:
                    # meaning it's a wall and insert it into wall list
                    self.walls.append((x, y))  # (x, y)
                elif self.grid[y][x] == 0:
                    # food
                    self.foodGrid[y, x] = 1
                    self.paths.append((x, y))

        # special simple case: only 1 food
        if self.gridName == "xsmallGrid":
            self.foodGrid = np.zeros((yA, xA))
            self.foodGrid[int(yA / 2), int(xA / 2)] = 1
        return self

    def reset(self):
        self.grid = []
        self.walls = []
        self.score = 0
        self.oldScore = 0
        self.currentSteps = 0
        self.won = self.lost = False
        self.create()
        return self

    def timeScore(self):
        self.score -= 1

    def scoreShow(self):
        label = self.labelFont.render("CURRENT SCORE", True, self.MUTED)
        value = self.valueFont.render(str(self.score), True, self.TEXT)
        self.screen.blit(label, (607, 158))
        self.screen.blit(value, (607, 181))

    def episodeShow(self):
        episodeText = self.font.render(str(self.episode), True, self.TEXT)
        label = self.labelFont.render("EPISODE", True, self.MUTED)
        self.screen.blit(label, (607, 261))
        self.screen.blit(episodeText, (607, 283))

    """show training or testing"""

    def phaseShow(self, state):
        stateText = self.font.render(state, True, self.TEXT)
        label = self.labelFont.render("PHASE", True, self.MUTED)
        self.screen.blit(label, (757, 261))
        self.screen.blit(stateText, (757, 283))

    def speedShow(self, phase="Training"):
        isTraining = phase == "Training"
        delay = self.trainDelay if isTraining else self.testDelay
        label = self.labelFont.render(phase.upper() + " SPEED", True, self.MUTED)
        value = self.smallFont.render(str(delay) + " ms / frame", True, self.TEXT)
        controls = self.smallFont.render(
            "- slower   + faster   Q quit", True, self.MUTED
        )
        self.screen.blit(label, (52, 480))
        self.screen.blit(value, (177, 478))
        self.screen.blit(controls, (355, 480))

    def tileCenter(self, x, y):
        return (
            self.boardX
            + (self.margin + self.width) * x
            + self.margin
            + self.width // 2,
            self.boardY
            + (self.margin + self.height) * y
            + self.margin
            + self.height // 2,
        )

    # update map and food
    def mapShow(self):
        self.screen.fill(self.BACKGROUND)
        pygame.draw.rect(self.screen, self.SURFACE, (28, 22, 884, 76), border_radius=18)
        pygame.draw.circle(self.screen, self.WALL_HIGHLIGHT, (58, 60), 15)
        pygame.draw.circle(self.screen, self.SURFACE, (58, 60), 7)
        title = self.headingFont.render("PACMAN", True, self.TEXT)
        subtitle = self.smallFont.render(
            "A smarter way through the maze", True, self.MUTED
        )
        self.screen.blit(title, (86, 35))
        self.screen.blit(subtitle, (88, 66))

        pygame.draw.rect(
            self.screen, self.SURFACE, (28, 116, 520, 330), border_radius=18
        )
        pygame.draw.rect(
            self.screen, self.SURFACE, (568, 116, 344, 330), border_radius=18
        )
        pygame.draw.rect(
            self.screen, self.SURFACE_LIGHT, (592, 143, 296, 80), border_radius=14
        )
        pygame.draw.rect(
            self.screen, self.SURFACE_LIGHT, (592, 239, 140, 78), border_radius=14
        )
        pygame.draw.rect(
            self.screen, self.SURFACE_LIGHT, (744, 239, 144, 78), border_radius=14
        )
        pygame.draw.rect(
            self.screen, self.SURFACE_LIGHT, (592, 333, 296, 88), border_radius=14
        )
        mazeLabel = self.labelFont.render("THE MAZE", True, self.MUTED)
        algorithmLabel = self.labelFont.render("ALGORITHM", True, self.MUTED)
        gridLabel = self.labelFont.render("MAP", True, self.MUTED)
        self.screen.blit(mazeLabel, (52, 126))
        self.screen.blit(algorithmLabel, (607, 344))
        self.screen.blit(gridLabel, (757, 344))
        algorithmValue = self.smallFont.render("Q-Learning", True, self.TEXT)
        gridValue = self.smallFont.render(self.gridName, True, self.TEXT)
        self.screen.blit(algorithmValue, (607, 367))
        self.screen.blit(gridValue, (757, 367))

        pygame.draw.rect(
            self.screen,
            (16, 12, 27),
            (
                self.boardX - 7,
                self.boardY - 7,
                self.boardWidth + 14,
                self.boardHeight + 14,
            ),
            border_radius=12,
        )
        pygame.draw.rect(
            self.screen,
            self.WALL_HIGHLIGHT,
            (
                self.boardX - 7,
                self.boardY - 7,
                self.boardWidth + 14,
                self.boardHeight + 14,
            ),
            width=1,
            border_radius=12,
        )
        xA = len(self.grid[0])
        yA = len(self.grid)
        for x in range(xA):
            for y in range(yA):
                tileX = self.boardX + (self.margin + self.width) * x + self.margin
                tileY = self.boardY + (self.margin + self.height) * y + self.margin
                pygame.draw.rect(
                    self.screen,
                    self.GREY,
                    (tileX, tileY, self.width, self.height),
                    border_radius=5,
                )
                if self.foodGrid[y][x] == 1:
                    pygame.draw.circle(
                        self.screen, self.WHITE, self.tileCenter(x, y), 2
                    )
        for wall in self.walls:
            tileX = self.boardX + (self.margin + self.width) * wall[0] + self.margin
            tileY = self.boardY + (self.margin + self.height) * wall[1] + self.margin
            pygame.draw.rect(
                self.screen,
                self.BLUE,
                (tileX, tileY, self.width, self.height),
                border_radius=5,
            )
            pygame.draw.rect(
                self.screen,
                self.WALL_HIGHLIGHT,
                (tileX + 2, tileY + 2, self.width - 4, self.height - 4),
                width=1,
                border_radius=4,
            )
        pygame.draw.rect(
            self.screen, self.SURFACE, (28, 464, 884, 52), border_radius=16
        )

    """check if episode ends and change lost/won"""

    def terminal(self, pacman, ghosts):
        if self.foodGrid.sum() == 0:
            self.won = True
            return True

        for ghost in ghosts:
            if (pacman.x, pacman.y) == (ghost.x, ghost.y):
                self.lost = True
                return True

        if self.currentSteps >= self.maxSteps:
            self.lost = True
            return True

        return False

    """when terminal, update score and print message"""

    def terScore(self):
        if self.lost:
            self.score -= 300
            # print((str)(self.episode) + " Lost! Final Score = " + (str)(self.score))
        elif self.won:
            self.score += 300
            # print((str)(self.episode) + " Won! Final Score = " + (str)(self.score))

    def terminal_BU(self):
        if self.lost:
            # print((str)(self.episode) + " Lost! Final Score = " + (str)(self.score))
            return True
        elif self.won:
            print((str)(self.episode) + " Won! Final Score = " + (str)(self.score))
            return True
        return False


class Agent:

    def __init__(self, x, y, game):
        self.x = x
        self.y = y
        self.previousX = x
        self.previousY = y
        self.game = game

    def isWall(self, x, y):
        if (x, y) in self.game.walls:
            return True
        else:
            return False

    def getPosition(self, x, y, action):
        if action != None:
            if action == 0:
                return (x - 1, y)
            elif action == 1:
                return (x + 1, y)
            elif action == 2:
                return (x, y - 1)
            elif action == 3:
                return (x, y + 1)
        return (x, y)

    def getLegalActions(self, x, y):
        """0, 1, 2, 3 for left, right, up, down"""
        list = []
        if self.isWall(x - 1, y) == False:
            list.append(0)
        if self.isWall(x + 1, y) == False:
            list.append(1)
        if self.isWall(x, y - 1) == False:
            list.append(2)
        if self.isWall(x, y + 1) == False:
            list.append(3)
        return list

    def getLegalNeighbors(self, x, y):
        list = []
        if self.isWall(x - 1, y) == False:
            list.append((x - 1, y))
        if self.isWall(x + 1, y) == False:
            list.append((x + 1, y))
        if self.isWall(x, y - 1) == False:
            list.append((x, y - 1))
        if self.isWall(x, y + 1) == False:
            list.append((x, y + 1))
        return list

    def takeLegalAction(self, action):
        if action != None:
            if action == 0:
                self.x = self.x - 1
            if action == 1:
                self.x = self.x + 1
            if action == 2:
                self.y = self.y - 1
            if action == 3:
                self.y = self.y + 1
        # return self

    def getDistance(self, xs, ys, xd, yd):
        return abs(xs - xd) + abs(ys - yd)


class Ghost(Agent):

    def __init__(self, game, color):
        self.game = game
        self.action = randint(0, 3)
        self.color = color
        grid = game.grid
        xA = len(grid[0])
        yA = len(grid)
        # spawn at center of the map
        x = (int)(xA / 2)
        y = (int)(yA / 2)
        if game.gridName == "miniGrid":
            x, y = xA - 2, yA - 2
        Agent.__init__(self, x, y, game)

    def reset(self):
        grid = self.game.grid
        xA = len(grid[0])
        yA = len(grid)
        # spawn at center of the map
        x = (int)(xA / 2)
        y = (int)(yA / 2)
        if self.game.gridName == "miniGrid":
            x, y = xA - 2, yA - 2
        Agent.__init__(self, x, y, self.game)

    def show(self, progress=1):
        game = self.game
        color = game.RED
        if self.color == "GREEN":
            color = game.GREEN
        if self.color == "PINK":
            color = game.PINK
        dx = self.x - self.previousX
        dy = self.y - self.previousY
        if dx and dy:
            if progress < 0.5:
                x = self.previousX + dx * progress * 2
                y = self.previousY
            else:
                x = self.x
                y = self.previousY + dy * (progress - 0.5) * 2
        else:
            x = self.previousX + dx * progress
            y = self.previousY + dy * progress
        center = game.tileCenter(x, y)
        look = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}.get(self.action, (1, 0))
        draw_ghost(
            game.screen,
            center[0],
            center[1],
            10,
            color,
            look,
            pygame.time.get_ticks() / 1000,
        )

    def getReverseAction(self, action):
        if action == 0 or action == 1:
            return 1 - action
        if action == 2 or action == 3:
            return 5 - action

    def move(self):
        legalActions = self.getLegalActions(self.x, self.y)
        if not legalActions:
            return

        if len(legalActions) == 1:
            self.action = legalActions[0]
        elif (
            len(legalActions) > 2
            or len(legalActions) <= 2
            and self.action not in legalActions
        ):
            """remove backward action"""
            rev = self.getReverseAction(self.action)
            options = [action for action in legalActions if action != rev]
            self.action = options[randint(0, len(options) - 1)]
            # otherwise same action

        if self.action not in legalActions:
            self.action = legalActions[randint(0, len(legalActions) - 1)]

        nextX, nextY = self.getPosition(self.x, self.y, self.action)
        if abs(nextX - self.x) + abs(nextY - self.y) == 1:
            self.x, self.y = nextX, nextY

    def getPossibleNextPositions(self):
        legalActions = self.getLegalActions(self.x, self.y)
        if not legalActions:
            return [(self.x, self.y)]

        if len(legalActions) == 1:
            possibleActions = legalActions
        elif len(legalActions) > 2 or self.action not in legalActions:
            reverseAction = self.getReverseAction(self.action)
            possibleActions = [
                action for action in legalActions if action != reverseAction
            ]
        else:
            possibleActions = [self.action]

        return [self.getPosition(self.x, self.y, action) for action in possibleActions]


class Pacman(Agent):

    def __init__(self, game):
        self.game = game

        self.epsilon = game.epsilon
        self.gamma = game.gamma
        self.initialFoodCount = int(game.foodGrid.sum())
        self.lr = game.lr
        self.reward = 0
        self.Q = Counter()
        self.action = 1

        x, y = 1, 1
        Agent.__init__(self, x, y, game)

    def reset(self):
        paths = self.game.paths
        x, y = 1, 1
        # if self.game.gridName == "largeGrid":
        # #     index = randint(0, len(paths)-1)
        # #     x, y = paths[index]
        # #
        # # elif self.game.gridName == "miniGrid" or self.game.gridName == "xsmallGrid":
        # #         x, y = 1, 1
        #
        # # else:
        #     grid = self.game.grid
        #     xA = len(grid[0])
        #     yA = len(grid)
        #     start = [(1, 1),
        #              (1, yA - 2),
        #              (xA - 2, 1),
        #              (xA - 2, yA - 2)]
        #     index = randint(0, len(start) - 1)
        #     x, y = start[index]

        self.action = 1
        Agent.__init__(self, x, y, self.game)

    def takeLegalAction(self, action):
        if action is None:
            return
        old_position = (self.x, self.y)
        Agent.takeLegalAction(self, action)
        if (self.x, self.y) != old_position:
            self.action = action

    """transform list of lists to tuple of strings"""

    def getTS_String(self, input):
        out = []
        for row in input:
            out.append([])
            for item in row:
                if item == 1:
                    out[-1].append("1")
                else:
                    out[-1].append("0")
            out[-1] = "".join(out[-1])
        out = tuple(out)
        return out

    """transform to tuples of tuple"""

    def getTS(self, input):
        out = []
        for row in input:
            out.append([])
            for item in row:
                if item == 1:
                    out[-1].append(1)
                else:
                    out[-1].append(0)
            out[-1] = tuple(out[-1])
        out = tuple(out)
        return out

    """Build a compact state from Pacman, nearest targets, and safe exits."""

    def getNearestTargetFeatures(self, targets):
        start = (self.x, self.y)
        if start in targets:
            return None, 0

        queue = deque([(start[0], start[1], None, 0)])
        visited = {start}
        while queue:
            x, y, firstAction, distance = queue.popleft()
            for action in self.getLegalActions(x, y):
                position = self.getPosition(x, y, action)
                if position in visited:
                    continue
                nextAction = action if firstAction is None else firstAction
                nextDistance = distance + 1
                if position in targets:
                    return nextAction, nextDistance
                visited.add(position)
                queue.append((position[0], position[1], nextAction, nextDistance))
        return None, None

    def getState(self, ghosts):
        foodY, foodX = np.nonzero(self.game.foodGrid)
        foodTargets = set(zip(foodX.tolist(), foodY.tolist()))
        foodDirection, foodDistance = self.getNearestTargetFeatures(foodTargets)

        ghostTargets = {(ghost.x, ghost.y) for ghost in ghosts}
        ghostDirection, ghostDistance = self.getNearestTargetFeatures(ghostTargets)

        legalActions = self.getLegalActions(self.x, self.y)
        dangerousPositions = self.getDangerousPositions(ghosts)
        escapeRoutes = sum(
            self.getPosition(self.x, self.y, action) not in dangerousPositions
            for action in legalActions
        )

        return (
            (self.x, self.y),
            foodDirection,
            foodDistance,
            ghostDirection,
            ghostDistance,
            escapeRoutes,
        )

    def hasFood(self, x, y):
        if self.game.foodGrid[y][x] == 1:
            return True
        else:
            return False

    def eatFood(self, x, y):
        self.game.foodGrid[y][x] = 0
        remainingFood = self.initialFoodCount - int(self.game.foodGrid.sum())
        self.game.score += 10 + 5 * remainingFood

    def hasGhost(self, x, y, ghosts):
        for ghost in ghosts:
            if (ghost.x, ghost.y) == (x, y):
                return True
        return False

    def show(self, progress=1):
        game = self.game
        x = self.previousX + (self.x - self.previousX) * progress
        y = self.previousY + (self.y - self.previousY) * progress
        center = game.tileCenter(x, y)
        direction_deg = {0: 180, 1: 0, 2: 90, 3: 270}.get(self.action, 0)
        animation = pygame.time.get_ticks() * 0.006
        mouth_deg = 4 + 28 * (0.5 - 0.5 * math.cos(animation))
        draw_pacman(
            game.screen, center[0], center[1], 10, direction_deg, mouth_deg, None
        )

    def getReward(self):
        diff = self.game.score - self.game.oldScore
        if self.game.won:
            self.reward = 300
        elif self.game.lost:
            self.reward = -300
        elif diff > 0:  # eat food
            self.reward = diff
        else:  # elapsed time
            self.reward = -1

        self.game.oldScore = self.game.score

    def chooseAction(self, actions, values):
        if not actions:
            raise ValueError("No legal actions available.")

        if np.random.random() < self.epsilon:
            return np.random.choice(actions)

        bestValue = max(values)

        bestActions = [
            action
            for action, value in zip(actions, values)
            if np.isclose(value, bestValue)
        ]

        return np.random.choice(bestActions)

    def getDangerousPositions(self, ghosts):
        dangerousPositions = set()
        for ghost in ghosts:
            dangerousPositions.add((ghost.x, ghost.y))
            dangerousPositions.update(ghost.getPossibleNextPositions())
        return dangerousPositions

    def getSafeActions(self, actions, ghosts):
        dangerousPositions = self.getDangerousPositions(ghosts)
        safeActions = [
            action
            for action in actions
            if self.getPosition(self.x, self.y, action) not in dangerousPositions
        ]
        return safeActions or actions

    """____________Pacman Ctrl Algorithms____________"""

    def QLearning(self, ghosts):
        self.game.currentSteps += 1
        s = self.getState(ghosts)

        actions = self.getLegalActions(self.x, self.y)
        safeActions = self.getSafeActions(actions, ghosts)
        a = self.chooseAction(
            safeActions, [self.Q[(s, action)] for action in safeActions]
        )

        self.takeLegalAction(a)
        if self.hasFood(self.x, self.y):
            self.eatFood(self.x, self.y)

        ter = self.game.terminal(self, ghosts)  # walked into ghost, or won
        if not ter:
            for g in ghosts:
                g.move()
            ter = self.game.terminal(self, ghosts)  # ghost walked into Pacman
        if ter:
            self.game.terScore()
        self.getReward()

        nextValue = 0
        if not ter:
            sn = self.getState(ghosts)
            nextActions = self.getLegalActions(self.x, self.y)
            nextActions = self.getSafeActions(nextActions, ghosts)
            nextValue = max(self.Q[(sn, action)] for action in nextActions)

        self.Q[(s, a)] += self.lr * (
            self.reward + self.gamma * nextValue - self.Q[(s, a)]
        )
        return ter


class Run:

    def __init__(self, game, pacman, ghosts):
        self.trainResults = []
        self.testResults = []
        self.results = []
        self.winRate = []
        self.scores = []
        self.game = game
        self.pacman = pacman
        self.ghosts = ghosts
        self.initialEpsilon = pacman.epsilon
        self.checkpointPath = "pacman_checkpoint.pkl"
        self.saveButtonRect = pygame.Rect(770, 472, 120, 34)

    def updateEpsilon(self, completedEpisode):
        epsilonFloor = min(0.05, self.initialEpsilon)
        decayEpisodes = max(1, int(self.game.trainEpi * 0.8))
        progress = min(1.0, completedEpisode / decayEpisodes)
        self.pacman.epsilon = (
            self.initialEpsilon + (epsilonFloor - self.initialEpsilon) * progress
        )

    def startStep(self):
        for agent in [self.pacman] + self.ghosts:
            agent.previousX = agent.x
            agent.previousY = agent.y

    def renderFrame(self, progress, phase):
        self.game.mapShow()
        self.pacman.show(progress)
        for ghost in self.ghosts:
            ghost.show(progress)
        self.game.scoreShow()
        self.game.episodeShow()
        self.game.phaseShow(phase)
        self.game.speedShow(phase)
        if phase == "Training":
            pygame.draw.rect(
                self.game.screen,
                self.game.WALL,
                self.saveButtonRect,
                border_radius=10,
            )
            saveText = self.game.smallFont.render("SAVE Q TABLE", True, self.game.TEXT)
            saveRect = saveText.get_rect(center=self.saveButtonRect.center)
            self.game.screen.blit(saveText, saveRect)

    def animateStep(self, duration, phase):
        duration = max(0, duration)
        start = pygame.time.get_ticks()
        clock = pygame.time.Clock()

        while True:
            elapsed = pygame.time.get_ticks() - start
            progress = 1 if duration == 0 else min(1, elapsed / duration)
            self.renderFrame(progress, phase)
            pygame.display.flip()
            if progress >= 1:
                break
            clock.tick(60)

    def saveTrainingCheckpoint(self, completedEpisode):
        checkpoint = {
            "episode": completedEpisode,
            "gridName": self.game.gridName,
            "stateFormat": 2,
            "epsilon": self.pacman.epsilon,
            "Q": self.pacman.Q,
            "trainResults": self.trainResults,
            "scores": self.scores,
        }
        temporaryPath = self.checkpointPath + ".tmp"
        with open(temporaryPath, "wb") as checkpointFile:
            pickle.dump(checkpoint, checkpointFile)
        os.replace(temporaryPath, self.checkpointPath)

    def loadTrainingCheckpoint(self, checkpointPath):
        with open(checkpointPath, "rb") as checkpointFile:
            checkpoint = pickle.load(checkpointFile)

        if not isinstance(checkpoint, dict) or "Q" not in checkpoint:
            raise ValueError(f"Invalid training checkpoint: {checkpointPath}")
        if checkpoint.get("stateFormat") != 2:
            raise ValueError(
                f"Checkpoint {checkpointPath} uses an incompatible Q-state format. "
                "Train a new Q table for the current state representation."
            )
        checkpointGrid = checkpoint.get("gridName")
        if checkpointGrid is not None and checkpointGrid != self.game.gridName:
            raise ValueError(
                f"Checkpoint grid {checkpointGrid!r} does not match "
                f"selected grid {self.game.gridName!r}."
            )
        self.pacman.Q = checkpoint["Q"]
        print(
            f"Loaded episode {checkpoint.get('episode', 'unknown')} "
            f"checkpoint from {checkpointPath}"
        )

    def train(self):
        self.startStep()

        # time penalty
        self.game.timeScore()

        # agents action
        # for ghost in self.ghosts:
        #     # ghost.move()
        #     ghost.move()
        return self.pacman.QLearning(self.ghosts)

    def test(self):
        self.pacman.epsilon = 0
        self.pacman.lr = 0

        self.startStep()

        # time penalty
        self.game.timeScore()

        # agents actions
        # for ghost in self.ghosts:
        #     ghost.move()
        return self.pacman.QLearning(self.ghosts)

    def final(self, mode):
        print("_________Conclusion___________")
        if mode == "train":
            won = self.trainResults.count("won")
            total = self.game.trainEpi
            print(
                "Winrate: "
                + str(won)
                + "/"
                + str(total)
                + "="
                + str(won / total * 100)
                + "%"
                + ", Average Score: "
                + str(sum(self.scores) / total)
            )
        else:
            won = self.testResults.count("won")
            total = self.game.testEpi
            print(
                "Winrate: "
                + str(won)
                + "/"
                + str(total)
                + "="
                + str(won / total * 100)
                + "%"
                + ", Average Score: "
                + str(sum(self.scores) / total)
            )

        # print((str)(self.results.count("won")) + "/" + (str)(self.game.trainEpi))
        return self.winRate

    def checkDone(self, training=False):
        done = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == KEYDOWN:
                if event.key == K_q:
                    done = True
                elif event.key in (K_MINUS, K_KP_MINUS):
                    delayName = "trainDelay" if training else "testDelay"
                    delay = getattr(self.game, delayName)
                    setattr(self.game, delayName, min(1000, delay + 5))
                elif event.key in (K_PLUS, K_EQUALS, K_KP_PLUS):
                    delayName = "trainDelay" if training else "testDelay"
                    delay = getattr(self.game, delayName)
                    setattr(self.game, delayName, max(0, delay - 5))
            if (
                training
                and event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.saveButtonRect.collidepoint(event.pos)
            ):
                completedEpisodes = len(self.trainResults)
                self.saveTrainingCheckpoint(completedEpisodes)
                print(
                    f"Saved episode {completedEpisodes} checkpoint to "
                    f"{self.checkpointPath}"
                )
        return done

    def loopTrain(self):
        while self.game.episode <= self.game.trainEpi:
            done = self.checkDone(training=True)
            epiEnd = self.train()
            self.animateStep(self.game.trainDelay, "Training")

            if epiEnd:
                if self.game.lost:
                    self.trainResults.append("lost")
                elif self.game.won:
                    self.trainResults.append("won")
                completedEpisode = self.game.episode
                self.winRate.append(self.trainResults.count("won") / completedEpisode)
                self.scores.append(self.game.score)
                self.updateEpsilon(completedEpisode)

                if completedEpisode % 100 == 0:
                    lastHundred = self.trainResults[-100:]
                    wins = lastHundred.count("won")
                    print(
                        f"Episodes {completedEpisode - 99}-{completedEpisode}: "
                        f"win rate {wins}/100 ({wins}%), "
                        f"epsilon {self.pacman.epsilon:.3f}"
                    )
                self.game.episode += 1
                self.game.reset()
                self.pacman.reset()
                for ghost in self.ghosts:
                    ghost.reset()

            """quit"""
            if done:
                return done

    def loopTest(self):
        while self.game.episode <= self.game.testEpi:
            done = self.checkDone()
            epiEnd = self.test()
            self.animateStep(self.game.testDelay, "Testing")

            if epiEnd:
                if self.game.lost:
                    self.testResults.append("lost")
                elif self.game.won:
                    self.testResults.append("won")
                self.winRate.append(self.testResults.count("won") / self.game.episode)
                self.scores.append(self.game.score)

                self.game.episode += 1
                self.game.reset()
                self.pacman.reset()
                for ghost in self.ghosts:
                    ghost.reset()

            """quit"""
            if done:
                return done

    def plotTwo(self, trainWR, testWR, labels):
        trainL = len(trainWR)
        testL = len(testWR)
        xTrain = np.arange(1, trainL + 1)
        xTest = np.arange(1, testL + 1)
        graph = []
        (gTrain,) = plt.plot(xTrain, trainWR, label=labels[0])
        (gTest,) = plt.plot(xTest, testWR, label=labels[1])
        graph.append(gTrain)
        graph.append(gTest)
        # for arg in args, lab in labels:
        #     g, = plt.plot(arg, label=lab)
        #     graph.append(g)
        plt.title("Winrate against Episode")
        plt.xlabel("Episode")
        plt.ylabel("Winrate")
        plt.legend(graph, labels)
        plt.show()

    def plot(self, WR, label):
        l = len(WR)
        xl = np.arange(1, l + 1)
        (g,) = plt.plot(xl, WR, label=label)
        graph = [g]
        plt.title("Winrate against Episode")
        plt.xlabel("Episode")
        plt.ylabel("Winrate")
        plt.legend(graph, label)
        plt.show()

    def flow(self, phase="train", checkpointPath="pacman_checkpoint.pkl"):
        self.checkpointPath = checkpointPath
        if phase == "train":
            print("_____________TRAINING_______________")
            try:
                done = self.loopTrain()
                if not done:
                    self.final("train")
            finally:
                pygame.quit()
            return

        self.loadTrainingCheckpoint(checkpointPath)
        self.game.episode = 1
        print("_____________TESTING_______________")
        done = self.loopTest()
        if not done:
            self.final("test")
        pygame.quit()

    """_______________MCTS______________"""

    def initialState(self):

        self.game.reset()
        self.pacman.reset()
        ghost = self.ghosts[0]
        ghost.reset()

        state = self.pacman.getState(self.ghosts)

        # grid =   [[1, 1, 1, 1, 1, 1],
        #           [1, 0, 0, 0, 0, 1],
        #           [1, 0, 1, 1, 0, 1],
        #           [1, 0, 1, 3, 0, 1],
        #           [1, 0, 1, 1, 0, 1],
        #           [1, 0, 0, 0, 0, 1],
        #           [1, 1, 1, 1, 1, 1]]
        # xA, yA = len(grid[0]), len(grid)
        #
        # px, py = 1, 1
        # gx, gy = xA-2, yA-2
        #
        # pac = np.zeros((yA, xA))
        # gho = np.zeros((yA, xA))
        # food = np.zeros((yA, xA))
        #
        # pac[py, px] = 1
        # gho[gy, gx] = 1
        # food[yA//2, xA//2] = 1
        #
        # pacS = self.pacman.getTS(pac)
        # ghoS = self.pacman.getTS(gho)
        # foodS = self.pacman.getTS(food)
        # wallS = self.pacman.getTS(grid)
        #
        # state = (pacS, ghoS, foodS, wallS)

        return State(state)

    def MCST_Train(self, mode):
        # node = self.loopTrain_MCTS()
        # print(node.ValueAll)
        iniS = self.initialState()
        root = TreeNode(iniS, None, None)
        trainEpi = self.game.trainEpi

        for t in range(trainEpi):
            t += 1
            node = root
            S = iniS.clone()
            visited = [root]
            visitedS = [S]
            print("________________________________iteration: ", t)

            # while not terminal
            while S.getReward() == 0:
                print(0)
                pac = S.state[0]
                # for row in pac:
                #     print(row)
                # print("________________")
                # if node.S in visitedS:
                #     break

                # select(fully expanded and non terminal)
                if node.untriedMoves == []:

                    node = node.choose()
                    S.act(node.action)
                    visited.append(node)
                    print(1)

                # expand
                elif node.S not in visitedS:
                    a = np.random.choice(node.untriedMoves)
                    S.act(a)
                    # add child and traverse down the tree
                    node = node.addChild(a, S)
                    visited.append(node)
                    print(2)

                    # # rollout
                    # input = S.state[S.player]
                    # grid = S.state[3]
                    # aList = S.getLegalActions(input, grid)
                    # while aList != []:
                    #     S.act(np.random.choice(aList))
                    # #     # print(3)

            # backpropagate
            for node in visited:
                node.update(S.getReward())

                print(3)

        # print info
        if mode:
            print(root.treeToStr(0))
        else:
            print(root.childrenToStr())

        # return the most visited move
        return sorted(root.children, key=lambda c: c.visits)[-1].action

    # def MCST_Train_outofbound(self, mode):
    #     # node = self.loopTrain_MCTS()
    #     # print(node.ValueAll)
    #     iniS = self.initialState()
    #     root = TreeNode(iniS, None, None)
    #     trainEpi = self.game.trainEpi
    #
    #     for t in range(trainEpi):
    #
    #         node = root
    #         S = iniS.clone()
    #         print("iteration: ", t)
    #
    #         while(S.getReward()==0):
    #
    #             # select(fully expanded and non terminal)
    #             if node.untriedMoves == [] and node.children != []:
    #                 node = node.choose()
    #                 S.act(node.action)
    #                 # print(1)
    #
    #             # expand
    #             elif node.untriedMoves != []: #non terminal
    #                 a = np.random.choice(node.untriedMoves)
    #                 S.act(a)
    #                 # add child and traverse down the tree
    #                 node = node.addChild(a, S)
    #                 # print(2)
    #
    #                 # rollout(non terminal)
    #                 input = S.state[S.player]
    #                 grid = S.state[3]
    #                 aList = S.getLegalActions(input, grid)
    #                 while aList != []:
    #                     S.act(np.random.choice(aList))
    #                     # print(3)
    #
    #         # backpropagate
    #         while node != None:
    #             node.update(S.getReward())
    #             node = node.parent
    #             # print(4)
    #
    #     # print info
    #     if mode:
    #         print(root.treeToStr(0))
    #     else:
    #         print(root.childrenToStr())
    #
    #     # return the most visited move
    #     return sorted(root.children, key=lambda c: c.visited)[-1].move

    # def MCST_Train_BU(self, mode):
    #     # node = self.loopTrain_MCTS()
    #     # print(node.ValueAll)
    #     iniState = self.initialState()
    #     root = TreeNode(iniState, None, None)
    #     trainEpi = self.game.trainEpi
    #
    #     for t in range(trainEpi):
    #
    #         node = root
    #         state = deepcopy(iniState)
    #         print("iteration: ", t)
    #
    #         # select(fully expanded and non terminal)
    #         while node.untriedMoves == [] and node.children != [] and state.getReward() == 0:
    #             node = node.choose()
    #             state.act(node.action)
    #             print(1)
    #
    #         # expand
    #         if node.untriedMoves != [] and state.getReward() == 0: #non terminal
    #             a = np.random.choice(node.untriedMoves)
    #             state.act(a)
    #             # add child and traverse down the tree
    #             node = node.addChild(a, state)
    #             print(2)
    #
    #         # rollout(non terminal)
    #         input = state.state[state.player]
    #         grid = state.state[3]
    #         while state.getLegalActions(input, grid) != []\
    #                 and state.getReward() == 0:
    #             state.act(np.random.choice(state.getLegalActions(input, grid)))
    #             print(3)
    #
    #         # backpropagate
    #         while node != None:
    #             node.update(state.getReward())
    #             node = node.parent
    #             print(4)
    #
    #     # print info
    #     if mode:
    #         print(root.treeToStr(0))
    #     else:
    #         print(root.childrenToStr())
    #
    #     # return the most visited move
    #     return sorted(root.children, key=lambda c: c.visited)[-1].move


class TreeNode:

    def __init__(self, S, a, parent):
        """state, action, parent"""
        self.S = S
        self.action = a
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        if self.S.player == 0:
            self.untriedMoves = self.S.getLegalActions(self.S.state[0], self.S.state[3])
        else:
            self.untriedMoves = self.S.getLegalActions(self.S.state[1], self.S.state[3])
        self.player = S.player

    def choose(self):
        """return nextnode based on UCB1/UCT value"""
        chosen = None
        maxValue = -1.0e10  # initialize some very small value
        for child in self.children:
            # uct/ucb: upper confidence bounds for trees; 1e-6 to avoid divide 0
            uct = child.wins / (child.visits + 1e-6) + np.sqrt(
                2 * np.log(self.visits + 1) / (child.visits + 1e-6)
            )
            if uct > maxValue:
                chosen = child
                maxValue = uct
        return chosen

    def addChild(self, action, S):
        # remove action from untriedMoves
        # add new child node for this action
        # return the child node
        newNode = TreeNode(S, action, self)
        self.untriedMoves.remove(action)
        self.children.append(newNode)
        return newNode

    def update(self, reward):
        self.visits += 1
        self.wins += reward

    """_________Visualize Tree___________"""

    def indentStr(self, indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def treeToStr(self, indent):
        s = self.indentStr(indent) + str(self)
        for c in self.children:
            s += c.treeToStr(indent + 1)
        return s

    def childrenToStr(self):
        s = ""
        for c in self.children:
            s += str(c) + "\n"
        return s


class State:

    def __init__(self, state):
        # pacman: 0 ; ghost: 1
        # pacman move first(root)
        self.player = 0
        """pac, ghosts, food, wall"""
        self.state = state

    """____Read state channel, find next state, build state channel____"""

    def findOnes(self, input):
        """find the 1s in the input tuple"""
        oneList = []
        xA, yA = len(input[0]), len(input)
        for y in range(yA):
            for x in range(xA):
                if input[y][x] == 1:
                    oneList.append((x, y))
        return oneList

    def nextXY(self, x, y, a):
        if a == 0:
            return (x - 1, y)
        if a == 1:
            return (x + 1, y)
        if a == 2:
            return (x, y - 1)
        if a == 3:
            return (x, y + 1)

    def buildState(self, list, xA, yA):
        matrix = []
        for i in range(yA):
            matrix.append([])
            for j in range(xA):
                if (i, j) in list:
                    matrix[-1].append(1)
                else:
                    matrix[-1].append(0)
            matrix[-1] = tuple(matrix[-1])
        return tuple(matrix)

    """__________________update state channels_________________"""

    def pacAct(self, input, a):
        xA, yA = len(input[0]), len(input)

        # build new pacman state matrix
        curxy = self.findOnes(input)
        cx, cy = curxy[0]
        nx, ny = self.nextXY(cx, cy, a)
        pacNS = self.buildState([(nx, ny)], xA, yA)
        return pacNS, (nx, ny)

    def foodAct(self, input, sp):
        nx, ny = sp
        xA, yA = len(input[0]), len(input)
        if input[ny][nx] == 1:
            foodList = self.findOnes(input)
            foodList.remove((nx, ny))
            foodNS = self.buildState(foodList, xA, yA)
            return foodNS
        else:
            return input

    def ghoAct(self, input, a):
        xA, yA = len(input[0]), len(input)

        # build new pacman state matrix
        curxy = self.findOnes(input)
        cx, cy = curxy[0]
        nx, ny = self.nextXY(cx, cy, a)
        # nextxy = []
        # for i in range(len(curxy)):
        #     cx, cy = curxy[i]
        #     nxy = self.nextXY(cx, cy, a)
        #     nextxy.append(nxy)
        ghostNS = self.buildState([(nx, ny)], xA, yA)
        # ghostNS = self.buildState(nextxy, xA, yA)
        return ghostNS

    """_______________others_________________"""

    def act(self, action):
        """act and update state"""

        # pacman
        if self.player == 0:
            statePac, sp = self.pacAct(self.state[0], action)
            stateFood = self.foodAct(self.state[2], sp)
            self.state = (statePac, self.state[1], stateFood, self.state[3])
        # ghost
        else:
            stateGho = self.ghoAct(self.state[1], action)
            self.state = (self.state[0], stateGho, self.state[2], self.state[3])

        self.player = 1 - self.player

    def clone(self):
        newS = State(self.state)
        return newS

    def getLegalActions(self, input, grid):
        xA, yA = len(input[0]), len(input)

        curxy = self.findOnes(input)
        cx, cy = curxy[0]
        aList = []
        if cx > 0 and grid[cy][cx - 1] == 0:
            aList.append(0)
        if cx < xA - 1 and grid[cy][cx + 1] == 0:
            aList.append(1)
        if cy > 0 and grid[cy - 1][cx] == 0:
            aList.append(2)
        if cy < yA - 1 and grid[cy + 1][cx] == 0:
            aList.append(3)
        # print(aList)
        return aList

    def getReward(self):
        state = self.state
        # for matrix in state:
        #     for row in matrix:
        #         print(row)
        #     print("\n")
        # lost
        pac = self.findOnes(state[0])[0]
        gho = self.findOnes(state[1])[0]
        if pac == gho:
            return -1
        # win

        if sum(list(map(sum, state[2]))) == 0:
            return 1

        # foodList = self.findOnes(state[2])[0]
        # if foodList == []:
        #     return 1
        # not terminal
        return 0
