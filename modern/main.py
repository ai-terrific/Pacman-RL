#!/usr/bin/env python3
"""
Pac-Q: watch a Pac-Man agent learn with tabular Q-learning, live.

Eat every pellet while avoiding the ghosts.

Install:  pip install pygame numpy
Run:      python pacq.py

Controls (or click the buttons):
  Space  pause / resume          N          advance one step
  Up / Down  change speed        (starts slow so you can follow it)
  D      demo the greedy policy  H          toggle visit heatmap
  S      save Q-table            L          load Q-table
  R      reset the agent         Esc        quit
"""

import math
import os
import random
import sys
import time
from collections import deque

import numpy as np
import pygame

# =============================================================================
#  Environment
# =============================================================================

MAZE = [
    "###################",
    "#.................#",
    "#.###...#.###...#.#",
    "#.................#",
    "#.###.#...#.###...#",
    "#.................#",
    "#...###.#.###...#.#",
    "#.................#",
    "#.###.###.###.###.#",
    "#.................#",
    "#.###...#.###...#.#",
    "#.................#",
    "###################",
]
PAC_START = (11, 9)
GHOST_STARTS = [(1, 1), (1, 17), (11, 1), (11, 17)]
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left (row, col)
ACTION_ANGLE = [90, 0, 270, 180]  # for drawing Pac-Man's mouth


class Ghost:
    def __init__(self, start):
        self.pos = start


class PacmanEnv:
    """
    Pac-Man collects every pellet while avoiding four ghosts.

    The state records Pac-Man's cell, the nearest ghost's direction and
    distance, and the direction to the nearest remaining pellet. Legal-action
    masking prevents the learner from choosing moves into walls.
    """

    def __init__(self, max_steps=2400, chase_prob=0.18, seed=None):
        self.rng = random.Random(seed)
        self.rows, self.cols = len(MAZE), len(MAZE[0])
        self.max_steps = max_steps
        self.chase_prob = chase_prob
        self.wall = [[ch == "#" for ch in row] for row in MAZE]
        self.n_cells = self.rows * self.cols
        self.n_states = self.n_cells * 3 * 3 * 3 * 5

        # Where each action leads (bumping into a wall means staying put).
        self.move_to, self.open_nbrs = {}, {}
        for r in range(self.rows):
            for c in range(self.cols):
                if self.wall[r][c]:
                    continue
                targets, opens = [], []
                for dr, dc in ACTIONS:
                    nr, nc = r + dr, c + dc
                    if self.wall[nr][nc]:
                        targets.append((r, c))
                    else:
                        targets.append((nr, nc))
                        opens.append((nr, nc))
                self.move_to[(r, c)] = targets
                self.open_nbrs[(r, c)] = opens

        self.dist = self._all_pairs_distance()
        self.dist_l = self.dist.tolist()
        taken = {PAC_START, *GHOST_STARTS}
        self.initial_dots = {cell for cell in self.move_to if cell not in taken}
        self.reset()

    def _flat(self, cell):
        return cell[0] * self.cols + cell[1]

    def _all_pairs_distance(self):
        dist = np.full((self.n_cells, self.n_cells), 999, dtype=np.int16)
        for start in self.move_to:
            s = self._flat(start)
            dist[s, s] = 0
            queue = deque([start])
            while queue:
                cur = queue.popleft()
                for nxt in self.open_nbrs[cur]:
                    if dist[s, self._flat(nxt)] == 999:
                        dist[s, self._flat(nxt)] = dist[s, self._flat(cur)] + 1
                        queue.append(nxt)
        return dist

    # ---------------------------------------------------------------- core API
    def reset(self):
        self.pac = PAC_START
        self.ghosts = [Ghost(start) for start in GHOST_STARTS]
        self.dots = set(self.initial_dots)
        # Penalty rows: 0 where a pellet exists, 999 elsewhere.
        self.pen_all = np.full(self.n_cells, 999, dtype=np.int16)
        for cell in self.dots:
            self.pen_all[self._flat(cell)] = 0
        self.steps = 0
        return self.state()

    def step(self, action):
        """Returns (state, reward, done, terminal, outcome)."""
        self.steps += 1
        # A small living cost encourages efficient routes without drowning out
        # the pellet and survival rewards.
        reward = -1
        old_pac = self.pac
        old_pos = [g.pos for g in self.ghosts]
        old_ghost_distance = min(
            self.dist[self._flat(old_pac)][self._flat(g.pos)] for g in self.ghosts
        )
        old_pellet_distance = self._nearest_pellet_distance(old_pac)

        new_pac = self.move_to[old_pac][action]
        if new_pac == old_pac:
            reward -= 5.0  # wall bumps are discouraged; they are also masked below
        self.pac = new_pac

        if new_pac in self.dots:
            self.dots.discard(new_pac)
            self.pen_all[self._flat(new_pac)] = 999
            reward += 20.0

        # Potential-based progress shaping: reward movement that gets closer
        # to the nearest remaining pellet. The pellet reward handles collection.
        new_pellet_distance = self._nearest_pellet_distance(new_pac)
        if old_pellet_distance < 999 and new_pellet_distance < 999:
            reward += 0.6 * (old_pellet_distance - new_pellet_distance)

        caught = any(g.pos == new_pac for g in self.ghosts)
        if not caught:
            for i, g in enumerate(self.ghosts):
                self._move_ghost(g)
                crossed = g.pos == old_pac and new_pac == old_pos[i]
                if g.pos == new_pac or crossed:
                    caught = True
                    break

        if not caught:
            new_ghost_distance = min(
                self.dist[self._flat(new_pac)][self._flat(g.pos)] for g in self.ghosts
            )
            # Small bounded bonus for opening space from ghosts; large enough
            # to teach caution, but smaller than collecting a pellet.
            reward += max(
                -2.0, min(2.0, 0.4 * (new_ghost_distance - old_ghost_distance))
            )

        if caught:
            return self.state(), reward - 500.0, True, True, "caught"
        if not self.dots:
            return self.state(), reward + 500.0, True, True, "win"
        if self.steps >= self.max_steps:
            # The episode is over; do not bootstrap from a state that will reset.
            return self.state(), reward - 25.0, True, True, "timeout"
        return self.state(), reward, False, False, None

    def valid_actions(self, cell=None):
        """Actions that move to a different, walkable cell (no wall bumps)."""
        cell = self.pac if cell is None else cell
        return [a for a, target in enumerate(self.move_to[cell]) if target != cell]

    def _nearest_pellet_distance(self, cell):
        if not self.dots:
            return 0
        distances = self.dist[self._flat(cell)]
        return min(int(distances[self._flat(dot)]) for dot in self.dots)

    def _move_ghost(self, g):
        options = self.open_nbrs[g.pos]
        d = self.dist_l
        if self.rng.random() < self.chase_prob:
            target = self.pac
            target_flat = self._flat(target)
            best = min(d[self._flat(n)][target_flat] for n in options)
            options = [n for n in options if d[self._flat(n)][target_flat] == best]
        g.pos = self.rng.choice(options)

    # ------------------------------------------------------------------ state
    def _features(self):
        pac = self.pac
        p = self._flat(pac)
        distances = self.dist_l[p]
        nearest = min(self.ghosts, key=lambda ghost: distances[self._flat(ghost.pos)])
        distance = distances[self._flat(nearest.pos)]
        side_r = (nearest.pos[0] > pac[0]) - (nearest.pos[0] < pac[0]) + 1
        side_c = (nearest.pos[1] > pac[1]) - (nearest.pos[1] < pac[1]) + 1
        near = 0 if distance <= 2 else (1 if distance <= 5 else 2)
        pellet_dir = self._direction_to(self.pen_all) if self.dots else 4
        return side_r, side_c, near, pellet_dir

    def state(self):
        side_r, side_c, near, pellet_dir = self._features()
        idx = self.pac[0] * self.cols + self.pac[1]
        idx = ((idx * 3 + side_r) * 3 + side_c) * 3 + near
        return idx * 5 + pellet_dir

    def _direction_to(self, penalty):
        best_a, best_d = 4, 10_000
        for a, target in enumerate(self.move_to[self.pac]):
            if target == self.pac:
                continue
            d = int((self.dist[self._flat(target)] + penalty).min())
            if d < best_d:
                best_a, best_d = a, d
        return best_a


# =============================================================================
#  Agent and trainer
# =============================================================================


class QAgent:
    def __init__(
        self,
        n_states,
        alpha=0.2,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.9975,
        seed=None,
    ):
        self.Q = np.zeros((n_states, 4), dtype=np.float32)
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.rng = random.Random(seed)

    def act(self, s, valid_actions=None, greedy=False):
        valid = list(valid_actions) if valid_actions else list(range(4))
        if not greedy and self.rng.random() < self.eps:
            return self.rng.choice(valid)
        q = self.Q[s, valid]
        best_value = q.max()
        best = [valid[i] for i, value in enumerate(q) if value == best_value]
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    def learn(self, s, a, r, s2, terminal, next_valid_actions=None):
        if terminal:
            target = r
        else:
            valid = list(next_valid_actions) if next_valid_actions else list(range(4))
            target = r + self.gamma * float(self.Q[s2, valid].max())
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


class Trainer:
    """Runs the training loop in small slices so the UI stays responsive."""

    def __init__(self):
        self.env = PacmanEnv()
        self.agent = QAgent(self.env.n_states)
        self.state = self.env.reset()
        self.demo = False

        self.episodes = 0
        self.total_steps = 0
        self.wins = self.deaths = self.timeouts = 0
        self.recent_reward = deque(maxlen=100)
        self.recent_win = deque(
            maxlen=100
        )  # rolling win rate: latest 100 training episodes

        self.ep_reward = 0.0
        self.last_outcome, self.last_reward, self.last_steps = None, 0.0, 0
        self.event_id = 0
        self.last_action = 1
        self._snapshot_prev()
        self.visits = [0] * self.env.n_cells

    def _snapshot_prev(self):
        self.prev_pac = self.env.pac
        self.prev_ghosts = [g.pos for g in self.env.ghosts]

    def set_demo(self, on):
        self.demo = on
        self.state = self.env.reset()
        self.ep_reward = 0.0
        self._snapshot_prev()

    def advance(self, n, budget=None):
        env, agent = self.env, self.agent
        t0 = time.perf_counter()
        done_steps = 0
        for i in range(n):
            if budget and i and (i & 63) == 0 and time.perf_counter() - t0 > budget:
                break
            self._snapshot_prev()
            s = self.state
            a = agent.act(s, valid_actions=env.valid_actions(), greedy=self.demo)
            s2, r, done, terminal, outcome = env.step(a)
            self.last_action = a
            self.ep_reward += r
            if not self.demo:
                agent.learn(
                    s,
                    a,
                    r,
                    s2,
                    terminal,
                    next_valid_actions=env.valid_actions(),
                )
                self.visits[env.pac[0] * env.cols + env.pac[1]] += 1
            self.state = s2
            done_steps += 1
            if done:
                self._end_episode(outcome)
        self.total_steps += done_steps
        return done_steps

    def _end_episode(self, outcome):
        self.last_outcome, self.last_reward = outcome, self.ep_reward
        self.last_steps = self.env.steps
        self.event_id += 1
        if not self.demo:
            self.episodes += 1
            self.wins += outcome == "win"
            self.deaths += outcome == "caught"
            self.timeouts += outcome == "timeout"
            self.recent_reward.append(self.ep_reward)
            self.recent_win.append(outcome == "win")
            self.agent.decay()
        self.ep_reward = 0.0
        self.state = self.env.reset()
        self._snapshot_prev()

    # --------------------------------------------------------------- readouts
    @property
    def win_rate(self):
        return (
            100.0 * sum(self.recent_win) / len(self.recent_win)
            if self.recent_win
            else 0.0
        )

    @property
    def avg_reward(self):
        return (
            sum(self.recent_reward) / len(self.recent_reward)
            if self.recent_reward
            else 0.0
        )

    def save(self, path):
        np.save(path, self.agent.Q)

    def load(self, path):
        q = np.load(path)
        if q.shape != self.agent.Q.shape:
            raise ValueError("That Q-table was saved for a different maze or rules")
        self.agent.Q = q.astype(np.float32)
        self.agent.eps = self.agent.eps_end


# =============================================================================
#  Look and feel
# =============================================================================

W, H = 1500, 860
CELL = 40
MAZE_X, MAZE_Y = 40, 114

INK = (232, 230, 255)  # primary text
MUTED = (140, 137, 199)  # secondary text
NIGHT_TOP = (22, 18, 62)  # background gradient, top
NIGHT_BOT = (11, 10, 34)  # background gradient, bottom
PANEL = (28, 25, 78, 225)
PANEL_EDGE = (110, 100, 220, 80)
WALL_FILL = (28, 46, 150)
WALL_EDGE = (74, 124, 255)
FLOOR = (14, 12, 42)
PELLET = (255, 226, 170)
PAC_YELLOW = (255, 210, 63)
GHOST_CORAL = (255, 92, 122)
GHOST_CYAN = (110, 230, 255)
MINT = (76, 242, 176)
AMBER = (255, 184, 77)
SKY = (96, 200, 255)
GHOST_COLORS = [GHOST_CORAL, GHOST_CYAN, AMBER, MINT]


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))


def make_background():
    surf = pygame.Surface((W, H))
    for y in range(H):
        pygame.draw.line(surf, lerp_color(NIGHT_TOP, NIGHT_BOT, y / H), (0, y), (W, y))
    return surf


def make_glow(radius, color, strength=70):
    surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    for i in range(radius, 0, -1):
        alpha = int(strength * (1 - i / radius) ** 2)
        pygame.draw.circle(surf, (*color, alpha), (radius, radius), i)
    return surf


def make_maze_surface(env):
    """Walls and floor never change, so paint them once."""
    w, h = env.cols * CELL, env.rows * CELL
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(surf, FLOOR, surf.get_rect(), border_radius=14)
    inset = 6
    for r in range(env.rows):
        for c in range(env.cols):
            if not env.wall[r][c]:
                continue
            x, y = c * CELL, r * CELL
            rect = pygame.Rect(x + inset, y + inset, CELL - 2 * inset, CELL - 2 * inset)
            pygame.draw.rect(surf, WALL_FILL, rect, border_radius=10)
            pygame.draw.rect(surf, WALL_EDGE, rect, width=2, border_radius=10)
    for r in range(env.rows):
        for c in range(env.cols):
            if not env.wall[r][c]:
                continue
            x, y = c * CELL, r * CELL
            if c + 1 < env.cols and env.wall[r][c + 1]:
                bridge = pygame.Rect(
                    x + CELL - inset - 3,
                    y + inset + 2,
                    2 * inset + 6,
                    CELL - 2 * inset - 4,
                )
                pygame.draw.rect(surf, WALL_FILL, bridge)
            if r + 1 < env.rows and env.wall[r + 1][c]:
                bridge = pygame.Rect(
                    x + inset + 2,
                    y + CELL - inset - 3,
                    CELL - 2 * inset - 4,
                    2 * inset + 6,
                )
                pygame.draw.rect(surf, WALL_FILL, bridge)
    return surf


class Fonts:
    def __init__(self):
        names = "segoeui,sfprodisplay,helveticaneue,inter,dejavusans,arial"
        self.xs = pygame.font.SysFont(names, 14)
        self.sm = pygame.font.SysFont(names, 16)
        self.md = pygame.font.SysFont(names, 20, bold=True)
        self.lg = pygame.font.SysFont(names, 34, bold=True)
        self.xl = pygame.font.SysFont(names, 46, bold=True)


def draw_text(surf, txt, font, color, pos, anchor="topleft"):
    img = font.render(txt, True, color)
    rect = img.get_rect(**{anchor: pos})
    surf.blit(img, rect)
    return rect


_card_cache = {}


def draw_card(surf, rect, radius=18):
    key = (rect.w, rect.h, radius)
    if key not in _card_cache:
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, PANEL, s.get_rect(), border_radius=radius)
        pygame.draw.rect(s, PANEL_EDGE, s.get_rect(), width=1, border_radius=radius)
        _card_cache[key] = s
    surf.blit(_card_cache[key], rect.topleft)


def draw_pacman(surf, cx, cy, r, direction_deg, mouth_deg, glow):
    if glow:
        surf.blit(glow, (cx - glow.get_width() // 2, cy - glow.get_height() // 2))
    pts = [(cx, cy)]
    start, end = direction_deg + mouth_deg, direction_deg + 360 - mouth_deg
    n = 26
    for i in range(n + 1):
        a = math.radians(start + (end - start) * i / n)
        pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
    pygame.draw.polygon(surf, PAC_YELLOW, pts)


def draw_ghost(
    surf,
    cx,
    cy,
    r,
    color,
    look,
    t,
    glow=None,
):
    if glow:
        surf.blit(glow, (cx - glow.get_width() // 2, cy - glow.get_height() // 2))
    pts = []
    for i in range(21):  # dome
        a = math.radians(180 - 180 * i / 20)
        pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
    base, teeth = cy + r * 0.92, 4
    wob = math.sin(t * 9) * r * 0.06
    for i in range(teeth * 2 + 1):  # wavy hem, right to left
        x = cx + r - (2 * r) * i / (teeth * 2)
        up = i % 2 == 1
        pts.append((x, base - (r * 0.32 if up else 0) + (wob if up else -wob)))
    pygame.draw.polygon(surf, color, pts)
    for side in (-1, 1):
        ex, ey = cx + side * r * 0.36, cy - r * 0.1
        pygame.draw.circle(surf, (255, 255, 255), (ex, ey), r * 0.3)
        pygame.draw.circle(
            surf,
            (30, 30, 110),
            (ex + look[0] * r * 0.12, ey + look[1] * r * 0.12),
            r * 0.15,
        )


class Button:
    def __init__(self, rect, label, on_click, is_on=None):
        self.rect = pygame.Rect(rect)
        self.label, self.on_click, self.is_on = label, on_click, is_on

    def draw(self, surf, fonts, mouse):
        label = self.label() if callable(self.label) else self.label
        hover = self.rect.collidepoint(mouse)
        active = self.is_on() if self.is_on else False
        fill = (62, 54, 150) if hover else (40, 35, 105)
        if active:
            fill = (70, 58, 170)
        pygame.draw.rect(surf, fill, self.rect, border_radius=12)
        edge = SKY if active else (96, 88, 200)
        pygame.draw.rect(
            surf, edge, self.rect, width=2 if active else 1, border_radius=12
        )
        draw_text(surf, label, fonts.sm, INK, self.rect.center, "center")


class Slider:
    def __init__(self, rect, label, vmin, vmax, get, put, fmt="{:.2f}"):
        self.rect = pygame.Rect(rect)
        self.label, self.vmin, self.vmax = label, vmin, vmax
        self.get, self.put, self.fmt = get, put, fmt

    def draw(self, surf, fonts):
        v = self.get()
        t = max(0.0, min(1.0, (v - self.vmin) / (self.vmax - self.vmin)))
        draw_text(surf, self.label, fonts.sm, MUTED, (self.rect.x, self.rect.y - 24))
        draw_text(
            surf,
            self.fmt.format(v),
            fonts.sm,
            INK,
            (self.rect.right, self.rect.y - 24),
            "topright",
        )
        track = pygame.Rect(self.rect.x, self.rect.centery - 3, self.rect.w, 6)
        pygame.draw.rect(surf, (18, 16, 54), track, border_radius=3)
        fill = track.copy()
        fill.w = int(track.w * t)
        pygame.draw.rect(surf, SKY, fill, border_radius=3)
        kx = self.rect.x + int(self.rect.w * t)
        pygame.draw.circle(surf, SKY, (kx, track.centery), 10)
        pygame.draw.circle(surf, INK, (kx, track.centery), 6)

    def hit(self, pos):
        return self.rect.inflate(24, 28).collidepoint(pos)

    def drag(self, x):
        t = max(0.0, min(1.0, (x - self.rect.x) / self.rect.w))
        self.put(self.vmin + t * (self.vmax - self.vmin))


# =============================================================================
#  App
# =============================================================================

# environment steps per frame at 60 fps; the first few are slow enough to follow by eye
SPEEDS = [0.06, 0.12, 0.5, 2, 10, 50, 300, 3000, 30000]
SPEED_NAMES = [
    "Stroll",
    "Watch",
    "Gentle",
    "Brisk",
    "Quick",
    "Fast",
    "Faster",
    "Turbo",
    "Max",
]
DEFAULT_SPEED = 1  # "Watch": about 7 steps per second
QTABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtable.npy")


def main():
    pygame.init()
    pygame.display.set_caption("Pac-Q: Q-learning lab")
    try:  # SCALED lets the window shrink to fit smaller screens
        screen = pygame.display.set_mode((W, H), pygame.SCALED | pygame.RESIZABLE)
    except pygame.error:
        screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    fonts = Fonts()
    background = make_background()

    trainer = Trainer()
    load_warning = None
    try:
        trainer.load(QTABLE_PATH)
    except FileNotFoundError:
        pass
    except ValueError:
        load_warning = (
            "Saved Q-table uses a different maze or state format; starting fresh."
        )

    state = {
        "trainer": trainer,
        "paused": False,
        "speed": DEFAULT_SPEED,
        "heatmap": False,
        "acc": 0.0,
        "toast": ("", 0.0),
        "flash": None,  # (color, strength)
        "result_message": None,
        "result_until": 0.0,
        "seen_event": 0,
        "drag": None,  # slider being dragged
    }
    maze_surf = make_maze_surface(state["trainer"].env)
    glows = {
        "pellet": make_glow(12, PELLET, 60),
        "pac": make_glow(30, PAC_YELLOW, 90),
        **{
            f"ghost{i}": make_glow(30, color, 90)
            for i, color in enumerate(GHOST_COLORS)
        },
    }
    heat_cell = pygame.Surface((CELL - 6, CELL - 6), pygame.SRCALPHA)

    def toast(msg):
        state["toast"] = (msg, time.time() + 2.5)

    if load_warning:
        toast(load_warning)

    def toggle_pause():
        state["paused"] = not state["paused"]

    def step_once():
        state["paused"] = True
        state["trainer"].advance(1)

    def faster():
        state["speed"] = min(len(SPEEDS) - 1, state["speed"] + 1)

    def slower():
        state["speed"] = max(0, state["speed"] - 1)

    def toggle_demo():
        t = state["trainer"]
        t.set_demo(not t.demo)
        state["acc"] = 0.0
        if t.demo and state["speed"] > 2:
            state["speed"] = 2

    def toggle_heat():
        state["heatmap"] = not state["heatmap"]

    def save():
        state["trainer"].save(QTABLE_PATH)
        toast("Saved the Q-table to qtable.npy")

    def load():
        try:
            state["trainer"].load(QTABLE_PATH)
            toast("Loaded qtable.npy (exploration set to its minimum)")
        except FileNotFoundError:
            toast("No qtable.npy yet. Press Save first.")
        except ValueError as e:
            toast(str(e))

    def reset():
        state["trainer"] = Trainer()
        state["acc"] = 0.0
        state["seen_event"] = 0
        state["result_message"] = None
        toast("Agent reset. Learning starts from scratch.")

    # ---- buttons (two rows of five under the maze)
    bw, bh, gap, bx = 140, 38, 8, 36

    def slot(row, col):
        return (bx + col * (bw + gap), 696 + row * (bh + 8), bw, bh)

    buttons = [
        Button(
            slot(0, 0),
            lambda: "Resume" if state["paused"] else "Pause",
            toggle_pause,
            lambda: state["paused"],
        ),
        Button(slot(0, 1), "Step", step_once),
        Button(slot(0, 2), "Slower", slower),
        Button(slot(0, 3), "Faster", faster),
        Button(slot(0, 4), lambda: SPEED_NAMES[state["speed"]], lambda: None),
        Button(slot(1, 0), "Demo policy", toggle_demo, lambda: state["trainer"].demo),
        Button(slot(1, 1), "Heatmap", toggle_heat, lambda: state["heatmap"]),
        Button(slot(1, 2), "Save", save),
        Button(slot(1, 3), "Load", load),
        Button(slot(1, 4), "Reset", reset),
    ]
    key_actions = {
        pygame.K_SPACE: toggle_pause,
        pygame.K_n: step_once,
        pygame.K_UP: faster,
        pygame.K_DOWN: slower,
        pygame.K_d: toggle_demo,
        pygame.K_h: toggle_heat,
        pygame.K_s: save,
        pygame.K_l: load,
        pygame.K_r: reset,
    }

    # ---- sliders for the learning settings
    def set_eps_floor(v):
        a = state["trainer"].agent
        a.eps_end = v
        a.eps = max(a.eps, v)

    def set_chase(v):
        state["trainer"].env.chase_prob = v

    def agent_attr(name):
        def put(v):
            setattr(state["trainer"].agent, name, v)

        return put

    sx, sw = 860, 592

    sliders = [
        Slider(
            (sx, 184, sw, 20),
            "Learning rate (alpha)",
            0.02,
            0.8,
            lambda: state["trainer"].agent.alpha,
            agent_attr("alpha"),
        ),
        Slider(
            (sx, 232, sw, 20),
            "Discount factor (gamma)",
            0.5,
            0.999,
            lambda: state["trainer"].agent.gamma,
            agent_attr("gamma"),
        ),
        Slider(
            (sx, 280, sw, 20),
            "Minimum exploration",
            0.0,
            0.3,
            lambda: state["trainer"].agent.eps_end,
            set_eps_floor,
        ),
        Slider(
            (sx, 328, sw, 20),
            "Ghost aggression",
            0.0,
            0.9,
            lambda: state["trainer"].env.chase_prob,
            set_chase,
        ),
    ]

    def cell_px(cell):
        return (
            MAZE_X + cell[1] * CELL + CELL // 2,
            MAZE_Y + cell[0] * CELL + CELL // 2,
        )

    running = True
    while running:
        clock.tick(60)
        now = time.time()
        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key in key_actions:
                    key_actions[ev.key]()
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for b in buttons:
                    if b.rect.collidepoint(ev.pos):
                        b.on_click()
                for s in sliders:
                    if s.hit(ev.pos):
                        state["drag"] = s
                        s.drag(ev.pos[0])
            elif ev.type == pygame.MOUSEMOTION and state["drag"]:
                state["drag"].drag(ev.pos[0])
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                state["drag"] = None

        trainer = state["trainer"]
        speed = SPEEDS[state["speed"]]

        # ----------------------------------------------------------- simulate
        lerp_t = 1.0
        if not state["paused"]:
            if speed < 1:
                state["acc"] += speed
                n = int(state["acc"])
                state["acc"] -= n
                lerp_t = state["acc"]
                trainer.advance(n)
            else:
                trainer.advance(int(speed), budget=0.012)
        if trainer.event_id != state["seen_event"]:
            state["seen_event"] = trainer.event_id
            won = trainer.last_outcome == "win"
            state["result_message"] = (
                "You are a winner" if won else "You are lost",
                MINT if won else GHOST_CORAL,
            )
            state["result_until"] = now + 3.0
            if speed <= 2:
                color = {"win": MINT, "caught": GHOST_CORAL}.get(trainer.last_outcome)
                if color:
                    state["flash"] = (color, 70)

        # --------------------------------------------------------------- draw
        screen.blit(background, (0, 0))
        env = trainer.env
        clock_t = time.time()

        # header
        draw_text(screen, "Pac-Q", fonts.xl, PAC_YELLOW, (28, 22))
        draw_text(
            screen,
            "Collect every pellet while avoiding four ghosts.",
            fonts.sm,
            MUTED,
            (30, 76),
        )
        if trainer.demo:
            pill, pill_color = "Demo: greedy policy, no learning", SKY
        elif state["paused"]:
            pill, pill_color = "Paused", AMBER
        else:
            pill, pill_color = f"Training at {SPEED_NAMES[state['speed']]} speed", MINT
        pr = draw_text(screen, pill, fonts.md, INK, (W - 52, 42), "midright")
        pygame.draw.circle(screen, pill_color, (pr.x - 16, pr.centery), 7)

        # ------------------------------------------------------- maze column
        maze_card = pygame.Rect(24, 104, 800, 580)
        draw_card(screen, maze_card, 22)
        screen.blit(maze_surf, (MAZE_X, MAZE_Y))

        if state["heatmap"]:
            top = max(trainer.visits) or 1
            for r in range(env.rows):
                for c in range(env.cols):
                    v = trainer.visits[r * env.cols + c]
                    if env.wall[r][c] or v == 0:
                        continue
                    heat_cell.fill((0, 0, 0, 0))
                    pygame.draw.rect(
                        heat_cell,
                        (*SKY, int(40 + 150 * math.sqrt(v / top))),
                        heat_cell.get_rect(),
                        border_radius=10,
                    )
                    screen.blit(
                        heat_cell, (MAZE_X + c * CELL + 3, MAZE_Y + r * CELL + 3)
                    )

        for r, c in env.dots:
            cx, cy = cell_px((r, c))
            screen.blit(glows["pellet"], (cx - 12, cy - 12))
            pygame.draw.circle(screen, PELLET, (cx, cy), 4)

        def cell_center(cell, prev, t):
            x = lerp(prev[1], cell[1], t)
            y = lerp(prev[0], cell[0], t)
            return (
                int(MAZE_X + x * CELL + CELL / 2),
                int(MAZE_Y + y * CELL + CELL / 2),
            )

        pac_c = cell_center(env.pac, trainer.prev_pac, lerp_t)
        for i, g in enumerate(env.ghosts):
            gcx, gcy = cell_center(g.pos, trainer.prev_ghosts[i], lerp_t)
            dx, dy = pac_c[0] - gcx, pac_c[1] - gcy
            norm = math.hypot(dx, dy) or 1
            look = (dx / norm, dy / norm)
            draw_ghost(
                screen,
                gcx,
                gcy,
                13,
                GHOST_COLORS[i],
                look,
                clock_t,
                glows[f"ghost{i}"],
            )
        mouth = 8 + 30 * abs(math.sin(clock_t * 10))
        draw_pacman(
            screen,
            pac_c[0],
            pac_c[1],
            13,
            ACTION_ANGLE[trainer.last_action],
            mouth,
            glows["pac"],
        )

        if state["flash"]:
            color, strength = state["flash"]
            veil = pygame.Surface((env.cols * CELL, env.rows * CELL), pygame.SRCALPHA)
            pygame.draw.rect(
                veil, (*color, int(strength)), veil.get_rect(), border_radius=14
            )
            screen.blit(veil, (MAZE_X, MAZE_Y))
            strength -= 5
            state["flash"] = (color, strength) if strength > 0 else None

        if state["result_message"] and now < state["result_until"]:
            message, color = state["result_message"]
            result = fonts.lg.render(message, True, color)
            result_rect = result.get_rect(
                center=(MAZE_X + env.cols * CELL // 2, MAZE_Y + env.rows * CELL // 2)
            )
            panel = pygame.Surface(
                (result_rect.w + 40, result_rect.h + 24), pygame.SRCALPHA
            )
            pygame.draw.rect(
                panel,
                (14, 12, 42, 225),
                panel.get_rect(),
                border_radius=14,
            )
            pygame.draw.rect(
                panel,
                (*color, 210),
                panel.get_rect(),
                width=2,
                border_radius=14,
            )
            screen.blit(panel, (result_rect.x - 20, result_rect.y - 12))
            screen.blit(result, result_rect)

        # Episode status and remaining pellet count under the maze.
        draw_text(
            screen,
            "Goal: collect every pellet and avoid the ghosts",
            fonts.sm,
            MUTED,
            (MAZE_X, 638),
        )
        outcomes = {
            "win": ("Cleared the maze", MINT),
            "caught": ("Caught by a ghost", GHOST_CORAL),
            "timeout": ("Ran out of time", AMBER),
            None: ("No episode finished yet", MUTED),
        }
        text, color = outcomes[trainer.last_outcome]
        if trainer.last_outcome:
            text += f"  ({trainer.last_reward:+.0f} reward, {trainer.last_steps} steps)"
        draw_text(screen, "Last episode:", fonts.sm, MUTED, (MAZE_X, 662))
        draw_text(screen, text, fonts.sm, color, (MAZE_X + 100, 662))
        pellets_left = len(env.dots)
        draw_text(
            screen,
            f"Pellets left: {pellets_left}   Step {env.steps}/{env.max_steps}",
            fonts.sm,
            INK,
            (maze_card.right - 20, 662),
            "topright",
        )

        for b in buttons:
            b.draw(screen, fonts, mouse)
        draw_text(
            screen,
            "Space pause  |  N step  |  Up/Down speed  |  D demo  |  H heatmap  |  "
            "S save  |  L load  |  R reset",
            fonts.xs,
            MUTED,
            (30, 786),
        )

        # legend
        legend = pygame.Rect(24, 806, 800, 48)
        draw_card(screen, legend)
        ly = legend.centery
        pygame.draw.circle(screen, PELLET, (legend.x + 100, ly), 4)
        draw_text(screen, "Pellet", fonts.sm, INK, (legend.x + 122, ly), "midleft")
        for i, color in enumerate(GHOST_COLORS):
            draw_ghost(screen, legend.x + 388 + i * 20, ly - 2, 8, color, (0, 0), 0)
        draw_text(screen, "Ghosts (4)", fonts.sm, INK, (legend.x + 480, ly), "midleft")

        # ----------------------------------------------------- middle column
        settings = pygame.Rect(840, 104, 636, 284)
        draw_card(screen, settings)
        draw_text(
            screen,
            "Learning settings",
            fonts.md,
            INK,
            (settings.x + 20, settings.y + 14),
        )
        for s in sliders:
            s.draw(screen, fonts)

        # ------------------------------------------------------ right column
        cards = [
            ("Episodes", f"{trainer.episodes:,}", INK),
            ("Last 100 win rate", f"{trainer.win_rate:.1f}%", MINT),
            ("Avg reward", f"{trainer.avg_reward:.0f}", SKY),
            ("Pellets left", f"{len(trainer.env.dots):,}", AMBER),
        ]
        cw, ch, cgap, cx0 = 312, 104, 12, 840
        for i, (label, value, color) in enumerate(cards):
            rect = pygame.Rect(
                cx0 + (i % 2) * (cw + cgap),
                414 + (i // 2) * (ch + cgap),
                cw,
                ch,
            )
            draw_card(screen, rect)
            draw_text(screen, label, fonts.sm, MUTED, (rect.x + 16, rect.y + 12))
            draw_text(screen, value, fonts.lg, color, (rect.x + 16, rect.y + 38))

        msg, until = state["toast"]
        if now < until:
            tr = draw_text(screen, msg, fonts.sm, INK, (W // 2, H - 14), "center")
            pygame.draw.rect(screen, SKY, tr.inflate(28, 14), width=1, border_radius=10)

        pygame.display.flip()

    state["trainer"].save(QTABLE_PATH)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
