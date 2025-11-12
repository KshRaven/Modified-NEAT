
from .functional import INT, FLOAT, BOOL, ARRAY_1D
from .player import Players, PLAYERS_TYPE

from numba import prange
from numba.experimental import jitclass

import pygame
import math
import random
import numpy as np


class Ball:
    MAX_VEL = 5
    RADIUS = 7

    def __init__(self, x, y):
        self.x = self.original_x = x
        self.y = self.original_y = y
        
        angle = self._get_random_angle(-30, 30, [0])
        pos = 1 if random.random() < 0.5 else -1

        self.x_vel = pos * abs(math.cos(angle) * self.MAX_VEL)
        self.y_vel = math.sin(angle) * self.MAX_VEL

    def _get_random_angle(self, min_angle, max_angle, excluded):
        angle = 0
        while angle in excluded:
            angle = math.radians(random.randrange(min_angle, max_angle))

        return angle

    def draw(self, win):
        pygame.draw.circle(win, (255, 255, 255), (self.x, self.y), self.RADIUS)

    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y

        angle = self._get_random_angle(-30, 30, [0])
        x_vel = abs(math.cos(angle) * self.MAX_VEL)
        y_vel = math.sin(angle) * self.MAX_VEL

        self.y_vel = y_vel
        self.x_vel *= -1


@jitclass([
    ('MAX_VEL', FLOAT),
    ('RADIUS', FLOAT),
    ('MAX_DRAW', INT),
    ('LAUNCH_ANGLE', INT),
    ('players', PLAYERS_TYPE),
    ('total', INT),
    ('window_width', INT),
    ('window_height', INT),
    ('original_x', INT),
    ('original_y', INT),
    ('x', ARRAY_1D),
    ('y', ARRAY_1D),
    ('x_vel', ARRAY_1D),
    ('y_vel', ARRAY_1D),
])
class Balls(object):
    def __init__(self, players: Players,
                 x: int, y: int, window_shape: tuple[int, int],
                 max_vel: int | float = 5, radius: int = 7, max_draw: int = 1):
        self.MAX_VEL = max_vel
        self.RADIUS = radius
        self.MAX_DRAW = max_draw
        self.LAUNCH_ANGLE = 30

        self.players = players
        self.total = self.players.total
        self.window_width, self.window_height = window_shape
        self.original_x = min(max(x, self.RADIUS), self.window_width - self.RADIUS)
        self.original_y = min(max(self.window_height - y, self.RADIUS), self.window_height - self.RADIUS)
        self.x = np.full((self.total,), self.original_x, dtype=FLOAT)
        self.y = np.full((self.total,), self.original_y, dtype=FLOAT)

        # Restrict angle between -30° and +30° (converted to radians)
        angle = self._get_random_angle(-self.LAUNCH_ANGLE, self.LAUNCH_ANGLE, [0])
        pos = np.full((self.total,), 1 if random.random() < 0.5 else -1, dtype=INT)
        left, right = self.players.left, self.players.right
        angle[right] = angle[left]
        pos[right] = pos[left]

        self.x_vel = pos * np.abs(np.cos(angle) * self.MAX_VEL)
        self.y_vel = np.sin(angle) * self.MAX_VEL

    def _get_random_angle(self, min_angle: int | float, max_angle: int | float, excluded: list[int] | None):
        """Generate random launch angles (in radians) between min_angle and max_angle degrees."""
        if excluded is None:
            excluded = [0]
        # convert to radians
        min_rad, max_rad = np.radians(min_angle), np.radians(max_angle)
        excluded_rads = np.radians(np.array([excluded]))

        angle = np.zeros((self.total,), dtype=FLOAT)
        not_filled = np.full((self.total,), True, dtype=BOOL)
        count = np.count_nonzero(not_filled).item()
        while count > 0:
            angle[not_filled] = np.random.uniform(min_rad, max_rad, size=(count,))
            for i in prange(len(angle)):
                if not_filled[i] and np.all(angle[i] != excluded_rads):
                    not_filled[i] = False
            count = np.count_nonzero(not_filled).item()

        return angle

    def reset(self, complete=True):
        restart = self.players.to_restart()
        complete |= self.players.total != self.total # or np.all(restart == True).item()
        if complete:
            self.total = self.players.total
            self.x = np.full((self.total,), self.original_x, dtype=FLOAT)
            self.y = np.full((self.total,), self.original_y, dtype=FLOAT)
        else:
            self.x[restart] = self.original_x
            self.y[restart] = self.original_y

        # Restrict angle between -30° and +30° (converted to radians)
        angle = self._get_random_angle(-self.LAUNCH_ANGLE, self.LAUNCH_ANGLE, [0])
        pos = np.full((self.total,), 1 if random.random() < 0.5 else -1, dtype=INT)
        left, right = self.players.left, self.players.right
        angle[right] = angle[left]
        pos[right] = pos[left]

        if complete:
            self.x_vel = pos * np.abs(np.cos(angle) * self.MAX_VEL)
            self.y_vel = np.sin(angle) * self.MAX_VEL
        else:
            _pos = pos[restart]
            _angle = angle[restart]
            self.x_vel[restart] = _pos * np.abs(np.cos(_angle) * self.MAX_VEL)
            self.y_vel[restart] = np.sin(_angle) * self.MAX_VEL

    def restart(self):
        self.reset(complete=False)

    def move(self):
        active = self.players.game_active
        self.x[active] += self.x_vel[active]
        self.y[active] += self.y_vel[active]

    def render(self, window: pygame.Surface, ranking: list[int]):
        # Ranking should be sorted by best at last index
        best_index = ranking[-1]
        pygame.draw.circle(window, (255, 255, 255), (self.x[best_index], self.y[best_index]), self.RADIUS)


if __name__ == "__main__":
    import random

    test_players = Players(lives=1)
    test_balls = Balls(test_players, 500, 500, (1000, 1000))

    test_players.reset(10)
    test_balls.reset()
    print(f"x: {test_balls.x}")
    print(f"y: {test_balls.y}")

    for i in range(test_players.total // 2):
        if random.random() < 0.5:
            test_players.lives[i*2] = 0
            test_players.lives[i*2+1] = 0
    ga = test_players.game_active
    print(f"active: {[(ga[i*2] and ga[i*2+1]).item() for i in range(test_players.total // 2)]}")

    for _ in range(10):
        test_balls.move()
    print(f"x: {test_balls.x.round()}")
    print(f"y: {test_balls.y.round()}")

    for i in range(test_players.total):
        test_players.lives[i] = test_players.lives_total
    test_balls.restart()
    for _ in range(10):
        test_balls.move()
    print(f"x: {test_balls.x.round()}")
    print(f"y: {test_balls.y.round()}")

    pass