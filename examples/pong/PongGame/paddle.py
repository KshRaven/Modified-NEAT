
from .functional import INT, FLOAT, BOOL, ARRAY_1D, MASK_1D
from .player import Players, PLAYERS_TYPE

from numba.experimental import jitclass

import pygame
import numpy as np


class Paddle:
    VEL = 4
    WIDTH = 20
    HEIGHT = 100

    def __init__(self, x, y):
        self.x = self.original_x = x
        self.y = self.original_y = y

    def draw(self, win):
        pygame.draw.rect(
            win, (255, 255, 255), (self.x, self.y, self.WIDTH, self.HEIGHT))

    def move(self, up=True):
        if up:
            self.y -= self.VEL
        else:
            self.y += self.VEL

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y

@jitclass([
    ('VEL', FLOAT),
    ('WIDTH', INT),
    ('HEIGHT', INT),
    ('players', PLAYERS_TYPE),
    ('total', INT),
    ('window_width', INT),
    ('window_height', INT),
    ('original_x', INT),
    ('original_y', INT),
    ('x', ARRAY_1D),
    ('y', ARRAY_1D),
    ('prev_y', ARRAY_1D),
])
class Paddles(object):
    def __init__(self, players: Players,
                 x: int, y: int, window_shape: tuple[int, int],
                 velocity: float = 4, width: int = 20, height: int = 100):
        self.VEL = velocity
        self.WIDTH = width
        self.HEIGHT = height

        self.players = players
        self.total = self.players.total
        self.window_width, self.window_height = window_shape
        self.original_x = min(max(x, 0), self.window_width // 2 - self.WIDTH)
        self.original_y = min(max(self.window_height - y - self.HEIGHT // 2, 0), self.window_height - self.HEIGHT // 2)
        self.x = np.full((self.total,), self.original_x, dtype=FLOAT)
        self.x[self.players.right] = self.window_width - self.original_x - self.WIDTH
        self.y = np.full((self.total,), self.original_y, dtype=FLOAT)
        self.prev_y = self.y.copy()

    def reset(self, complete=True):
        restart = self.players.to_restart()
        complete |= self.players.total != self.total # or np.all(restart == True).item()
        if complete:
            self.total = self.players.total
            self.x = np.full((self.total,), self.original_x, dtype=FLOAT)
            self.x[self.players.right] = self.window_width - self.original_x - self.WIDTH
            self.y = np.full((self.total,), self.original_y, dtype=FLOAT)
            self.prev_y = self.y.copy()
        else:
            self.x[restart] = self.original_x
            self.x[restart & self.players.right] = self.window_width - self.original_x - self.WIDTH
            self.y[restart] = self.original_y
            self.prev_y[restart] = self.y[restart].copy()

    def restart(self):
        # It might be okay for the paddles to continue from their positions instead of resetting it
        self.reset(complete=False)

    def move(self, direction: np.ndarray):
        self.prev_y = self.y.copy()
        moving = direction != 0
        self.y[moving] += (np.pow(-1, direction[moving]) * self.VEL)
        self.y = np.clip(self.y, 0, self.window_height - self.HEIGHT)

    @property
    def stationary(self):
        return self.y == self.prev_y

    def render(self, window: pygame.Surface, ranking: list[int]):
        # Ranking should be sorted by best at last index
        best_index = ranking[-1]
        pygame.draw.rect(
            window, (255, 255, 255), (self.x[best_index], self.y[best_index], self.WIDTH, self.HEIGHT)
        )

    def __str__(self):
        return f"Paddles(origin=({self.original_x}, {self.original_y}), "\
               f"velocity={self.VEL}, width={self.WIDTH}, height={self.HEIGHT})"


if __name__ == "__main__":
    import random

    test_players = Players()
    test_paddles = Paddles(test_players, 0, 500, (1000, 1000))
    print(test_paddles)

    test_players.reset(10)
    test_paddles.reset()
    print(f"\nx : {test_paddles.x}\ny : {test_paddles.y}")

    for i in range(test_players.total // 2):
        if random.random() < 0.5:
            test_players.lives[i*2] = 0
            test_players.lives[i*2+1] = 0
    ga = test_players.game_active
    print(f"\nactive: {[(ga[i*2] and ga[i*2+1]).item() for i in range(test_players.total // 2)]}")

    for i in range(20):
        test_paddles.move(np.random.randint(0, 3, (10,)))
    print(f"\nx : {test_paddles.x}\ny : {test_paddles.y}")
    test_paddles.restart()
    print(f"\nx : {test_paddles.x}\ny : {test_paddles.y}")

    pass