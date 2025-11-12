
from .functional import INT, BOOL, FLOAT, MASK_1D, COUNTER_1D, ARRAY_1D

from numba.experimental import jitclass

import numpy as np


@jitclass([
    ('total', INT),
    ('opponent', MASK_1D),
    ('lives_total', INT),
    ('lives', COUNTER_1D),
    ('scores', COUNTER_1D),
    ('hits', COUNTER_1D),
    ('disqualified', MASK_1D),
    ('frames_done', COUNTER_1D),
    ('fitness', ARRAY_1D),
])
class Players(object):
    def __init__(self, lives=3):
        self.total = 2
        self.opponent = np.full((self.total,), False, dtype=BOOL)
        self.opponent[(np.arange(self.total) % 2) != 0] = True
        self.lives_total = lives
        self.lives = np.full((self.total,), self.lives_total, dtype=INT)
        self.scores = np.full((self.total,), 0, dtype=INT)
        self.hits = np.full((self.total,), 0, dtype=INT)
        self.disqualified = np.full((self.total,), False, dtype=BOOL)
        self.frames_done = np.full((self.total,), 0, dtype=INT)
        self.fitness = np.full((self.total,), 0, dtype=FLOAT)

    def reset(self, total: int | list[int]):
        assert total % 2 == 0
        self.total = total
        self.opponent = np.full((self.total,), False, dtype=BOOL)
        self.opponent[(np.arange(self.total) % 2) != 0] = True
        self.lives = np.full((self.total,), self.lives_total, dtype=INT)
        self.scores = np.full((self.total,), 0, dtype=INT)
        self.hits = np.full((self.total,), 0, dtype=INT)
        self.disqualified = np.full((self.total,), False, dtype=BOOL)
        self.frames_done = np.full((self.total,), 0, dtype=INT)
        self.fitness = np.full((self.total,), 0, dtype=FLOAT)

    def restart(self):
        self.frames_done[self.disqualified == True] = 0
        self.disqualified[self.disqualified == True] = False

    @property
    def left(self):
        return ~self.opponent

    @property
    def right(self):
        return self.opponent

    @property
    def active(self):
        return self.lives > 0

    @property
    def active_total(self):
        return np.count_nonzero(self.active).item()

    @property
    def game_active(self):
        active = self.active
        left, right = self.left, self.right
        active[left] |= active[right]
        active[right] |= active[left]
        return active

    def to_restart(self):
        activate = self.disqualified
        left, right = self.left, self.right
        activate[left] |= activate[right]
        activate[right] |= activate[left]
        return activate

    def update(self, deflections: np.ndarray, strikes: np.ndarray, verbose=False):
        left, right = self.left, self.right
        left_strikes = strikes[left]
        right_strikes = strikes[right]
        if np.any((left_strikes == right_strikes) & (left_strikes == True)):
            raise ValueError("You cannot strike a left and right at the same time")

        # Gain a point for deflecting the ball
        active = self.active
        self.hits[active & deflections] += 1
        if verbose:
            for index, loss in enumerate(active & deflections):
                if loss:
                    print(f"Index {index} has hit a ball")

        # Lose a life when a player misses a ball
        self.lives[strikes] -= 1
        self.lives = np.clip(self.lives, 0, self.lives_total)
        self.disqualified[strikes] = True
        if verbose:
            for index, loss in enumerate(strikes):
                if loss:
                    print(f"Index {index} has lost a life")

        # Update player activity
        active = self.active

        # Opposite side gains a point when a paddle misses
        scored = strikes.copy()
        scored[left] = right_strikes
        scored[right] = left_strikes
        self.scores[active & scored] += 1

        self.frames_done[self.active] += 1

    @property
    def best_index(self):
        total_score = self.scores + self.hits
        # minimum, maximum = total_score.min(), total_score.max()
        # probs = np.random.rand(*total_score.shape) + ((total_score - minimum) / (maximum - minimum + 1e-6)) * 0.50
        # return probs.argmax().item()
        return total_score.argmax().item()

    def __str__(self):
        return f"Players(players={self.total}, active={self.active_total}, best_index={self.best_index})"


PLAYERS_TYPE = Players.class_type.instance_type


if __name__ == '__main__':
    import random

    test_lives = 3
    players = Players(lives=test_lives)
    print(players)

    players.reset(10)
    print(players)
    print(f"left : {players.left}")
    print(f"right : {players.right}")

    for _ in range(test_lives):
        print("\n")
        deflection = np.random.randint(2, size=players.total).astype(bool)
        strike = np.random.randint(2, size=players.total).astype(bool)
        # print(f"strikes : {strike}")
        for i in range(len(strike)):
            if strike[i]:
                if i % 2 == 0 and strike[i+1]:
                    if random.random() < 0.5:
                        strike[i] = False
                if i % 2 == 1 and strike[i-1]:
                    strike[i] = False
        deflection &= strike == False
        print(f"deflection : {deflection}")
        print(f"strikes : {strike}")
        players.update(deflection, strike)
        print(players)
        print(f"lives  : {players.lives}")
        print(f"active : {players.active.astype(int)}")
        print(f"hits   : {players.hits}")
        print(f"scores : {players.scores}")
