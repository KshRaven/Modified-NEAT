
from .functional import INT, BOOL, FLOAT, MASK_1D, COUNTER_1D, ARRAY_1D

from numba.experimental import jitclass
from numpy import ndarray as CPUArray

import numpy as np


INF_MIN = FLOAT(-np.finfo(np.float32).max.item())


@jitclass([
    ('total', INT),
    ('lives_total', INT),
    ('lives', COUNTER_1D),
    ('scores', COUNTER_1D),
    ('disqualified', MASK_1D),
    ('completed', MASK_1D),
    ('frames_done', COUNTER_1D),
    ('fitness', ARRAY_1D),
])
class Players(object):
    def __init__(self, lives=3):
        self.total = 1
        self.lives_total = lives
        self.lives = np.full((self.total,), self.lives_total, dtype=INT)
        self.scores = np.full((self.total,), 0, dtype=INT)
        self.disqualified = np.full((self.total,), False, dtype=BOOL)
        self.completed = np.full((self.total,), False, dtype=BOOL)
        self.frames_done = np.full((self.total,), 0, dtype=INT)
        self.fitness = np.full((self.total,), 0, dtype=FLOAT)

    def reset(self, total: int | list[int]):
        self.total = total
        self.lives = np.full((self.total,), self.lives_total, dtype=INT)
        self.scores = np.full((self.total,), 0, dtype=INT)
        self.disqualified = np.full((self.total,), False, dtype=BOOL)
        self.completed = np.full((self.total,), False, dtype=BOOL)
        self.frames_done = np.full((self.total,), 0, dtype=INT)
        self.fitness = np.full((self.total,), 0, dtype=FLOAT)

    def restart(self):
        self.frames_done[self.disqualified] = 0
        self.disqualified[self.disqualified] = False

    @property
    def active(self):
        return self.lives > 0

    @property
    def active_total(self):
        return np.count_nonzero(self.active).item()

    def to_restart(self):
        return self.disqualified

    def update(self,
               ate_food: CPUArray, hit_wall: CPUArray, hit_self: CPUArray, completed: CPUArray,
               distances: CPUArray, hiatus: CPUArray, moved_closer: CPUArray,
               verbose=False):
        died = hit_wall | hit_self
        self.lives[died] -= 1
        self.lives = np.clip(self.lives, 0, self.lives_total)
        self.disqualified[died] = True
        active = self.active & ~completed
        inactive = ~active
        if np.any(np.isnan(distances[active])):
            raise ValueError("An active player cannot have an NaN distance value")
        _hiatus = np.clip(hiatus, 1, None)
        _moved_closer_p = active & moved_closer
        _moved_closer_n = active & (~moved_closer)
        self.scores[ate_food & active] += 1
        self.fitness[ate_food & active] += 100
        self.fitness[died] -= 100
        self.fitness[_moved_closer_p] += 1
        self.fitness[_moved_closer_n] -= 1 * distances[_moved_closer_n] * _hiatus[_moved_closer_n]
        # self.fitness[active] += 0.00000 + ((-1e-0) * distances[active] * _hiatus[active])
        self.fitness[inactive] -= 1e-0 * self.frames_done[inactive]
        self.fitness[completed] += 10000
        self.completed[completed] |= True

        self.frames_done += 1

        if verbose:
            for index, loss in enumerate(ate_food):
                if loss:
                    print(f"Index {index} ate food")
            for index, loss in enumerate(hit_wall):
                if loss:
                    print(f"Index {index} hit a wall")
            for index, loss in enumerate(hit_self):
                if loss:
                    print(f"Index {index} hit themselves")

    @property
    def best_index(self):
        total_score = self.scores * (self.lives + 1)
        total_score[~self.active] = -np.inf # += INF_MIN
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
