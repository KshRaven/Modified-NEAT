"""Per-player lives, score and fitness (reward) bookkeeping."""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy import ndarray as Array

from .config import RewardConfig, normalize_hiatus
from .constants import NP_FLOAT, NP_INT

__all__ = ["Players"]

_NO_LIMIT = 1 << 60


@njit(parallel=True, fastmath=True, cache=True)
def _update(
    # State, mutated in place
    lives: Array, deaths: Array, scores: Array, true_scores: Array,
    disqualified: Array, completed: Array, frames_done: Array,
    fitness: Array, prev_fitness: Array,
    # Per-frame signals
    ate: Array, hit_wall: Array, hit_self: Array, finished: Array, hiatus: Array, length: Array,
    shape_pos: Array, shape_neg: Array,
    # Scalars
    lives_total: int, max_hiatus: int, shaping: bool,
    food_reward: float, death_penalty: float, complete_reward: float, reward_offset: float, crash_scale: float,
):
    """Update lives, scores and fitness of every player. Returns the mask of players that lost a life."""
    players_total = lives.shape[0]
    eliminated = np.zeros(players_total, dtype=np.bool_)

    for t in prange(players_total):
        # A player without lives is out: its fitness is frozen
        if lives[t] <= 0:
            prev_fitness[t] = fitness[t]
            frames_done[t] += 1
            continue

        elim = hit_wall[t] or hit_self[t] or (hiatus[t] >= max_hiatus)
        eliminated[t] = elim

        # Lives and scores
        if elim:
            disqualified[t] = True
            lives[t] -= 1
            deaths[t] += 1
        if lives[t] < 0:           lives[t] = 0
        if lives[t] > lives_total: lives[t] = lives_total
        if finished[t]:            completed[t] = True
        if ate[t]:                 scores[t] += 1
        if scores[t] > true_scores[t]:
            true_scores[t] = scores[t]

        # Rewards scale with lives left, penalties with deaths
        _lives  = lives[t] + 1
        _deaths = deaths[t] + 1

        prev_fitness[t] = fitness[t]

        fitness[t] += reward_offset
        if shaping:
            fitness[t] += shape_pos[t] * _lives
            fitness[t] -= shape_neg[t] * _deaths
        if ate[t]:      fitness[t] += food_reward * _lives * (1.0 + 0.1 * scores[t])
        if elim:        fitness[t] -= death_penalty * _deaths * crash_scale
        if finished[t]: fitness[t] += complete_reward * _lives

        frames_done[t] += 1

    return eliminated


class Players:
    """Lives, deaths, score and fitness of every player, stored as arrays."""

    def __init__(self, lives: int = 3, max_hiatus: int | None = 100, reward: RewardConfig | None = None):
        self.reward: RewardConfig = reward if reward is not None else RewardConfig()
        self.reward_scale:  float = float(self.reward.reward_scale)     # multiplies the reward from get_reward()
        self.reward_offset: float = float(self.reward.reward_offset)    # added to every living player's reward
        self.total:       int = 1
        self.lives_total: int = int(lives)
        self.max_hiatus = max_hiatus                                    # None = a snake never starves
        self._init_arrays()

    def _init_arrays(self) -> None:
        n = (self.total,)
        self.lives        = np.full(n, self.lives_total, dtype=NP_INT)
        self.deaths       = np.full(n, 0, dtype=NP_INT)
        self.scores       = np.full(n, 0, dtype=NP_INT)     # food eaten in the current life
        self.true_scores  = np.full(n, 0, dtype=NP_INT)     # best score over all lives
        self.disqualified = np.full(n, False, dtype=bool)
        self.completed    = np.full(n, False, dtype=bool)
        self.frames_done  = np.full(n, 0, dtype=NP_INT)
        self.fitness      = np.full(n, 0, dtype=NP_FLOAT)
        self.prev_fitness = np.full(n, 0, dtype=NP_FLOAT)

    def reset(self, total: int) -> None:
        """New episode with `total` players."""
        self.total = int(total)
        self._init_arrays()

    def restart(self, mask: Array | None = None) -> None:
        """Re-activate disqualified players (the masked ones when given); a new life restarts the score."""
        if mask is None:
            mask = self.disqualified.copy()
        self.scores[mask] = 0
        self.disqualified[mask] = False

    @property
    def max_hiatus(self) -> int | None:
        """Frames without food before a life is lost; None = no limit. None / inf / 0 all mean no limit."""
        return self._max_hiatus

    @max_hiatus.setter
    def max_hiatus(self, value) -> None:
        self._max_hiatus = normalize_hiatus(value)

    @property
    def hiatus_limit(self) -> int:
        """Frames without food that cost a life, as an integer for the kernels (huge when there is no limit)."""
        return int(self.max_hiatus) if self.max_hiatus is not None else _NO_LIMIT

    @property
    def active(self) -> Array:
        """Players that still have lives left."""
        return self.lives > 0

    @property
    def active_total(self) -> int:
        return int(np.count_nonzero(self.active))

    def update(
        self,
        ate: Array,             # bool (N,) ate food this frame
        hit_wall: Array,        # bool (N,) hit a wall (only possible when wrap is off)
        hit_self: Array,        # bool (N,) hit its own body
        finished: Array,        # bool (N,) filled the whole board
        hiatus: Array,          # int  (N,) frames since the last meal
        length: Array,          # int  (N,) snake length
        shape_pos: Array | None = None,     # float (N,) dense positive shaping
        shape_neg: Array | None = None,     # float (N,) dense negative shaping
        verbose: bool = False,
        crash_scale: float = 1.0,           # multiplier on the death penalty (curriculum)
    ) -> Array:
        """Advance lives, scores and fitness by one frame. Returns the mask of players that lost a life."""
        if shape_pos is None:
            shape_pos = np.zeros(self.total, dtype=NP_FLOAT)
        if shape_neg is None:
            shape_neg = np.zeros(self.total, dtype=NP_FLOAT)
        rw = self.reward
        eliminated = _update(
            self.lives, self.deaths, self.scores, self.true_scores,
            self.disqualified, self.completed, self.frames_done,
            self.fitness, self.prev_fitness,
            ate, hit_wall, hit_self, finished, hiatus, length,
            shape_pos, shape_neg,
            self.lives_total, self.hiatus_limit, bool(rw.shaping),
            float(rw.food_reward), float(rw.death_penalty), float(rw.complete_reward), float(self.reward_offset), float(crash_scale),
        )

        if verbose:
            for name, flags in (
                ("ate food", ate), ("hit a wall", hit_wall), ("hit themselves", hit_self),
                ("starved", hiatus >= self.hiatus_limit),
            ):
                for index in np.flatnonzero(flags):
                    print(f"Index {index} {name}")
        return eliminated

    @property
    def best_index(self) -> int:
        score = self.true_scores.astype(NP_FLOAT) * (self.lives + 1)
        score[~self.active] = -np.inf
        return int(score.argmax())

    def get_reward(self) -> Array:
        """Fitness gained since the previous `update`, times `reward_scale`."""
        return (self.fitness - self.prev_fitness) * self.reward_scale

    @property
    def scaled_fitness(self) -> Array:
        """Accumulated fitness in the units the learner sees (fitness x `reward_scale`)."""
        return self.fitness * self.reward_scale

    def __str__(self) -> str:
        return f"Players(players={self.total}, active={self.active_total}, best_index={self.best_index})"