"""Per-player lives, score and fitness (reward) bookkeeping."""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy import ndarray as Array

from .config import RewardConfig
from .constants import NP_FLOAT, NP_INT
from .util import grid

__all__ = ["Players"]


@njit(parallel=True, fastmath=True, cache=True)
def _update(
    # State, mutated in place
    lives: Array, deaths: Array, scores: Array, true_scores: Array,
    disqualified: Array, completed: Array, frames_done: Array,
    fitness: Array, prev_fitness: Array,
    # Per-frame signals
    checked: Array, finished: Array, out_of_bounds: Array, hiatus: Array,
    moved_forward: Array, tile_stag: Array, vel_norm: Array, shape_pos: Array, shape_neg: Array,
    corner_tile_curr: Array, straight_tile_curr: Array,
    on_road: Array,
    # Scalars
    lives_total: int, max_hiatus: int,
    shaping: bool, crash_penalty: float, crash_speed_k: float, crash_scale: float, reward_offset: float,
):
    """Update lives, scores and fitness of every player. Returns the mask of players that lost a life."""
    players_total = lives.shape[0]
    eliminated = np.zeros(players_total, dtype=np.bool_)
    total, indices = grid((players_total,))

    for i in prange(total):
        t = indices[i]

        # A player without lives is out: its fitness is frozen
        if lives[t] <= 0:
            prev_fitness[t] = fitness[t]
            frames_done[t] += 1
            continue

        elim = out_of_bounds[t] or (hiatus[t] >= max_hiatus) or tile_stag[t] or (not moved_forward[t])
        eliminated[t] = elim

        # Lives and scores
        if elim:        disqualified[t] = True
        if finished[t]: completed[t]    = True
        if elim:        lives[t]       -= 1
        if lives[t] < 0:           lives[t] = 0
        if lives[t] > lives_total: lives[t] = lives_total
        if elim:        deaths[t]      += 1

        if checked[t]:  scores[t] += 1
        if finished[t]: scores[t] += 10
        if elim:        scores[t] -= 1
        if scores[t] > true_scores[t]:
            true_scores[t] = scores[t]

        # Rewards scale with lives left, penalties with deaths
        _lives  = lives[t] + 1
        _deaths = deaths[t] + 1
        if on_road[t]: _lives  *= 2
        else:          _deaths *= 2

        prev_fitness[t] = fitness[t]

        # Per-frame reward
        fitness[t] += reward_offset
        if shaping:
            fitness[t] += shape_pos[t] * _lives
            fitness[t] -= shape_neg[t] * _deaths
        else:
            if on_road[t]: fitness[t] += 5.0 * _lives
            else:          fitness[t] -= 2.0 * _deaths

        # Checkpoint and lap rewards
        if checked[t]:
            if straight_tile_curr[t]: fitness[t] += 250.0 * _lives
            if corner_tile_curr[t]:   fitness[t] += 300.0 * _lives
        if finished[t]: fitness[t] += 2000.0 * _lives

        # Crash penalty, heavier at speed and boosted by `crash_scale` early in the curriculum
        if elim:
            if shaping:
                fitness[t] -= crash_penalty * crash_scale * (1.0 + crash_speed_k * abs(vel_norm[t])) * _deaths
            else:
                if straight_tile_curr[t]: fitness[t] -= 30.0 * _deaths
                if corner_tile_curr[t]:   fitness[t] -= 30.0 * _deaths

        frames_done[t] += 1

    return eliminated


class Players:
    """Lives, deaths, score and fitness of every player, stored as arrays."""

    def __init__(self, lives: int = 3, max_hiatus: int = 100, reward: RewardConfig | None = None):
        self.reward: RewardConfig = reward if reward is not None else RewardConfig()
        self.reward_scale:  float = float(self.reward.reward_scale)     # multiplies the reward from get_reward()
        self.reward_offset: float = float(self.reward.reward_offset)    # added to every living player's reward
        self.total:       int = 1
        self.lives_total: int = int(lives)
        self.max_hiatus:  int = int(max_hiatus)
        self._init_arrays()

    def _init_arrays(self) -> None:
        n = (self.total,)
        self.lives        = np.full(n, self.lives_total, dtype=NP_INT)
        self.deaths       = np.full(n, 0, dtype=NP_INT)
        self.scores       = np.full(n, 0, dtype=NP_INT)
        self.true_scores  = np.full(n, 0, dtype=NP_INT)
        self.disqualified = np.full(n, False, dtype=bool)
        self.completed    = np.full(n, False, dtype=bool)
        self.frames_done  = np.full(n, 0, dtype=NP_INT)
        self.fitness      = np.full(n, 0, dtype=NP_FLOAT)
        self.prev_fitness = np.full(n, 0, dtype=NP_FLOAT)

    def reset(self, total: int) -> None:
        """New episode with `total` players."""
        self.total = int(total)
        self._init_arrays()

    def restart(self) -> None:
        """Re-activate disqualified players."""
        self.disqualified[:] = False

    @property
    def active(self) -> Array:
        """Players that still have lives left."""
        return self.lives > 0

    @property
    def active_total(self) -> int:
        return int(np.count_nonzero(self.active))

    def update(
        self,
        checked: Array,             # bool (N,) passed a checkpoint this frame
        finished: Array,            # bool (N,) completed a lap this frame
        out_of_bounds: Array,       # bool (N,) crashed or left the track
        hiatus: Array,              # int  (N,) frames since the last forward move
        moved_forward: Array,       # bool (N,) heading within the accepted arc
        tile_stag: Array,           # bool (N,) stuck on one tile too long
        corner_tile_curr: Array,    # bool (N,) current tile is a turn
        straight_tile_curr: Array,  # bool (N,) current tile is a straight or the start
        vel_norm: Array,            # float (N,) speed / max speed
        on_road: Array,             # bool (N,) body on the road only
        shape_pos: Array | None = None,     # float (N,) dense positive shaping
        shape_neg: Array | None = None,     # float (N,) dense negative shaping
        crash_scale: float = 1.0,           # multiplier on the crash penalty
        verbose: bool = False,
    ) -> Array:
        """Advance lives, scores and fitness by one frame. Returns the mask of players that lost a life."""
        if shape_pos is None:
            shape_pos = np.zeros(self.total, dtype=NP_FLOAT)
        if shape_neg is None:
            shape_neg = np.zeros(self.total, dtype=NP_FLOAT)
        eliminated = _update(
            self.lives, self.deaths, self.scores, self.true_scores,
            self.disqualified, self.completed, self.frames_done,
            self.fitness, self.prev_fitness,
            checked, finished, out_of_bounds, hiatus, moved_forward, tile_stag,
            vel_norm, shape_pos, shape_neg,
            corner_tile_curr, straight_tile_curr,
            on_road,
            self.lives_total, self.max_hiatus,
            bool(self.reward.shaping), float(self.reward.crash_penalty), float(self.reward.crash_speed_k),
            float(crash_scale), float(self.reward_offset),
        )

        if verbose:
            for name, flags in (
                ("passed checkpoint", checked), ("completed a lap", finished),
                ("went out of bounds", out_of_bounds), ("stagnated too long", hiatus >= self.max_hiatus),
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