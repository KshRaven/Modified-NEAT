"""Reward shaping. Per-frame lives, score and fitness live in player.py."""
from __future__ import annotations

import numpy as np
from numpy import ndarray as Array

from . import kernels
from .config import RewardConfig
from .constants import NP_FLOAT

__all__ = ["RewardShaper"]


class RewardShaper:
    """Dense shaping terms for every snake: progress towards the food, time, starvation and looping."""

    def __init__(self, config: RewardConfig, max_hiatus: int = 100):
        self.config = config
        self.max_hiatus = int(max_hiatus)
        self.allocate(1)

    def allocate(self, total: int) -> None:
        """Per-snake arrays, recreated on every reset."""
        n = (total,)
        self.shape_pos = np.zeros(n, dtype=NP_FLOAT)
        self.shape_neg = np.zeros(n, dtype=NP_FLOAT)
        self.prev_dist = np.zeros(n, dtype=NP_FLOAT)    # food distance of the previous frame, in cells

    def update(self, dist: Array, ate: Array, died: Array, hiatus: Array, low_usage: Array, active: Array) -> None:
        """Compute `shape_pos` / `shape_neg` for this frame. Call before `Players.update`."""
        rw = self.config
        if not rw.shaping:
            self.shape_pos[:] = 0.0
            self.shape_neg[:] = 0.0
            return
        kernels.reward_shape(
            self.shape_pos, self.shape_neg,
            dist, self.prev_dist, ate, died, hiatus, low_usage, active,
            int(self.max_hiatus), float(rw.closer_weight), float(rw.away_weight), float(rw.time_penalty),
            float(rw.hiatus_weight), float(rw.loop_weight),
        )
