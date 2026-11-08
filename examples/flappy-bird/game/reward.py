"""Reward shaping and the learning curriculum. Per-frame lives, score and fitness live in player.py.

The dense terms (see `kernels.reward.reward_shape`) pull the bird to the centre of the next gap; the optional
`gap` mode adds the safe corridor between the pipe edges. A skill curriculum (survive first, then be precise) is graded
from the finished episode on every reset.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray as Array

from . import kernels
from .config import RewardConfig
from .constants import NP_FLOAT

__all__ = ["RewardShaper"]


class RewardShaper:
    """Dense shaping terms for every bird plus the skill curriculum."""

    def __init__(self, config: RewardConfig, max_hiatus: int | None = 300, world=None):
        self.config = config
        self.max_hiatus = None if max_hiatus is None else int(max_hiatus)   # None = no limit
        self.world = world
        self.skill_value = float(config.skill_start)
        self.gap_on = False                     # corridor terms of the current episode (resolved on allocate)
        self.total = 0
        self.allocate(1)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    @property
    def skill(self) -> float:
        """Curriculum progress in [0, 1] (1 when the curriculum is off)."""
        return float(self.skill_value) if self.config.curriculum else 1.0

    @property
    def crash_scale(self) -> float:
        """Multiplier on the death cost, larger early in the curriculum."""
        return 1.0 + self.config.early_crash_mult * (1.0 - self.skill)

    def grade_episode(self, pipes: Array, deaths: Array) -> None:
        """Update the skill from the finished episode. Call on reset, before the stats are cleared."""
        if pipes is None or deaths is None or pipes.size == 0 or pipes.size != deaths.size:
            return
        self.skill_value = kernels.next_skill(self.skill_value, pipes, deaths, self.config)

    def _resolve_gap(self) -> bool:
        mode = self.config.mode
        if mode == "staged":
            return self.skill >= self.config.staged_switch
        return mode == "gap"

    @property
    def mode_name(self) -> str:
        return "gap" if self.gap_on else "basic"

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate(self, total: int) -> None:
        """Per-bird arrays, recreated on every reset (the mode of the episode is chosen here)."""
        n = (int(total),)
        self.total = int(total)
        self.gap_on = self._resolve_gap()
        self.shape_pos = np.zeros(n, dtype=NP_FLOAT)
        self.shape_neg = np.zeros(n, dtype=NP_FLOAT)
        self.prev_err = np.zeros(n, dtype=NP_FLOAT)     # distance to the gap centre of the previous frame, pixels

    def _hiatus_ref(self) -> int:
        """Frames the hiatus terms are measured against: the limit, or a few pipe spacings when there is none."""
        if self.max_hiatus is not None:
            return int(self.max_hiatus)
        w = self.world
        return int(4 * w.pipe_spacing / max(w.config.velocity, 1e-6)) if w is not None else 200

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def update(
        self, err: Array, moved: Array, died: Array, hiatus: Array, low_usage: Array,
        active: Array, flap: Array, cy: Array, gap_top: float, gap_bot: float, dist_x: float,
    ) -> None:
        """Compute `shape_pos` / `shape_neg` for this frame. Call after the move, before `Players.update`.
        `moved` = the target pipe changed this frame (no progress term)."""
        rw, w = self.config, self.world
        if not rw.shaping:
            self.shape_pos[:] = 0.0
            self.shape_neg[:] = 0.0
            return
        kernels.reward_shape(
            self.shape_pos, self.shape_neg,
            err, self.prev_err, moved, died, hiatus, low_usage, active, flap, cy,
            float(gap_top), float(gap_bot), float(dist_x), float(w.height), float(w.floor_y),
            bool(self.gap_on), int(self.max_hiatus or 1 << 30), self._hiatus_ref(), float(self.skill), float(rw.distance_unit),
            float(rw.closer_weight), float(rw.away_weight), float(rw.time_penalty), float(rw.hiatus_weight),
            float(rw.loop_weight), float(rw.alive_weight), float(rw.death_dist_weight), float(rw.slow_cap),
            float(rw.approach_dist), float(rw.corridor_margin), float(rw.corridor_weight), float(rw.outside_weight),
            float(rw.edge_weight), float(rw.edge_frac), float(rw.flap_weight),
        )