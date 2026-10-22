"""Reward shaping and the learning curriculum. Per-frame lives, score and fitness live in player.py."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray as Array

from . import kernels
from .config import RewardConfig
from .constants import NP_FLOAT

if TYPE_CHECKING:
    from .car import Cars

__all__ = ["RewardShaper"]


class RewardShaper:
    """Dense shaping terms for every car plus the skill curriculum (easy first, then fast)."""

    def __init__(self, config: RewardConfig, car_config):
        self.config = config
        self.car_config = car_config
        self.skill_value = float(config.skill_start)
        self.corner_f1 = self.corner_f2 = self.corner_f1_curb = self.a_dec = 0.0
        self.allocate(1)

    def allocate(self, total: int) -> None:
        """Per-car arrays, recreated on every reset."""
        n = (total,)
        self.shape_pos = np.zeros(n, dtype=NP_FLOAT)
        self.shape_neg = np.zeros(n, dtype=NP_FLOAT)
        self.target_speed = np.ones(n, dtype=NP_FLOAT)
        self.prev_rotation = np.zeros(n, dtype=NP_FLOAT)
        self.prev_off = np.zeros(n, dtype=bool)
        self.tr_pos = np.zeros(n, dtype=NP_FLOAT)
        self.tr_neg = np.zeros(n, dtype=NP_FLOAT)
        self.tr_edge = np.ones(n, dtype=NP_FLOAT)

    def setup(self, cars: Cars) -> None:
        """Corner speeds derived from the car physics and the track size."""
        self.corner_f1, self.corner_f2, self.corner_f1_curb, self.a_dec = kernels.corner_speeds(
            self.config, self.car_config, float(cars.track.size),
            cars.max_vel_lin, cars.max_vel_ang, cars.acc_lin, cars.coeff_brake,
        )

    @property
    def skill(self) -> float:
        """Curriculum progress in [0, 1] (1 when the curriculum is off)."""
        return float(self.skill_value) if self.config.curriculum else 1.0

    @property
    def crash_scale(self) -> float:
        """Multiplier on the crash penalty, larger early in the curriculum."""
        return 1.0 + self.config.early_crash_mult * (1.0 - self.skill)

    def grade_episode(self, checks_done: Array, deaths: Array | None) -> None:
        """Update the skill from the finished episode. Call on reset, before the stats are cleared."""
        if deaths is None or checks_done.size == 0 or checks_done.size != deaths.size:
            return
        self.skill_value = kernels.next_skill(self.skill_value, checks_done, deaths, self.config)

    def update(self, cars: Cars, sg: dict, rotation: Array, vel_norm: Array, prev_progress: Array) -> None:
        """Compute `shape_pos` / `shape_neg` for this frame. `sg` is the signal buffer from `kernels.signals`."""
        rw = self.config
        if rw.shaping:
            if cars.race_on:
                kernels.traffic_shape(
                    self.tr_pos, self.tr_neg, self.tr_edge,
                    cars.x, cars.y, cars.angle, cars.vel_lin, cars.race_active,
                    cars.progress, prev_progress, sg["on_road"], sg["phase_align"], sg["corner_curr"],
                    float(cars.width), float(cars.height), float(cars.track.size), float(cars.max_vel_lin),
                    float(rw.traffic_range * cars.height), float(rw.traffic_gate), float(rw.edge_guard),
                    float(rw.veer_weight), float(rw.closing_weight), float(rw.lane_weight), float(rw.overtake_bonus),
                )
            else:
                self.tr_pos[:] = 0.0
                self.tr_neg[:] = 0.0
                self.tr_edge[:] = 1.0
            kernels.reward_shape(
                self.shape_pos, self.shape_neg, self.target_speed,
                self.tr_pos, self.tr_neg, self.tr_edge,
                cars.progress, prev_progress, cars.in_tile, cars.vel_lin, vel_norm,
                rotation, self.prev_rotation,
                cars.tile_idx, cars.t_turn, cars.t_next, sg["corner_curr"],
                sg["on_road"], cars.on_curb, cars.on_grass, cars.on_gravel, cars.bumped, self.prev_off,
                float(cars.track.size), float(cars.max_vel_lin), float(self.a_dec),
                float(self.corner_f1), float(self.corner_f2), float(self.corner_f1_curb),
                int(rw.lookahead_tiles), int(rw.corner_window), float(self.skill),
                float(rw.progress_weight), float(rw.time_penalty), float(rw.survival_bonus),
                float(rw.overspeed_weight), float(rw.overspeed_tolerance), float(rw.underspeed_weight),
                float(rw.steer_weight), float(rw.jitter_weight),
                float(rw.curb_weight), float(rw.curb_straight_mult), float(rw.curb_corner_discount),
                float(rw.grass_weight), float(rw.gravel_weight), float(rw.bump_weight),
                float(rw.offroad_progress), float(rw.offroad_entry_penalty),
                float(rw.offroad_corner_mult), float(rw.exit_gain), float(rw.early_surface_mult),
            )
        self.prev_rotation[:] = rotation
        self.prev_off[:] = cars.on_grass | cars.on_gravel