"""Numba kernels operating on per-bird arrays and the pipe ring."""
from __future__ import annotations

from . import physics, pipes, reward, sensors
from .physics import spawn, step, collide, low_usage
from .pipes import init_pipes, scroll, targets
from .reward import next_skill, reward_shape
from .sensors import (
    get_state, N_FEATURES_BASIC, N_FEATURES_FULL, N_FEATURES_FULL_MOVING, N_FEATURES_EGO,
)

__all__ = [
    # modules
    "physics", "pipes", "reward", "sensors",
    # physics
    "spawn", "step", "collide", "low_usage",
    # pipes
    "init_pipes", "scroll", "targets",
    # sensors
    "get_state", "N_FEATURES_BASIC", "N_FEATURES_FULL", "N_FEATURES_FULL_MOVING", "N_FEATURES_EGO",
    # reward
    "reward_shape", "next_skill",
]
