"""Numba kernels operating on per-snake arrays."""
from __future__ import annotations

from . import movement, reward, sensors
from .movement import food_distance, low_usage, place_food, spawn, step
from .reward import reward_shape
from .sensors import N_FEATURES, cast_rays, get_state, grid_state

__all__ = [
    # modules
    "movement", "reward", "sensors",
    # movement
    "spawn", "step", "place_food", "food_distance", "low_usage",
    # sensors
    "cast_rays", "get_state", "grid_state", "N_FEATURES",
    # reward
    "reward_shape",
]
