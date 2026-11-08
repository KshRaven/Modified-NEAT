"""Numba kernels operating on per-snake arrays."""
from __future__ import annotations

from . import movement, reward, road, sensors
from .movement import food_distance, low_usage, place_food, spawn, step
from .reward import next_skill, reward_shape
from .road import build_roads, commit, road_available, scan, spawn_road
from .sensors import N_FEATURES, N_FEATURES_EGO, N_FEATURES_EGO_PLAIN, N_FEATURES_STRATEGY, GRID_CHANNELS, cast_rays, ego_state, get_state, grid_state, strategy_state

__all__ = [
    # modules
    "movement", "reward", "road", "sensors",
    # movement
    "spawn", "step", "place_food", "food_distance", "low_usage",
    # road
    "build_roads", "spawn_road", "scan", "commit", "road_available",
    # sensors
    "cast_rays", "get_state", "grid_state", "strategy_state", "ego_state", "N_FEATURES", "N_FEATURES_STRATEGY", "N_FEATURES_EGO", "N_FEATURES_EGO_PLAIN", "GRID_CHANNELS",
    # reward
    "reward_shape", "next_skill",
]