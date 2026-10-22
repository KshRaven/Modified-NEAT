"""Numba kernels operating on per-car arrays."""
from __future__ import annotations

from . import physics, race, reward, sensors, tracking
from .physics import blocked, drive, steer, violations, walls
from .race import collide, slipstream, trail_points, trail_push
from .reward import corner_factor, corner_speeds, next_skill, reward_shape, traffic_shape
from .sensors import get_state, lidar_cars, pack_layers, tile_beam, update_lidar, update_regions, update_surface
from .tracking import progress, signals, update_location

__all__ = [
    # modules
    "physics", "race", "reward", "sensors", "tracking",
    # physics
    "steer", "drive", "violations", "blocked", "walls",
    # sensors
    "pack_layers", "update_surface", "update_lidar", "lidar_cars", "update_regions", "tile_beam", "get_state",
    # tracking
    "update_location", "signals", "progress",
    # reward
    "corner_factor", "traffic_shape", "reward_shape", "corner_speeds", "next_skill",
    # race
    "slipstream", "collide", "trail_push", "trail_points",
]