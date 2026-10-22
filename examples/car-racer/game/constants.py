"""Global constants shared across the package."""
from __future__ import annotations

import os

import numpy as np

__all__ = [
    "CURR_DIR", "IMAGE_DIR",
    "NP_INT", "NP_FLOAT",
    "COLOR_ROAD", "COLOR_GRASS", "COLOR_CURB_PRI", "COLOR_CURB_SEC", "COLOR_START_PRI",
    "COLOR_START_SEC", "COLOR_GRAVEL", "COLOR_EMPTY", "COLOR_LIDAR",
    "SURFACE_ROAD", "SURFACE_CURB", "SURFACE_GRASS", "SURFACE_GRAVEL", "SURFACE_VOID", "SURFACE_CAR",
    "L_ROAD", "L_CURB", "L_GRASS", "L_GRAVEL", "L_TRACK", "L_BOUND", "L_FINISH",
    "T_START", "T_STRAIGHT", "T_LEFT", "T_RIGHT",
    "TRAIL_RESET_CARS", "DRAFT_TREND_EPS", "DRAFT_VISIBLE", "DRAFT_TOWER_MARK", "STREAK_REAR_CARS",
]

# Paths
CURR_DIR  = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(CURR_DIR, "imgs")

# Numba dtypes NOTE: Left for reference purposes
# INT         = types.int64
# FLOAT       = types.float64
# STR         = typeof('str')
# BOOL        = types.bool
# ARRAY_1D    = types.Array(FLOAT, 1, 'C')
# COUNTER_1D  = types.Array(INT, 1, 'C')
# MASK_1D     = types.Array(BOOL, 1, 'C')

# Array dtypes
NP_INT   = np.int64
NP_FLOAT = np.float64

# Track colors
COLOR_ROAD      = (31, 32, 34)      # Rich gray
COLOR_GRASS     = (21, 71, 52)      # Forest Green
COLOR_CURB_PRI  = (187, 0, 0)       # Scarlet Red
COLOR_CURB_SEC  = (240, 240, 240)   # White
COLOR_START_PRI = (245, 245, 245)   # White
COLOR_START_SEC = (15, 15, 15)      # Black
COLOR_GRAVEL    = (137, 81, 41)     # Brown
COLOR_EMPTY     = (54, 69, 79)      # Sage Gray
COLOR_LIDAR     = (255, 191, 0)     # Yellow

# Surface codes (Cars.surface and the LiDAR hit map)
SURFACE_ROAD   = np.int8(0)
SURFACE_CURB   = np.int8(1)
SURFACE_GRASS  = np.int8(2)
SURFACE_GRAVEL = np.int8(3)
SURFACE_VOID   = np.int8(4)     # off track / out of bounds
SURFACE_CAR    = np.int8(5)     # LiDAR hit on another car

# Bit flags of the packed uint8 (W, H) surface map
L_ROAD, L_CURB, L_GRASS, L_GRAVEL, L_TRACK, L_BOUND, L_FINISH = 1, 2, 4, 8, 16, 32, 64

# Tile type codes (values of functional.Type)
T_START, T_STRAIGHT, T_LEFT, T_RIGHT = 0, 1, 2, 3

# Slipstream
TRAIL_RESET_CARS = 4.0      # pose jump (in car lengths) treated as a respawn; clears the trail
DRAFT_TREND_EPS  = 1e-4     # draft level changes below this keep the previous gain/decay color
DRAFT_VISIBLE    = 0.05     # minimum draft level for a streak to be drawn
DRAFT_TOWER_MARK = 0.15     # draft level that adds "*" to the leaderboard row
STREAK_REAR_CARS = 0.5      # streak start behind the car center, in car lengths