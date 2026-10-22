"""Global constants shared across the package."""
from __future__ import annotations

import numpy as np

__all__ = [
    "NP_INT", "NP_FLOAT", "NP_CELL",
    "CELL_EMPTY", "CELL_FOOD", "CELL_BODY", "CELL_HEAD",
    "ACT_LEFT", "ACT_STRAIGHT", "ACT_RIGHT",
    "DIR_DX", "DIR_DY",
    "COLOR_EMPTY", "COLOR_EMPTY_ALT", "COLOR_WALL", "COLOR_WRAP", "COLOR_FOOD", "COLOR_FOOD_LEAF",
    "COLOR_HEAD", "COLOR_BODY", "COLOR_BODY_ALT", "COLOR_RAY", "COLOR_RAY_FOOD", "COLOR_TEXT",
]

# Array dtypes
NP_INT   = np.int64
NP_FLOAT = np.float64
NP_CELL  = np.int8

# Cell codes of the occupancy map (Grid.cells)
CELL_EMPTY = 0
CELL_FOOD  = 1
CELL_BODY  = 2
CELL_HEAD  = 3

# Relative actions (the snake turns, it never reverses)
ACT_LEFT, ACT_STRAIGHT, ACT_RIGHT = 0, 1, 2

# Direction vectors, index = Direction value (RIGHT, UP, LEFT, DOWN). Screen coordinates: y grows down.
DIR_DX = np.array([1, 0, -1, 0], dtype=NP_INT)
DIR_DY = np.array([0, -1, 0, 1], dtype=NP_INT)

# Colors
COLOR_EMPTY      = (18, 22, 28)
COLOR_EMPTY_ALT  = (22, 27, 34)
COLOR_WALL       = (255, 165, 0)     # solid boundary (wrap off)
COLOR_WRAP       = (70, 110, 160)    # dashed boundary (wrap on)
COLOR_FOOD       = (225, 45, 45)
COLOR_FOOD_LEAF  = (60, 190, 90)
COLOR_HEAD       = (0, 255, 120)
COLOR_BODY       = (0, 110, 255)
COLOR_BODY_ALT   = (0, 90, 220)
COLOR_RAY        = (255, 191, 0)
COLOR_RAY_FOOD   = (255, 80, 220)
COLOR_TEXT       = (235, 235, 235)