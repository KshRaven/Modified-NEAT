"""Global constants shared across the package."""
from __future__ import annotations

import numpy as np

__all__ = [
    "NP_INT", "NP_FLOAT",
    "ACT_NOOP", "ACT_FLAP",
    "COLOR_SKY_TOP", "COLOR_SKY_BOTTOM", "COLOR_HILL", "COLOR_CLOUD",
    "COLOR_PIPE", "COLOR_PIPE_DARK", "COLOR_PIPE_LIGHT",
    "COLOR_FLOOR", "COLOR_FLOOR_ALT", "COLOR_FLOOR_EDGE",
    "COLOR_RAY", "COLOR_GAP", "COLOR_HITBOX", "COLOR_TEXT",
]

# Array dtypes
NP_INT   = np.int64
NP_FLOAT = np.float64

# Actions: the bird either does nothing or flaps (jumps)
ACT_NOOP, ACT_FLAP = 0, 1

# Colors (the birds themselves are drawn in rainbow hues, see sprite.py)
COLOR_SKY_TOP    = (78, 192, 202)
COLOR_SKY_BOTTOM = (176, 226, 226)
COLOR_HILL       = (140, 205, 160)
COLOR_CLOUD      = (245, 252, 250)
COLOR_PIPE       = (115, 191, 46)
COLOR_PIPE_DARK  = (63, 117, 26)
COLOR_PIPE_LIGHT = (176, 232, 96)
COLOR_FLOOR      = (222, 216, 149)
COLOR_FLOOR_ALT  = (205, 197, 128)
COLOR_FLOOR_EDGE = (115, 191, 46)
COLOR_RAY        = (255, 60, 60)
COLOR_GAP        = (255, 255, 255)
COLOR_HITBOX     = (255, 255, 0)
COLOR_TEXT       = (255, 255, 255)
