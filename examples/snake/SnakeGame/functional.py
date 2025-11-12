
import numpy as np

from numba import njit, prange, types, typeof
from enum import Enum

INT         = types.int64
FLOAT       = types.float64
STR         = typeof('str')
BOOL        = types.bool
ARRAY_1D    = types.Array(FLOAT, 1, 'C')
COUNTER_1D  = types.Array(INT, 1, 'C')
MASK_1D     = types.Array(BOOL, 1, 'C')

# -------------------- Enumerations -------------------- #

class Direction(Enum):
    RIGHT   = 0
    UP      = 1
    LEFT    = 2
    DOWN    = 3

class GridEnum(Enum):
    Empty       = 0
    Boundary    = 1
    Food        = 2
    SnakeHead   = 3