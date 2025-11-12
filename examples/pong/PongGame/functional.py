
import numpy as np

from numba import njit, prange, types, typeof

INT         = types.int64
FLOAT       = types.float64
STR         = typeof('str')
BOOL        = types.bool
ARRAY_1D    = types.Array(FLOAT, 1, 'C')
COUNTER_1D  = types.Array(INT, 1, 'C')
MASK_1D     = types.Array(BOOL, 1, 'C')
