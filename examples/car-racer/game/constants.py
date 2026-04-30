import os
import numpy as np

from numba import types, typeof


CURR_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(CURR_DIR, "imgs")

INT         = types.int64
FLOAT       = types.float64
STR         = typeof('str')
BOOL        = types.bool
ARRAY_1D    = types.Array(FLOAT, 1, 'C')
COUNTER_1D  = types.Array(INT, 1, 'C')
MASK_1D     = types.Array(BOOL, 1, 'C')


NP_INT      = np.int64
NP_FLOAT    = np.float64


COLOR_ROAD      = (31, 32, 34) # Rich gray
COLOR_GRASS     = (21, 71, 52) # Forest Green
COLOR_CURB_PRI  = (187, 0, 0) # Scarlet Red
COLOR_CURB_SEC  = (240, 240, 240) # White
COLOR_START_PRI = (245, 245, 245) # White
COLOR_START_SEC = (15, 15, 15) # Black
COLOR_GRAVEL    = (137, 81, 41) # Brown
COLOR_EMPTY     = (54, 69, 79) # Sage Gray
COLOR_LIDAR     = (255, 191, 0) # Yellow
