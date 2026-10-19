from ModifiedNEAT.base.functional import calc_grid, calc

from numba import njit
from numpy import ndarray as CPUArray

import numpy as np


@njit
def get_enumeration(array: CPUArray):
    positions: list[tuple[int, ...]] = [index for index, value in np.ndenumerate(array)]
    count = len(positions)
    return positions, count
