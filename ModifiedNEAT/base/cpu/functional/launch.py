import numpy as np
from numba import njit
from numpy import ndarray as CPUArray


@njit
def get_enumeration(array: CPUArray):
    positions: list[tuple[int, ...]] = [index for index, value in np.ndenumerate(array)]
    count = len(positions)
    return positions, count
