
from numba import njit, types, prange
from numba.typed import List, Dict
from numpy import ndarray as CPUArray

import math
import numpy as np


def calc_grid(*block_sizes: int, tpb: int = 4, multiplier: float = None):
    for _ in range(3-len(block_sizes)):
        block_sizes += (1,)
    if multiplier is None:
        multiplier = 1
    return calc(block_sizes, tpb, multiplier)


def calc(tasks: tuple[int, ...], threads: int, multiplier: float):
    ndim = len(tasks)
    bpg: tuple[int] = (math.ceil((tasks[0] * multiplier) / threads),)
    tpb = (threads,)
    if ndim == 1:
        return bpg + tpb
    else:
        for dim_idx in range(1, ndim):
            t = threads
            if dim_idx == 1:
                t = min(10, t)
            if dim_idx == 2:
                t = min(10, t)
                # print(tpb, tasks[dim_idx], t, tasks[dim_idx] / t)
            bpg += (math.ceil((tasks[dim_idx] * multiplier) / t * 1),)
            if len(tasks) > 1:
                tpb += (t,)
        return bpg, tpb


@njit
def get_enumeration(array: CPUArray):
    positions: list[tuple[int, ...]] = [index for index, value in np.ndenumerate(array)]
    count = len(positions)
    return positions, count