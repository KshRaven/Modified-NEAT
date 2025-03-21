
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray as GPUArray
from typing import Union

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


@cuda.jit(device=True)
def handle_invalids(value: float):
    if value == np.nan:
        raise ValueError(f"nan value found")
    elif abs(value) == np.inf:
        raise ValueError(f"inf value found")
    else:
        return value


@cuda.jit(device=True)
def get_value(source: GPUArray, g: int, x: int, y: int) -> Union[int, float, bool]:
    if source.ndim == 3:
        return handle_invalids(source[g, x, y])
    elif source.ndim == 2:
        return handle_invalids(source[g, x])
    elif source.ndim == 1:
        return handle_invalids(source[g])
    else:
        raise NotImplementedError('Cannot get source array value')


@cuda.jit(device=True)
def set_value(source: GPUArray, g: int, x: int, y: int, value: Union[int, float, bool]):
    if source.ndim == 3:
        source[g, x, y] = value
    elif source.ndim == 2:
        source[g, x] = value
    elif source.ndim == 1:
        source[g] = value
    else:
        raise NotImplementedError('Cannot set value to update array')
