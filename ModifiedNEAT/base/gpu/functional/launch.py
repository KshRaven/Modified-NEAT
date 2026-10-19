from ModifiedNEAT.base.functional import calc_grid, calc

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray as GPUArray
from typing import Union

import numpy as np


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
