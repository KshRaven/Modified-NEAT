from ModifiedNEAT.base import functional as shared
from ModifiedNEAT.base.functional import SEED, set_seed, clamp, threads_total

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray as GPUArray

import numpy as np
import cupy as cp


def get_rng_states(kernel_shape: tuple, seed: int = None, get_normal=True, use_cuda=True) -> tuple[cp.ndarray, int]:
    if seed is None:
        seed = shared.SEED
    total = threads_total(kernel_shape)
    if get_normal:
        rng_states = cp.random.normal(size=total, dtype=cp.float32) if use_cuda \
            else np.random.normal(size=total).astype(cp.float32)
    else:
        rng_states = cp.random.uniform(size=total, dtype=cp.float32) if use_cuda \
            else np.random.uniform(size=total).astype(cp.float32)
    return rng_states, total


@cuda.jit(device=True)
def gauss(states: GPUArray, index: int) -> float:
    """
    Get the value at a given index for a Normal Distribution GPU Array
    :param states: 1D GPUArray for all max possible threads
    :param index: int
    :return:
    """
    return states[index]


@cuda.jit(device=True)
def prob(states: GPUArray, index: int) -> float:
    """
    Get the value at a given index for a Probability (Uniform[0, 1]) GPU Array
    :param states: 1D GPUArray for all max possible threads
    :param index: int
    :return:
    """
    return states[index]


@cuda.jit(device=True)
def normal(states: GPUArray, index: int, mean: float, std: float) -> float:
    return mean + gauss(states, index) * std


@cuda.jit(device=True)
def uniform(states: GPUArray, index: int, minimum: float, maximum: float) -> float:
    difference = maximum - minimum
    return minimum + prob(states, index) * difference
