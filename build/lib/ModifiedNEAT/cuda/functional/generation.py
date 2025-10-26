
from numba import cuda, njit
from numba.cuda.cudadrv.devicearray import DeviceNDArray as GPUArray
# from numba.cuda import random

import numpy as np
import cupy as cp

SEED = int(np.random.randint(0, int(2 ** 20)))


def set_seed(seed: int):
    global SEED
    SEED = seed


def get_rng_states(kernel_shape: tuple, seed: int = None, get_normal=True, use_cuda=True) -> tuple[cp.ndarray, int]:
    if seed is None:
        seed = SEED
    threads_total = int(np.prod([np.prod(v) for v in kernel_shape]))
    # rng_states = random.create_xoroshiro128p_states(threads_total, seed=seed)
    if get_normal:
        rng_states = cp.random.normal(size=threads_total, dtype=cp.float32) if use_cuda \
            else np.random.normal(size=threads_total).astype(cp.float32)
    else:
        rng_states = cp.random.uniform(size=threads_total, dtype=cp.float32) if use_cuda \
            else np.random.uniform(size=threads_total).astype(cp.float32)
    return rng_states, threads_total


@njit
def clamp(value: float, minimum: float, maximum: float):
    return min(max(value, minimum), maximum)


@cuda.jit(device=True)
def gauss(states: GPUArray, index: int) -> float:
    """
    Get the value at a given index for a Normal Distribution GPU Array
    :param states: 1D GPUArray for all max possible threads
    :param index: int
    :return:
    """
    # return random.xoroshiro128p_normal_float64(states, index)
    return states[index]


@cuda.jit(device=True)
def prob(states: GPUArray, index: int) -> float:
    """
    Get the value at a given index for a Probability (Uniform[0, 1]) GPU Array
    :param states: 1D GPUArray for all max possible threads
    :param index: int
    :return:
    """
    # return random.xoroshiro128p_uniform_float64(states, index)
    return states[index]


@cuda.jit(device=True)
def normal(states: GPUArray, index: int, mean: float, std: float) -> float:
    return mean + gauss(states, index) * std


@cuda.jit(device=True)
def uniform(states: GPUArray, index: int, minimum: float, maximum: float) -> float:
    difference = maximum - minimum
    return minimum + prob(states, index) * difference
