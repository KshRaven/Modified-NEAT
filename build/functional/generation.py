
from numba import cuda, njit
from numba.cuda.cudadrv.devicearray import DeviceNDArray as GPUArray
from numba.cuda import random

import numpy as np

SEED = int(np.random.randint(0, int(1e2)))


def get_rng_states(kernel_shape: tuple, seed: int = None):
    if seed is None:
        seed = SEED
    threads_total = int(np.prod([np.prod(v) for v in kernel_shape]))
    # rng_states = random.create_xoroshiro128p_states(threads_total, seed=seed)
    rng_states = cuda.to_device(np.random.normal(size=threads_total))
    return rng_states, threads_total


@njit
def clamp(value: float, minimum: float, maximum: float):
    return min(max(value, minimum), maximum)


@cuda.jit(device=True)
def gauss(states: GPUArray, index: int):
    # return random.xoroshiro128p_normal_float64(states, index)
    return states[index]


@cuda.jit(device=True)
def prob(states: GPUArray, index: int):
    # return random.xoroshiro128p_uniform_float64(states, index)
    return clamp(((states[index]+1)/2), 0, 1)


@cuda.jit(device=True)
def normal(states: GPUArray, index: int, mean: float, std: float):
    return mean + gauss(states, index) * std


@cuda.jit(device=True)
def uniform(states: GPUArray, index: int, a: float, b: float):
    return a + (b - a) * prob(states, index)
