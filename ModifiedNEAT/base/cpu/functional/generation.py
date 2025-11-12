
from numba import njit
from numpy import ndarray as CPUArray

import numpy as np
import cupy as cp

SEED = int(np.random.randint(0, int(2 ** 20)))


def set_seed(seed: int):
    global SEED
    SEED = seed


def get_rng_states(kernel_shape: tuple, seed: int = None, get_normal=True, dtype=np.float64) -> tuple[cp.ndarray, int]:
    if seed is None:
        seed = SEED
    threads_total = int(np.prod([np.prod(v) for v in kernel_shape]))
    if get_normal:
        rng_states = np.random.normal(size=threads_total).astype(dtype)
    else:
        rng_states = np.random.uniform(size=threads_total).astype(dtype)
    return rng_states, threads_total


@njit
def clamp(value: float, minimum: float, maximum: float):
    return min(max(value, minimum), maximum)


@njit
def gauss(states: CPUArray, index: int) -> float:
    """
    Get the value at a given index for a Normal Distribution GPU Array
    :param states: 1D CPUArray for all max possible threads
    :param index: int
    :return:
    """
    return states[index]


@njit
def prob(states: CPUArray, index: int) -> float:
    """
    Get the value at a given index for a Probability (Uniform[0, 1]) GPU Array
    :param states: 1D CPUArray for all max possible threads
    :param index: int
    :return:
    """
    return states[index]


@njit
def normal(states: CPUArray, index: int, mean: float, std: float) -> float:
    return mean + gauss(states, index) * std


@njit
def uniform(states: CPUArray, index: int, minimum: float, maximum: float) -> float:
    difference = maximum - minimum
    return minimum + prob(states, index) * difference
