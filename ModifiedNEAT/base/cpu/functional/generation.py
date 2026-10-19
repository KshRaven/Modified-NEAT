from ModifiedNEAT.base import functional as shared
from ModifiedNEAT.base.functional import SEED, set_seed, clamp, threads_total

from numba import njit
from numpy import ndarray as CPUArray

import numpy as np


def get_rng_states(kernel_shape: tuple, seed: int = None, get_normal=True, dtype=np.float64) -> tuple[np.ndarray, int]:
    if seed is None:
        seed = shared.SEED
    total = threads_total(kernel_shape)
    if get_normal:
        rng_states = np.random.normal(size=total).astype(dtype)
    else:
        rng_states = np.random.uniform(size=total).astype(dtype)
    return rng_states, total


@njit
def gauss(states: CPUArray, index: int) -> float:
    """
    Get the value at a given index for a Normal Distribution CPU Array
    :param states: 1D CPUArray for all max possible threads
    :param index: int
    :return:
    """
    return states[index]


@njit
def prob(states: CPUArray, index: int) -> float:
    """
    Get the value at a given index for a Probability (Uniform[0, 1]) CPU Array
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
