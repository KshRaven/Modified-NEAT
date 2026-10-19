"""CPU initialization kernels (numba, prange). The orchestration lives in ``ModifiedNEAT.base.initialization``."""

from .functional import clamp, normal, uniform

from numba import njit, prange
from numpy import ndarray as CPUArray

import numpy as np


@njit
def initialize_genome(parameter: CPUArray, g: int, x: int, y: int, init_type: int, mean: float, std: float,
                      norm_min: float, norm_max: float, rng_states: CPUArray, rng_index: int):
    uni_min = max(norm_min, (mean - (2 * std)))
    uni_max = min(norm_max, (mean + (2 * std)))
    # for index, value in np.ndenumerate(parameter):
    if init_type == 0:
        parameter[g, x, y] = clamp(
            normal(rng_states, rng_index, mean, std),
            norm_min, norm_max
        )
    elif init_type == 1:
        parameter[g, x, y] = uniform(rng_states, rng_index, uni_min, uni_max)


@njit(parallel=True) # , cache=True)
def initialize(parameter: CPUArray, init_type: int, mean: float, std: float, minimum: float, maximum: float, rng_states):
    ret_code: int = 0
    if parameter.ndim != 3: ret_code = 1

    if not ret_code:
        s_g, s_x, s_y = parameter.shape
        # Parameter shape (genomes, *spatial_dims)
        g_lim = parameter.shape[0]
        x_lim = 1 if parameter.ndim <= 1 else parameter.shape[1]
        y_lim = 1 if parameter.ndim <= 2 else parameter.shape[2]

        # Flat index -> (genome, x, y) without materializing a list of index tuples
        total = s_g * s_x * s_y
        yz = s_x * s_y
        for i in prange(total):
            # prange indices are unsigned: cast to int64 so // and % stay integers (uint64 with int64 -> float64)
            flat = np.int64(i)
            genome_idx = flat // yz
            x = (flat % yz) // s_y
            y = flat % s_y

            # Linearized thread index
            rng_index = (y * s_x * s_g) + (x * s_g) + genome_idx

            if genome_idx < g_lim and x < x_lim and y < y_lim:
                initialize_genome(
                    parameter, genome_idx, x, y, init_type, mean, std, minimum, maximum, rng_states, rng_index
                )
            else:
                ret_code = 2

    return ret_code
