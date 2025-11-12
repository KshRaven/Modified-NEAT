
from ModifiedNEAT.nn import NeatModule
from ModifiedNEAT.config import Config
from .functional import calc_grid, get_rng_states, clamp, normal, uniform
from ModifiedNEAT.util.fancy_text import CM, Fore

from numba import njit, prange
from numpy import ndarray as CPUArray

import numpy as np
import torch
import time as clock
import gc


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


@njit(parallel=True)
def _execute(parameter: CPUArray, init_type: int, mean: float, std: float, minimum: float, maximum: float, rng_states):
    if parameter.ndim != 3:
        raise ValueError(f"Expected a 3D array, got {parameter.ndim}")

    s_g, s_x, s_y = parameter.shape
    # Parameter shape (genomes, *spatial_dims)
    g_lim = parameter.shape[0]
    x_lim = 1 if parameter.ndim <= 1 else parameter.shape[1]
    y_lim = 1 if parameter.ndim <= 2 else parameter.shape[2]

    for genome_idx in prange(s_g):
        for x in prange(s_x):
            for y in prange(s_y):
                # Linearized thread index
                rng_index = (y * s_x * s_g) + (x * s_g) + genome_idx

                if genome_idx < g_lim and x < x_lim and y < y_lim:
                    initialize_genome(
                        parameter, genome_idx, x, y, init_type, mean, std, minimum, maximum, rng_states, rng_index
                    )
                    pass
                else:
                    raise ValueError("Out of bounds")


def initialize(config: Config, module: NeatModule, tpb=1, verbose: int = None):
    ts = clock.perf_counter()
    if config.genome.init_type == 'normal':
        init_type = 0
    elif config.genome.init_type == 'uniform':
        init_type = 1
    else:
        raise NotImplementedError(f"Unsupported NEAT Genome init type '{config.genome.init_type}'")

    for param in module.neat_parameters():
        dtype = param.data.dtype if param.data.dtype != torch.bfloat16 else torch.float32
        # Clone the existing parameter
        array = param.data.clone().to(dtype)
        original_shape = param.data.shape
        # In case dims are greater than 3, flatten the extra dimensions
        if array.ndim > 3:
            array = array.reshape(*array.shape[:2], -1)
        elif array.ndim < 3:
            for _ in range(3-array.ndim):
                array = array.unsqueeze(-1)
        # Convert param to numpy array
        array = np.asarray(array)
        # Get kernel dims
        kernel_shape = calc_grid(*array.shape, tpb=tpb)
        # Get rng states for each element in the param
        rng_states, threads_total = get_rng_states(kernel_shape, config.general.seed, get_normal=True)

        try:
            # Run kernel
            _execute(
                array, init_type, config.genome.weight_init_mean, config.genome.weight_init_std,
                config.genome.weight_min_value, config.genome.weight_max_value, rng_states
            )
        except Exception as e:
            print(param.dtype, param.device, kernel_shape, threads_total, array.shape, original_shape)
            raise e

        # Copy data back to parameter
        param.data.copy_(torch.from_dlpack(array.reshape(original_shape)))

        # Remove data from GPU
        del rng_states, array

    # Remove data
    gc.collect()

    # Limit init values according the module specifications
    module.update_limit()
    for m in module.neat_modules():
        m.update_limit()

    if verbose and verbose >= 2:
        print(f"{CM('Initialized genomes in ', Fore.CYAN)} in {round(clock.perf_counter() - ts, 2)}s")
