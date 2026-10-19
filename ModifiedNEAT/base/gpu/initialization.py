"""GPU initialization kernels (numba.cuda). The orchestration lives in ``ModifiedNEAT.base.initialization``."""

from .functional import clamp, normal, uniform

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray as GPUArray


@cuda.jit(device=True)
def initialize_genome(parameter: GPUArray, g: int, x: int, y: int, init_type: int, mean: float, std: float,
                      norm_min: float, norm_max: float, rng_states: GPUArray, rng_index: int):
    uni_min = max(norm_min, (mean - (2 * std)))
    uni_max = min(norm_max, (mean + (2 * std)))
    if parameter.ndim == 1:
        raise NotImplementedError()
    #     if init_type == 0:
    #         parameter[g] = clamp(normal(rng_states, rng_index, mean, std), norm_min, norm_max)
    #     elif init_type == 1:
    #         parameter[g] = uniform(rng_states, rng_index, uni_min, uni_max)
    elif parameter.ndim == 2:
        raise NotImplementedError()
    #     if init_type == 0:
    #         parameter[g, x] = clamp(normal(rng_states, rng_index, mean, std), norm_min, norm_max)
    #     elif init_type == 1:
    #         parameter[g, x] = uniform(rng_states, rng_index, uni_min, uni_max)
    if parameter.ndim == 3:
        if init_type == 0:
            parameter[g, x, y] = clamp(
                normal(rng_states, rng_index, mean, std),
                norm_min, norm_max
            )
        elif init_type == 1:
            parameter[g, x, y] = uniform(rng_states, rng_index, uni_min, uni_max)
    else:
        raise NotImplementedError('Yet to be implemented for params with more than 3 dims')


@cuda.jit
def _execute(parameter: GPUArray, init_type: int, mean: float, std: float, minimum: float, maximum: float, rng_states):
    genome_idx, x, y = cuda.grid(3)
    # Parameter shape (genomes, *spatial_dims)
    g_lim = parameter.shape[0]
    x_lim = 1 if parameter.ndim <= 1 else parameter.shape[1]
    y_lim = 1 if parameter.ndim <= 2 else parameter.shape[2]
    s_g, s_x, s_y = parameter.shape # cuda.gridsize(3)

    # Linearized thread index
    rng_index = (y * s_x * s_g) + (x * s_g) + genome_idx

    if genome_idx < g_lim and x < x_lim and y < y_lim:
        initialize_genome(parameter, genome_idx, x, y, init_type, mean, std, minimum, maximum, rng_states, rng_index)
