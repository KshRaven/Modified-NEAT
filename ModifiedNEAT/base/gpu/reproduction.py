"""GPU crossover kernel (numba.cuda). The orchestration lives in ``ModifiedNEAT.base.reproduction``."""

from .functional import prob, get_value, set_value

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from typing import Union

import cupy as cp

GPUArray = Union[DeviceNDArray, cp.ndarray]


@cuda.jit(device=True)
def _crossover(value1: float, value2: float, states: GPUArray, index: int):
    if prob(states, index) >= 0.5:
        return value1
    else:
        return value2


@cuda.jit() # debug=True, opt=False)
def crossover(genus: int, source: GPUArray, updates: GPUArray, parents: GPUArray, genera: GPUArray, filled: GPUArray,
              probabilities: GPUArray, equal_sources: bool):
    genome_idx, x, y = cuda.grid(3)
    # Parameter shape (genomes, *spatial_dims)
    g_lim = updates.shape[0]
    x_lim = 1 if updates.ndim <= 1 else updates.shape[1]
    y_lim = 1 if updates.ndim <= 2 else updates.shape[2]
    s_g, s_x, s_y = updates.shape # cuda.gridsize(3)

    # Linearized thread index
    rng_index = (y * s_x * s_g) + (x * s_g) + genome_idx

    if genome_idx < g_lim and x < x_lim and y < y_lim:
        parent0, parent1, parent_   = parents[genome_idx]
        genus0, genus1, genus_      = genera[genome_idx]
        filled0, filled1, filled_   = filled[genome_idx, x, y]
        if equal_sources or (genus0 == genus1 == genus):
            # For elite genomes or clones
            if (parent0 == parent1) and (genus == genus0 == genus1):
                value = get_value(source, parent0, x, y)
                filled[genome_idx, x, y, 0] = True
                filled[genome_idx, x, y, 1] = True
            # For child crossover
            else:
                already_filled = filled0 or filled1

                # If not filled, fill from current available source
                value = get_value(updates, genome_idx, x, y) # Get value in case param has been filled from another source
                if not already_filled:
                    if genus == genus0:
                        value = get_value(source, parent0, x, y)
                        filled[genome_idx, x, y, 0] = filled0 = True
                    elif genus == genus1:
                        value = get_value(source, parent1, x, y)
                        filled[genome_idx, x, y, 1] = filled1 = True

                # Crossover when both parents' values are available
                if genus == genus0 and not filled0 and filled1:
                    value_c = get_value(source, parent0, x, y)
                    value = _crossover(value, value_c, probabilities, rng_index)
                    filled[genome_idx, x, y, 0] = True
                if genus == genus1 and not filled1 and filled0:
                    value_c = get_value(source, parent1, x, y)
                    value = _crossover(value, value_c, probabilities, rng_index)
                    filled[genome_idx, x, y, 1] = True
        else:
            # Emergency fill on unmatched param groups
            if not filled_ and genus == genus_:
                # TODO: Should random emergency parents be used instead of random cloning?
                value = get_value(source, parent_, x, y)
                filled[genome_idx, x, y, 2] = True
            else:
                value = get_value(updates, genome_idx, x, y)

        set_value(updates, genome_idx, x, y, value)
