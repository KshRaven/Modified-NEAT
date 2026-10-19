"""CPU crossover kernel (numba, prange). The orchestration lives in ``ModifiedNEAT.base.reproduction``."""

from .functional import prob

from numba import njit, prange
from numpy import ndarray as CPUArray

import numpy as np


@njit
def _crossover(value1: float, value2: float, states: CPUArray, index: int):
    if prob(states, index) >= 0.5:
        return value1
    else:
        return value2


@njit(parallel=True) # , cache=True) # nogil=True)
# TODO: Caching seems to work for now since no custom Classes or prints are used
def crossover(genus: int, source: CPUArray, updates: CPUArray, parents: CPUArray, genera: CPUArray, filled: CPUArray,
              probabilities: CPUArray, equal_sources: bool):
    # TODO: Ensure no breaks, raises or other multiple return points in order for parallel to work.
    #       Avoid assertions too even though most uses have not raised NumbaPerformanceWarning.
    ret_code: int = 0
    if updates.ndim != 3: ret_code = 1

    if not ret_code:
        s_g, s_x, s_y = updates.shape # cuda.gridsize(3)
        # Parameter shape (genomes, *spatial_dims)
        g_lim = updates.shape[0]
        x_lim = 1 if updates.ndim <= 1 else updates.shape[1]
        y_lim = 1 if updates.ndim <= 2 else updates.shape[2]

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
                parent0, parent1, parent_   = parents[genome_idx]
                genus0, genus1, genus_      = genera[genome_idx]
                filled0, filled1, filled_   = filled[genome_idx, x, y]
                if equal_sources or (genus0 == genus1 == genus):
                    # For elite genomes or clones
                    if (parent0 == parent1) and (genus == genus0 == genus1):
                        value = source[parent0, x, y]
                        filled[genome_idx, x, y, :2] = True
                    # For child crossover
                    else:
                        already_filled = filled0 or filled1

                        # If not filled, fill from current available source
                        value = updates[genome_idx, x, y].item() # Get value in case param has been filled from another source
                        if not already_filled:
                            if genus == genus0:
                                value = source[parent0, x, y]
                                filled[genome_idx, x, y, 0] = filled0 = True
                            elif genus == genus1:
                                value = source[parent1, x, y]
                                filled[genome_idx, x, y, 1] = filled1 = True

                        # Crossover when both parents' values are available
                        if genus == genus0 and not filled0 and filled1:
                            value_c = source[parent0, x, y].item()
                            value = _crossover(value, value_c, probabilities, rng_index)
                            filled[genome_idx, x, y, 0] = True
                        if genus == genus1 and not filled1 and filled0:
                            value_c = source[parent1, x, y].item()
                            value = _crossover(value, value_c, probabilities, rng_index)
                            filled[genome_idx, x, y, 1] = True
                else:
                    # Emergency fill on unmatched param groups
                    if not filled_ and genus == genus_:
                        # TODO: Should random emergency parents be used instead of random cloning?
                        value = source[parent_, x, y]
                        filled[genome_idx, x, y, 2] = True
                    else:
                        value = updates[genome_idx, x, y]

                updates[genome_idx, x, y] = value
            else: ret_code = 2

    return ret_code
