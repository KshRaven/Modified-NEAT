"""CPU distance kernels (numba, prange). The species grouping lives in ``ModifiedNEAT.base.speciation``."""

from ModifiedNEAT.config import Config
from ModifiedNEAT.nn.base import NeatModule
from ModifiedNEAT.base.util import parameter_dtype, reshape_to_ndim

from numba import njit, prange
from numpy import ndarray as CPUArray

import numpy as np


@njit
def calc_distance(parameter: CPUArray, total_distance: CPUArray,
                  genome0: int, genome1: int, x: int,
                  compatibility_weight_coefficient: float, compatibility_disjoint_coefficient: float):
    """
    Returns the genetic distance between this genome and the other. This distance value
    is used to compute genome compatibility for speciation.
    """

    value0 = parameter[genome0, x]
    value1 = parameter[genome1, x]

    current_distance = 0 # total_distance[genome0, genome1]
    disjoint_value   = 0
    if value0 == 0:
        disjoint_value += 1
    if value1 == 0:
        disjoint_value = disjoint_value + 1
    if value0 != 0 and value1 != 0:
        current_distance = current_distance + (abs(value0 - value1) * compatibility_weight_coefficient)
    current_distance = current_distance + disjoint_value * compatibility_disjoint_coefficient

    # Parameter shape = (genomes, x, y | 1)
    x_s = parameter.shape[1] # [:]
    size = x_s # * y_s
    # if size != 0:
    current_distance = current_distance / size

    return current_distance


@njit(parallel=True) # , cache=True)
# NOTE: Same fix as crossover(): a single return point and ret_code instead of
#       raising inside a prange body. Only the outer `genome0` loop is a prange -
#       each iteration exclusively owns row `genome0` of total_distance (lower
#       triangle writes total_distance[genome0, genome1>=genome0], upper triangle
#       writes total_distance[genome0, genome1<genome0] after reading the already
#       completed lower triangle) so there is no cross-thread write conflict.
#       The inner loops stay plain `range` - nesting prange inside prange gives
#       numba nothing extra here and only muddies which loop actually gets parallelized.
def get_distance(parameter: CPUArray, total_distance: CPUArray,
                 compatibility_weight_coefficient: float, compatibility_disjoint_coefficient: float):
    ret_code: int = 0
    if parameter.ndim != 2: ret_code = 1

    if not ret_code:
        g_lim = total_distance.shape[0]
        x_lim = parameter.shape[1]

        # TODO: The nested pranges may raise performance warning so find a way to combine
        #       and iterate over a single integer like other parralel functions
        # Fill the lower triangle
        # Rows i and (g_lim - 1 - i) are handled by the same iteration to balance the triangular workload.
        # Each row is owned by exactly one iteration, the diagonal is skipped (it only ever added 0.0), and
        # the per-element sum is accumulated in a local (same order as before, so results are identical).
        half = (g_lim + 1) // 2
        for r in prange(half):
            row = np.int64(r)  # prange index is unsigned; keep everything int64 (uint64 with int64 -> float64)
            for side in range(2):
                genome0 = row if side == 0 else g_lim - 1 - row
                if side == 1 and genome0 == row:
                    continue
                for genome1 in range(genome0 + 1, g_lim):
                    acc = total_distance[genome0, genome1]
                    for x in range(x_lim):
                        acc += calc_distance(
                            parameter, total_distance, genome0, genome1, x,
                            compatibility_weight_coefficient, compatibility_disjoint_coefficient
                        )
                    total_distance[genome0, genome1] = acc
        # Fill the upper triangle
        for genome0 in prange(g_lim):
            for genome1 in prange(genome0):
                # Assert limits
                if genome0 < g_lim and genome1 < g_lim:
                    distance = total_distance[genome1, genome0]
                    total_distance[genome0, genome1] = distance
                else: ret_code = 2

    return ret_code


def compute_distances(config: Config, module: NeatModule, dtype, tpb: int = 10, verbose: int = None) -> np.ndarray:
    """Genetic distance matrix of the module's genomes (symmetric, zero diagonal), summed over all parameters."""
    total_distance = np.zeros((module.genome_num, module.genome_num), dtype)

    for param in module.neat_parameters():
        array = np.asarray(reshape_to_ndim(param.data.to(parameter_dtype(param.data.dtype)), 2)[0])

        try:
            ret_code = get_distance(
                array, total_distance,
                config.genome.compatibility_weight_coefficient,
                config.genome.compatibility_disjoint_coefficient
            )
            if ret_code == 1: raise ValueError(f"Expected a 2D array, got {array.ndim}")
            if ret_code == 2: raise ValueError("Out of bounds")
        except Exception as e:
            print(param.dtype, param.device, array.shape, param.data.shape)
            raise e

        # Release memory
        del array

    return total_distance
