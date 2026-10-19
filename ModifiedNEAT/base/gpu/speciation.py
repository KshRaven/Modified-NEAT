"""GPU distance kernels (numba.cuda). The species grouping lives in ``ModifiedNEAT.base.speciation``."""

from ModifiedNEAT.config import Config
from ModifiedNEAT.nn.base import NeatModule
from ModifiedNEAT.base.util import parameter_dtype, reshape_to_ndim
from ModifiedNEAT.util.fancy_text import CM, Fore
from .functional import calc_grid

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray as GPUArray
from typing import Union

import numpy as np
import cupy as cp


_Index  = Union[int, tuple[int, ...]]
_Number = Union[float, int]


@cuda.jit(device=True) # , cache=True)
def atomic_add(array: GPUArray, index: _Index, value: _Number) -> None:
    cuda.atomic.add(array, index, value)


@cuda.jit(device=True)
def calc_distance(parameter: GPUArray, total_distance: GPUArray, match_counter: GPUArray,
                  genome0: int, genome1: int, x: int,
                  compatibility_weight_coefficient: float, compatibility_disjoint_coefficient: float):
    """
    Returns the genetic distance between this genome and the other. This distance value
    is used to compute genome compatibility for speciation.
    """

    value0 = parameter[genome0, x]
    value1 = parameter[genome1, x]

    current_distance = 0 # total_distance[genome0, genome1]
    disjoint_value = 0
    if value0 == 0:
        disjoint_value = disjoint_value + 1
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

    atomic_add(total_distance, (genome0, genome1), current_distance)

    # Add to count if values are the same to flag genomes that haven't been mutated properly
    if value0 == value1:
        cuda.atomic.add(match_counter, (genome0, genome1), 1)


@cuda.jit
def get_distance(parameter: GPUArray, total_distance: GPUArray, match_counter: GPUArray,
                 compatibility_weight_coefficient: float, compatibility_disjoint_coefficient: float):
    # select one genome
    x, genome0, genome1 = cuda.grid(3)
    g_lim = total_distance.shape[0]
    x_lim = parameter.shape[1]
    # Upper triangle only (diagonal skipped, mirrored on the host) - same logic as the CPU get_distance
    if genome0 < genome1 and genome1 < g_lim and x < x_lim:
        calc_distance(parameter, total_distance, match_counter, genome0, genome1, x,
                      compatibility_weight_coefficient, compatibility_disjoint_coefficient)
        # cuda.syncthreads()


def compute_distances(config: Config, module: NeatModule, dtype, tpb: int = 10, verbose: int = None) -> np.ndarray:
    """
    Genetic distance matrix of the module's genomes (symmetric, zero diagonal) as a host array. The kernel fills the
    upper triangle only; it is mirrored here. Also verifies that no two genomes fully match each other.
    """
    genomes_total = module.genome_num
    total_distance = cp.zeros((genomes_total, genomes_total), dtype)
    faulty_counter = cp.zeros((genomes_total, genomes_total), int)

    for p_idx, param in enumerate(module.neat_parameters()):
        match_counter = cp.zeros((genomes_total, genomes_total), int)
        values_total = np.prod(param.original_shape).item()

        genome_num = len(module.mapping)
        array = cp.asarray(reshape_to_ndim(param.data.to(parameter_dtype(param.data.dtype)), 2, cp)[0])
        axes = array.shape[1:]
        kernel_shape = calc_grid(*axes, genome_num, genome_num, tpb=tpb)

        get_distance[*kernel_shape](
            array, total_distance, match_counter,
            config.genome.compatibility_weight_coefficient,
            config.genome.compatibility_disjoint_coefficient
        )

        # Add count when genome parameters fully match
        faulty_counter[match_counter >= values_total] += 1

        if verbose and verbose >= 4:
            max_match_count = cp.max(match_counter).item()
            print(f"Parameter {p_idx} had a max match count of {max_match_count} / {values_total}")

    neat_params_total = len(module.neat_parameters())
    reproduction_error = faulty_counter >= neat_params_total
    if cp.any(reproduction_error):
        error_count = cp.sum(reproduction_error).item()
        raise ValueError(f"Genomes cannot 100% match each other! \n"
                         f"{error_count} cases confirmed! Error in crossover or mutation of values.")

    if verbose and verbose >= 4:
        print(f"Maximum fault count was {cp.max(faulty_counter).item()} / {neat_params_total}")

    # Mirror the upper triangle (diagonal is 0)
    return (total_distance + total_distance.T).get()
