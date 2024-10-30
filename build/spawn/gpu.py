

from build.spawn import Weights, Bias, Layer, Network, Genome

from numba import types, typeof, njit, optional, prange, cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray as GPUArray
from numpy import ndarray as CPUArray

import numpy as np


@cuda.jit(device=True)
def reproduce(
        weights: GPUArray, biases: GPUArray, parents: tuple[Genome, Genome],
        genome_idx: int, network_idx: int, layer_idx: int,
):
    # Assuming parent1 has the higher fitness
    parent1, parent2 = parents
    weights


