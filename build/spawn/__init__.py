

from numba.cuda.cudadrv.devicearray import DeviceNDArray as GPUArray

Weights = GPUArray
Bias    = GPUArray
Layer   = tuple[Weights, Bias]
Network = tuple[tuple[Layer, ...], bool]
Genome  = tuple[Network, ...]
