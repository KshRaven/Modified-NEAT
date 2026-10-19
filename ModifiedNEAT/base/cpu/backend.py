from ModifiedNEAT.base.backend import Backend
from ModifiedNEAT.base import seeding
from .functional import get_rng_states
from .initialization import initialize as _initialize_kernel
from .mutation import mutate as _mutate_kernel
from .reproduction import crossover as _crossover_kernel
from .speciation import compute_distances as _compute_distances

import gc
import numpy as np


class CpuBackend(Backend):
    """Numba (prange) implementation of the device interface, running on numpy arrays."""
    name = 'cpu'
    xp = np
    has_fallback = False

    def asarray(self, array):
        return np.asarray(array)

    def zeros(self, shape: tuple):
        return np.zeros(shape, np.float64)

    def release(self):
        gc.collect()

    def uniform(self, shape: tuple):
        return seeding.numpy_rng().random(size=shape)

    def normal_states(self, kernel_shape: tuple, seed: int = None):
        return get_rng_states(kernel_shape, seed, get_normal=True)

    @staticmethod
    def _check(ret_code: int, ndim: int):
        if ret_code == 1: raise ValueError(f"Expected a 3D array, got {ndim}")
        if ret_code == 2: raise ValueError("Out of bounds")

    def initialize(self, array, init_type, mean, std, minimum, maximum, rng_states, kernel_shape, info=()):
        try:
            self._check(_initialize_kernel(array, init_type, mean, std, minimum, maximum, rng_states), array.ndim)
        except Exception as e:
            print(*info)
            raise e

    def crossover(self, genus, source, updates, parents, genera, filled, probabilities, equal_sources, kernel_shape):
        self._check(
            _crossover_kernel(genus, source, updates, parents, genera, filled, probabilities, equal_sources),
            updates.ndim
        )

    def mutate(self, updates, children, *args, kernel_shape):
        self._check(_mutate_kernel(updates, children, *args), updates.ndim)

    def compute_distances(self, config, module, dtype, tpb, verbose=None):
        return _compute_distances(config, module, dtype, tpb, verbose)


BACKEND = CpuBackend()
