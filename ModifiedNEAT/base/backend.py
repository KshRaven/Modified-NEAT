"""
Device backend interface.

Everything that is identical on CPU and GPU (orchestration of initialization, reproduction and speciation) is
written once in ``initialization.py`` / ``reproduction.py`` / ``speciation.py`` and talks to the device only through
a :class:`Backend`. ``cpu/backend.py`` and ``gpu/backend.py`` implement it with the numba prange / numba.cuda kernels
that live in the device packages.
"""

import numpy as np


class Backend(object):
    #: Device name ('cpu' or 'cuda')
    name: str = 'abstract'
    #: Array module used on this device (numpy or cupy)
    xp = np
    #: Whether `crossover_fallback` is available as a cross-check when the primary crossover result is invalid
    has_fallback: bool = False

    # ---------- Array management ---------- #
    def asarray(self, array):
        """Convert a torch tensor / numpy array to an array on this device (zero copy when possible)."""
        raise NotImplementedError

    def to_host(self, array) -> np.ndarray:
        """Bring a device array back to host memory (no-op on CPU)."""
        return array

    def zeros(self, shape: tuple):
        """float64 zeros on this device."""
        raise NotImplementedError

    def release(self):
        """Release temporary memory (garbage collection / memory pool)."""
        raise NotImplementedError

    # ---------- Random numbers ---------- #
    def uniform(self, shape: tuple):
        """Uniform [0, 1) values of the given shape."""
        raise NotImplementedError

    def normal_states(self, kernel_shape: tuple, seed: int = None):
        """Standard normal values, one per thread. Returns (states, threads_total)."""
        raise NotImplementedError

    # ---------- Kernels ---------- #
    def initialize(self, array, init_type: int, mean: float, std: float, minimum: float, maximum: float,
                       rng_states, kernel_shape: tuple, info: tuple = ()):
        raise NotImplementedError

    def crossover(self, genus: int, source, updates, parents, genera, filled, probabilities, equal_sources: bool,
                  kernel_shape: tuple):
        raise NotImplementedError

    def crossover_fallback(self, genus: int, source, updates, parents, genera, filled, probabilities,
                           equal_sources: bool, cache: dict):
        raise NotImplementedError

    def mutate(self, updates, children, *args, kernel_shape: tuple):
        """args = (*mutation_arguments(...), probabilities, normals)"""
        raise NotImplementedError

    def compute_distances(self, config, module, dtype, tpb: int, verbose: int = None) -> np.ndarray:
        """
        Genetic distance matrix (genomes x genomes) of the module's genomes as a host float array. Symmetric, with a
        zero diagonal.
        """
        raise NotImplementedError
