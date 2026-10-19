from ModifiedNEAT.base.backend import Backend
from ModifiedNEAT.base import seeding
from ModifiedNEAT.base.cpu.reproduction import crossover as _crossover_cpu
from .functional import get_rng_states
from .initialization import _execute as _initialize_kernel
from .mutation import mutate as _mutate_kernel
from .reproduction import crossover as _crossover_kernel
from .speciation import compute_distances as _compute_distances

import gc
import cupy as cp


class GpuBackend(Backend):
    """numba.cuda / cupy implementation of the device interface (the CPU crossover is the cross-check fallback)."""
    name = 'cuda'
    xp = cp
    has_fallback = True

    def asarray(self, array):
        return cp.asarray(array)

    def to_host(self, array):
        return array.get()

    def zeros(self, shape: tuple):
        return cp.zeros(shape)

    def release(self):
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    def uniform(self, shape: tuple):
        return seeding.cupy_rng().random(size=shape, dtype=cp.float32)

    def normal_states(self, kernel_shape: tuple, seed: int = None):
        return get_rng_states(kernel_shape, seed, get_normal=True, use_cuda=True)

    def initialize(self, array, init_type, mean, std, minimum, maximum, rng_states, kernel_shape, info=()):
        _initialize_kernel[*kernel_shape](array, init_type, mean, std, minimum, maximum, rng_states)

    def crossover(self, genus, source, updates, parents, genera, filled, probabilities, equal_sources, kernel_shape):
        _crossover_kernel[*kernel_shape](genus, source, updates, parents, genera, filled, probabilities, equal_sources)

    def crossover_fallback(self, genus, source, updates, parents, genera, filled, probabilities, equal_sources, cache):
        """Run the CPU crossover on host copies and write the result back (host copies of shared inputs made once)."""
        if 'args' not in cache:
            cache['args'] = (parents.get(), genera.get(), probabilities.get())
        parents_cpu, genera_cpu, probabilities_cpu = cache['args']
        updates_cpu, filled_cpu = updates.get(), filled.get()
        filled_cpu[:] = False
        _crossover_cpu(
            genus, source.get(), updates_cpu, parents_cpu, genera_cpu, filled_cpu, probabilities_cpu, equal_sources
        )
        updates[...] = cp.asarray(updates_cpu)
        filled[...] = cp.asarray(filled_cpu)

    def mutate(self, updates, children, *args, kernel_shape):
        _mutate_kernel[*kernel_shape](updates, children, *args)

    def compute_distances(self, config, module, dtype, tpb, verbose=None):
        return _compute_distances(config, module, dtype, tpb, verbose)


BACKEND = GpuBackend()
