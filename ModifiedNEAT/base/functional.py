"""
Device independent helpers shared by the CPU and GPU implementations.

Device specific counterparts (RNG sampling helpers, element accessors, ...) live in
``cpu/functional`` and ``gpu/functional``.
"""

from numba import njit

import math
import numpy as np

from ModifiedNEAT.base.seeding import set_seed, get_seed  # noqa: F401  (set_seed re-exported for compatibility)


def __getattr__(name):
    # Legacy ``functional.SEED`` now reflects the seed set through ``config.general.seed`` / ``set_seed``
    if name == 'SEED':
        return get_seed()
    raise AttributeError(name)


def calc_grid(*block_sizes: int, tpb: int = 4, multiplier: float = None):
    """Compute (blocks per grid, threads per block) for a 3D launch grid."""
    for _ in range(3 - len(block_sizes)):
        block_sizes += (1,)
    if multiplier is None:
        multiplier = 1
    return calc(block_sizes, tpb, multiplier)


def calc(tasks: tuple[int, ...], threads: int, multiplier: float):
    ndim = len(tasks)
    bpg: tuple[int] = (math.ceil((tasks[0] * multiplier) / threads),)
    tpb = (threads,)
    if ndim == 1:
        return bpg + tpb
    else:
        for dim_idx in range(1, ndim):
            t = threads
            if dim_idx == 1:
                t = min(10, t)
            if dim_idx == 2:
                t = min(10, t)
            bpg += (math.ceil((tasks[dim_idx] * multiplier) / t * 1),)
            if len(tasks) > 1:
                tpb += (t,)
        return bpg, tpb


def threads_total(kernel_shape: tuple) -> int:
    """Total number of threads (= number of RNG values needed) of a launch grid."""
    return int(np.prod([np.prod(v) for v in kernel_shape]))


@njit
def clamp(value: float, minimum: float, maximum: float):
    return min(max(value, minimum), maximum)
