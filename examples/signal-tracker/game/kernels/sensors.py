"""Frame / history assembly kernels. A frame is [cursor, cue, tag] per cursor (+ optional instance noise)."""
from __future__ import annotations

from numba import njit, prange
from numpy import ndarray as Array

__all__ = ["fill_state", "push_history", "reset_history"]


@njit(parallel=True, cache=True)
def fill_state(out: Array, x: Array, cue: float, tag: float, noise: Array, use_noise: bool):
    """out (N, 3) = [cursor, cue, tag] (+ noise (N, 3) when use_noise)."""
    for p in prange(out.shape[0]):
        a = x[p]
        b = cue
        c = tag
        if use_noise:
            a += noise[p, 0]
            b += noise[p, 1]
            c += noise[p, 2]
        out[p, 0] = a
        out[p, 1] = b
        out[p, 2] = c


@njit(parallel=True, cache=True)
def push_history(hist: Array, out: Array, frame: Array):
    """Shift every history (N, L, F) one step left (oldest dropped), append `frame` (N, F); `out` gets a copy."""
    L = hist.shape[1]
    F = hist.shape[2]
    for p in prange(hist.shape[0]):
        for t in range(L - 1):
            for f in range(F):
                v = hist[p, t + 1, f]
                hist[p, t, f] = v
                out[p, t, f] = v
        for f in range(F):
            hist[p, L - 1, f] = frame[p, f]
            out[p, L - 1, f] = frame[p, f]


@njit(parallel=True, cache=True)
def reset_history(hist: Array, out: Array, frame: Array):
    """Fill every history slot with the first frame (start of an episode); `out` gets a copy."""
    for p in prange(hist.shape[0]):
        for t in range(hist.shape[1]):
            for f in range(hist.shape[2]):
                hist[p, t, f] = frame[p, f]
                out[p, t, f] = frame[p, f]
