"""Cursor movement kernel."""
from __future__ import annotations

from numba import njit, prange
from numpy import ndarray as Array

__all__ = ["move"]


@njit(parallel=True, cache=True)
def move(x: Array, scores: Array, speed: float):
    """In place: x += speed * clip(score, -1, 1), then x clipped to the track [-1, 1]."""
    for p in prange(x.shape[0]):
        s = scores[p]
        if s > 1.0:
            s = 1.0
        elif s < -1.0:
            s = -1.0
        v = x[p] + speed * s
        if v > 1.0:
            v = 1.0
        elif v < -1.0:
            v = -1.0
        x[p] = v
