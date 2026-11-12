"""Reward + error bookkeeping kernel."""
from __future__ import annotations

from numba import njit, prange
from numpy import ndarray as Array

__all__ = ["step_reward"]


@njit(parallel=True, cache=True)
def step_reward(
    rewards: Array, score_sum: Array, err_sum: Array, err_decoy: Array,
    x: Array, target: float, scale: float, decoy: bool,
):
    """rewards = scale * (1 - min(|x - target|, 1)); also accumulates the score and the absolute tracking error
    (total and on decoy frames, so the error on clean frames is the difference)."""
    for p in prange(x.shape[0]):
        e = abs(x[p] - target)
        r = scale * (1.0 - (1.0 if e > 1.0 else e))
        rewards[p] = r
        score_sum[p] += r
        err_sum[p] += e
        if decoy:
            err_decoy[p] += e
