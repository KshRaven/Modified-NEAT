"""Reward shaping kernel."""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy import ndarray as Array

__all__ = ["reward_shape"]


@njit(parallel=True, fastmath=True, cache=True)
def reward_shape(
    shape_pos: Array, shape_neg: Array,
    dist: Array, prev_dist: Array, ate: Array, died: Array, hiatus: Array, low_usage: Array, active: Array,
    max_hiatus: int, closer_weight: float, away_weight: float, time_penalty: float,
    hiatus_weight: float, loop_weight: float,
):
    """Dense per-frame shaping, in cells.

    Positive: cells gained towards the food. Negative: cells lost (weighted a bit higher so that
    shuffling back and forth never pays), a flat time cost, a cost growing as the snake goes
    `max_hiatus` frames without food, and a cost for looping or spiralling.
    The frame of a meal or a respawn carries no progress term (the food moved).
    """
    for p in prange(dist.shape[0]):
        shape_pos[p] = 0.0
        shape_neg[p] = 0.0
        if not active[p]:
            continue
        if not ate[p] and not died[p]:
            delta = prev_dist[p] - dist[p]
            if delta > 0.0:
                shape_pos[p] += closer_weight * delta
            else:
                shape_neg[p] += away_weight * -delta
        shape_neg[p] += time_penalty
        shape_neg[p] += hiatus_weight * min(hiatus[p] / max(max_hiatus, 1), 1.0)
        if low_usage[p]:
            shape_neg[p] += loop_weight
