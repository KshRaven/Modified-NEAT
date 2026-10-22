"""Sensor kernels: rays along the three relative directions and the packed state vector."""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from numpy import ndarray as Array

from ..constants import CELL_BODY, CELL_FOOD

__all__ = ["cast_rays", "get_state", "grid_state", "N_FEATURES"]

N_FEATURES = 12     # heading(2) + food offset(2) + food distance(1) + obstacle rays(3) + food rays(3) + length(1)

_DX = np.array([1, 0, -1, 0], dtype=np.int64)
_DY = np.array([0, -1, 0, 1], dtype=np.int64)


@njit(parallel=True, cache=True)
def cast_rays(
    cells: Array, body_x: Array, body_y: Array, head_idx: Array, direction: Array,
    wrap: bool, max_len: int, obstacle: Array, food: Array,
):
    """Rays to the left (0), straight (1) and right (2) of the heading.

    obstacle[p, r] = cells to the first wall or body part (max_len + 1 when open),
    food[p, r]     = cells to the food along the ray (0 when not seen).
    With `wrap` a ray continues on the opposite side of the board.
    """
    total, w, h = cells.shape
    for p in prange(total):
        hx, hy = body_x[p, head_idx[p]], body_y[p, head_idx[p]]
        for r in range(3):
            d = (direction[p] + (1 if r == 0 else 0 if r == 1 else 3)) % 4
            x, y = hx, hy
            obstacle[p, r] = max_len + 1
            food[p, r] = 0
            for s in range(1, max_len + 1):
                x += _DX[d]
                y += _DY[d]
                if wrap:
                    x %= w
                    y %= h
                elif x < 0 or y < 0 or x >= w or y >= h:
                    obstacle[p, r] = s
                    break
                v = cells[p, x, y]
                if v >= CELL_BODY:
                    obstacle[p, r] = s
                    break
                if v == CELL_FOOD and food[p, r] == 0:
                    food[p, r] = s


@njit(parallel=True, fastmath=True, cache=True)
def get_state(
    out: Array, direction: Array, length: Array, dist: Array, dx: Array, dy: Array,
    obstacle: Array, food: Array, w: int, h: int, wrap: bool, max_len: int,
):
    """Pack the per-snake feature vector (all values roughly in [-1, 1]).

    0-1  heading (cos, sin)            2-3  food offset (dx, dy) normalized by the board side
    4    food distance / max distance  5-7  obstacle distance per ray (0 = adjacent, 1 = open)
    8-10 food seen per ray (0 = not seen, 1 = adjacent) 11 snake length / board cells
    """
    max_dist = (w // 2 + h // 2) if wrap else (w - 1 + h - 1)
    nx = (w / 2.0) if wrap else float(max(w - 1, 1))
    ny = (h / 2.0) if wrap else float(max(h - 1, 1))
    for p in prange(out.shape[0]):
        ang = direction[p] * 0.5 * math.pi
        out[p, 0] = round(math.cos(ang))
        out[p, 1] = round(math.sin(ang))
        out[p, 2] = dx[p] / nx
        out[p, 3] = dy[p] / ny
        out[p, 4] = dist[p] / max(max_dist, 1)
        for r in range(3):
            out[p, 5 + r] = (obstacle[p, r] - 1) / max_len
            out[p, 8 + r] = 0.0 if food[p, r] == 0 else 1.0 - (food[p, r] - 1) / max_len
        out[p, 11] = length[p] / (w * h)


@njit(parallel=True, cache=True)
def grid_state(cells: Array, out: Array):
    """Board as one normalized channel: empty -1, food -0.33, body 0.33, head 1."""
    total, w, h = cells.shape
    for p in prange(total):
        for x in range(w):
            for y in range(h):
                v = cells[p, x, y]
                out[p, 0, x, y] = -1.0 if v == 0 else (-0.33 if v == CELL_FOOD else (0.33 if v == CELL_BODY else 1.0))
