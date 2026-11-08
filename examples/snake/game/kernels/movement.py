"""Movement kernels: spawn, step (with optional boundary wrap), food placement and distances.

Every snake is a ring buffer of its cells: segment i (0 = head) sits at index (head_idx - i) mod capacity.
The occupancy map `cells` (players, W, H) is kept in sync so collisions are O(1).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from numpy import ndarray as Array

from ..constants import CELL_BODY, CELL_EMPTY, CELL_FOOD, CELL_HEAD

__all__ = ["spawn", "step", "place_food", "food_distance", "low_usage"]

_DX = np.array([1, 0, -1, 0], dtype=np.int64)
_DY = np.array([0, -1, 0, 1], dtype=np.int64)


@njit(parallel=True, cache=True)
def spawn(
    cells: Array, body_x: Array, body_y: Array, head_idx: Array, length: Array, direction: Array,
    hiatus: Array, action_count: Array, mask: Array, init_dir: Array, init_len: int, wrap: bool,
):
    """(Re)place the masked snakes at the board centre, heading `init_dir[p]`, tail behind the head."""
    total, w, h = cells.shape
    cx, cy = w // 2, h // 2
    for p in prange(total):
        if not mask[p]:
            continue
        cells[p, :, :] = CELL_EMPTY
        d = init_dir[p]
        # Keep a centred spawn where possible, but shift the head just enough that its tail
        # remains on-board with solid walls. This also makes runtime wrap toggling safe on
        # boards where init_len is longer than half a dimension.
        sx, sy = cx, cy
        if not wrap:
            if d == 0:       # facing right; body extends left from the head
                sx = max(sx, init_len - 1)
            elif d == 2:     # facing left; body extends right from the head
                sx = min(sx, w - init_len)
            elif d == 1:     # facing up; body extends down from the head
                sy = min(sy, h - init_len)
            else:            # facing down; body extends up from the head
                sy = max(sy, init_len - 1)
        direction[p] = d
        length[p] = init_len
        head_idx[p] = init_len - 1
        hiatus[p] = 0
        action_count[p, :] = 0
        for i in range(init_len):                       # i = 0 is the head
            x = sx - _DX[d] * i
            y = sy - _DY[d] * i
            if wrap:
                x %= w
                y %= h
            idx = init_len - 1 - i
            body_x[p, idx] = x
            body_y[p, idx] = y
            cells[p, x, y] = CELL_HEAD if i == 0 else CELL_BODY


@njit(parallel=True, cache=True)
def step(
    cells: Array, body_x: Array, body_y: Array, head_idx: Array, length: Array, direction: Array,
    hiatus: Array, action_count: Array, actions: Array, active: Array, wrap: bool,
    ate: Array, hit_wall: Array, hit_self: Array,
):
    """Advance every active snake one cell. Actions: 0 left, 1 straight, 2 right.

    A snake that would hit a wall or itself does not move; the flags tell the player logic to
    take a life. Moving into the cell the tail leaves this frame is allowed (unless food was eaten).
    With `wrap` a snake leaving one side of the board appears on the opposite side.
    """
    total, w, h = cells.shape
    cap = body_x.shape[1]
    for p in prange(total):
        ate[p] = False
        hit_wall[p] = False
        hit_self[p] = False
        if not active[p]:
            continue

        act = actions[p]
        d = (direction[p] + 1 - act) % 4
        direction[p] = d
        action_count[p, act] += 1

        hi = head_idx[p]
        hx, hy = body_x[p, hi], body_y[p, hi]
        nx, ny = hx + _DX[d], hy + _DY[d]
        if wrap:
            nx %= w
            ny %= h
        elif nx < 0 or ny < 0 or nx >= w or ny >= h:
            hit_wall[p] = True
            hiatus[p] += 1
            continue

        n = length[p]
        ti = (hi - (n - 1)) % cap
        tx, ty = body_x[p, ti], body_y[p, ti]
        v = cells[p, nx, ny]
        eats = v == CELL_FOOD
        if v == CELL_BODY or v == CELL_HEAD:
            if eats or nx != tx or ny != ty:            # the tail cell is free if the tail moves away
                hit_self[p] = True
                hiatus[p] += 1
                continue

        if eats:
            length[p] = n + 1
            ate[p] = True
            hiatus[p] = 0
        else:
            cells[p, tx, ty] = CELL_EMPTY               # tail leaves
            hiatus[p] += 1
        if cells[p, hx, hy] == CELL_HEAD:
            cells[p, hx, hy] = CELL_BODY
        hi = (hi + 1) % cap
        head_idx[p] = hi
        body_x[p, hi] = nx
        body_y[p, hi] = ny
        cells[p, nx, ny] = CELL_HEAD


@njit(parallel=True, cache=True)
def place_food(cells: Array, food_x: Array, food_y: Array, mask: Array, rnd: Array):
    """Put new food on a random empty cell of every masked board (`rnd` in [0, 1) picks which).

    The old food cell is cleared first. A full board leaves food_x = food_y = -1.
    """
    total, w, h = cells.shape
    for p in prange(total):
        if not mask[p]:
            continue
        if food_x[p] >= 0 and cells[p, food_x[p], food_y[p]] == CELL_FOOD:
            cells[p, food_x[p], food_y[p]] = CELL_EMPTY
        free = 0
        for x in range(w):
            for y in range(h):
                if cells[p, x, y] == CELL_EMPTY:
                    free += 1
        if free == 0:
            food_x[p] = -1
            food_y[p] = -1
            continue
        k = min(int(rnd[p] * free), free - 1)
        for x in range(w):
            for y in range(h):
                if cells[p, x, y] == CELL_EMPTY:
                    if k == 0:
                        cells[p, x, y] = CELL_FOOD
                        food_x[p] = x
                        food_y[p] = y
                        k = -1
                        break
                    k -= 1
            if k < 0:
                break


@njit(parallel=True, cache=True)
def food_distance(
    body_x: Array, body_y: Array, head_idx: Array, food_x: Array, food_y: Array,
    w: int, h: int, wrap: bool, dist: Array, dx_out: Array, dy_out: Array,
):
    """Manhattan distance in cells to the food (shortest way round when wrapping) and the signed
    offset (dx, dy) of the food relative to the head. No food -> distance 0."""
    for p in prange(dist.shape[0]):
        if food_x[p] < 0:
            dist[p] = 0.0
            dx_out[p] = 0.0
            dy_out[p] = 0.0
            continue
        dx = food_x[p] - body_x[p, head_idx[p]]
        dy = food_y[p] - body_y[p, head_idx[p]]
        if wrap:
            if dx > w / 2: dx -= w
            elif dx < -w / 2: dx += w
            if dy > h / 2: dy -= h
            elif dy < -h / 2: dy += h
        dx_out[p] = dx
        dy_out[p] = dy
        dist[p] = abs(dx) + abs(dy)


@njit(parallel=True, cache=True)
def low_usage(action_count: Array, thresh: float, min_samples: int, out: Array):
    """True when some action is used less than `thresh` x the most used one (spiralling / looping)."""
    for p in prange(action_count.shape[0]):
        total = action_count[p, 0] + action_count[p, 1] + action_count[p, 2]
        out[p] = False
        if total < min_samples:
            continue
        top = max(action_count[p, 0], max(action_count[p, 1], action_count[p, 2]))
        limit = math.ceil(top * thresh)
        for a in range(3):
            if action_count[p, a] < limit:
                out[p] = True
