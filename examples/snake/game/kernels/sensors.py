"""Sensor kernels: rays along the three relative directions and the packed state vector."""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from numpy import ndarray as Array

from ..constants import CELL_BODY, CELL_FOOD
from .road import K_DEAD, K_FREE, K_ROAD, K_SHORT, K_SLOW, K_UNSAFE

__all__ = ["cast_rays", "get_state", "grid_state", "strategy_state", "ego_state",
           "N_FEATURES", "N_FEATURES_STRATEGY", "N_FEATURES_EGO", "N_FEATURES_EGO_PLAIN", "GRID_CHANNELS"]

N_FEATURES = 12     # heading(2) + food offset(2) + food distance(1) + obstacle rays(3) + food rays(3) + length(1)
GRID_CHANNELS = 3          # grid state: body age, food, next-move cells
N_FEATURES_EGO_PLAIN = 8    # ego state without the road geometry (pure sensing)
N_FEATURES_EGO = 13         # ego state with the road geometry
N_FEATURES_STRATEGY = 20    # egocentric: food bearing + distance, 3 rays, 3 moves x (grade, road gain, room, tail), road gap, in-order, length

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
def grid_state(
    out: Array, body_x: Array, body_y: Array, head_idx: Array, length: Array, direction: Array,
    food_x: Array, food_y: Array, active: Array, w: int, h: int, wrap: bool,
):
    """Board as an image, GRID_CHANNELS channels of shape (w, h), indexed [channel, x, y]. Purely visual: no road,
    no grade, no distance; everything a player would see on the screen.

    0  body    snake cells by age: head = 1, then falling in equal steps to 1 / length at the tail; empty = 0.
               The slope shows the body order and therefore which way the snake is facing.
    1  food    1 on the food cell, 0 elsewhere.
    2  moves   the three cells the next move can enter: left of the head = 0.33, straight ahead = 0.66,
               right of the head = 1.0 (0 elsewhere). This ties the relative actions (left / straight / right)
               to the picture, so the network does not need to infer the heading from the body.
    """
    cap = body_x.shape[1]
    for p in prange(out.shape[0]):
        for c in range(out.shape[1]):
            for x in range(w):
                for y in range(h):
                    out[p, c, x, y] = 0.0
        if not active[p]:
            continue
        L = length[p]
        hi = head_idx[p]
        for i in range(L):
            idx = (hi - i) % cap
            out[p, 0, body_x[p, idx], body_y[p, idx]] = 1.0 - i / L
        if food_x[p] >= 0:
            out[p, 1, food_x[p], food_y[p]] = 1.0
        hx = body_x[p, hi]
        hy = body_y[p, hi]
        for r in range(3):
            d = (direction[p] + (1 if r == 0 else 0 if r == 1 else 3)) % 4
            nx = hx + _DX[d]
            ny = hy + _DY[d]
            if wrap:
                nx %= w
                ny %= h
            elif nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            out[p, 2, nx, ny] = (r + 1) / 3.0


@njit(parallel=True, fastmath=True, cache=True)
def strategy_state(
    out: Array, direction: Array, length: Array, dist: Array, dx: Array, dy: Array, obstacle: Array,
    kind: Array, gain: Array, space: Array, tail_ok: Array, phi: Array, synced: Array,
    w: int, h: int, wrap: bool, max_len: int,
):
    """Egocentric strategy features (N_FEATURES_STRATEGY values in [-1, 1]).

    Nothing here depends on the compass heading: the snake is always "facing forward", like the car's lidar. The
    food is given as a bearing (phase difference between the heading and the direction to the food) plus a
    distance, and every other feature is indexed by the relative moves left / straight / right.

    0      food bearing: angle between the heading and the food / pi, -1..1 (0 = dead ahead, > 0 = to the left,
           +-1 = behind)
    1      food distance / longest possible distance
    2-4    free distance ahead per ray (left, straight, right): 0 = wall or body adjacent, 1 = open
    5-16   per relative move (left, straight, right), 4 values each:
             grade   -1 wall/body, -0.5 breaks road order, 0.5 free / safe but not shorter, 1 road or shortcut
             gain    road cells the move saves / (w + h), clipped (a shortcut is > 1 cell, a bad hop is < 0)
             room    free squares reachable after the move / snake length, capped at 2 and halved
                     (< 0.5: a pocket smaller than the snake; with no way to the tail it is a trap)
             tail    1 when that region touches the tail (the snake can always follow its tail out)
    17     road distance to the food / cells (how far the food is along the one-way road)
    18     body in road order (1) or not (0)
    19     snake length / board cells
    """
    max_dist = (w // 2 + h // 2) if wrap else (w - 1 + h - 1)
    for p in prange(out.shape[0]):
        d = direction[p]
        hx = _DX[d]
        hy = _DY[d]
        left = (d + 1) % 4
        fwd = dx[p] * hx + dy[p] * hy                      # components of the food offset in the heading frame
        lat = dx[p] * _DX[left] + dy[p] * _DY[left]
        out[p, 0] = math.atan2(lat, fwd) / math.pi if (fwd != 0.0 or lat != 0.0) else 0.0
        out[p, 1] = dist[p] / max(max_dist, 1)
        for r in range(3):
            out[p, 2 + r] = (obstacle[p, r] - 1) / max_len
            k = kind[p, r]
            if k == K_DEAD:
                g = -1.0
            elif k == K_UNSAFE:
                g = -0.5
            elif k == K_ROAD or k == K_SHORT:
                g = 1.0
            else:
                g = 0.5
            o = 5 + 4 * r
            out[p, o] = g
            out[p, o + 1] = min(max(gain[p, r] / (w + h), -1.0), 1.0)
            out[p, o + 2] = min(space[p, r] / max(length[p], 1), 2.0) * 0.5
            out[p, o + 3] = tail_ok[p, r]
        out[p, 17] = phi[p] / (w * h)
        out[p, 18] = 1.0 if synced[p] else 0.0
        out[p, 19] = length[p] / (w * h)


@njit(parallel=True, fastmath=True, cache=True)
def ego_state(
    out: Array, body_x: Array, body_y: Array, head_idx: Array, length: Array, direction: Array,
    food_x: Array, food_y: Array, obstacle: Array, rank: Array, active: Array,
    w: int, h: int, wrap: bool, max_len: int, use_road: bool,
):
    """Egocentric state. No feature says whether a move is good: the snake has to work that out itself.

    Everything is measured from the head and in the heading frame (forward / left), so the compass direction and
    the absolute position never enter, exactly like the car's lidar and its tile-relative values. The board is
    never given as coordinates.

    Sensing (always present, 8 values)
      0  food bearing   angle between the heading and the food, atan2(left, forward) / pi.
                        0 = dead ahead, +0.5 = to the left, -0.5 = to the right, +-1 = behind.
      1  food distance  Manhattan distance head -> food (shortest way round with wrap) / (w + h).
      2  tail bearing   same as 0, for the last body cell.
      3  tail distance  same as 1, for the last body cell.
      4  ray left       cells from the head to the first wall or body part along the left ray, scaled to
      5  ray ahead      0..1: 0 = blocked right next to the head, 1 = open to the ray length (max_len).
      6  ray right
      7  length         snake length / number of cells.

    Road geometry (only with use_road, 5 more values). The road is a one-way ring through every cell, the same
    way a track tile has an entry and an exit. Only positions on the ring are given, never a verdict:
      8  road offset of the cell on the left of the head, i.e. how many road steps that cell lies ahead of the
      9  road offset of the cell straight ahead           head on the ring, / number of cells. 1/N = the cell
      10 road offset of the cell on the right             the road itself leads to; close to 1 = it lies just
                                                          behind the head. 0 = off the board (no wrap).
      11 road offset of the food     road steps from the head to the food, / number of cells.
      12 road offset of the tail     road steps from the head to the tail, / number of cells. Road cells between
                                     the head and this offset are the ones the body does not hold in road order.
    A hop is safe when the cell is free (rays) and its offset is below the tail offset; it is a shortcut when the
    offset is also larger than 1/N and not past the food offset. The network has to find that rule through the
    reward; the state only provides the numbers.
    """
    n_cells = w * h
    cap = body_x.shape[1]
    scale = float(w + h)
    for p in prange(out.shape[0]):
        for k in range(out.shape[1]):
            out[p, k] = 0.0
        if not active[p]:
            continue
        d = direction[p]
        hi = head_idx[p]
        hx = body_x[p, hi]
        hy = body_y[p, hi]
        ti = (hi - (length[p] - 1)) % cap
        tx = body_x[p, ti]
        ty = body_y[p, ti]
        fx = _DX[d]
        fy = _DY[d]
        lx = _DX[(d + 1) % 4]
        ly = _DY[(d + 1) % 4]

        for t in range(2):          # 0 = food, 1 = tail
            if t == 0:
                ox = food_x[p] - hx
                oy = food_y[p] - hy
            else:
                ox = tx - hx
                oy = ty - hy
            if wrap:                # shortest way round
                if ox > w // 2: ox -= w
                elif ox < -(w // 2): ox += w
                if oy > h // 2: oy -= h
                elif oy < -(h // 2): oy += h
            ahead = ox * fx + oy * fy
            side = ox * lx + oy * ly
            if t == 0 and food_x[p] < 0:
                continue
            out[p, 2 * t] = math.atan2(side, ahead) / math.pi if (ahead != 0 or side != 0) else 0.0
            out[p, 2 * t + 1] = (abs(ox) + abs(oy)) / scale
        for r in range(3):
            out[p, 4 + r] = (obstacle[p, r] - 1) / max_len
        out[p, 7] = length[p] / n_cells

        if use_road:
            hc = hy * w + hx
            for r in range(3):
                dd = (d + (1 if r == 0 else 0 if r == 1 else 3)) % 4
                nx = hx + _DX[dd]
                ny = hy + _DY[dd]
                if wrap:
                    nx %= w
                    ny %= h
                elif nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue
                out[p, 8 + r] = ((rank[p, ny * w + nx] - rank[p, hc]) % n_cells) / n_cells
            if food_x[p] >= 0:
                out[p, 11] = ((rank[p, food_y[p] * w + food_x[p]] - rank[p, hc]) % n_cells) / n_cells
            out[p, 12] = ((rank[p, ty * w + tx] - rank[p, hc]) % n_cells) / n_cells