"""World kernels: pipe generation, scrolling (with vertical movement) and target pipe lookup.

The pipes live in a small fixed ring of arrays; a pipe that left the screen on the left is recycled behind the
last one. All random draws come from `rnd` (P, 4), filled from the Env-private generator, so a seed fully
determines the pipe sequence.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from numpy import ndarray as Array

__all__ = ["init_pipes", "scroll", "targets"]


@njit(cache=True)
def _place(
    k: int, x: float, px: Array, gap_top: Array, gap_bot: Array, vdir: Array, gid: Array, rnd: Array, g: int,
    lo: float, hi: float, gap_lo: int, gap_hi: int, speed_y: float, move_chance: float,
):
    gap = gap_lo + int(rnd[k, 0] * (gap_hi - gap_lo + 1))
    gap = min(gap, gap_hi)
    span = max(hi - lo - gap, 1.0)
    can_move = speed_y > 0.0 and gap < hi - lo and rnd[k, 1] < move_chance
    top = lo + float(int(rnd[k, 2] * span))
    px[k] = x
    gap_top[k] = top
    gap_bot[k] = top + gap
    vdir[k] = (-1 if rnd[k, 3] < 0.5 else 1) if can_move else 0
    gid[k] = g


@njit(cache=True)
def init_pipes(
    px: Array, gap_top: Array, gap_bot: Array, vdir: Array, gid: Array, rnd: Array, first_x: float, pw: float,
    spawn_width: float, lo: float, hi: float, gap_lo: int, gap_hi: int, speed_y: float, move_chance: float,
) -> int:
    """Fill the ring with pipes one spacing apart, first one at `first_x`. Returns the next pipe id."""
    x = first_x
    for k in range(px.shape[0]):
        _place(k, x, px, gap_top, gap_bot, vdir, gid, rnd, k, lo, hi, gap_lo, gap_hi, speed_y, move_chance)
        x += pw + spawn_width
    return px.shape[0]


@njit(cache=True)
def scroll(
    px: Array, gap_top: Array, gap_bot: Array, vdir: Array, gid: Array, rnd: Array,
    velocity: float, speed_y: float, pw: float, spawn_width: float, lo: float, hi: float,
    gap_lo: int, gap_hi: int, move_chance: float, next_gid: int, pass_off: float, bird_x: float,
):
    """Move every pipe left (and up / down), recycle the ones that left the screen.

    Returns (pipes whose pass point crossed `bird_x` this frame, next pipe id).
    """
    crossed = 0
    for k in range(px.shape[0]):
        before = px[k] + pass_off
        px[k] -= velocity
        if before >= bird_x and px[k] + pass_off < bird_x:
            crossed += 1
        if vdir[k] != 0:
            if vdir[k] > 0:                                  # moving down
                if gap_bot[k] + speed_y < hi:
                    gap_top[k] += speed_y
                    gap_bot[k] += speed_y
                else:
                    vdir[k] = -1
                    gap_top[k] -= speed_y
                    gap_bot[k] -= speed_y
            else:                                            # moving up
                if gap_top[k] - speed_y > lo:
                    gap_top[k] -= speed_y
                    gap_bot[k] -= speed_y
                else:
                    vdir[k] = 1
                    gap_top[k] += speed_y
                    gap_bot[k] += speed_y
    for k in range(px.shape[0]):
        if px[k] + pw < 0.0:
            last = px[0]
            for j in range(px.shape[0]):
                if px[j] > last:
                    last = px[j]
            _place(k, last + pw + spawn_width, px, gap_top, gap_bot, vdir, gid, rnd, next_gid,
                   lo, hi, gap_lo, gap_hi, speed_y, move_chance)
            next_gid += 1
    return crossed, next_gid


@njit(cache=True)
def targets(px: Array, clear_off: float, bird_x: float) -> tuple[int, int]:
    """Indices of the next two pipes whose right edge (`px + clear_off`) has not passed `bird_x` yet (nearest first)."""
    best, second = -1, -1
    for k in range(px.shape[0]):
        if px[k] + clear_off < bird_x:
            continue
        if best < 0 or px[k] < px[best]:
            second = best
            best = k
        elif second < 0 or px[k] < px[second]:
            second = k
    if best < 0:                                             # nothing ahead (cannot happen with a full ring)
        best = int(np.argmax(px))
    if second < 0:
        second = best
    return best, second
