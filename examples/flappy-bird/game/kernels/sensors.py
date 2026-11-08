"""Sensor kernels: the packed state vector of every bird.

State layouts (dead / out birds read -1 everywhere):
  basic (3)  y / H, (y - gap_top) / H, (y - gap_bot) / H                                         (the original state)
  full  (5)  basic + dx to the pipe's left / right edge / W
        (7)  + gap_top / H and (floor - gap_bot) / H of the pipe when pipes move vertically      (the original state)
  ego  (10)  relative to the next gap, nothing absolute:
             0 bird centre - gap centre              / H
             1 last displacement                     / terminal velocity   (how fast it rises / falls)
             2 bird centre - gap top                 / H
             3 bird centre - gap bottom              / H
             4 pipe left edge  - bird right edge     / W
             5 pipe right edge - bird left edge      / W
             6 bird centre - gap centre of the pipe after it / H
             7 vertical speed of the next gap        / terminal velocity   (0 = static)
             8 free height above the floor           / H
             9 frames since the last flap            / 30 (clipped to 1)
"""
from __future__ import annotations

from numba import njit, prange
from numpy import ndarray as Array

__all__ = ["get_state", "N_FEATURES_BASIC", "N_FEATURES_FULL", "N_FEATURES_FULL_MOVING", "N_FEATURES_EGO"]

N_FEATURES_BASIC = 3
N_FEATURES_FULL = 5
N_FEATURES_FULL_MOVING = 7
N_FEATURES_EGO = 10

MODE_BASIC, MODE_FULL, MODE_EGO = 0, 1, 2


@njit(parallel=True, cache=True)
def get_state(
    out: Array, mode: int, x: Array, y: Array, disp: Array, tick: Array, active: Array,
    bw: float, bh: float, width: float, height: float, floor_y: float, terminal: float,
    px: Array, gap_top: Array, gap_bot: Array, vdir: Array, pw: float, t1: int, t2: int, speed_y: float,
):
    cols = out.shape[1]
    gt, gb = gap_top[t1], gap_bot[t1]
    gc = 0.5 * (gt + gb)
    gc2 = 0.5 * (gap_top[t2] + gap_bot[t2])
    for p in prange(out.shape[0]):
        if not active[p]:
            for c in range(cols):
                out[p, c] = -1.0
            continue
        cy = y[p] + 0.5 * bh
        if mode == MODE_EGO:
            out[p, 0] = (cy - gc) / height
            out[p, 1] = disp[p] / terminal
            out[p, 2] = (cy - gt) / height
            out[p, 3] = (cy - gb) / height
            out[p, 4] = (px[t1] - (x[p] + bw)) / width
            out[p, 5] = (px[t1] + pw - x[p]) / width
            out[p, 6] = (cy - gc2) / height
            out[p, 7] = vdir[t1] * speed_y / terminal
            out[p, 8] = (floor_y - (y[p] + bh)) / height
            out[p, 9] = min(tick[p] / 30.0, 1.0)
        else:
            out[p, 0] = y[p] / height
            out[p, 1] = (y[p] - gt) / height
            out[p, 2] = (y[p] - gb) / height
            if mode == MODE_FULL:
                out[p, 3] = (px[t1] - x[p]) / width
                out[p, 4] = (px[t1] + pw - x[p]) / width
                if cols > 5:
                    out[p, 5] = gt / height
                    out[p, 6] = (floor_y - gb) / height
