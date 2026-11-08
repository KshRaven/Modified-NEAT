"""Bird kernels: spawn, flight physics (raw kinematics, no sprite masks) and analytic collision.

A flap sets the velocity to -jump_vel and restarts the tick counter t. Every frame t grows by one and the bird
moves by `vel * t + fall_coeff * t^2` pixels (clipped to the terminal velocity; +jump_offset extra pixels while
rising). The hitbox is an ellipse inscribed in the sprite box, tested against the pipe rectangles directly.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy import ndarray as Array

__all__ = ["spawn", "step", "collide", "low_usage"]


@njit(parallel=True, cache=True)
def spawn(
    x: Array, y: Array, vel: Array, tick: Array, height: Array, tilt: Array, disp: Array,
    hiatus: Array, action_count: Array, mask: Array, init_x: float, init_y: float,
):
    """(Re)place the masked birds at (init_x, init_y), at rest."""
    for p in prange(x.shape[0]):
        if not mask[p]:
            continue
        x[p] = init_x
        y[p] = init_y
        vel[p] = 0.0
        tick[p] = 0
        height[p] = init_y
        tilt[p] = 0.0
        disp[p] = 0.0
        hiatus[p] = 0
        action_count[p, 0] = 0
        action_count[p, 1] = 0


@njit(parallel=True, cache=True)
def step(
    y: Array, vel: Array, tick: Array, height: Array, tilt: Array, disp: Array,
    hiatus: Array, action_count: Array, flap: Array, active: Array,
    fall_coeff: float, terminal: float, jump_vel: float, jump_offset: float, max_rot: float, ang_vel: float,
    draft_y: float = 0.0,
):
    """Advance every active bird one frame (flap, fall, tilt). `draft_y` scales the displacement (see DraftConfig)."""
    for p in prange(y.shape[0]):
        if not active[p]:
            continue
        if flap[p]:
            vel[p] = -jump_vel
            tick[p] = 0
            height[p] = y[p]
            action_count[p, 1] += 1
        else:
            action_count[p, 0] += 1
        tick[p] += 1
        t = float(tick[p])
        d = vel[p] * t + fall_coeff * t * t
        if d >= terminal:
            d = terminal
        if d < 0.0:
            d -= jump_offset
        if draft_y != 0.0:                              # y draft (+ = up): rises grow, falls shrink; (- = down) the reverse
            d = d * (1.0 + draft_y) if d < 0.0 else d * (1.0 - draft_y)
        y[p] += d
        disp[p] = d
        hiatus[p] += 1

        if d < 0.0 or y[p] < height[p] + 50.0:          # rising or just flapped: nose up
            if tilt[p] < max_rot:
                tilt[p] = max_rot
        elif tilt[p] > -90.0:                           # falling: nose dives
            tilt[p] = max(tilt[p] - ang_vel, -90.0)


@njit(cache=True)
def _ellipse_rect(cx: float, cy: float, ax: float, ay: float, x0: float, y0: float, x1: float, y1: float) -> bool:
    """Ellipse (centre, semi-axes) against an axis-aligned rectangle. The y axis is stretched so the ellipse is a circle."""
    k = ax / ay
    ny = min(max(cy * k, y0 * k), y1 * k)
    nx = min(max(cx, x0), x1)
    dx, dy = cx - nx, cy * k - ny
    return dx * dx + dy * dy < ax * ax


@njit(parallel=True, cache=True)
def collide(
    x: Array, y: Array, bw: float, bh: float, shrink: float,
    px: Array, gap_top: Array, gap_bot: Array, pw: float, floor_y: float,
    active: Array, hit_pipe: Array, hit_bound: Array,
):
    """Flags the birds that touch a pipe or leave the playfield (ceiling / floor)."""
    ax, ay = 0.5 * bw * shrink, 0.5 * bh * shrink
    for p in prange(x.shape[0]):
        hit_pipe[p] = False
        hit_bound[p] = False
        if not active[p]:
            continue
        if y[p] <= 0.0 or y[p] + bh >= floor_y:
            hit_bound[p] = True
            continue
        cx, cy = x[p] + 0.5 * bw, y[p] + 0.5 * bh
        for k in range(px.shape[0]):
            if cx + ax <= px[k] or cx - ax >= px[k] + pw:
                continue
            if (_ellipse_rect(cx, cy, ax, ay, px[k], -1.0e9, px[k] + pw, gap_top[k])
                    or _ellipse_rect(cx, cy, ax, ay, px[k], gap_bot[k], px[k] + pw, 1.0e9)):
                hit_pipe[p] = True
                break


@njit(parallel=True, cache=True)
def low_usage(action_count: Array, thresh: float, min_samples: int, out: Array):
    """True when flap or no-flap is used less than `thresh` x the most used one (never flapping / spamming)."""
    for p in prange(action_count.shape[0]):
        total = action_count[p, 0] + action_count[p, 1]
        out[p] = False
        if total < min_samples:
            continue
        top = max(action_count[p, 0], action_count[p, 1])
        limit = np.ceil(top * thresh)
        if action_count[p, 0] < limit or action_count[p, 1] < limit:
            out[p] = True