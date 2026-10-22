"""Race kernels: slipstream, slipstream trail history and car-to-car collisions.

Angle convention: heading vector = (-sin a, -cos a), `a` in degrees, increasing `a` turns left on screen.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from numpy import ndarray as Array

from ..config import CollisionConfig, SlipstreamConfig
from ..util import grid

__all__ = ["slipstream", "collide", "trail_push", "trail_points"]


@njit(parallel=True, fastmath=True, cache=True)
def _slipstream(
    x: Array, y: Array, angle: Array, vel_norm: Array, active: Array, level: Array,
    out: Array,
    car_len: float, rng: float, cone: float, max_heading_diff: float, min_leader_speed: float,
    distance_falloff: float, angle_falloff: float, ramp: float, decay: float,
):
    """Per-follower draft update. The leader scan is a serial inner loop."""
    total, indices = grid((x.shape[0],))
    for i in prange(total):
        f = indices[i]
        target = 0.0
        if active[f]:
            rad = angle[f] * math.pi / 180.0
            fx = -math.sin(rad)
            fy = -math.cos(rad)
            for j in range(total):
                if j != f and active[j] and vel_norm[j] > min_leader_speed:
                    ddx = x[j] - x[f]
                    ddy = y[j] - y[f]
                    ahead = ddx * fx + ddy * fy
                    side  = ddx * fy - ddy * fx
                    dist  = math.hypot(ddx, ddy)
                    theta = abs(math.atan2(side, max(ahead, 1e-6)))
                    d_ang = abs((angle[j] - angle[f] + 180.0) % 360.0 - 180.0)
                    if ahead > 0.5 * car_len and dist < rng and theta < cone and d_ang < max_heading_diff:
                        s = (
                            min(max(1.0 - dist / rng, 0.0), 1.0) ** distance_falloff
                            * min(max(1.0 - theta / cone, 0.0), 1.0) ** angle_falloff
                            * min(max(vel_norm[j], 0.0), 1.0)
                        )
                        if s > target: target = s
        rate = ramp if target > level[f] else decay
        out[f] = min(max(level[f] + rate * (target - level[f]), 0.0), 1.0)


def slipstream(
    x: Array, y: Array, angle: Array, vel_norm: Array, active: Array,
    level: Array, car_len: float, cfg: SlipstreamConfig,
) -> Array:
    """New draft level (N,) in [0, 1].

    For follower i and leader j: s = (1 - dist/range)^a * (1 - theta/cone)^b * leader_speed,
    where theta is the angle between heading i and the line to j. The strongest wake wins and the
    level eases towards it."""
    if x.shape[0] < 2:
        return np.zeros_like(level)

    out = np.empty_like(level)
    _slipstream(
        x, y, angle, vel_norm, active, level, out,
        float(car_len), float(cfg.range_cars * car_len), math.radians(cfg.cone_deg),
        float(cfg.max_heading_diff_deg), float(cfg.min_leader_speed),
        float(cfg.distance_falloff), float(cfg.angle_falloff), float(cfg.ramp), float(cfg.decay),
    )
    return out


# Deterministic jitter hash for particle trails (no RNG state, safe inside prange)
_HASH_CAR  = 73856093
_HASH_AGE  = 19349663
_HASH_TIME = 83492791
_HASH_MIX  = 40503          # odd 16-bit multiplier that decorrelates the second jitter axis
_HASH_MASK = 0xFFFF
_HASH_HALF = 32768.0        # maps [0, 65535] to [-1, 1)
_JITTER_HOLD_FRAMES = 3     # jitter changes every N frames
_MIN_SPAN = 1               # avoids dividing by zero when a single point is drawn


@njit(parallel=True, fastmath=True, cache=True)
def _trail_push(
    tx: Array, ty: Array, ta: Array, head: Array, count: Array,
    x: Array, y: Array, angle: Array, active: Array, jump: float,
):
    """Append the current pose of every car to its ring buffer.
    A jump larger than `jump` px (respawn) clears that car's history."""
    total, indices = grid((x.shape[0],))
    cap = tx.shape[1]
    for i in prange(total):
        c = indices[i]
        if not active[c]:
            count[c] = 0
            continue
        h = head[c]
        if count[c] > 0 and math.hypot(x[c] - tx[c, h], y[c] - ty[c, h]) > jump:
            count[c] = 0
        h = (h + 1) % cap
        tx[c, h] = x[c]
        ty[c, h] = y[c]
        ta[c, h] = angle[c]
        head[c] = h
        if count[c] < cap:
            count[c] += 1


@njit(cache=True, fastmath=True)
def _trail_points(
    tx: Array, ty: Array, ta: Array, head: Array, count: Array, c: int,
    n: int, rear: float, side: float, step: int, spread: float, frame: int, out: Array,
) -> int:
    """Fill `out` (>= n, 2) with the streak polyline of car `c`, newest to oldest, offset sideways by `side` px
    and behind the car by `rear` px. With spread > 0 the points get a hash jitter that grows with age.
    Returns the number of points written."""
    cap = tx.shape[1]
    n = min(n, count[c])
    m = 0
    k = 0
    while k < n:
        idx = (head[c] - k) % cap
        rad = ta[c, idx] * math.pi / 180.0
        fx = -math.sin(rad)
        fy = -math.cos(rad)
        px = tx[c, idx] - fx * rear - fy * side
        py = ty[c, idx] - fy * rear + fx * side
        if spread > 0.0:
            hsh = ((c * _HASH_CAR) ^ (k * _HASH_AGE) ^ ((frame // _JITTER_HOLD_FRAMES) * _HASH_TIME)) & _HASH_MASK
            r1 = hsh / _HASH_HALF - 1.0
            r2 = ((hsh * _HASH_MIX) & _HASH_MASK) / _HASH_HALF - 1.0
            age = k / max(n - 1, _MIN_SPAN)   # 0 at the car, 1 at the tail
            px += r1 * spread * age
            py += r2 * spread * age
        out[m, 0] = px
        out[m, 1] = py
        m += 1
        k += step
    return m


def trail_push(tx, ty, ta, head, count, x, y, angle, active, jump: float) -> None:
    _trail_push(tx, ty, ta, head, count, x, y, angle, active, float(jump))


def trail_points(tx, ty, ta, head, count, c, n, rear, side, step, spread, frame, out) -> int:
    return int(_trail_points(
        tx, ty, ta, head, count, int(c), int(n), float(rear), float(side), int(step),
        float(spread), int(frame), out,
    ))


@njit(parallel=True, fastmath=True, cache=True)
def _collide_near(x: Array, y: Array, active: Array, nearest: Array):
    """Broad phase: distance from every active car to its nearest active neighbour."""
    total, indices = grid((x.shape[0],))
    for i in prange(total):
        c = indices[i]
        best = np.inf
        if active[c]:
            for j in range(total):
                if j != c and active[j]:
                    d = math.hypot(x[c] - x[j], y[c] - y[j])
                    if d < best: best = d
        nearest[c] = best


@njit(parallel=True, fastmath=True, cache=True)
def _collide_forces(
    x: Array, y: Array, angle: Array, vel_lin: Array, active: Array,
    push_x: Array, push_y: Array, dv_x: Array, dv_y: Array, dang: Array, touched: Array,
    car_w: float, car_h: float, radius_scale: float, restitution: float, separation: float,
    max_shove: float, spin: float, max_spin: float,
):
    """Contact response per car: push-out, impulse and yaw torque accumulated from every neighbour
    (2 circles per car, up to 4 circle pairs per neighbour), reading only pre-collision state."""
    total, indices = grid((x.shape[0],))
    r   = car_w * radius_scale
    off = max(car_h / 2.0 - r, 0.0)
    for i in prange(total):
        c = indices[i]
        px = 0.0; py = 0.0; vx = 0.0; vy = 0.0; torque = 0.0
        hit = False
        if active[c]:
            rad_c = angle[c] * math.pi / 180.0
            fcx = -math.sin(rad_c); fcy = -math.cos(rad_c)
            for j in range(total):
                if j != c and active[j]:
                    rad_j = angle[j] * math.pi / 180.0
                    fjx = -math.sin(rad_j); fjy = -math.cos(rad_j)

                    # Overlapping circle pairs share the impulse
                    cnt = 0
                    for a in range(2):
                        sa = 1.0 - 2.0 * a
                        ax = x[c] + sa * fcx * off; ay = y[c] + sa * fcy * off
                        for b in range(2):
                            sb = 1.0 - 2.0 * b
                            bx = x[j] + sb * fjx * off; by = y[j] + sb * fjy * off
                            if max(math.hypot(ax - bx, ay - by), 1e-6) < 2.0 * r: cnt += 1

                    if cnt > 0:
                        hit = True
                        w = 1.0 / cnt
                        for a in range(2):
                            sa = 1.0 - 2.0 * a
                            ax = x[c] + sa * fcx * off; ay = y[c] + sa * fcy * off
                            for b in range(2):
                                sb = 1.0 - 2.0 * b
                                bx = x[j] + sb * fjx * off; by = y[j] + sb * fjy * off
                                dx = ax - bx; dy = ay - by
                                dist = max(math.hypot(dx, dy), 1e-6)
                                if dist < 2.0 * r:
                                    nx = dx / dist; ny = dy / dist
                                    pen = 2.0 * r - dist
                                    ux = fcx * vel_lin[c] - fjx * vel_lin[j]
                                    uy = fcy * vel_lin[c] - fjy * vel_lin[j]
                                    closing = max(-(ux * nx + uy * ny), 0.0)
                                    J = (1.0 + restitution) * 0.5 * closing * w
                                    k = pen * w * 0.5 * separation
                                    px += nx * k;  py += ny * k
                                    vx += nx * J;  vy += ny * J
                                    lx = 0.5 * (ax + bx) - x[c]
                                    ly = 0.5 * (ay + by) - y[c]
                                    torque += lx * (ny * J) - ly * (nx * J)

            mag = math.hypot(px, py)
            cap = car_w * max_shove
            if mag > cap:
                sc = cap / max(mag, 1e-9)
                px *= sc; py *= sc

        push_x[c] = px; push_y[c] = py
        dv_x[c] = vx;   dv_y[c] = vy
        dang[c] = min(max(-spin * torque / max(car_h / 2.0, 1e-6), -max_spin), max_spin)
        touched[c] = hit


@njit(parallel=True, fastmath=True, cache=True)
def _collide_apply(
    x: Array, y: Array, angle: Array, vel_lin: Array, vel_ang: Array, disp_x: Array, disp_y: Array,
    push_x: Array, push_y: Array, dv_x: Array, dv_y: Array, dang: Array, touched: Array,
    min_speed_abs: float, lateral_transfer: float,
):
    total, indices = grid((x.shape[0],))
    for i in prange(total):
        c = indices[i]
        rad = angle[c] * math.pi / 180.0
        fx = -math.sin(rad); fy = -math.cos(rad)
        x[c] += push_x[c]
        y[c] += push_y[c]
        forward = dv_x[c] * fx + dv_y[c] * fy
        lat_x = dv_x[c] - fx * forward
        lat_y = dv_y[c] - fy * forward
        if touched[c] and vel_lin[c] > 0:
            vel_lin[c] = max(vel_lin[c] + forward, min_speed_abs)
        # disp_* hold the negated displacement (position -= disp)
        disp_x[c] -= lat_x * lateral_transfer
        disp_y[c] -= lat_y * lateral_transfer
        if touched[c]:
            vel_ang[c] += dang[c]


def collide(
    x: Array, y: Array, angle: Array, vel_lin: Array, vel_ang: Array,
    disp_x: Array, disp_y: Array, active: Array,
    car_w: float, car_h: float, max_vel: float, cfg: CollisionConfig,
) -> Array:
    """Resolve car overlaps in place. Returns a bool (N,) mask of cars hit this frame.

    Each car is two circles. Response: push-out along the contact normal, an impulse changing forward
    speed and adding a sideways slide (disp_x/y), and a yaw kick on vel_ang from off-center contact."""
    N = x.shape[0]
    touched = np.zeros(N, dtype=np.bool_)
    if N < 2:
        return touched

    # Broad phase: cars can only touch when their centers are closer than one car length
    nearest = np.empty(N, dtype=np.float64)
    _collide_near(x, y, active, nearest)
    if nearest.min() >= car_h:
        return touched

    push_x, push_y, dv_x, dv_y, dang = (np.zeros(N, dtype=np.float64) for _ in range(5))
    _collide_forces(
        x, y, angle, vel_lin, active, push_x, push_y, dv_x, dv_y, dang, touched,
        float(car_w), float(car_h), float(cfg.radius_scale), float(cfg.restitution),
        float(cfg.separation), float(cfg.max_shove), float(cfg.spin), float(cfg.max_spin),
    )
    if touched.any():
        _collide_apply(
            x, y, angle, vel_lin, vel_ang, disp_x, disp_y,
            push_x, push_y, dv_x, dv_y, dang, touched,
            float(cfg.min_speed * max_vel), float(cfg.lateral_transfer),
        )
    return touched
