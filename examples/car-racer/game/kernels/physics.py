"""Per-car motion kernels: steering, driving and solid walls."""
from __future__ import annotations

import math

from numba import njit, prange
from numpy import ndarray as Array

from ..constants import L_TRACK
from ..util import grid, sign, wrap180

__all__ = [
    "steer", "drive", "violations", "blocked", "walls",
]


@njit(parallel=True, fastmath=True, cache=True)
def steer(
    rotation: Array, vel_ang: Array, on_gravel: Array, on_grass: Array, on_curb: Array,
    friction: Array, handling: Array, slip: Array,
    coef: Array,   # (3, 4): rows friction/handling/slip, cols road/curb/grass/gravel
    max_vel_ang: float, acc_ang: float, coeff_checker: float,
):
    """Surface coefficients and steering for every car.

    `handling` scales both how fast the yaw rate builds and its cap.
    """
    total, indices = grid((vel_ang.shape[0],))
    for i in prange(total):
        c = indices[i]

        col = 0
        if   on_gravel[c]: col = 3
        elif on_grass[c]:  col = 2
        elif on_curb[c]:   col = 1
        friction[c] = coef[0, col]
        handling[c] = coef[1, col]
        slip[c]     = coef[2, col]

        h   = handling[c]
        cap = max_vel_ang * h
        rot = rotation[c]
        va  = vel_ang[c]
        if sign(rot) != sign(va):
            rot *= 1.0 + abs(va / max_vel_ang) * coeff_checker
        vel_ang[c] = min(max(va + acc_ang * h * rot, -cap), cap)


@njit(parallel=True, fastmath=True, cache=True)
def drive(
    acceleration: Array, vel_lin: Array, vel_ang: Array, angle: Array, x: Array, y: Array,
    disp_x: Array, disp_y: Array, vel_norm: Array,
    friction: Array, slip: Array,
    on_gravel: Array, draft: Array, gravel_noise: Array,
    max_vel_lin: float, acc_lin: float, coeff_brake: float, coeff_reverse: float,
    reverse_enabled: bool, drafting: bool, ss_bonus: float, ss_push: float,
    lateral_grip: float, grip_relief: float, slide_drag: float,
):
    """Speed, heading and displacement of every car.

    Friction scales engine force and the surface top speed. Speed above the top speed is
    shed at engine-brake rate. The travel direction follows the nose at a rate limited by
    lateral grip / speed, so slipping redirects speed without adding any.
    """
    total, indices = grid((x.shape[0],))
    disp_max = max_vel_lin * (1.0 + ss_bonus) if drafting else max_vel_lin
    for i in prange(total):
        c = indices[i]
        f = friction[c]

        top  = max_vel_lin * f
        push = 0.0
        if drafting:
            top = max_vel_lin * f * (1.0 + ss_bonus * draft[c])
            if acceleration[c] > 0.0: push = ss_push * acc_lin * draft[c]

        acc = acceleration[c]
        v   = vel_lin[c]
        if sign(acc) != sign(v):
            acc *= 1.0 + abs(v / max_vel_lin) * coeff_brake
        dv = acc_lin * acc * f + push
        lo = -max_vel_lin if reverse_enabled else 0.0
        if v > top:
            nv = min(v + dv, v - acc_lin)          # engine braking
            if dv >= -acc_lin: nv = max(nv, top)   # never below the surface top speed
        else:
            nv = min(v + dv, top)
        v = max(nv, lo)
        if reverse_enabled and v < 0.0: v *= coeff_reverse
        vel_lin[c] = v

        vn = max(v / max_vel_lin, -1.0)    # may exceed 1.0 while drafting
        vel_norm[c] = vn
        avn = abs(vn)

        a = (angle[c] + vel_ang[c]) % 360.0
        if on_gravel[c]: a += gravel_noise[c] * avn
        angle[c] = a
        rad = a * math.pi / 180.0

        # Path direction chases the nose at a rate of lateral grip / speed
        s       = slip[c]
        nose    = rad if v >= 0.0 else rad + math.pi
        speed   = abs(v)
        pdx = disp_x[c]; pdy = disp_y[c]
        pspeed  = math.sqrt(pdx * pdx + pdy * pdy)
        phi     = math.atan2(pdx, pdy) if pspeed > 1.0e-3 else nose
        delta   = (nose - phi + math.pi) % (2.0 * math.pi) - math.pi
        w_grip  = lateral_grip * (1.0 - s) / max(max(pspeed, speed), 0.5)       # rad / frame
        step    = min(abs(delta), w_grip * (1.0 - s * avn))
        relief  = max(1.0 - speed / max(grip_relief * max_vel_lin, 1.0e-6), 0.0)
        step   += relief * (abs(delta) - step)
        phi    += step if delta >= 0.0 else -step
        misalign = abs((nose - phi + math.pi) % (2.0 * math.pi) - math.pi)
        m  = speed * (1.0 - slide_drag * (1.0 - math.cos(misalign)) * 0.5)      # sliding scrubs speed
        dx = m * math.sin(phi)
        dy = m * math.cos(phi)
        n = math.sqrt(dx * dx + dy * dy)
        if n > disp_max:
            dx *= disp_max / n
            dy *= disp_max / n
        disp_x[c] = dx
        disp_y[c] = dy
        x[c] -= dx
        y[c] -= dy


@njit(cache=True)
def violations(
    layers: Array, tile_map: Array, t_next: Array, t_prev: Array, cell_size: float, cur: int,
    x: float, y: float, ang: float, hw: float, hl: float, block_prev: bool,
) -> int:
    """Number of body probes (3x3 grid over the footprint) outside the track, plus 100 if the center
    is in a tile the car may not enter. Only the current and next tile are open (and the
    previous one when `block_prev` is False).
    """
    gw, gh = layers.shape
    a  = ang * math.pi / 180.0
    fx = -math.sin(a); fy = -math.cos(a)
    rx = -fy;          ry = fx
    bad = 0
    for sl in range(-1, 2):
        for sf in range(-1, 2):
            px = x + sl * hw * rx + sf * hl * fx
            py = y + sl * hw * ry + sf * hl * fy
            xi = int(px); yi = int(py)
            if px < 0.0 or py < 0.0 or xi >= gw or yi >= gh:
                bad += 1
            elif not (layers[xi, yi] & L_TRACK):
                bad += 1
    kx = int(abs(x) / cell_size)
    ky = int(abs(y) / cell_size)
    if kx < tile_map.shape[0] and ky < tile_map.shape[1]:
        t = tile_map[kx, ky]
        if t >= 0 and t != cur and t != t_next[cur] and (block_prev or t != t_prev[cur]):
            bad += 100
    return bad


@njit(cache=True)
def blocked(
    layers: Array, tile_map: Array, t_next: Array, t_prev: Array, cell_size: float, cur: int,
    block_prev: bool, px: float, py: float,
) -> bool:
    """True if the point is off the track or inside a tile the car may not enter."""
    gw, gh = layers.shape
    xi = int(px); yi = int(py)
    if px < 0.0 or py < 0.0 or xi >= gw or yi >= gh:
        return True
    if not (layers[xi, yi] & L_TRACK):
        return True
    kx = int(abs(px) / cell_size)
    ky = int(abs(py) / cell_size)
    if kx < tile_map.shape[0] and ky < tile_map.shape[1]:
        t = tile_map[kx, ky]
        if t >= 0 and t != cur and t != t_next[cur] and (block_prev or t != t_prev[cur]):
            return True
    return False


@njit(parallel=True, fastmath=True, cache=True)
def walls(
    layers: Array, tile_map: Array, t_next: Array, t_prev: Array, cell_size: float,
    x: Array, y: Array, angle: Array, prev_x: Array, prev_y: Array, prev_angle: Array,
    vel_lin: Array, vel_ang: Array, disp_x: Array, disp_y: Array, tile_idx: Array,
    hw: float, hl: float, block_prev: bool, damping: float, align: float,
):
    """Solid walls for restrict_movement 1-2: the track edge and every tile except the current and next
    (`block_prev` also closes the previous tile).

    A blocked move is projected onto the wall so the car slides along it. While sliding, the nose
    turns towards the sliding direction by `align` and speed is scaled by `damping`.
    """
    total, indices = grid((x.shape[0],))
    ring = 12
    for i in prange(total):
        c   = indices[i]
        cur = tile_idx[c]
        base = violations(layers, tile_map, t_next, t_prev, cell_size, cur, prev_x[c], prev_y[c], prev_angle[c], hw, hl, block_prev)
        v0 = violations(layers, tile_map, t_next, t_prev, cell_size, cur, x[c], y[c], angle[c], hw, hl, block_prev)
        if v0 == 0 or v0 <= base:
            continue

        mx = x[c] - prev_x[c]; my = y[c] - prev_y[c]
        mlen = math.sqrt(mx * mx + my * my)

        # Wall normal from a ring of probes
        sx = 0.0; sy = 0.0
        r = max(hw, hl)
        for k in range(ring):
            th = 2.0 * math.pi * k / ring
            ox = math.cos(th); oy = math.sin(th)
            if blocked(layers, tile_map, t_next, t_prev, cell_size, cur, block_prev, x[c] + r * ox, y[c] + r * oy):
                sx += ox; sy += oy
        nl = math.sqrt(sx * sx + sy * sy)
        if nl > 1.0e-6:
            nx = -sx / nl; ny = -sy / nl
        elif mlen > 1.0e-9:
            nx = -mx / mlen; ny = -my / mlen                    # no probe hit: head-on
        else:
            nx = 0.0; ny = 0.0

        # Project the motion onto the wall
        into = mx * nx + my * ny
        if into < 0.0:
            tx = mx - into * nx; ty = my - into * ny
        else:
            tx = mx; ty = my

        chosen = -1
        px = prev_x[c]; py = prev_y[c]; pa = angle[c]
        for k in range(4):
            f = 1.0 if k < 2 else (0.5 if k == 2 else 0.0)
            na = angle[c] if k == 0 else prev_angle[c]
            qx = prev_x[c] + tx * f; qy = prev_y[c] + ty * f
            v = violations(layers, tile_map, t_next, t_prev, cell_size, cur, qx, qy, na, hw, hl, block_prev)
            if v == 0 or v <= base:
                chosen = k; px = qx; py = qy; pa = na
                break
        if chosen < 0:
            px = prev_x[c]; py = prev_y[c]; pa = prev_angle[c]
            vel_ang[c] = 0.0

        # Align the nose with the wall
        slide = math.sqrt((px - prev_x[c]) ** 2 + (py - prev_y[c]) ** 2)
        if align > 0.0 and slide > 1.0e-3:
            want = math.atan2(-(px - prev_x[c]), -(py - prev_y[c])) * 180.0 / math.pi
            if vel_lin[c] < 0.0: want += 180.0
            na = (pa + align * wrap180(want - pa)) % 360.0
            if violations(layers, tile_map, t_next, t_prev, cell_size, cur, px, py, na, hw, hl, block_prev) <= max(base, 0):
                pa = na

        x[c] = px; y[c] = py; angle[c] = pa
        disp_x[c] = prev_x[c] - px
        disp_y[c] = prev_y[c] - py
        keep = slide / mlen if mlen > 1.0e-9 else 0.0
        vel_lin[c] *= min(keep, 1.0) * damping
