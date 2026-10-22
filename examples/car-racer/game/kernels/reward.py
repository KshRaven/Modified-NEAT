"""Reward shaping kernels."""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from numpy import ndarray as Array

from ..util import grid

__all__ = [
    "corner_factor", "traffic_shape", "reward_shape",
    "corner_speeds", "next_skill",
]


@njit(fastmath=True, cache=True)
def corner_factor(m: float, f1: float, f2: float) -> float:
    """Safe speed fraction for corner severity m (0 straight, 1 = 90 degrees, 2 = U-turn)."""
    if m <= 1.0: return 1.0 - (1.0 - f1) * m * m
    return f1 - (f1 - f2) * (m - 1.0)


@njit(parallel=True, fastmath=True, cache=True)
def traffic_shape(
    extra_pos: Array, extra_neg: Array, edge_mult: Array,
    x: Array, y: Array, angle: Array, vel_lin: Array, active: Array,
    progress: Array, prev_progress: Array, on_road: Array, align: Array, corner_curr: Array,
    car_w: float, car_h: float, size: float, max_vel_lin: float,
    reach: float, gate: float, edge_guard: float, w_veer: float, w_close: float, w_lane: float, w_pass: float,
):
    """Race traffic shaping from true car positions.

    For the nearest car in the cone ahead (`reach` px, +-`gate` car widths):
      edge_mult   scales off-road costs so dodging onto grass is never cheaper than passing
      veer        cost for heading away from the track direction on a straight
      closing     cost for running up to a car in the same lane
    Also adds a per-frame bonus for riding level with a car in the next lane and a one-off
    bonus for a clean pass.
    """
    total, indices = grid((x.shape[0],))
    for i in prange(total):
        c = indices[i]
        extra_pos[c] = 0.0
        extra_neg[c] = 0.0
        edge_mult[c] = 1.0
        if not active[c]:
            continue
        fx = -math.sin(angle[c] * math.pi / 180.0)
        fy = -math.cos(angle[c] * math.pi / 180.0)

        best_gap = 1.0e18
        best_lat = 0.0
        best_v   = 0.0
        found    = False
        lane     = False
        passed   = False
        for j in range(x.shape[0]):
            if j == c or not active[j]:
                continue
            rx = x[j] - x[c]; ry = y[j] - y[c]
            ahead = rx * fx + ry * fy
            lat   = abs(-rx * fy + ry * fx)
            if ahead > 0.0 and ahead < reach and lat < gate * car_w and ahead < best_gap:
                best_gap = ahead; best_lat = lat; best_v = vel_lin[j]; found = True
            if (ahead > -0.6 * car_h and ahead < 1.2 * car_h and lat >= 0.9 * car_w
                    and lat < gate * car_w and vel_lin[c] >= 0.95 * vel_lin[j]):
                lane = True
            if (rx * rx + ry * ry < (2.5 * car_h) ** 2
                    and prev_progress[c] <= prev_progress[j] and progress[c] > progress[j]
                    and abs(progress[c] - prev_progress[c]) < 0.5 * size
                    and abs(progress[j] - prev_progress[j]) < 0.5 * size):
                passed = True

        if on_road[c]:
            if lane:   extra_pos[c] += w_lane
            if passed: extra_pos[c] += w_pass
        if found:
            near = 1.0 - best_gap / reach                         # 0 far, 1 touching
            edge_mult[c] = 1.0 + (edge_guard - 1.0) * near
            if not corner_curr[c]:
                extra_neg[c] += w_veer * (1.0 - align[c]) * near
            closing = max(vel_lin[c] - best_v, 0.0) / max_vel_lin
            same    = max(1.0 - best_lat / car_w, 0.0)
            tight   = max(1.0 - best_gap / (0.5 * reach), 0.0)
            extra_neg[c] += w_close * closing * same * tight


@njit(parallel=True, fastmath=True, cache=True)
def reward_shape(
    shape_pos: Array, shape_neg: Array, target_norm: Array,
    extra_pos: Array, extra_neg: Array, edge_mult: Array,
    progress: Array, prev_progress: Array, in_tile: Array, vel_lin: Array, vel_norm: Array,
    rotation: Array, prev_rotation: Array,
    tile_idx: Array, t_turn: Array, t_next: Array, corner_curr: Array,
    on_road: Array, on_curb: Array, on_grass: Array, on_gravel: Array, bumped: Array, prev_off: Array,
    size: float, max_vel_lin: float, a_dec: float, f1: float, f2: float, f1c: float,
    lookahead: int, window: int, skill: float,
    w_prog: float, w_time: float, w_surv: float, w_over: float, over_tol: float, w_under: float,
    w_steer: float, w_jit: float,
    w_curb: float, curb_straight: float, curb_disc: float, w_grass: float, w_gravel: float, w_bump: float,
    off_progress: float, w_entry: float, off_corner: float, exit_gain: float, early_surface: float,
):
    """Dense reward shaping with a survive-first curriculum.

    Corners within `lookahead` tiles are graded and turned into a braking-distance speed
    profile, for road and for curb. The profile gives `target_norm`.

    Positive: progress along the center line (credit capped at the target speed as `skill` rises),
              survival bonus, `extra_pos`.
    Negative: time, overspeed, underspeed, steering on straights, steering jitter, curb / grass /
              gravel (scaled up while `skill` is low), bumps, `extra_neg`.
    """
    total, indices = grid((progress.shape[0],))
    early = 1.0 + early_surface * (1.0 - skill)
    for i in prange(total):
        c  = indices[i]
        t0 = tile_idx[c]

        vt = max_vel_lin
        vtc = max_vel_lin
        d_corner = 1.0e9
        for j in range(lookahead + 1):
            tj = t0
            for _ in range(j):
                tj = t_next[tj]
            if t_turn[tj] == 0.0:
                continue
            cum = 0.0; cmin = 0.0; cmax = 0.0; tw = tj
            for _ in range(window):
                cum += t_turn[tw]
                cmin = min(cmin, cum); cmax = max(cmax, cum)
                tw = t_next[tw]
            m = min(max(abs(cum), 0.5 * (cmax - cmin)), 2.0)
            d  = 0.0 if j == 0 else max(j * size - in_tile[c], 0.0)
            vc  = corner_factor(m, f1, f2) * max_vel_lin
            vcc = max(corner_factor(m, f1c, f2 * f1c / max(f1, 1.0e-6)), corner_factor(m, f1, f2)) * max_vel_lin
            if j == 0:
                # past the apex the car may accelerate towards the exit
                ex = min(max((in_tile[c] / size - 0.5) * 2.0, 0.0), 1.0) * exit_gain
                vc  += ex * (max_vel_lin - vc)
                vcc += ex * (max_vel_lin - vcc)
            vt  = min(vt,  math.sqrt(vc  * vc  + 2.0 * a_dec * d))
            vtc = min(vtc, math.sqrt(vcc * vcc + 2.0 * a_dec * d))
            d_corner = min(d_corner, d)
        if on_curb[c]: vt = max(vt, vtc)
        tn = vt / max_vel_lin
        target_norm[c] = tn

        # 0 = open road, 1 = at the corner
        near = 1.0 - min(max((d_corner - 0.3 * size) / (0.7 * size), 0.0), 1.0)
        open_road = 1.0 - near

        pos = 0.0
        neg = w_time
        dp = progress[c] - prev_progress[c]
        if abs(dp) > 0.5 * size: dp = 0.0                 # lap wrap or respawn
        if dp > 0.0:
            cap = min(1.0, vt / max(vel_lin[c], 1.0e-6))
            credit = w_prog * dp / size * (1.0 - skill * (1.0 - cap))
            if on_grass[c] or on_gravel[c]: credit *= off_progress
            pos += credit
        else:
            neg += w_prog * (-dp) / size

        vn = vel_norm[c]
        excess = max(vn - tn * (1.0 + over_tol), 0.0)
        neg += skill * w_over * excess * min(excess * 4.0, 1.0)
        if on_road[c] or on_curb[c]:
            neg += skill * w_under * max(0.85 * tn - vn, 0.0)

        neg += w_steer * abs(rotation[c]) * open_road + w_jit * abs(rotation[c] - prev_rotation[c])

        if (on_grass[c] or on_gravel[c]) and not prev_off[c]: neg += w_entry * early
        em = edge_mult[c]
        loose = early * (1.0 + off_corner * near)
        if on_grass[c]:    neg += w_grass * em * loose
        elif on_gravel[c]: neg += w_gravel * em * loose
        elif on_curb[c]:   neg += w_curb * em * (open_road * curb_straight + near * (1.0 - curb_disc))
        pos += extra_pos[c]
        neg += extra_neg[c]
        if on_road[c]:     pos += w_surv
        if bumped[c]:      neg += w_bump

        shape_pos[c] = pos
        shape_neg[c] = neg


def corner_speeds(
    reward, car, size: float, max_vel_lin: float, max_vel_ang: float, acc_lin: float, coeff_brake: float,
) -> tuple[float, float, float, float]:
    """Physics-derived corner speeds for the reward.

    Returns `(f1, f2, f1_curb, a_dec)`: safe speed fraction of a 90 degree corner and of a U-turn on the road,
    the 90 degree fraction on the curb, and the braking deceleration used for the speed profile.
    """
    radius = reward.corner_line_radius * size / 2.0
    if reward.corner_speed_ratio is not None:
        f1 = float(reward.corner_speed_ratio)
    else:
        omega  = max_vel_ang * car.handling_road * math.pi / 180.0
        v_grip = math.sqrt(car.lateral_grip * (1.0 - car.slip_road) * radius)
        f1 = reward.corner_safety * min(omega * radius, v_grip) / max_vel_lin
    f1_road = min(max(f1, 0.25), 1.0)
    f2 = f1_road * (reward.uturn_ratio if f1_road < 1.0 else 1.0)
    a_dec = acc_lin * (1.0 + 0.5 * coeff_brake) * reward.braking_margin

    omega_c  = max_vel_ang * car.handling_curb * math.pi / 180.0
    v_grip_c = math.sqrt(car.lateral_grip * (1.0 - car.slip_curb) * radius)
    f1_curb = reward.corner_safety * min(omega_c * radius, v_grip_c) / max_vel_lin
    if reward.corner_speed_ratio is not None:
        f1_curb = f1 * car.handling_curb / max(car.handling_road, 1.0e-6)
    return f1_road, f2, min(max(f1_curb, f1_road), 1.0), a_dec


def next_skill(skill: float, checks: Array, deaths: Array, reward) -> float:
    """Curriculum skill after an episode, from the clean tiles driven per crash."""
    mean_checks, mean_deaths = float(np.mean(checks)), float(np.mean(deaths))
    if mean_checks + mean_deaths <= 0.0:
        return skill
    quality = mean_checks / (mean_checks + mean_deaths * reward.skill_tiles_per_crash)
    x = min(max((quality - reward.skill_lo) / max(reward.skill_hi - reward.skill_lo, 1.0e-6), 0.0), 1.0)
    x = x * x * (3.0 - 2.0 * x)
    return skill + reward.skill_rate * (x - skill)
