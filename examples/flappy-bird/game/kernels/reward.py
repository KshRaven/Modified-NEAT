"""Reward shaping kernels."""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy import ndarray as Array

__all__ = ["reward_shape", "next_skill"]


@njit(parallel=True, fastmath=True, cache=True)
def reward_shape(
    shape_pos: Array, shape_neg: Array,
    err: Array, prev_err: Array, moved: Array, died: Array, hiatus: Array, low_usage: Array,
    active: Array, flap: Array, cy: Array,
    gap_top: float, gap_bot: float, dist_x: float, height: float, floor_y: float,
    gap_on: bool, max_hiatus: int, hiatus_ref: int, skill: float, unit: float,
    closer_w: float, away_w: float, time_pen: float, hiatus_w: float, loop_w: float, alive_w: float,
    death_dist_w: float, slow_cap: float,
    approach: float, margin: float, corridor_w: float, outside_w: float, edge_w: float, edge_frac: float, flap_w: float,
):
    """Dense per-frame shaping. `err` = |bird centre - gap centre| of the next pipe in pixels, `moved` = the target
    pipe changed this frame (no progress term), `dist_x` = distance to the next pipe's left edge.

    Every mode: flat time cost, a cost growing as the bird goes `max_hiatus` frames without passing a pipe, a cost for
    low flap usage, credit for moving towards the gap centre (a bit more is charged for moving away, so jittering never
    pays), a living bonus x (1 - skill) fading with the hiatus (living must never be worth less than dying), and a
    cost for dying that grows with the distance to the gap.

    Gap mode adds, while a pipe is within `approach` pixels:
      + corridor_weight x (0.5 + 0.5 skill) per frame between the gap edges (kept `margin` pixels clear)
      - outside_weight per unit outside that corridor
    and always: - edge_weight when the bird hugs the ceiling / floor, - flap_weight for a flap while already above the corridor.
    """
    edge = max(edge_frac * height, 1.0)
    for p in prange(err.shape[0]):
        shape_pos[p] = 0.0
        shape_neg[p] = 0.0
        if not active[p]:
            continue
        pos = 0.0
        neg = time_pen + hiatus_w * min(hiatus[p] / max(max_hiatus, 1), 1.0)
        if low_usage[p]:
            neg += loop_w

        if not moved[p] and not died[p]:
            delta = (prev_err[p] - err[p]) / unit
            if delta > 0.0:
                pos += closer_w * delta
            else:
                neg += away_w * -delta

        if died[p]:
            neg += death_dist_w * min(err[p] / unit, slow_cap)
        else:
            pos += alive_w * (1.0 - skill) * (1.0 - min(hiatus[p] / max(hiatus_ref, 1), 1.0))

        if gap_on and not died[p]:
            lo = gap_top + margin
            hi = gap_bot - margin
            if dist_x < approach:
                if cy[p] >= lo and cy[p] <= hi:
                    pos += corridor_w * (0.5 + 0.5 * skill)
                else:
                    out = (lo - cy[p]) if cy[p] < lo else (cy[p] - hi)
                    neg += outside_w * min(out / unit, slow_cap)
            if cy[p] < edge:
                neg += edge_w * (1.0 - cy[p] / edge)
            elif cy[p] > floor_y - edge:
                neg += edge_w * min((cy[p] - (floor_y - edge)) / edge, 1.0)
            if flap[p] and cy[p] < lo:
                neg += flap_w
        shape_pos[p] = pos
        shape_neg[p] = neg


def next_skill(skill: float, pipes: Array, deaths: Array, reward) -> float:
    """Curriculum skill after an episode, from the pipes passed per death (best life of every bird)."""
    mean_pipes, mean_deaths = float(np.mean(pipes)), float(np.mean(deaths))
    if mean_pipes + mean_deaths <= 0.0:
        return skill
    quality = mean_pipes / (mean_pipes + mean_deaths * reward.skill_pipes_per_death)
    x = min(max((quality - reward.skill_lo) / max(reward.skill_hi - reward.skill_lo, 1.0e-6), 0.0), 1.0)
    x = x * x * (3.0 - 2.0 * x)
    return float(min(max(skill + reward.skill_rate * (x - skill), 0.0), 1.0))
