"""Reward shaping kernels."""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy import ndarray as Array

from .road import K_DEAD, K_ROAD, K_SHORT, K_SLOW, K_UNSAFE

__all__ = ["reward_shape", "next_skill"]


@njit(parallel=True, fastmath=True, cache=True)
def reward_shape(
    shape_pos: Array, shape_neg: Array,
    dist: Array, prev_dist: Array, ate: Array, moved: Array, died: Array, hiatus: Array, low_usage: Array,
    active: Array, length: Array, act: Array, last_turn: Array, last_in: Array,
    kind: Array, gain: Array, trap: Array, synced: Array,
    road_on: bool, max_hiatus: int, cells: int, skill: float, crash: float,
    closer_weight: float, away_weight: float, time_penalty: float, hiatus_weight: float, loop_weight: float,
    w_road: float, w_prog: float, short_mult: float, slow_cap: float,
    w_unsafe: float, w_desync: float, w_trap: float, w_nascar: float, w_fear: float, fear_frac: float,
    w_alive: float, hiatus_ref: int, refund: float,
):
    """Dense per-frame shaping, in cells. `kind`, `gain`, `trap`, `synced` describe the move that was taken
    (graded by `road.scan` before the move); `crash` = 1 + early_crash_mult * (1 - skill).

    Every mode: flat time cost, a cost growing as the snake goes `max_hiatus` frames without food, a cost for
    looping, and a cost for moving into a pocket it cannot leave (flood fill).

    Without a road, or while the body is out of road order: cells gained towards the food are credited, cells
    lost are charged a bit more so that shuffling never pays. The frame of a meal carries no progress term.

    With a road (survive first, then fast):
      + road_weight x (1 - skill) for every safe move (staying in road order is the first lesson)
      + progress_weight x road cells removed from the way to the food. A plain step removes 1; the extra cells
        of a shortcut are credited at short_mult x skill, so shortcuts only pay once the snake is reliable.
      - progress_weight x cells added (overshooting the food costs a lap, capped at slow_cap)
      + alive_weight x (1 - skill) for every living frame, fading to zero as the hiatus grows (so stalling
        pays nothing after one lap of the board). Living must never cost more than dying: a snake with no
        skill yet is out of road order within a few moves, and per-frame costs larger than the death penalty
        make "ram the wall" the best policy. The bonus is the survive-first gradient.
      - unsafe_weight x crash once for the move that breaks road order, a mild desync_weight per frame while
        the body stays out of order (the generic food shaping keeps the snake hunting meanwhile), and
        `refund` of the exit cost back when the body is in road order again. Road order is a potential:
        leaving and re-entering nets (almost) nothing, so a short snake that drifts in and out of order by
        chance is not charged again and again.
      - nascar_weight for reversing the turn direction, fear_weight for swinging off the road into the food
        once the board is `fear_frac` full (the "mods" of the dhcr solver)
    """
    for p in prange(dist.shape[0]):
        shape_pos[p] = 0.0
        shape_neg[p] = 0.0
        if not active[p]:
            continue
        pos = 0.0
        neg = time_penalty + hiatus_weight * min(hiatus[p] / max(max_hiatus, 1), 1.0)
        if low_usage[p]:
            neg += loop_weight
        in_order = synced[p]

        if (not road_on) or (not in_order):
            if not moved[p] and not died[p]:
                delta = prev_dist[p] - dist[p]
                if delta > 0.0:
                    pos += closer_weight * delta
                else:
                    neg += away_weight * -delta

        if road_on:
            k = kind[p]
            if not died[p]:
                pos += w_alive * (1.0 - skill) * (1.0 - min(hiatus[p] / max(hiatus_ref, 1), 1.0))
            if not in_order:
                neg += w_desync
            else:
                if not last_in[p] and not died[p]:          # back in road order: the exit cost is (mostly) repaid
                    pos += w_unsafe * crash * refund
                if k == K_DEAD and not died[p]:
                    k = K_UNSAFE                    # moved into the tail cell: legal, but it is not on the road
                if k == K_ROAD or k == K_SHORT or k == K_SLOW:
                    if not died[p]:
                        pos += w_road * (1.0 - skill)
                        if not moved[p]:
                            g = max(gain[p], -slow_cap)
                            if g > 0.0:
                                pos += w_prog * (min(g, 1.0) + skill * short_mult * max(g - 1.0, 0.0))
                            else:
                                neg += w_prog * -g
                elif k == K_UNSAFE and not died[p]:
                    neg += w_unsafe * crash
            last_in[p] = True if died[p] else in_order      # a respawn starts in road order
            if ate[p] and kind[p] != K_ROAD and length[p] >= fear_frac * cells:
                neg += w_fear
            a = act[p]
            code = 1 if a == 0 else (2 if a == 2 else 0)
            if code != 0:
                if last_turn[p] != 0 and last_turn[p] != code:
                    neg += w_nascar
                last_turn[p] = code

        if trap[p] and not died[p]:
            neg += w_trap * crash
        shape_pos[p] = pos
        shape_neg[p] = neg


def next_skill(skill: float, food: Array, deaths: Array, reward) -> float:
    """Curriculum skill after an episode, from the meals eaten per death (best life of every snake)."""
    mean_food, mean_deaths = float(np.mean(food)), float(np.mean(deaths))
    if mean_food + mean_deaths <= 0.0:
        return skill
    quality = mean_food / (mean_food + mean_deaths * reward.skill_food_per_death)
    x = min(max((quality - reward.skill_lo) / max(reward.skill_hi - reward.skill_lo, 1.0e-6), 0.0), 1.0)
    x = x * x * (3.0 - 2.0 * x)
    return skill + reward.skill_rate * (x - skill)