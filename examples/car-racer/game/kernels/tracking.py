"""Track progress kernels: tile transitions, laps and per-step signals."""
from __future__ import annotations

import math

from numba import njit, prange
from numpy import ndarray as Array

from ..constants import T_LEFT, T_RIGHT
from ..util import grid, wrap180

__all__ = [
    "update_location", "signals", "progress",
]


@njit(parallel=True, fastmath=True, cache=True)
def update_location(
    tile_map: Array, t_next: Array, t_start: Array, cell_size: float,
    x: Array, y: Array, on_track: Array, tile_idx: Array,
    started: Array, finished: Array, crashed: Array, tile_crash: Array,
    laps_done: Array, checks_done: Array, tile_stag_steps: Array,
    lap_pending: Array, on_finish: Array, strict: bool, restrict_movement: int,
):
    """Tile transitions: checkpoints, laps and wrong-tile crashes.

    strict=False: a lap counts when the car enters the start tile.
    strict=True:  entering the start tile arms the lap, it counts when the car overlaps
                  the finish line (or, as a fallback, when it leaves the start tile).
    """
    total, indices = grid((x.shape[0],))
    for i in prange(total):
        c = indices[i]
        if strict and lap_pending[c] and on_finish[c]:
            lap_pending[c] = False
            laps_done[c]  += 1
            finished[c]    = True
        if on_track[c]:
            kx = int(abs(x[c]) / cell_size)
            ky = int(abs(y[c]) / cell_size)
            if kx < tile_map.shape[0] and ky < tile_map.shape[1]:
                new = tile_map[kx, ky]
                if new >= 0:
                    old = tile_idx[c]
                    if new == old or new == t_next[old]:
                        if new != old:
                            if t_start[old]:
                                started[c] = True
                                if lap_pending[c]:             # left the start tile without touching the line
                                    lap_pending[c] = False
                                    laps_done[c]  += 1
                                    finished[c]    = True
                            if t_start[new]:
                                if strict:
                                    lap_pending[c] = True
                                else:
                                    finished[c]   = True
                                    laps_done[c] += 1
                            checks_done[c]     += 1
                            tile_stag_steps[c]  = 0
                            tile_crash[c]       = False
                        else:
                            tile_stag_steps[c] += 1
                        tile_idx[c] = new
                    elif restrict_movement >= 3:
                        crashed[c] = True              # wrong tile, barriers open -> restart
                    else:
                        tile_stag_steps[c] += 1        # levels 0-2: no crash, the stagnation counter punishes it


@njit(parallel=True, fastmath=True, cache=True)
def signals(
    # outputs (rewritten every call)
    checked: Array, finished: Array, tile_stag: Array, on_road_eff: Array,
    moved_forward: Array, corner_curr: Array, corner_next: Array, straight_curr: Array,
    phase_align: Array, turning: Array, braking: Array, straight_aligned: Array,
    brake_check: Array, turn_align: Array,
    # persistent counters / flags (mutated in place)
    turn_steps: Array, brake_steps: Array, straight_steps: Array, used_brake: Array,
    corner_entry_speed: Array, crashed: Array, tile_crash: Array, crashes_done: Array, hiatus: Array,
    # current / previous state (read only)
    angle: Array, prev_angle: Array, tile_idx: Array, prev_tile: Array,
    checks_done: Array, prev_checks: Array, laps_done: Array, prev_laps: Array,
    tile_stag_steps: Array, prev_brake_steps: Array, vel_lin: Array, vel_norm: Array,
    on_road: Array, on_curb: Array, on_grass: Array, on_gravel: Array, on_track: Array,
    acceleration: Array, rotation: Array,
    t_type: Array, t_orient: Array, t_next: Array,
    max_hiatus: int, straight_tol: float, restrict_movement: int, race_on: bool, max_angle: float,
):
    """Reward signals, step counters and crash flags for every car."""
    total, indices = grid((angle.shape[0],))
    for i in prange(total):
        c  = indices[i]
        tc = tile_idx[c]
        pt = prev_tile[c]

        phase_curr  = wrap180(-(angle[c]      - t_orient[tc]))
        _phase_curr = wrap180(-(prev_angle[c] - t_orient[pt]))
        phase_align[c] = math.cos(phase_curr * math.pi / 180.0)

        typ      = t_type[tc]
        corner   = typ == T_LEFT or typ == T_RIGHT
        nxt_type = t_type[t_next[tc]]
        corner_curr[c]   = corner
        straight_curr[c] = not corner
        corner_next[c]   = nxt_type == T_LEFT or nxt_type == T_RIGHT

        offset = 0.0
        if   typ == T_LEFT:  offset = 45.0
        elif typ == T_RIGHT: offset = -45.0
        mf = vel_lin[c] > 0.0 and phase_curr < max_angle + offset and phase_curr > -max_angle + offset
        moved_forward[c] = mf or restrict_movement < 4      # the angle limit only counts from level 4

        oob = False
        if restrict_movement >= 3:                           # barriers open: leaving the track restarts
            oob = not on_track[c]
        if restrict_movement >= 5:                           # also completely off the road + curb
            oob = oob or not (on_road[c] or on_curb[c])

        chk = checks_done[c] > prev_checks[c]
        checked[c]   = chk
        finished[c]  = laps_done[c] > prev_laps[c]
        tile_stag[c] = restrict_movement >= 1 and tile_stag_steps[c] > max_hiatus

        road = on_road[c] and (not on_curb[c]) and (not on_grass[c]) and (not on_gravel[c]) and on_track[c]
        on_road_eff[c] = road
        delayed = chk or oob

        turn = (abs(phase_curr) < abs(_phase_curr)) and corner and road
        turning[c] = turn
        if turn:    turn_steps[c] += 1
        if delayed: turn_steps[c]  = 0

        brake = acceleration[c] < 0.0
        braking[c] = brake
        if brake: brake_steps[c] += 1
        else:     brake_steps[c]  = 0
        if brake: used_brake[c]   = True

        aligned = abs(phase_curr) <= straight_tol and (not corner) and road
        straight_aligned[c] = aligned
        if aligned: straight_steps[c] += 1
        if delayed: straight_steps[c]  = 0

        brake_check[c] = (chk and ((prev_brake_steps[c] > 0 and corner) or (not corner))) or (not chk)

        # Steering alignment (LEFT -1, RIGHT +1)
        req = 0.0
        if   typ == T_LEFT:  req = -1.0
        elif typ == T_RIGHT: req = 1.0
        ta = min(max(rotation[c] * req, -1.0), 1.0)
        turn_align[c] = ta if corner else 0.0

        # Corner entry speed
        if chk and corner: corner_entry_speed[c] = vel_norm[c]
        if not corner:     corner_entry_speed[c] = vel_norm[c]

        # Crash flags
        elim = oob
        if restrict_movement >= 4 and not race_on: elim = elim or (not mf)
        if elim:
            crashed[c]    = True
            tile_crash[c] = True
        if crashed[c]: crashes_done[c] += 1
        if mf or restrict_movement == 0: hiatus[c]  = 0
        else:                            hiatus[c] += 1


@njit(parallel=True, fastmath=True, cache=True)
def progress(
    progress: Array, in_tile: Array, x: Array, y: Array, tile_idx: Array, laps_done: Array,
    t_lap: Array, t_line: Array, n_tiles: int, size: float,
):
    """Distance driven along the track center line (race order key)."""
    total, indices = grid((x.shape[0],))
    for i in prange(total):
        c  = indices[i]
        t  = tile_idx[c]
        best_d, best = 1e18, 0.0
        for s in range(2):                      # entry -> center, center -> exit
            ax = t_line[t, s, 0];     ay = t_line[t, s, 1]
            bx = t_line[t, s + 1, 0]; by = t_line[t, s + 1, 1]
            vx = bx - ax; vy = by - ay
            L2 = vx * vx + vy * vy
            if L2 == 0.0: L2 = 1.0
            u  = min(max(((x[c] - ax) * vx + (y[c] - ay) * vy) / L2, 0.0), 1.0)
            qx = ax + vx * u - x[c]; qy = ay + vy * u - y[c]
            d2 = qx * qx + qy * qy
            if d2 < best_d:
                best_d = d2
                best   = s * (size / 2.0) + u * (size / 2.0)
        in_tile[c]  = best
        progress[c] = (laps_done[c] * n_tiles + t_lap[t]) * size + best
