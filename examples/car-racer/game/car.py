import math
import pygame
import pygame.surfarray
import numpy as np

from numba import njit, prange
from numpy import ndarray as Array
from pygame import Surface, Mask
from pygame.font import Font
from typing import Optional, Any

from .config import CarConfig, RaceConfig, RewardConfig, overwrite_config
from .functional import Type, Position
from .constants import NP_INT, NP_FLOAT, IMAGE_DIR, COLOR_LIDAR, COLOR_CURB_PRI, COLOR_CURB_SEC
from .util import scale_image, draw_arrow, grid, draw_panel
from .grid import Grid
from .tile import Tile
from .player import Players
from .sprite import make_car, make_halo, rainbow, HALO_COLORS, WHEEL_PAD, export_car_images
from .race import TrackGeometry, StartingGrid, slipstream, collide, trail_push, trail_points



# Surface integer codes (shared by Cars.surface and lidar_surface_hits)
SURFACE_ROAD   = np.int8(0)
SURFACE_CURB   = np.int8(1)
SURFACE_GRASS  = np.int8(2)
SURFACE_GRAVEL = np.int8(3)
SURFACE_VOID   = np.int8(4)  # out-of-bounds / off-track
SURFACE_CAR    = np.int8(5)  # LiDAR hit on another car (race.lidar_detect_cars)


# ---------------------------------------------------------------------------
# Surface layer bit-flags (packed into a single uint8 (W, H) map)
# ---------------------------------------------------------------------------
L_ROAD, L_CURB, L_GRASS, L_GRAVEL, L_TRACK, L_BOUND, L_FINISH = 1, 2, 4, 8, 16, 32, 64

# Slipstream rendering constants
TRAIL_RESET_CARS = 4.0    # a frame-to-frame jump above this many car lengths is a respawn -> clear trail
DRAFT_TREND_EPS  = 1e-4   # dead band: level changes smaller than this keep the previous gain/decay colour
DRAFT_VISIBLE    = 0.05   # below this draft level no streak is drawn (avoids a flickering stub)
DRAFT_TOWER_MARK = 0.15   # draft level at which the leaderboard tower shows the '*' marker
STREAK_REAR_CARS = 0.5    # streak starts at the rear bumper = half a car length behind the centre

# Tile type codes (functional.Type values)
T_START, T_STRAIGHT, T_LEFT, T_RIGHT = 0, 1, 2, 3


def pack_layers(road: Array, curb: Array, grass: Array, gravel: Array,
                track: Array, boundary: Array, finish: Array) -> Array:
    """Pack seven (W, H) bool arrays into one C-contiguous uint8 (W, H) map."""
    layers = (
        road.astype(np.uint8) * L_ROAD | curb.astype(np.uint8) * L_CURB
        | grass.astype(np.uint8) * L_GRASS | gravel.astype(np.uint8) * L_GRAVEL
        | track.astype(np.uint8) * L_TRACK | boundary.astype(np.uint8) * L_BOUND
        | finish.astype(np.uint8) * L_FINISH
    )
    return np.ascontiguousarray(layers, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Numba-accelerated helpers
# (same structure as backtest.trader / backtest.agent: module-level
#  @njit(parallel=True, fastmath=True, cache=True) kernels, a flat ``prange``
#  over independent cars, state mutated in place on pre-allocated buffers)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _sign(v: float) -> int:
    if v > 0.0: return 1
    if v < 0.0: return -1
    return 0


@njit(cache=True)
def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


@njit(parallel=True, fastmath=True, cache=True)
def _update_surface(layers: Array, x: Array, y: Array, surface: Array):
    """Centre-point surface lookup. Priority: road > curb > grass > gravel > void."""
    gw, gh = layers.shape
    total, indices = grid((x.shape[0],))
    for i in prange(total):
        c  = indices[i]
        xi = min(max(int(x[c]), 0), gw - 1)
        yi = min(max(int(y[c]), 0), gh - 1)
        v  = layers[xi, yi]
        if   v & L_ROAD:   surface[c] = 0
        elif v & L_CURB:   surface[c] = 1
        elif v & L_GRASS:  surface[c] = 2
        elif v & L_GRAVEL: surface[c] = 3
        else:              surface[c] = 4


@njit(parallel=True, fastmath=True, cache=True)
def _update_lidar(
    layers: Array, x: Array, y: Array, angle: Array, surface: Array,
    beam_offsets: Array, n_steps: int, step: float, max_dist: float,
    distances: Array, surface_hits: Array,
):
    """
    Ray-march every beam of every car (one prange item per car x beam).

    A beam stops at the first sample that leaves the home surface (and all inner
    layers), leaves the track, or lies on a tile boundary.  Results (in place):
        distances     (N, B) float64  normalised [0, 1]
        surface_hits  (N, B) int8     surface code at hit (4 = void / no hit)
    """
    gw, gh = layers.shape
    total, indices = grid((x.shape[0], beam_offsets.shape[0]))

    for i in prange(total):
        c, b = indices[i]

        a  = (angle[c] + beam_offsets[b]) * math.pi / 180.0
        dx = -math.sin(a)
        dy = -math.cos(a)
        cx = x[c]
        cy = y[c]

        home = int(surface[c])
        if home == 4: home = 3   # off-track car measures to the track boundary

        if   home == 0: inside_mask = L_ROAD
        elif home == 1: inside_mask = L_ROAD | L_CURB
        elif home == 2: inside_mask = L_ROAD | L_CURB | L_GRASS
        else:           inside_mask = L_TRACK

        first_hit = -1
        for d in range(n_steps):          # serial inner loop -> ``break`` is safe here
            t  = step * (d + 1)
            sx = int(cx + dx * t)
            sy = int(cy + dy * t)
            if sx < 0 or sx >= gw or sy < 0 or sy >= gh:
                first_hit = d
                break
            v = layers[sx, sy]
            if (not (v & inside_mask)) or (not (v & L_TRACK)) or (v & L_BOUND):
                first_hit = d
                break

        if first_hit < 0:
            distances[c, b]    = 1.0
            surface_hits[c, b] = 4
        else:
            idx = first_hit
            if home == 3: idx = max(first_hit - 1, 0)   # stay just inside the boundary
            dist = (idx + 1) * step
            distances[c, b] = dist / max_dist
            hx = int(cx + dx * dist)
            hy = int(cy + dy * dist)
            hit = 4
            if hx >= 0 and hx < gw and hy >= 0 and hy < gh:
                v = layers[hx, hy]
                if   v & L_GRAVEL: hit = 3      # outermost layer wins
                elif v & L_GRASS:  hit = 2
                elif v & L_CURB:   hit = 1
                elif v & L_ROAD:   hit = 0
            surface_hits[c, b] = hit


def _flame_color(base, hot, heat: float):
    """heat 1 -> `hot` (core), 0.5 -> `base`, 0 -> a dark ember of `base`."""
    if heat >= 0.5:
        u, a, b = (heat - 0.5) * 2.0, base, hot
    else:
        u, a, b = heat * 2.0, tuple(0.25 * v for v in base), base
    return (int(a[0] + (b[0] - a[0]) * u), int(a[1] + (b[1] - a[1]) * u), int(a[2] + (b[2] - a[2]) * u))


@njit(parallel=True, fastmath=True, cache=True)
def _lidar_cars(
    x: Array, y: Array, angle: Array, active: Array, beam_offsets: Array,
    max_dist: float, car_w: float, car_h: float, radius_scale: float,
    distances: Array, surface_hits: Array,
):
    """
    Shorten LiDAR beams where they hit another active car.  Each car is the same two
    circles used by the collision model; a beam is intersected analytically with them and
    keeps the nearer of (wall hit, car hit).  Car hits are tagged SURFACE_CAR (5).
    """
    total, indices = grid((x.shape[0], beam_offsets.shape[0]))
    r   = car_w * radius_scale
    off = max(car_h / 2.0 - r, 0.0)
    for i in prange(total):
        c, b = indices[i]
        if not active[c]:
            continue
        a  = (angle[c] + beam_offsets[b]) * math.pi / 180.0
        dx = -math.sin(a)
        dy = -math.cos(a)
        best = distances[c, b] * max_dist
        found = False
        for j in range(x.shape[0]):
            if j == c or not active[j]:
                continue
            rj = angle[j] * math.pi / 180.0
            fx = -math.sin(rj); fy = -math.cos(rj)
            for s in range(2):
                sg = 1.0 - 2.0 * s
                ox = x[j] + sg * fx * off - x[c]
                oy = y[j] + sg * fy * off - y[c]
                proj = ox * dx + oy * dy               # distance along the beam to closest approach
                if proj <= 0.0:
                    continue
                perp2 = ox * ox + oy * oy - proj * proj
                if perp2 >= r * r:
                    continue
                t = proj - math.sqrt(r * r - perp2)    # entry point of the circle
                if t < 0.0: t = 0.0
                if t < best:
                    best = t
                    found = True
        if found:
            distances[c, b]    = best / max_dist
            surface_hits[c, b] = 5


@njit(parallel=True, fastmath=True, cache=True)
def _update_regions(
    layers: Array, lut_off: Array, lut_n: Array, lut_id: Array, x: Array, y: Array, angle: Array,
    on_road: Array, on_curb: Array, on_grass: Array, on_gravel: Array,
    on_track: Array, on_finish: Array,
):
    """
    Which surfaces does the car BODY touch?  Replaces the per-car
    ``pygame.transform.rotate`` + ``mask.overlap`` calls.  ``lut_off[q, k]`` holds the
    set pixels of pygame's own rotated car mask (heading in 0.01 degree steps, identical
    masks de-duplicated through ``lut_id``) as offsets from the integer car centre, so the
    result matches the mask overlap and only needs lookups in the packed layer map.
    """
    gw, gh = layers.shape
    steps  = lut_id.shape[0]
    total, indices = grid((x.shape[0],))
    for i in prange(total):
        c  = indices[i]
        q  = lut_id[int(round(angle[c] * (steps / 360.0))) % steps]
        cx = int(math.floor(x[c] + 0.5))   # pygame rounds a float rect centre
        cy = int(math.floor(y[c] + 0.5))
        bits = 0
        for k in range(lut_n[q]):
            wx = cx + lut_off[q, k, 0]
            wy = cy + lut_off[q, k, 1]
            if wx >= 0 and wx < gw and wy >= 0 and wy < gh:
                bits |= layers[wx, wy]
        on_road[c]   = (bits & L_ROAD)   != 0
        on_curb[c]   = (bits & L_CURB)   != 0
        on_grass[c]  = (bits & L_GRASS)  != 0
        on_gravel[c] = (bits & L_GRAVEL) != 0
        on_track[c]  = (bits & L_TRACK)  != 0
        on_finish[c] = (bits & L_FINISH) != 0


@njit(parallel=True, fastmath=True, cache=True)
def _update_location(
    tile_map: Array, t_next: Array, t_start: Array, cell_size: float,
    x: Array, y: Array, on_track: Array, tile_idx: Array,
    started: Array, finished: Array, crashed: Array, tile_crash: Array,
    laps_done: Array, checks_done: Array, tile_stag_steps: Array,
    lap_pending: Array, on_finish: Array, strict: bool,
):
    """Tile transition tracking (checkpoints / laps / wrong-tile crashes).

    strict=False: a lap is counted when the car enters the start tile.
    strict=True : entering the start tile only arms the lap; it is counted when the car overlaps the
                  finish line mask (or, as a fallback, when it leaves the start tile)."""
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
                    else:
                        crashed[c] = True


@njit(parallel=True, fastmath=True, cache=True)
def _steer(
    rotation: Array, vel_ang: Array, on_gravel: Array, on_grass: Array, on_curb: Array,
    friction: Array, handling: Array, slip: Array,
    coef: Array,   # (3, 4): rows friction/handling/slip, cols road/curb/grass/gravel
    max_vel_ang: float, acc_ang: float, coeff_checker: float,
):
    """
    Per-car surface coefficients + steering.

    `handling` is steering authority: it scales both how fast the yaw rate builds and its cap, so a
    value above 1 makes the body rotate more eagerly.  How far the *path* follows the nose is not
    decided here (see slip in `_drive`).
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
        if _sign(rot) != _sign(va):
            rot *= 1.0 + abs(va / max_vel_ang) * coeff_checker
        vel_ang[c] = min(max(va + acc_ang * h * rot, -cap), cap)


@njit(parallel=True, fastmath=True, cache=True)
def _drive(
    acceleration: Array, vel_lin: Array, vel_ang: Array, angle: Array, x: Array, y: Array,
    disp_x: Array, disp_y: Array, vel_norm: Array,
    friction: Array, slip: Array,
    on_gravel: Array, draft: Array, gravel_noise: Array,
    max_vel_lin: float, acc_lin: float, coeff_brake: float, coeff_reverse: float,
    reverse_enabled: bool, drafting: bool, ss_bonus: float, ss_push: float,
    lateral_grip: float, grip_relief: float, slide_drag: float,
):
    """
    Longitudinal speed, heading and the velocity vector of every car.

    * friction (traction) scales the engine / brake force and the surface's top speed. Speed is never
      multiplied down: over the top speed the excess is shed at engine-brake rate, coasting keeps momentum.
    * The path direction chases the nose at a rate limited by lateral grip / speed (lag grows with the
      surface's slip and with speed), so sliding only redirects speed and never adds any; sliding
      sideways scrubs a little speed (slide_drag).
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
        if _sign(acc) != _sign(v):
            acc *= 1.0 + abs(v / max_vel_lin) * coeff_brake
        dv = acc_lin * acc * f + push
        lo = -max_vel_lin if reverse_enabled else 0.0
        if v > top:
            nv = min(v + dv, v - acc_lin)          # engine braking sheds the excess ...
            if dv >= -acc_lin: nv = max(nv, top)   # ... down to the surface top speed, not below it
        else:
            nv = min(v + dv, top)
        v = max(nv, lo)
        if reverse_enabled and v < 0.0: v *= coeff_reverse
        vel_lin[c] = v

        vn = min(max(v / max_vel_lin, -1.0), 1.0)
        vel_norm[c] = vn
        avn = abs(vn)

        a = (angle[c] + vel_ang[c]) % 360.0
        if on_gravel[c]: a += gravel_noise[c] * avn
        angle[c] = a
        rad = a * math.pi / 180.0

        # Path direction vs nose direction. The nose rotates freely (handling), but the PATH can only
        # turn towards it at the rate the tyres allow:  turn rate = lateral grip / speed.  That is what
        # makes full speed impossible in a tight corner (the car slides wide) and makes grass / gravel
        # (high slip = little grip) barely follow a rotated nose.  Near standstill grip is restored.
        s       = slip[c]
        nose    = rad if v >= 0.0 else rad + math.pi
        speed   = abs(v)
        pdx = disp_x[c]; pdy = disp_y[c]
        pspeed  = math.sqrt(pdx * pdx + pdy * pdy)
        phi     = math.atan2(pdx, pdy) if pspeed > 1.0e-3 else nose
        delta   = (nose - phi + math.pi) % (2.0 * math.pi) - math.pi
        w_grip  = lateral_grip * (1.0 - s) / max(max(pspeed, speed), 0.5)       # rad / frame
        step    = min(abs(delta), w_grip * (1.0 - s * avn))                      # slip also lags the path
        relief  = max(1.0 - speed / max(grip_relief * max_vel_lin, 1.0e-6), 0.0)
        step   += relief * (abs(delta) - step)
        phi    += step if delta >= 0.0 else -step
        misalign = abs((nose - phi + math.pi) % (2.0 * math.pi) - math.pi)
        m  = speed * (1.0 - slide_drag * (1.0 - math.cos(misalign)) * 0.5)      # sliding sideways scrubs speed
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


@njit(parallel=True, fastmath=True, cache=True)
def _signals(
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
    max_hiatus: int, straight_tol: float, restrict_movement: int, race_on: bool,
):
    """Derived reward-shaping signals, step counters and crash flags per car."""
    total, indices = grid((angle.shape[0],))
    for i in prange(total):
        c  = indices[i]
        tc = tile_idx[c]
        pt = prev_tile[c]

        phase_curr  = _wrap180(-(angle[c]      - t_orient[tc]))
        _phase_curr = _wrap180(-(prev_angle[c] - t_orient[pt]))
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
        mf = vel_lin[c] > 0.0 and phase_curr < 135.0 + offset and phase_curr > -135.0 + offset
        moved_forward[c] = mf

        oob = not on_track[c]
        if restrict_movement >= 2 and not race_on:
            oob = oob or on_grass[c] or on_gravel[c]

        chk = checks_done[c] > prev_checks[c]
        checked[c]   = chk
        finished[c]  = laps_done[c] > prev_laps[c]
        tile_stag[c] = tile_stag_steps[c] > max_hiatus

        road = on_road[c] and (not on_curb[c]) and (not on_grass[c]) and (not on_gravel[c]) and on_track[c]
        on_road_eff[c] = road
        delayed = chk or oob

        # Steps
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

        # Turn-direction alignment (LEFT -> -1, RIGHT -> +1, else 0)
        req = 0.0
        if   typ == T_LEFT:  req = -1.0
        elif typ == T_RIGHT: req = 1.0
        ta = min(max(rotation[c] * req, -1.0), 1.0)
        turn_align[c] = ta if corner else 0.0

        # Corner-entry speed snapshot
        if chk and corner: corner_entry_speed[c] = vel_norm[c]
        if not corner:     corner_entry_speed[c] = vel_norm[c]

        # Crash / elimination flags
        elim = oob
        if restrict_movement > 0 and not race_on: elim = elim or (not mf)
        if elim:
            crashed[c]    = True
            tile_crash[c] = True
        if crashed[c]: crashes_done[c] += 1
        if mf: hiatus[c]  = 0
        else:  hiatus[c] += 1


@njit(parallel=True, fastmath=True, cache=True)
def _traffic_shape(
    extra_pos: Array, extra_neg: Array, edge_mult: Array,
    x: Array, y: Array, angle: Array, vel_lin: Array, active: Array,
    progress: Array, prev_progress: Array, on_road: Array, align: Array, corner_curr: Array,
    car_w: float, car_h: float, size: float, max_vel_lin: float,
    reach: float, gate: float, edge_guard: float, w_veer: float, w_close: float, w_lane: float, w_pass: float,
):
    """
    Race-mode traffic shaping from true car positions (independent of LiDAR beam count / density).

    For the nearest car in the cone ahead (`reach` px, +-`gate` car widths):
      * edge_mult   : off-road costs grow as it gets closer, so dodging a car by leaving the road is
                      never cheaper than passing it on the road
      * veer        : cost for heading away from the track direction on a straight while blocked
      * closing     : cost for running up to a car in the same lane
    plus a per-frame bonus for being level with a car in the next lane (on the road, not slower) and a
    one-off bonus for completing a clean pass.
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
            near = 1.0 - best_gap / reach                         # 0 far ... 1 touching
            edge_mult[c] = 1.0 + (edge_guard - 1.0) * near
            if not corner_curr[c]:
                extra_neg[c] += w_veer * (1.0 - align[c]) * near
            closing = max(vel_lin[c] - best_v, 0.0) / max_vel_lin
            same    = max(1.0 - best_lat / car_w, 0.0)
            tight   = max(1.0 - best_gap / (0.5 * reach), 0.0)
            extra_neg[c] += w_close * closing * same * tight


@njit(parallel=True, fastmath=True, cache=True)
def _reward_shape(
    shape_pos: Array, shape_neg: Array, target_norm: Array,
    extra_pos: Array, extra_neg: Array, edge_mult: Array,
    progress: Array, prev_progress: Array, in_tile: Array, vel_lin: Array, vel_norm: Array,
    rotation: Array, prev_rotation: Array,
    tile_idx: Array, t_turn: Array, t_next: Array, corner_curr: Array,
    on_road: Array, on_curb: Array, on_grass: Array, on_gravel: Array, bumped: Array, prev_off: Array,
    size: float, max_vel_lin: float, a_dec: float, f1: float, f2: float,
    lookahead: int, window: int,
    w_prog: float, w_time: float, w_surv: float, w_over: float, w_steer: float, w_jit: float,
    w_curb: float, curb_disc: float, w_grass: float, w_gravel: float, w_bump: float,
    off_progress: float, w_entry: float,
):
    """
    Dense time-trial shaping.  Looks `lookahead` tiles down the track, grades every corner it
    finds (net heading change over `window` tiles: a 90 degree bend = 1, a U-turn = 2, an S of two
    opposite bends = 0.5 because it can be driven as a diagonal), turns that into a safe speed and
    a braking-distance speed profile, and then

      +  progress along the centre line, credited only up to the target speed
      -  speed above the target, steering on open straights, steering jitter, surface contact,
         car-to-car contact and a small constant time cost.
    """
    total, indices = grid((progress.shape[0],))
    for i in prange(total):
        c  = indices[i]
        t0 = tile_idx[c]

        vt = max_vel_lin
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
            if m <= 1.0: f = 1.0 - (1.0 - f1) * m * m
            else:        f = f1 - (f1 - f2) * (m - 1.0)
            d = 0.0 if j == 0 else max(j * size - in_tile[c], 0.0)
            vc = f * max_vel_lin
            vt = min(vt, math.sqrt(vc * vc + 2.0 * a_dec * d))   # can still brake down to vc by the corner
            d_corner = min(d_corner, d)
        tn = vt / max_vel_lin
        target_norm[c] = tn

        pos = 0.0
        neg = w_time
        dp = progress[c] - prev_progress[c]
        if abs(dp) > 0.5 * size: dp = 0.0                 # lap wrap / respawn jump, not driving
        if dp > 0.0:
            credit = w_prog * dp * min(1.0, vt / max(vel_lin[c], 1.0e-6)) / size
            if on_grass[c] or on_gravel[c]: credit *= off_progress
            pos += credit
        else:
            neg += w_prog * (-dp) / size
        neg += w_over * max(vel_norm[c] - tn, 0.0)

        open_road = min(max((d_corner - 0.3 * size) / (0.7 * size), 0.0), 1.0)
        neg += w_steer * abs(rotation[c]) * open_road + w_jit * abs(rotation[c] - prev_rotation[c])

        if (on_grass[c] or on_gravel[c]) and not prev_off[c]: neg += w_entry   # first touch of loose surface
        em = edge_mult[c]
        if on_grass[c]:    neg += w_grass * em
        elif on_gravel[c]: neg += w_gravel * em
        elif on_curb[c]:   neg += w_curb * em * ((1.0 - curb_disc) if corner_curr[c] else 1.0)
        pos += extra_pos[c]
        neg += extra_neg[c]
        if on_road[c]:     pos += w_surv
        if bumped[c]:      neg += w_bump

        shape_pos[c] = pos
        shape_neg[c] = neg


@njit(fastmath=True, cache=True)
def _tile_beam(px, py, dx, dy, x0, x1, y0, y1):
    """
    Signed distance along (dx, dy) from (px, py) to the FAR boundary of the axis-aligned tile [x0,x1]x[y0,y1],
    i.e. the far end of the line/tile intersection (slab method).
      inside the tile : the usual distance to the boundary ahead (>= 0, at most the tile diagonal)
      outside the tile: the beam keeps searching along its line, so one of two opposite beams goes negative
                        (closest boundary is behind it) and the other overshoots the diagonal (farthest boundary)
    """
    t_far = 1e18
    if abs(dx) > 1e-9:
        ta = (x0 - px) / dx
        tb = (x1 - px) / dx
        t_far = min(t_far, max(ta, tb))
    if abs(dy) > 1e-9:
        ta = (y0 - py) / dy
        tb = (y1 - py) / dy
        t_far = min(t_far, max(ta, tb))
    return t_far


@njit(parallel=True, fastmath=True, cache=True)
def _get_state(
    state: Array, x: Array, y: Array, angle: Array, vel_lin: Array, vel_ang: Array,
    tile_idx: Array, t_orient: Array, t_next: Array, t_check: Array, t_center: Array,
    lidar_distances: Array, surface: Array,
    half: float, tile_size: float, max_vel_lin: float, max_vel_ang: float,
    n_tiles: int, use_tile_lidar: bool, use_surface: bool, use_lidar: bool,
):
    """
    Observation rows (features roughly within [-1, 1]):
        [dist x n_tiles, phase x n_tiles, vel_lin, vel_ang, heading,
         (tile_lidar x4 if use_tile_lidar), (surface if use_surface), (lidar beams if use_lidar)]

    dist_k     : 1 - distance to the exit-boundary midpoint (t_check) of the k-th tile ahead / (half * (k + 1)), clipped
    phase_k    : wrapped heading difference to the k-th tile's exit orientation / 180
    tile_lidar : 4 beams at car heading +0/+90/+180/+270 deg measuring the distance to the CURRENT tile's boundary
                 (not surfaces). Normalised by the tile diagonal and mapped to [-1, 1] (no clipping): inside the
                 tile every beam is in [-1, 1]; once the car is outside, one beam of an opposite pair goes
                 below -1 (nearest boundary now behind it) and the other above +1 (farthest boundary of the tile).
    """
    total, indices = grid((x.shape[0],))
    beams = lidar_distances.shape[1]
    base = 2 * n_tiles + 3
    diag = tile_size * math.sqrt(2.0)
    hs = tile_size * 0.5
    for i in prange(total):
        c  = indices[i]
        t  = tile_idx[c]
        # Euclidean distance to Nth tile's exit boundary center and phase difference to its exit orientaion
        for k in range(n_tiles):
            state[c, k] = min(max(1.0 - math.hypot(t_check[t, 0] - x[c], t_check[t, 1] - y[c]) / (half * (k + 1.0)), -1.0), 1.0)
            state[c, n_tiles + k] = _wrap180(t_orient[t] - angle[c]) / 180.0
            t = t_next[t]
        state[c, base - 3] = vel_lin[c] / max_vel_lin                                               # Linear velocity
        state[c, base - 2] = vel_ang[c] / max_vel_ang                                               # Angular velocity
        state[c, base - 1] = (angle[c] - 180.0) / (-180.0)                                          # Car orientation
        col = base
        if use_tile_lidar:
            t0 = tile_idx[c]
            cx = t_center[t0, 0]
            cy = t_center[t0, 1]
            for k in range(4):
                a = (angle[c] + 90.0 * k) * math.pi / 180.0
                d = _tile_beam(x[c], y[c], -math.sin(a), -math.cos(a), cx - hs, cx + hs, cy - hs, cy + hs)
                state[c, col + k] = (d / diag) * 2.0 - 1.0
            col += 4
        if use_surface:
            # Surface under the car centre (codes 0..4: road, curb, grass, gravel, void -> [-1, 1])
            state[c, col] = (float(surface[c]) / 4.0) * 2.0 - 1.0
            col += 1
        if use_lidar:
            for b in range(beams):
                state[c, col + b] = lidar_distances[c, b] * 2.0 - 1.0


@njit(parallel=True, fastmath=True, cache=True)
def _progress(
    progress: Array, in_tile: Array, x: Array, y: Array, tile_idx: Array, laps_done: Array,
    t_lap: Array, t_line: Array, n_tiles: int, size: float,
):
    """Distance driven along the track centre line (race order key)."""
    total, indices = grid((x.shape[0],))
    for i in prange(total):
        c  = indices[i]
        t  = tile_idx[c]
        best_d, best = 1e18, 0.0
        for s in range(2):                      # entry->centre, centre->exit
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


class Cars:
    def __init__(
        self, 
        grid: Grid, players: Players, config: CarConfig | None = None, 
        race: RaceConfig | None = None,
        reward: RewardConfig | None = None,
        **params: dict[str, Any]
    ):
        if config is None:
            config = CarConfig()
        self.race = race if race is not None else RaceConfig()
        self.reward = reward if reward is not None else RewardConfig()

        overwrite_config(config, params)
        self.config = config
        
        self.grid = grid
        self.players = players
        self.grid_mapping: dict[Position, Tile] = {tile.position: tile for tile in self.grid._cells.flatten()}
        self.reverse_mapping: dict[Tile, Position] = {tile: position for position, tile in self.grid_mapping.items()}

        # Procedural car (see sprite.py): neutral body drives the masks / LiDAR footprint, the per-car
        # rainbow variants are built lazily in _sprite() and drawn with the same footprint.
        self.bot_car_img = make_car((150, 150, 160), config.scale)
        export_car_images(IMAGE_DIR)                    # static r/g/b/black/white previews, only if missing
        # Env-private RNG (gravel jitter): never touches np.random's global state
        self.rng = np.random.default_rng()
        self._sprites: dict[tuple[int, int], Surface] = {}
        self._halos: list[Surface] | None = None
        self.labels: list[str] | None = None            # per-car text shown on the top-3 halos / leaderboard
        self._label_cache: dict[tuple, Surface] = {}
        self.mask = pygame.mask.from_surface(self.bot_car_img)
        self._car_lut_off, self._car_lut_n, self._car_lut_id = self._build_car_lut(self.bot_car_img)
        self.width, self.height = self.bot_car_img.get_size()
        self.total = 1

        self.init_x, self.init_y, self.init_angle = 0.0, 0.0, 0.0
        
        # Physics and control configuration
        self.use_lidar: bool        = config.use_lidar
        self.state_tiles: int       = max(1, int(config.state_tiles))   # tiles ahead in the dist/phase features
        self.use_tile_lidar: bool   = config.use_tile_lidar   # 4 beams to the current tile's boundary
        self.use_surface: bool      = config.use_surface      # surface code under the car centre
        self.max_vel_lin: float     = config.max_linear_velocity
        self.max_vel_ang: float     = config.max_angular_velocity
        self.acc_lin: float         = config.linear_acceleration
        self.acc_ang: float         = config.angular_acceleration
        self.coeff_reverse: float   = config.reverse_coeff
        self.coeff_brake: float     = config.brake_coeff
        self.coeff_checker: float   = config.checker_coeff
        self.min_brake_steps: int   = config.min_brake_steps
        self.min_straight_steps: int = config.min_straight_steps
        self.straight_alignment_tolerance: float = config.straight_alignment_tolerance
        self.cutoff: float          = config.cutoff
        self.toggle_reverse: bool   = config.toggle_reverse
        self.full_restart: bool     = config.full_restart
        self.restrict_movement: bool = config.restrict_movement
        self.frames_per_tile: int   = config.frames_per_tile
        self.discrete: bool         = config.discrete
        self.endless: bool          = config.endless

        # Surface friction: velocity multiplier applied every frame (1.0 = no decay)
        self.friction_road:   float = config.friction_road
        self.friction_curb:   float = config.friction_curb
        self.friction_grass:  float = config.friction_grass
        self.friction_gravel: float = config.friction_gravel
        self.friction_min, self.friction_max = (
            func([self.friction_road, self.friction_curb, self.friction_grass, self.friction_gravel])
            for func in [min, max]
        )

        # Surface handling: angular velocity multiplier (1.0 = full control)
        self.handling_road:   float = config.handling_road
        self.handling_curb:   float = config.handling_curb
        self.handling_grass:  float = config.handling_grass
        self.handling_gravel: float = config.handling_gravel
        self.handling_min, self.handling_max = (
            func([self.handling_road, self.handling_curb, self.handling_grass, self.handling_gravel])
            for func in [min, max]
        )

        # Surface slip: percentage of previous displacement of current frames displacement (0.0 = none)
        self.slip_road:   float = config.slip_road
        self.slip_curb:   float = config.slip_curb
        self.slip_grass:  float = config.slip_grass
        self.slip_gravel: float = config.slip_gravel
        self.slip_min, self.slip_max = (
            func([self.slip_road, self.slip_curb, self.slip_grass, self.slip_gravel])
            for func in [min, max]
        )

        # ------------------------------------------------------------------
        # LiDAR configuration
        # ------------------------------------------------------------------
        # lidar_angles: number of evenly-spaced beams cast from the car centre.
        #   Beams are relative to the car's own heading (0° = forward,
        #   90° = right, 180° = behind, 270° = left).
        # lidar_max_dist: maximum ray length in pixels (default = 2.5 tiles).
        # lidar_step: sampling stride in pixels; smaller = more accurate but slower.
        self.lidar_angles: int   = config.lidar_angles
        self.lidar_max_dist: float = config.lidar_max_dist if config.lidar_max_dist is not None else float(self.grid.cell_size) * 2.5
        self.lidar_step: int    = config.lidar_step
        self._lidar_n_steps: int = max(1, int(self.lidar_max_dist / self.lidar_step))

        # Fixed, car-relative beam offsets (degrees, evenly spaced over 360°)
        self._lidar_beam_offsets: Array = np.linspace(
            0.0, 360.0, self.lidar_angles, endpoint=False, dtype=NP_FLOAT
        )

        # Cached numpy bool arrays extracted from grid masks (filled in reset())
        self._lidar_road_arr:   Array | None = None
        self._lidar_curb_arr:   Array | None = None
        self._lidar_grass_arr:  Array | None = None
        self._lidar_gravel_arr: Array | None = None
        # Union of all surface layers — used as the hard track-boundary clamp
        # that prevents beams from escaping through void cells and hitting a
        # road segment on the far side of a U-turn.
        self._lidar_track_arr:  Array | None = None
        # Tile boundaries (1-pixel borders between tiles) — prevents beams from
        # crossing tile edges and detecting surfaces from adjacent tiles in
        # different directions (e.g., U-turns).
        self._lidar_boundary_arr: Array | None = None
        self._lidar_layers: Array | None = None   # packed uint8 (W, H) map for the numba kernels
        self._lidar_D: int = 1                    # samples per beam
        self._coef: Array = np.array([
            [config.friction_road, config.friction_curb, config.friction_grass, config.friction_gravel],
            [config.handling_road, config.handling_curb, config.handling_grass, config.handling_gravel],
            [config.slip_road,     config.slip_curb,     config.slip_grass,     config.slip_gravel],
        ], dtype=NP_FLOAT)
        self.tiles: list[Tile] = []
        self._tile_id: dict[Tile, int] = {}
        self._lidar_gw: int = 0
        self._lidar_gh: int = 0

        # ------------------------------------------------------------------
        # Per-car state arrays (sized for total=1; rebuilt in reset())
        # ------------------------------------------------------------------
        self.x          = np.full((self.total,), self.init_x, dtype=NP_FLOAT)
        self.y          = np.full((self.total,), self.init_y, dtype=NP_FLOAT)
        self.disp_x     = np.full((self.total,), 0.0, dtype=NP_FLOAT)
        self.disp_y     = np.full((self.total,), 0.0, dtype=NP_FLOAT)
        self.vel_lin    = np.full((self.total,), 0.0, dtype=NP_FLOAT)
        self.vel_ang    = np.full((self.total,), 0.0, dtype=NP_FLOAT)
        self.angle      = np.full((self.total,), self.init_angle, dtype=NP_FLOAT)
        self.crashed    = np.full((self.total,), False, dtype=bool)
        self.finished   = np.full((self.total,), False, dtype=bool)
        self.started    = np.full((self.total,), False, dtype=bool)
        self.laps_done  = np.full((self.total,), 0, dtype=NP_INT)
        self.lap_pending = np.full((self.total,), False, dtype=bool)   # entered the start tile, line not crossed yet
        self.race_done = np.full((self.total,), False, dtype=bool)     # finished the race distance
        self.finish_order: list[int] = []
        self.finish_frame = np.full((self.total,), -1, dtype=NP_INT)
        self.race_frame = 0
        self.race_end_in: float | None = None
        self.checks_done = np.full((self.total,), 0, dtype=NP_INT)
        self.crashes_done = np.full((self.total,), 0, dtype=NP_INT)
        self.hiatus     = np.full((self.total,), 0, dtype=NP_INT)
        self.tile_stag_steps = np.full((self.total,), 0, dtype=NP_INT)
        self.brake_steps = np.full((self.total,), 0, dtype=NP_INT)
        self.turn_steps = np.full((self.total,), 0, dtype=NP_INT)
        self.straight_steps = np.full((self.total,), 0, dtype=NP_INT)
        self.on_road    = np.full((self.total,), False, dtype=bool)
        self.on_curb    = np.full((self.total,), False, dtype=bool)
        self.on_grass   = np.full((self.total,), False, dtype=bool)
        self.on_gravel  = np.full((self.total,), False, dtype=bool)
        self.on_track   = np.full((self.total,), True, dtype=bool)
        self.on_finish  = np.full((self.total,), False, dtype=bool)
        self.tile: Array[tuple[int], Tile] = np.full((self.total,), None, dtype=object)
        self.tile_idx = np.zeros((self.total,), dtype=NP_INT)   # compact tile ids (see _cache_tile_table)
        self.friction: Array | None = np.full_like(self.vel_lin, config.friction_road)
        self._friction_prev = self.friction.copy()
        self._slip_prev = np.full_like(self.vel_lin, config.slip_road)
        self.handling: Array | None = np.full_like(self.vel_lin, config.handling_road)
        self.slip: Array | None = np.full_like(self.vel_lin, config.slip_road)
        self.ranking: Array | None = None
        self.race_laps: float = float('inf')                       # laps to win (set by the Env)
        self.debug_car: int | None = None                           # car shown by the verbose overlays (None = leader)
        self._debug_idx: int | None = None
        self._debug_level: int = 0
        self.used_brake = np.full((self.total,), False, dtype=bool)
        self.tile_crash = np.full((self.total,), False, dtype=bool)

        # Corner-entry speed: recorded once when the car first enters a corner
        # tile; held constant until the car moves to the next tile.  Used by
        # Players.update() to award the entry-speed bonus.
        self.corner_entry_speed: Array = np.full((self.total,), 0.0, dtype=NP_FLOAT)

        # LiDAR output arrays
        # lidar_distances     (N, lidar_angles) float32, each beam's hit distance [0, 1]
        # lidar_surface_hits  (N, lidar_angles) int8, surface code at hit point
        # surface             (N,) int8, dominant surface under car centre
        self.lidar_distances:    Array = np.ones(
            (self.total, self.lidar_angles), dtype=NP_FLOAT
        )
        self.lidar_surface_hits: Array = np.full(
            (self.total, self.lidar_angles), SURFACE_VOID, dtype=np.int8
        )
        self.surface: Array = np.zeros((self.total,), dtype=np.int8)  # default: road

        # Race mode state (see game/race.py)
        self.track: TrackGeometry | None = None
        self.draft_level = np.zeros((self.total,), dtype=NP_FLOAT)   # slipstream strength [0, 1]
        self._init_trail()
        self._trail_overlay: Surface | None = None   # SRCALPHA layer for the flame trail
        self.bumped      = np.zeros((self.total,), dtype=bool)       # touched another car this frame
        self.progress    = np.zeros((self.total,), dtype=NP_FLOAT)   # px driven along the track
        self.in_tile     = np.zeros((self.total,), dtype=NP_FLOAT)
        self._prev_rotation = np.zeros((self.total,), dtype=NP_FLOAT)
        self._prev_off   = np.zeros((self.total,), dtype=bool)
        self.shape_pos   = np.zeros((self.total,), dtype=NP_FLOAT)
        self.shape_neg   = np.zeros((self.total,), dtype=NP_FLOAT)
        self.target_speed = np.ones((self.total,), dtype=NP_FLOAT)   # safe speed ahead / max speed
        self._tr_pos     = np.zeros((self.total,), dtype=NP_FLOAT)
        self._tr_neg     = np.zeros((self.total,), dtype=NP_FLOAT)
        self._tr_edge    = np.ones((self.total,), dtype=NP_FLOAT)
        self.race_order: Array | None = None                         # car indices, leader first

        self.usr_engaged = False
        self.font: Optional[Font] = None
        self.label_font: Optional[Font] = None

    def _init_trail(self) -> None:
        """Pose history (ring buffers) used to draw the curved slipstream streaks."""
        cap = self.race.slipstream.trail_length
        self._trail_x = np.zeros((self.total, cap), dtype=NP_FLOAT)
        self._trail_y = np.zeros((self.total, cap), dtype=NP_FLOAT)
        self._trail_a = np.zeros((self.total, cap), dtype=NP_FLOAT)
        self._trail_head = np.zeros((self.total,), dtype=np.int64)
        self._trail_count = np.zeros((self.total,), dtype=np.int64)
        self._trail_buf = np.empty((cap, 2), dtype=NP_FLOAT)   # scratch polyline for rendering
        self._draft_decay = np.zeros((self.total,), dtype=bool)  # True while the tow is wearing off
        self._frame = 0

    # ======================================================================
    # Spatial properties
    # ======================================================================

    @property
    def center(self):
        return self.x, self.y

    @property
    def position_front(self):
        radians = np.radians(self.angle)
        offset = self.height / 4
        return (self.x - np.sin(radians) * offset, self.y + np.cos(radians) * offset)

    @property
    def position_back(self):
        radians = np.radians(self.angle)
        offset = self.height / 4
        return (self.x + np.sin(radians) * offset, self.y - np.cos(radians) * offset)

    @property
    def center_front(self):
        radians = np.radians(self.angle)
        offset = self.height / 4
        return (self.center[0] - np.sin(radians) * offset,
                self.center[1] + np.cos(radians) * offset)

    @property
    def center_back(self):
        radians = np.radians(self.angle)
        offset = self.height / 4
        return (self.center[0] + np.sin(radians) * offset,
                self.center[1] - np.cos(radians) * offset)

    @property
    def mask_front(self):
        front_height = self.height // 2
        front_surf = self.img.subsurface((0, 0, self.width, front_height))
        return pygame.mask.from_surface(front_surf)

    @property
    def mask_back(self):
        front_height = self.height // 2
        back_height = self.height - front_height
        back_surf = self.img.subsurface((0, front_height, self.width, back_height))
        return pygame.mask.from_surface(back_surf)

    # ======================================================================
    # LiDAR internals
    # ======================================================================

    @staticmethod
    def _build_car_lut(img: Surface, steps: int = 36000) -> tuple[Array, Array, Array]:
        """
        Rotated car mask for every ``360 / steps`` degrees, as pixel offsets from the
        integer car centre (pygame places the rotated rect with ``center=(round(x), round(y))``
        i.e. topleft = centre - size // 2).  Identical masks are stored once.
        Built once at construction; used by ``_update_regions``.
        """
        unique: dict[bytes, int] = {}
        pts: list[Array] = []
        ids = np.zeros((steps,), dtype=NP_INT)
        for q in range(steps):
            rot  = pygame.transform.rotate(img, q * 360.0 / steps)
            w, h = rot.get_size()
            px, py = np.nonzero(pygame.surfarray.array_alpha(rot) > 127)   # Mask.from_surface threshold
            p = np.stack([px - w // 2, py - h // 2], axis=-1).astype(NP_INT)
            key = p.tobytes()
            if key not in unique:
                unique[key] = len(pts)
                pts.append(p)
            ids[q] = unique[key]
        k_max = max(1, max(len(p) for p in pts))
        off = np.zeros((len(pts), k_max, 2), dtype=NP_INT)
        n   = np.zeros((len(pts),), dtype=NP_INT)
        for u, p in enumerate(pts):
            off[u, :len(p)] = p
            n[u] = len(p)
        return off, n, ids

    def _cache_lidar_arrays(self) -> None:
        """
        Extract grid mask surfaces as numpy bool arrays for fast ray-marching.

        pygame.surfarray.array_alpha() returns a (W, H) uint8 array where
        alpha == 255 at set pixels and 0 elsewhere.  Indexing is [x, y]
        (column-major), matching the (cx, cy) → (x_int, y_int) lookups used
        in _update_lidar() and _update_surface().

        Called once from reset() after the grid has been fully initialised.

        In addition to the four surface arrays, we build:

        _lidar_track_arr  — boolean (W, H) that is True for every pixel that
            belongs to any non-empty track tile (road ∪ curb ∪ grass ∪ gravel).
            Used as a hard outer boundary so beams never escape the current tile
            cluster and spuriously hit a road segment on the far side of a U-turn.

        _lidar_boundary_arr — boolean (W, H) that is True at 1-pixel tile
            boundaries.  Used to clamp beams so they don't cross tile edges and
            incorrectly detect surfaces from adjacent tiles in different directions.
        """
        if not self.grid.initialized:
            return

        self._lidar_gw = self.grid.width
        self._lidar_gh = self.grid.height

        def _surf_to_bool(surf) -> Array:
            if surf is None:
                return np.zeros((self._lidar_gw, self._lidar_gh), dtype=bool)
            alpha = pygame.surfarray.pixels_alpha(surf)  # (W, H) uint8 view, no copy (the copy was the slow part)
            out = alpha > 0
            del alpha                                    # releases the surface lock
            return out

        self._lidar_road_arr   = _surf_to_bool(self.grid._surface_road)
        self._lidar_curb_arr   = _surf_to_bool(self.grid._surface_curb)
        self._lidar_grass_arr  = _surf_to_bool(self.grid._surface_grass)
        self._lidar_gravel_arr = _surf_to_bool(self.grid._surface_gravel)

        # Track boundary: union of all four surface layers.
        # Any pixel on-track is True; void / off-grid is False.
        # We derive this from the full_track_mask surface if available,
        # otherwise fall back to OR-ing the four layer arrays.
        if self.grid._surface_track is not None:
            self._lidar_track_arr = _surf_to_bool(self.grid._surface_track)
        else:
            self._lidar_track_arr = (
                self._lidar_road_arr
                | self._lidar_curb_arr
                | self._lidar_grass_arr
                | self._lidar_gravel_arr
            )

        # Tile boundary: thin 1-pixel lines marking tile edges.
        # Any pixel on a boundary is True; interior is False.
        # Used to prevent beams from crossing tile boundaries.
        if self.grid._surface_boundary is not None:
            self._lidar_boundary_arr = _surf_to_bool(self.grid._surface_boundary)
        else:
            self._lidar_boundary_arr = np.zeros((self._lidar_gw, self._lidar_gh), dtype=bool)

        # Single packed uint8 layer map consumed by the Numba kernels
        self._lidar_layers = pack_layers(
            self._lidar_road_arr, self._lidar_curb_arr, self._lidar_grass_arr,
            self._lidar_gravel_arr, self._lidar_track_arr, self._lidar_boundary_arr,
            _surf_to_bool(self.grid._surface_finish),
        )
        self._lidar_D = len(np.arange(self.lidar_step, self.lidar_max_dist + self.lidar_step, self.lidar_step))

    def _cache_tile_table(self) -> None:
        """
        Flatten Tile objects into numpy lookup tables so the step kernels never touch
        Python objects.  Row ``i`` of every ``t_*`` array describes ``self.tiles[i]``.
        """
        cont = self.grid.continuity
        tiles = list(cont.keys())
        known = set(tiles)
        tiles += [t for t in self.grid_mapping.values() if not t.empty and t not in known]
        self.tiles    = tiles
        self._tile_id = {t: i for i, t in enumerate(tiles)}
        self._start_id = self._tile_id[self.grid.start_cell]

        self.t_type   = np.array([t.type.value for t in tiles], dtype=NP_INT)
        self.t_orient = np.array([float(t.orientation) for t in tiles], dtype=NP_FLOAT)
        self.t_rot    = np.array([float(t.rotation) for t in tiles], dtype=NP_FLOAT)
        self.t_start  = self.t_type == Type.START.value
        self.t_turn   = np.where(self.t_type == Type.LEFT.value, 1.0, np.where(self.t_type == Type.RIGHT.value, -1.0, 0.0)).astype(NP_FLOAT)
        self.t_next   = np.array([self._tile_id.get(cont.get(t), i) for i, t in enumerate(tiles)], dtype=NP_INT)
        self.t_next2  = self.t_next[self.t_next]
        self.t_center = np.array([t.center_abs for t in tiles], dtype=NP_FLOAT)
        self.t_check  = np.array([t.center_check_abs for t in tiles], dtype=NP_FLOAT)

        cols = max(p[0] for p in self.grid_mapping) + 1
        rows = max(p[1] for p in self.grid_mapping) + 1
        self.tile_map = np.full((cols, rows), -1, dtype=NP_INT)
        for pos, tile in self.grid_mapping.items():
            if not tile.empty and tile in self._tile_id:
                self.tile_map[pos] = self._tile_id[tile]

    def _cache_progress_table(self) -> None:
        """Track centre-lines + lap index per tile for the race-order kernel."""
        T = len(self.tiles)
        self.t_lap  = np.zeros((T,), dtype=NP_INT)
        self.t_line = np.zeros((T, 3, 2), dtype=NP_FLOAT)
        for t, line in self.track.lines.items():
            i = self._tile_id[t]
            self.t_lap[i]  = self.track.lap_index[t]
            self.t_line[i] = np.stack(line)

    def _update_surface(self) -> None:
        """Centre-point surface lookup for every car (Numba-parallel, see ``_update_surface``)."""
        if self._lidar_layers is None:
            return
        _update_surface(self._lidar_layers, self.x, self.y, self.surface)

    def _update_lidar(self) -> None:
        """
        Cast `lidar_angles` rays from each active car's centre and record the
        distance (and surface type) at the edge of the car's current surface
        layer, with three hard guarantees:

        1. **Surface-relative origin** — the beam starts on whatever surface
           the car occupies (road, curb, grass, or gravel) and measures the
           distance to the outer edge of that surface.  If a car segment is on
           the curb the beam measures to the curb→grass boundary, not back to
           the road→curb boundary.

        2. **Track boundary clamp** — beams are hard-stopped at the outer edge
           of the track tile cluster (the gravel→void boundary, or the
           road/grass/curb boundary when gravel is absent).  This prevents a
           beam from passing through void cells and spuriously detecting a road
           segment on the far side of a U-turn.

        3. **Tile boundary clamp** — beams must not cross tile edges (the
           1-pixel boundaries between tiles). This prevents beams from
           incorrectly detecting surfaces on adjacent tiles that are oriented
           in different directions, which is especially critical in layouts
           with U-turns or other complex geometries.

        Surface priority (innermost → outermost):
            road (0) → curb (1) → grass (2) → gravel (3) → void (4)

        The beam stops at the first sample that is **outside** the car's
        current surface layer and all layers inner than it.  For example:
          • car on road   → beam stops at first non-road sample
          • car on curb   → beam stops at first non-(road∪curb) sample
          • car on grass  → beam stops at first non-(road∪curb∪grass) sample
          • car on gravel → beam stops at first non-track sample
                            (i.e. first void / out-of-bounds sample),
                            but it uses the last on-track step so the endpoint
                            stays just inside the gravel, not outside it.

        When a beam never exits the home surface (max distance reached), it
        returns distance=1.0 and surface_hit=SURFACE_VOID to signal "open".

        Fully vectorised: no Python loops over cars or beams.

        Outputs (written in-place):
          self.lidar_distances    (N, B) float64   normalised hit dist [0, 1]
          self.lidar_surface_hits (N, B) int8       surface code at hit point
        """
        if self._lidar_layers is None:
            return
        N, B = self.total, self.lidar_angles
        if self.lidar_distances.shape != (N, B):
            self.lidar_distances    = np.ones((N, B), dtype=NP_FLOAT)
            self.lidar_surface_hits = np.full((N, B), SURFACE_VOID, dtype=np.int8)
        _update_lidar(
            self._lidar_layers, self.x, self.y, self.angle, self.surface,
            self._lidar_beam_offsets, self._lidar_D, float(self.lidar_step), float(self.lidar_max_dist),
            self.lidar_distances, self.lidar_surface_hits,
        )
        if self.race_on and self.race.lidar_detect_cars and N > 1:
            _lidar_cars(
                self.x, self.y, self.angle, self.race_active, self._lidar_beam_offsets,
                float(self.lidar_max_dist), float(self.width), float(self.height),
                float(self.race.collision.radius_scale),
                self.lidar_distances, self.lidar_surface_hits,
            )

    # ======================================================================
    # Location / collision tracking
    # ======================================================================

    def _update_regions(self, index: int | None = None):
        """
        Refresh the on_road / on_curb / ... flags of the car body.  All cars at once by
        default (Numba-parallel); ``granularity == 'cell'`` keeps the pygame mask path.
        """
        if self.grid.granularity != 'cell' and self._lidar_layers is not None:
            _update_regions(
                self._lidar_layers, self._car_lut_off, self._car_lut_n, self._car_lut_id, self.x, self.y, self.angle,
                self.on_road, self.on_curb, self.on_grass, self.on_gravel, self.on_track, self.on_finish,
            )
            return

        for index in (range(self.total) if index is None else [index]):
            obj_center_point = (self.x[index].item(), self.y[index].item())
            obj = self.bot_car_img.copy()
            rot_obj = pygame.transform.rotate(obj, self.angle[index].item())
            rot_mask = pygame.mask.from_surface(rot_obj)
            rot_rect = rot_obj.get_rect(center=obj_center_point)
            x_t, y_t = rot_rect.topleft

            self.on_road  [index] = self.grid.on_road  (x_t, y_t, self.width, self.height, mask=rot_mask)
            self.on_track [index] = self.grid.on_track (x_t, y_t, self.width, self.height, mask=rot_mask)
            self.on_gravel[index] = self.grid.on_gravel(x_t, y_t, self.width, self.height, mask=rot_mask)
            self.on_grass [index] = self.grid.on_grass (x_t, y_t, self.width, self.height, mask=rot_mask)
            self.on_curb  [index] = self.grid.on_curb  (x_t, y_t, self.width, self.height, mask=rot_mask)
            self.on_finish[index] = self.grid.on_finish(x_t, y_t, self.width, self.height, mask=rot_mask)

    def _update_location(self):
        prev_idx = self.tile_idx.copy()
        self._update_regions()
        _update_location(
            self.tile_map, self.t_next, self.t_start, float(self.grid.cell_size),
            self.x, self.y, self.on_track, self.tile_idx,
            self.started, self.finished, self.crashed, self.tile_crash,
            self.laps_done, self.checks_done, self.tile_stag_steps,
            self.lap_pending, self.on_finish, bool(self.finish_line_laps),
        )
        if self.race_done.any():                       # a finished car rolls slowly: that must not cost lives
            self.tile_stag_steps[self.race_done] = 0
        # Keep the object view (used by rendering / race helpers) in sync with the ids
        for i in np.flatnonzero(self.tile_idx != prev_idx):
            self.tile[i] = self.tiles[self.tile_idx[i]]

    # ======================================================================
    # Main update
    # ======================================================================

    def update(self, inputs: Array, verbose=False):
        if self.discrete:
            raise NotImplementedError()

        # ------------------------------------------------------------------
        # Cache the previous states / counts
        # ------------------------------------------------------------------
        prev_checks = self.checks_done.copy()
        prev_laps   = self.laps_done.copy()
        prev_angle  = self.angle.copy()
        prev_tile   = self.tile_idx.copy()
        prev_brake  = self.brake_steps.copy()

        # ------------------------------------------------------------------
        # 0. Inputs  (N, 2) -> acceleration, rotation
        # ------------------------------------------------------------------
        inputs = np.asarray(inputs, dtype=NP_FLOAT)
        acceleration = np.ascontiguousarray(inputs[..., 0])
        rotation     = np.ascontiguousarray(inputs[..., 1])
        self.race_frame += 1
        if self.race_on and self.race_done.any():
            # After the flag a car keeps its steering but has no throttle: it brakes down to a slow roll
            # (never a full stop, a standstill would cost it lives)
            d = self.race_done
            crawl = 0.15 * self.max_vel_lin
            acceleration[d] = np.where(self.vel_lin[d] > crawl, -1.0, 0.05)

        # ------------------------------------------------------------------
        # 1-3. Surface coefficients + steering / handling
        # ------------------------------------------------------------------
        _steer(
            rotation, self.vel_ang, self.on_gravel, self.on_grass, self.on_curb,
            self.friction, self.handling, self.slip,
            self._coef, float(self.max_vel_ang), float(self.acc_ang), float(self.coeff_checker),
        )

        # ------------------------------------------------------------------
        # 4-5. Slipstream (cross-car), accelerate / brake, friction, move
        # ------------------------------------------------------------------
        drafting = bool(self.race_on and self.race.slipstream.enabled)
        ss = self.race.slipstream
        if drafting:
            prev_level = self.draft_level
            self.draft_level = slipstream(
                self.x, self.y, self.angle, np.clip(self.vel_lin / self.max_vel_lin, 0.0, 1.0),
                self.race_active, self.draft_level, self.height, ss,
            )
            # gaining (or holding) -> False, wearing off -> True; unchanged inside the dead band
            self._draft_decay = np.where(
                self.draft_level < prev_level - DRAFT_TREND_EPS, True,
                np.where(self.draft_level > prev_level + DRAFT_TREND_EPS, False, self._draft_decay),
            )
        # Gravel jitter is drawn here (not inside the prange kernel) so the env-private RNG (Game seed)
        # controls it and runs are reproducible, exactly like the original implementation.
        noise = np.zeros(self.total, dtype=NP_FLOAT)
        n_gravel = int(np.count_nonzero(self.on_gravel))
        if n_gravel:
            amp = float(self.config.gravel_noise)
            noise[self.on_gravel] = self.rng.uniform(-amp, amp, size=n_gravel)
        _drive(
            acceleration, self.vel_lin, self.vel_ang, self.angle, self.x, self.y,
            self.disp_x, self.disp_y, self._vel_norm,
            self.friction, self.slip,
            self.on_gravel, self.draft_level, noise,
            float(self.max_vel_lin), float(self.acc_lin), float(self.coeff_brake), float(self.coeff_reverse),
            bool(self.toggle_reverse or self.race_on), drafting,
            float(ss.top_speed_bonus), float(ss.push),
            float(self.config.lateral_grip), float(self.config.grip_relief), float(self.config.slide_drag),
        )
        vel_norm = self._vel_norm

        # Car-to-car contact (race mode): resolved after movement, before surface lookups
        if self.race_on and self.race.collision.enabled:
            self.bumped = collide(
                self.x, self.y, self.angle, self.vel_lin, self.vel_ang, self.disp_x, self.disp_y,
                self.race_active, self.width, self.height, self.max_vel_lin, self.race.collision,
            )

        # Slipstream trail history (final pose of this frame)
        if self.race_on and self.race.slipstream.enabled:
            trail_push(
                self._trail_x, self._trail_y, self._trail_a, self._trail_head, self._trail_count,
                self.x, self.y, self.angle, self.race_active, self.height * TRAIL_RESET_CARS,
            )

        # ------------------------------------------------------------------
        # Update location, surface flags, and LiDAR
        # ------------------------------------------------------------------
        self._update_location()
        self._update_surface()   # update self.surface from centre-point lookup
        self._update_lidar()     # cast beams; update lidar_distances / lidar_surface_hits
        prev_progress = self.progress.copy()
        self._compute_progress()

        # ------------------------------------------------------------------
        # Derived signals for reward shaping, counters and crash flags
        # ------------------------------------------------------------------
        sg = self._signal_buf
        _signals(
            sg['checked'], sg['finished'], sg['tile_stag'], sg['on_road'],
            sg['moved_forward'], sg['corner_curr'], sg['corner_next'], sg['straight_curr'],
            sg['phase_align'], sg['turning'], sg['braking'], sg['straight_aligned'],
            sg['brake_check'], sg['turn_align'],
            self.turn_steps, self.brake_steps, self.straight_steps, self.used_brake,
            self.corner_entry_speed, self.crashed, self.tile_crash, self.crashes_done, self.hiatus,
            self.angle, prev_angle, self.tile_idx, prev_tile,
            self.checks_done, prev_checks, self.laps_done, prev_laps,
            self.tile_stag_steps, prev_brake, self.vel_lin, vel_norm,
            self.on_road, self.on_curb, self.on_grass, self.on_gravel, self.on_track,
            acceleration, rotation,
            self.t_type, self.t_orient, self.t_next,
            int(self.players.max_hiatus), float(self.straight_alignment_tolerance),
            int(self.restrict_movement), bool(self.race_on),
        )

        # ------------------------------------------------------------------
        # Dense time-trial shaping (progress / target speed / smoothness / surfaces)
        # ------------------------------------------------------------------
        rw = self.reward
        if rw.shaping:
            if self.race_on:
                _traffic_shape(
                    self._tr_pos, self._tr_neg, self._tr_edge,
                    self.x, self.y, self.angle, self.vel_lin, self.race_active,
                    self.progress, prev_progress, sg['on_road'], sg['phase_align'], sg['corner_curr'],
                    float(self.width), float(self.height), float(self.track.size), float(self.max_vel_lin),
                    float(rw.traffic_range * self.height), float(rw.traffic_gate), float(rw.edge_guard),
                    float(rw.veer_weight), float(rw.closing_weight), float(rw.lane_weight), float(rw.overtake_bonus),
                )
            else:
                self._tr_pos[:] = 0.0; self._tr_neg[:] = 0.0; self._tr_edge[:] = 1.0
            _reward_shape(
                self.shape_pos, self.shape_neg, self.target_speed,
                self._tr_pos, self._tr_neg, self._tr_edge,
                self.progress, prev_progress, self.in_tile, self.vel_lin, vel_norm,
                rotation, self._prev_rotation,
                self.tile_idx, self.t_turn, self.t_next, sg['corner_curr'],
                sg['on_road'], self.on_curb, self.on_grass, self.on_gravel, self.bumped, self._prev_off,
                float(self.track.size), float(self.max_vel_lin), float(self._a_dec),
                float(self._corner_f1), float(self._corner_f2),
                int(rw.lookahead_tiles), int(rw.corner_window),
                float(rw.progress_weight), float(rw.time_penalty), float(rw.survival_bonus),
                float(rw.overspeed_weight), float(rw.steer_weight), float(rw.jitter_weight),
                float(rw.curb_weight), float(rw.curb_corner_discount),
                float(rw.grass_weight), float(rw.gravel_weight), float(rw.bump_weight),
                float(rw.offroad_progress), float(rw.offroad_entry_penalty),
            )
        self._prev_rotation[:] = rotation
        self._prev_off[:] = self.on_grass | self.on_gravel

        # ------------------------------------------------------------------
        # Player (reward) update
        # ------------------------------------------------------------------
        self.players.update(
            sg['checked'], sg['finished'],
            self.crashed, self.hiatus,
            sg['moved_forward'], sg['tile_stag'],
            sg['corner_curr'], sg['corner_next'],
            sg['straight_curr'],
            vel_norm, sg['phase_align'],
            sg['on_road'],
            sg['turning'], self.turn_steps, self.min_brake_steps,
            sg['braking'], self.brake_steps, self.min_brake_steps, sg['brake_check'],
            sg['straight_aligned'], self.straight_steps, self.min_straight_steps,
            sg['turn_align'],
            self.corner_entry_speed,
            self.tile_crash,
            None, verbose,
            shape_pos=self.shape_pos, shape_neg=self.shape_neg,
        )

        if self.race_on:
            self._update_race_order(recompute=False)
        elif np.count_nonzero(self.laps_done >= 1) <= 3:
            self.ranking = np.argsort(-self.players.scores)

        self.restart()
        self._compute_progress()   # respawned cars must not register a progress jump next frame

    def _compute_progress(self) -> None:
        """Distance driven along the centre line (+ distance inside the current tile)."""
        _progress(
            self.progress, self.in_tile, self.x, self.y, self.tile_idx, self.laps_done + self.lap_pending,
            self.t_lap, self.t_line, int(self.track.n_tiles), float(self.track.size),
        )

    def _setup_reward(self) -> None:
        """Physics-derived corner speed for the reward (what a 90 degree bend can be taken at)."""
        rw = self.reward
        size = float(self.track.size)
        if rw.corner_speed_ratio is not None:
            f1 = float(rw.corner_speed_ratio)
        else:
            radius = rw.corner_line_radius * size / 2.0
            omega  = self.max_vel_ang * self.config.handling_road * math.pi / 180.0       # rad / frame
            v_turn = omega * radius                                                         # yaw-rate limit
            grip   = self.config.lateral_grip * (1.0 - self.config.slip_road)
            v_grip = math.sqrt(grip * radius)                                               # tyre limit
            f1 = rw.corner_safety * min(v_turn, v_grip) / self.max_vel_lin
        self._corner_f1 = min(max(f1, 0.25), 1.0)
        self._corner_f2 = self._corner_f1 * (rw.uturn_ratio if self._corner_f1 < 1.0 else 1.0)
        self._a_dec = self.acc_lin * (1.0 + 0.5 * self.coeff_brake) * rw.braking_margin

    def _update_race_order(self, recompute: bool = True) -> None:
        """Live race positions from distance driven (laps are offset so the grid starts fair)."""
        if recompute:
            self._compute_progress()
        key = np.where(self.race_active, self.progress, -np.inf)
        for rank, i in enumerate(self.finish_order):      # finishers keep the order they crossed the line in
            if key[i] > -np.inf:
                key[i] = 1.0e15 - rank
        self.race_order = np.argsort(-key, kind='stable')
        self.ranking = self.race_order

    @property
    def race_active(self) -> Array:
        """Cars that take part physically in the race: those with lives left, plus the
        human-driven car in manual mode (it keeps driving after running out of lives)."""
        active = np.asarray(self.players.active).copy()
        if self.usr_engaged and active.size:
            active[-1] = True
        return active

    @property
    def finish_line_laps(self) -> bool:
        """Laps only count when the car crosses the finish line (race.finish_line_laps in race mode,
        car.finish_line_laps otherwise)."""
        return bool(self.race.finish_line_laps if self.race_on else self.config.finish_line_laps)

    def mark_finished(self, laps_target: float) -> int:
        """Flag cars that completed `laps_target` laps (call once per step). Returns how many have finished."""
        newly = (self.laps_done >= laps_target) & ~self.race_done & self.race_active
        if newly.any():
            for i in sorted(np.flatnonzero(newly), key=lambda j: -self.progress[j]):
                self.race_done[i] = True
                self.finish_frame[i] = self.race_frame
                self.finish_order.append(int(i))
        return len(self.finish_order)

    @property
    def race_on(self) -> bool:
        return self.race.enabled and self.track is not None

    # ======================================================================
    # Event-driven (human) update
    # ======================================================================

    def update_on_event(self, verbose=False, bot_inputs: Array | None = None):
        """Human drives the last car. `bot_inputs` (N, 2) optionally drives all the others."""
        keys = pygame.key.get_pressed()

        if not self.discrete:
            inputs = np.full((self.total, 2), 0.0, dtype=NP_FLOAT)
            if bot_inputs is not None:
                inputs[:-1] = bot_inputs[:-1]
            moving = self.vel_lin[-1].item() > 0
            if moving: inputs[-1, 0] = -1
            # if self.race_on:
            #     # On a grid the auto-brake must not stall the car: a standstill frame costs a
            #     # life / respawns it off its grid slot. Brake down to a slow crawl instead.
            #     crawl = 0.15 * self.max_vel_lin
            #     inputs[-1, 0] = -1 if self.vel_lin[-1].item() > crawl else 0.05

            if   keys[pygame.K_a]: inputs[-1, 1] =  1
            elif keys[pygame.K_d]: inputs[-1, 1] = -1
            else:                  self.vel_ang[-1] = 0
            if keys[pygame.K_w]: inputs[-1, 0] =  1
            if keys[pygame.K_s]:
                if moving: inputs[-1, 0] = -1 * self.coeff_brake
                if (self.toggle_reverse or self.race_on) and not moving: inputs[-1, 0] = -1 * self.coeff_reverse

            self.update(inputs, verbose=verbose)
        else:
            raise NotImplementedError()

    # ======================================================================
    # State representations
    # ======================================================================

    def get_state(self) -> Array:
        """
        Return the observation vector for all cars, shape (N, features).

        Layout (n = state_tiles, default 3, minimum 1):
        0 … n-1        dist      1 - distance to exit-boundary midpoint of tile k ahead (k=0 is current), clipped
        n … 2n-1       phase     heading offset to the exit orientation of tile k ahead
        2n             vel_lin   normalised linear speed
        2n+1           vel_ang   normalised angular speed
        2n+2           heading   absolute car heading
        Optional, in this order:
        use_tile_lidar (4)  beams at heading +0/+90/+180/+270 to the current tile's boundary, / tile diagonal, [-1, 1]
                            (below -1 / above +1 when the car is outside the tile, see _get_state)
        use_surface    (1)  surface code under the car centre, in [-1, 1]
        use_lidar      (B)  lidar_distances[b] per-beam normalised hit distance
        """
        n = self.state_tiles
        features = (2 * n + 3 + (4 if self.use_tile_lidar else 0) + (1 if self.use_surface else 0)
                    + (self.lidar_angles if self.use_lidar else 0))
        state = np.empty((self.total, features), dtype=NP_FLOAT)
        _get_state(
            state, self.x, self.y, self.angle, self.vel_lin, self.vel_ang,
            self.tile_idx, self.t_orient, self.t_next, self.t_check, self.t_center,
            self.lidar_distances, self.surface,
            float(np.sqrt((self.grid.cell_size ** 2) * 2) / 2), float(self.grid.cell_size),
            float(self.max_vel_lin), float(self.max_vel_ang), int(n),
            bool(self.use_tile_lidar), bool(self.use_surface), bool(self.use_lidar),
        )
        return state

    # ======================================================================
    # Collision helpers
    # ======================================================================

    def _collide(self, int_mask: Mask, ext_mask: Mask, x: int = 0, y: int = 0):
        offset = (self.x, self.y)
        if x != 0 or y != 0:
            offset = (self.x - x, self.y - y)
        _poi = [ext_mask.overlap(int_mask, (round(x), round(y))) for x, y in zip(*offset)]
        poi = np.array([list(point) if point is not None else [-1, -1] for point in _poi])
        has_collided = np.array([True if point is not None else False for point in _poi])
        return has_collided, poi

    def collide(self, mask: Mask, x: int = 0, y: int = 0):
        return self._collide(self.mask, mask, x, y)

    # ======================================================================
    # Rendering
    # ======================================================================

    def set_labels(self, labels: "list[str] | None") -> None:
        """Per-car display text (index -> label). ``None`` falls back to the car index."""
        self.labels = None if labels is None else [str(l) for l in labels]
        self._label_cache.clear()

    def label(self, idx: int) -> "str | None":
        if self.labels is not None and 0 <= idx < len(self.labels):
            return self.labels[idx]
        return None

    def _steer_angle(self, idx: int) -> float:
        """Visual front-wheel angle (deg, + = left). Follows the yaw rate; the lock shrinks with speed."""
        cfg = self.config
        turn = float(np.clip(self.vel_ang[idx] / max(self.max_vel_ang, 1e-6), -1.0, 1.0))
        spd = float(np.clip(abs(self.vel_lin[idx]) / max(self.max_vel_lin, 1e-6), 0.0, 1.0))
        lock = cfg.wheel_lock_low_speed + (cfg.wheel_lock_top_speed - cfg.wheel_lock_low_speed) * spd
        return turn * lock

    def _sprite(self, idx: int, steer: float = 0.0) -> Surface:
        step = max(float(self.config.wheel_angle_step), 0.5)
        bucket = int(round(steer / step))                  # quantised so sprites can be cached
        key = (idx, self.total, bucket)
        img = self._sprites.get(key)
        if img is None:
            if len(self._sprites) > 4096:
                self._sprites.clear()
            img = self._sprites[key] = make_car(rainbow(idx, self.total), self.config.scale,
                                                steer=bucket * step, pad=WHEEL_PAD)
        return img

    def _text(self, text: str, color) -> Surface:
        key = (text, tuple(color))
        surf = self._label_cache.get(key)
        if surf is None:
            if len(self._label_cache) > 256:
                self._label_cache.clear()
            surf = self._label_cache[key] = self.label_font.render(text, True, color)
        return surf

    def render(self, surface: Surface, verbose: int | bool = False, hud: bool = True, debug_panel: bool = True):
        """
        verbose is a level: 0 nothing extra, 1 stats panel, 2 + LiDAR beams, 3 + tile exit arrows
        (True counts as 3). The stats / arrows / LiDAR describe `debug_car` when set, else the leader
        (the human car in a manual run). debug_panel=False leaves the stats panel to `render_debug`,
        so a camera can draw it on the screen instead of the world.
        """
        level = 3 if verbose is True else int(verbose or 0)
        self._debug_level = level
        if self.font is None:
            self.font = pygame.font.Font(None, 17)
        if self.label_font is None:
            self.label_font = pygame.font.Font(None, 15)
        if self._halos is None:
            radius = int(max(self.width, self.height) * 0.8)
            self._halo_r = radius
            self._halos = [make_halo(c, radius) for c in HALO_COLORS]

        self._render_race(surface, hud=False)               # slipstream streaks, under the cars

        ranking = [int(i) for i in (self.ranking if self.ranking is not None else np.argsort(-self.players.fitness))[:3]]
        tags = []
        best_idx = ranking[0] if not self.usr_engaged and ranking else (self.total - 1)
        if self.debug_car is not None and 0 <= self.debug_car < self.total:
            best_idx = self.debug_car
        self._debug_idx = best_idx
        for idx, (x, y, angle, active) in enumerate(zip(*self.center, self.angle, self.players.active)):
            if not (active or (self.usr_engaged and idx == self.total - 1)):
                continue
            cx, cy = x.item(), y.item()
            rank = ranking.index(idx) if idx in ranking else -1

            if rank >= 0:                                   # halo under the car (never rotated)
                halo = self._halos[rank]
                surface.blit(halo, halo.get_rect(center=(cx, cy)))

            rot_img  = pygame.transform.rotate(self._sprite(idx, self._steer_angle(idx)), angle.item())
            surface.blit(rot_img, rot_img.get_rect(center=(cx, cy)))

            if self.usr_engaged and idx == self.total - 1:  # the human-driven car: thin white ring
                pygame.draw.circle(surface, (255, 255, 255), (int(cx), int(cy)), self._halo_r, width=1)
            if rank >= 0:
                tags.append((rank, idx, cx, cy))
            if idx == best_idx:
                if debug_panel and level >= 1:
                    self._render_debugging(surface, idx)    # debugging text
                if self.use_lidar and level >= 2:
                    self._render_lidar(surface, best_idx, cx, cy)
                if level >= 3:
                    tile_curr = self.tile[idx]
                    tile_next = self.grid.continuity[tile_curr]
                    tile_next_1 = self.grid.continuity[tile_next]
                    draw_arrow(surface, tile_curr.respawn_point, tile_curr.center_check_abs, color=(102, 0, 153))
                    draw_arrow(surface, tile_next.respawn_point, tile_next.center_check_abs, color=(255, 255, 255))
                    draw_arrow(surface, tile_next_1.respawn_point, tile_next_1.center_check_abs, color=(255, 255, 255))
                    position = (x, y)
                    draw_arrow(surface, position, tile_curr.center_check_abs, color=(255, 0, 0))
                    draw_arrow(surface, position, tile_next.center_check_abs, color=(0, 0, 255))
                    draw_arrow(surface, position, tile_next_1.center_check_abs, color=(0, 255, 0))

        # Labels follow their halo; drawn last so no car covers them
        if tags:
            sw, sh = surface.get_size()
            for rank, idx, cx, cy in tags:
                text = self.label(idx)
                if text is None:
                    text = str(idx)                          # no keys/mapping given: show the car index
                color = HALO_COLORS[rank]
                txt = self._text(f"{rank + 1}  {text}", (240, 240, 240))
                w, h = txt.get_width() + 8, txt.get_height() + 4
                ty = cy - self._halo_r - 6 - h
                if ty < 2:                                  # near the top edge: flip below the halo
                    ty = cy + self._halo_r + 6
                    anchor = (cx, cy + self._halo_r)
                else:
                    anchor = (cx, cy - self._halo_r)
                rect = pygame.Rect(0, 0, w, h)
                rect.centerx = int(min(max(cx, w // 2 + 2), sw - w // 2 - 2))
                rect.y = int(ty)
                pygame.draw.line(surface, color, anchor, (rect.centerx, rect.bottom if ty < cy else rect.top), 1)
                draw_panel(surface, rect, self.config.overlay_color, 170, radius=4)
                pygame.draw.rect(surface, color, rect, width=1, border_radius=4)
                surface.blit(txt, (rect.x + 4, rect.y + 2))

        if hud:
            self._render_race(surface, hud=True)            # race leaderboard, on top of everything

    def render_debug(self, surface: Surface) -> None:
        """Stats panel of the debug car (verbose level >= 1), drawn onto `surface` (e.g. the screen)."""
        idx = getattr(self, "_debug_idx", None)
        if self._debug_level >= 1 and idx is not None and 0 <= idx < self.total:
            self._render_debugging(surface, idx)

    def render_hud(self, surface: Surface) -> None:
        """Race leaderboard / driver panel only. Lets a camera draw it on the screen instead of the world."""
        self._render_race(surface, hud=True)

    def _draw_trail(self, pts: Array, color, ss, tail_alpha: float, dirty):
        """Draw one streak onto the shared SRCALPHA layer; returns the updated dirty rect.

        `pts` runs newest -> oldest.  Opacity falls linearly from ss.alpha_head at the car to
        `tail_alpha` at the last point, for the line, particle and flame styles alike.
        """
        if self._trail_overlay is None or self._trail_overlay.get_size() != self.screen_size():
            self._trail_overlay = Surface(self.screen_size(), pygame.SRCALPHA)
        ov = self._trail_overlay
        m = len(pts)
        span = max(m - 1, 1)
        pad = int(max(ss.flame_width, ss.particle_spread, 3.0)) + 3
        lo, lo_y = float(pts[:, 0].min()), float(pts[:, 1].min())
        hi, hi_y = float(pts[:, 0].max()), float(pts[:, 1].max())
        box = pygame.Rect(int(lo) - pad, int(lo_y) - pad, int(hi - lo) + 2 * pad, int(hi_y - lo_y) + 2 * pad)

        def alpha_at(k: int) -> float:
            return ss.alpha_head + (tail_alpha - ss.alpha_head) * (k / span)

        if ss.flame:
            L = ss.flame_layers
            for layer in range(L):                    # outer glow first, hot core last (painted on top)
                core = layer / max(L - 1, 1)          # 0 = outer ... 1 = core
                heat_layer = 0.35 + 0.65 * core
                for k in range(m):
                    age = k / span
                    fade = alpha_at(k) * (0.55 + 0.45 * core)     # outer glow is fainter than the core
                    if ss.flame_flicker > 0.0:
                        fade *= 1.0 + ss.flame_flicker * 0.5 * math.sin(self._frame * 0.9 + k * 1.7 + layer * 2.1)
                    alpha = int(255 * min(max(fade, 0.0), 1.0))
                    if alpha <= 0:
                        continue
                    rgb = _flame_color(color, ss.flame_hot, heat_layer * (1.0 - 0.65 * age))   # cools to the tail
                    radius = max(1.0, 0.5 * ss.flame_width * (1.0 - ss.flame_taper * age) * (1.0 - 0.6 * core))
                    pygame.draw.circle(ov, (*rgb, alpha), (int(pts[k, 0]), int(pts[k, 1])), int(round(radius)))
        elif ss.particles:
            for k in range(m):
                rgba = (*color, int(255 * min(max(alpha_at(k), 0.0), 1.0)))
                px, py = int(pts[k, 0]), int(pts[k, 1])
                if k < m // 2:
                    pygame.draw.circle(ov, rgba, (px, py), 2)
                else:
                    ov.fill(rgba, (px, py, 1, 1))
        else:                                         # line: one segment per pair of points
            for k in range(m - 1):
                rgba = (*color, int(255 * min(max(alpha_at(k), 0.0), 1.0)))
                pygame.draw.line(ov, rgba, (pts[k, 0], pts[k, 1]), (pts[k + 1, 0], pts[k + 1, 1]), 1)
        return box if dirty is None else dirty.union(box)

    def screen_size(self) -> tuple[int, int]:
        return self._screen_size

    def _render_race(self, surface: Surface, hud: bool = False) -> None:
        """Curved slipstream streaks (or particles) behind drafting cars + a small position tower."""
        if not self.race_on:
            return
        ss = self.race.slipstream
        if not hud:
            self._frame += 1
            self._screen_size = surface.get_size()
        if ss.enabled and not hud:
            cap = ss.trail_length
            step = ss.particle_step if ss.particles else 1
            spread = ss.particle_spread if ss.particles else 0.0
            side_px = self.width * ss.side_offset
            dirty = None
            for i in np.flatnonzero(self.draft_level > DRAFT_VISIBLE):
                if not self.players.active[i]:
                    continue
                lvl = float(self.draft_level[i])
                # streak length follows the draft level: full trail at max slipstream
                n = max(2, int(math.ceil(lvl * cap)))
                color = ss.color_decay if self._draft_decay[i] else ss.color_gain
                for side in (-1.0, 1.0):               # two streaks, either side of the rear
                    m = trail_points(
                        self._trail_x, self._trail_y, self._trail_a, self._trail_head, self._trail_count,
                        i, n, self.height * STREAK_REAR_CARS, side * side_px, step, spread, self._frame, self._trail_buf,
                    )
                    if m < 2:
                        continue
                    pts = self._trail_buf[:m]
                    frac = (n - 2) / max(cap - 2, 1)          # current streak length vs the maximum
                    tail_alpha = ss.alpha_tail_short + (ss.alpha_tail_long - ss.alpha_tail_short) * min(max(frac, 0.0), 1.0)
                    dirty = self._draw_trail(pts, color, ss, tail_alpha, dirty)

            if dirty is not None:                      # composite the translucent flame layer once
                dirty = dirty.clip(surface.get_rect())
                surface.blit(self._trail_overlay, dirty, dirty)
                self._trail_overlay.fill((0, 0, 0, 0), dirty)

        if hud and self.font is not None and self.race_order is not None:
            total = len(self.race_order)
            tower = self.race_order[:5]
            rows = []
            lead = int(self.race_order[0])
            done = max(0, int(self.laps_done[lead]))
            if math.isfinite(self.race_laps):
                head = f"Laps {min(done, int(self.race_laps))}/{int(self.race_laps)}"
            else:
                head = f"Laps {done}"
            if self.race_end_in is not None:
                head += f"  ends {self.race_end_in:.1f}s"
            rows.append(self.font.render(head, True, (255, 215, 0)))
            for row, idx in enumerate(tower):
                name = self.label(int(idx))
                tag = f"{row + 1} {int(idx)}" + (f" {name}" if name is not None else "")
                tag += " FIN" if self.race_done[idx] else (" *" if self.draft_level[idx] > DRAFT_TOWER_MARK else "")
                rows.append(self.font.render(tag, True, (230, 230, 230)))
            if rows:
                pad = 6
                w = max(r.get_width() for r in rows) + 2 * pad
                rect = pygame.Rect(surface.get_width() - w - 4, 4, w, len(rows) * 16 + 2 * pad - 4)
                draw_panel(surface, rect, self.config.overlay_color, self.config.overlay_alpha)
                for row, txt in enumerate(rows):
                    surface.blit(txt, (rect.x + pad, rect.y + pad + row * 16))
            if self.usr_engaged:
                me = self.total - 1
                pos = int(np.flatnonzero(self.race_order == me)[0]) + 1
                col = ss.color_decay if self._draft_decay[me] else ss.color_gain
                txt = self.font.render(f"P{pos}/{total}  Draft {self.draft_level[me] * 100:3.0f}%", True, col)
                rect = pygame.Rect(4, surface.get_height() - (txt.get_height() + 12), txt.get_width() + 12, txt.get_height() + 8)
                draw_panel(surface, rect, self.config.overlay_color, self.config.overlay_alpha)
                surface.blit(txt, (rect.x + 6, rect.y + 4))

    def _render_debugging(self, surface: Surface, index: int) -> None:
        """
        Render debug information showing what surface the car is currently on.
        Displays surface type and color coding on the screen.
        """
        
        if surface is None or self.font is None:
            return
        
        x, y, vel_lin, angle = tuple(array[index].item() for array in [self.x, self.y, self.vel_lin, self.angle])
        tile_curr = self.tile[index]
        tile_next = self.grid.continuity[tile_curr]
        
        # Check what surface the car is on
        surfaces_touching = []
        surface_colors = {
            'road': COLOR_CURB_SEC, # (90, 90, 90),      # Dark gray
            'curb': COLOR_CURB_SEC, # (220, 35, 35),     # Red
            'grass': COLOR_CURB_SEC, # (30, 110, 30),    # Green
            'gravel': COLOR_CURB_SEC, # (137, 81, 41),   # Brown
            'finish': COLOR_CURB_SEC, # (137, 81, 41),   # Brown
        }
        
        if self.on_road[index]: surfaces_touching.append('road')
        if self.on_curb[index]: surfaces_touching.append('curb')
        if self.on_grass[index]: surfaces_touching.append('grass')
        if self.on_gravel[index]: surfaces_touching.append('gravel')
        if self.on_finish[index]: surfaces_touching.append('finish')
        
        # Create status text
        if surfaces_touching:
            status_text = f"Surface: {', '.join(surfaces_touching).upper()}"
            status_color = surface_colors.get(surfaces_touching[0], (255, 255, 255))
        else:
            status_text = "Surface: OFF TRACK"
            status_color = (255, 255, 255)  # Light red for off-track

        # Text lines are collected first so one translucent panel can sit behind all of them
        lines: list[tuple[Surface, tuple[int, int]]] = []
        lines.append((self.font.render(status_text, True, status_color), (10, 10)))
        
        # Render position information
        phase_curr = -(angle - tile_curr.orientation)
        phase_curr = (phase_curr + 180) % 360 - 180 # wrap to [-180, 180]
        phase_next = -(angle - tile_next.orientation)
        phase_next = (phase_next + 180) % 360 - 180 
        pos_text = (
            f"Pos: ({x:.1f}, {y:.1f}) "
            f"Vel: {vel_lin:.1f} "
            f"Angle: {angle:.1f} "
            f"Rot: {tile_curr.orientation:.1f} "
            f"PhC: {phase_curr:.1f} "
            f"PhN: {phase_next:.1f} "
        )
        lines.append((self.font.render(pos_text, True, (200, 200, 200)), (10, 28)))

        # Render car status text
        score_text = (
            f"Checks: {self.checks_done[index]} "
            f"Laps: {self.laps_done[index]} "
            f"Crashes: {self.crashes_done[index]} "
        )
        lines.append((self.font.render(score_text, True, (200, 200, 200)), (10, 46)))

        # Render player status text
        score_text = (
            f"Hiatus: {self.hiatus[index]} "
            f"Reward: {self.players.get_reward()[index]:.1f} "
            f"Lives: {self.players.lives[index]} "
            f"Score: {self.players.true_scores[index]} "
        )
        lines.append((self.font.render(score_text, True, (200, 200, 200)), (10, 64)))

        # Render player status text
        score_text = (
            f"Brake: {self.brake_steps[index]} "
            f"Turn: {self.turn_steps[index]} "
        )
        lines.append((self.font.render(score_text, True, (200, 200, 200)), (10, 82)))

        # Translucent backdrop (cars on the track stay visible), then the text
        pad = 6
        panel = pygame.Rect(
            10 - pad, 10 - pad,
            max(t.get_width() for t, _ in lines) + 2 * pad,
            max(p[1] + t.get_height() for t, p in lines) - 10 + 2 * pad,
        )
        draw_panel(surface, panel, self.config.overlay_color, self.config.overlay_alpha)
        for text_surface, pos in lines:
            surface.blit(text_surface, pos)

    def _render_lidar(self, surface: Surface, car_idx: int, car_x: float, car_y: float) -> None:
        """
        Render the LiDAR system for a specific car as rays cast from its center.
        
        Args:
            surface: pygame Surface to draw on
            car_idx: Index of the car to render LiDAR for
            car_x: Car's x position
            car_y: Car's y position
        """
        if not self.use_lidar or self.lidar_distances is None:
            return
        
        from .constants import COLOR_LIDAR, COLOR_ROAD, COLOR_CURB_PRI, COLOR_GRASS, COLOR_GRAVEL
        
        # Surface type colors for different LiDAR hit types
        surface_colors = {
            SURFACE_ROAD:   (255, 255, 255), # COLOR_ROAD,
            SURFACE_CURB:   COLOR_CURB_PRI,
            SURFACE_GRASS:  COLOR_GRASS,
            SURFACE_GRAVEL: COLOR_GRAVEL,
            SURFACE_VOID:   COLOR_LIDAR,
            SURFACE_CAR:    (255, 80, 220),  # other car (race.lidar_detect_cars)
        }
        
        car_angle = self.angle[car_idx]
        distances = self.lidar_distances[car_idx]  # (lidar_angles,)
        surface_hits = self.lidar_surface_hits[car_idx]  # (lidar_angles,)
        
        # Draw each LiDAR beam
        for beam_idx in range(self.lidar_angles):
            # Calculate the absolute angle of this beam in world space
            beam_offset = self._lidar_beam_offsets[beam_idx]
            absolute_angle = car_angle + beam_offset
            angle_rad = np.radians(absolute_angle)
            
            # Calculate the distance this beam traveled (normalized to [0, 1])
            norm_distance = distances[beam_idx]
            pixel_distance = norm_distance * self.lidar_max_dist
            
            # Calculate endpoint of the beam
            # Convention: angle 0 = north (up), includes clockwise
            # x += sin(a), y -= cos(a)  (pygame y increases downward)
            end_x = car_x - np.sin(angle_rad) * pixel_distance
            end_y = car_y - np.cos(angle_rad) * pixel_distance
            
            # Get surface hit color
            surface_type = surface_hits[beam_idx]
            color = surface_colors.get(surface_type, COLOR_LIDAR)
            
            # Draw the beam as a line from car center to hit point
            pygame.draw.line(
                surface, color,
                (car_x, car_y), (end_x, end_y),
                width=1
            )
            
            # Draw a small circle at the hit point to mark where the beam stopped
            pygame.draw.circle(surface, color, (int(end_x), int(end_y)), radius=2)
        
        # Draw a circle around the car center to show the LiDAR origin
        pygame.draw.circle(surface, COLOR_LIDAR, (int(car_x), int(car_y)), radius=3, width=2)

    # ======================================================================
    # Reset / restart
    # ======================================================================

    def reset(self, total: int):
        if not self.grid.initialized:
            raise RuntimeError("Grid must be initialized.")

        self.grid_mapping    = {tile.position: tile for tile in self.grid._cells.flatten()}
        self.reverse_mapping = {tile: position for position, tile in self.grid_mapping.items()}

        self._cache_tile_table()

        self.init_x, self.init_y = self.grid.start_cell.center_abs
        self.init_angle = self.grid.start_cell.rotation
        self.total = total

        self.x           = np.full((self.total,), self.init_x, dtype=NP_FLOAT)
        self.y           = np.full((self.total,), self.init_y, dtype=NP_FLOAT)
        self.disp_x      = np.full((self.total,), 0.0, dtype=NP_FLOAT)
        self.disp_y      = np.full((self.total,), 0.0, dtype=NP_FLOAT)
        self.vel_lin     = np.full((self.total,), 0.0, dtype=NP_FLOAT)
        self.vel_ang     = np.full((self.total,), 0.0, dtype=NP_FLOAT)
        self.angle       = np.full((self.total,), self.init_angle, dtype=NP_FLOAT)
        self.crashed     = np.full((self.total,), False, dtype=bool)
        self.finished    = np.full((self.total,), False, dtype=bool)
        self.started     = np.full((self.total,), False, dtype=bool)
        self.laps_done   = np.full((self.total,), 0, dtype=NP_INT)
        self.lap_pending = np.full((self.total,), False, dtype=bool)   # entered the start tile, line not crossed yet
        self.race_done = np.full((self.total,), False, dtype=bool)     # finished the race distance
        self.finish_order: list[int] = []
        self.finish_frame = np.full((self.total,), -1, dtype=NP_INT)
        self.race_frame = 0
        self.race_end_in: float | None = None
        self.checks_done = np.full((self.total,), 0, dtype=NP_INT)
        self.crashes_done= np.full((self.total,), 0, dtype=NP_INT)
        self.tile_crash = np.full((self.total,), False, dtype=bool)
        self.hiatus      = np.full((self.total,), 0, dtype=NP_INT)
        self.tile_stag_steps = np.full((self.total,), 0, dtype=NP_INT)
        self.brake_steps = np.full((self.total,), 0, dtype=NP_INT)
        self.turn_steps  = np.full((self.total,), 0, dtype=NP_INT)
        self.straight_steps = np.full((self.total,), 0, dtype=NP_INT)
        self.on_road     = np.full((self.total,), False, dtype=bool)
        self.on_curb     = np.full((self.total,), False, dtype=bool)
        self.on_grass    = np.full((self.total,), False, dtype=bool)
        self.on_gravel   = np.full((self.total,), False, dtype=bool)
        self.on_track    = np.full((self.total,), True, dtype=bool)
        self.on_finish   = np.full((self.total,), False, dtype=bool)
        self.tile        = np.full((self.total,), self.grid.start_cell, dtype=object)
        self.tile_idx    = np.full((self.total,), self._start_id, dtype=NP_INT)
        self._vel_norm   = np.zeros((self.total,), dtype=NP_FLOAT)
        self._friction_prev = np.full_like(self.vel_lin, self.config.friction_road)
        self._slip_prev     = np.full_like(self.vel_lin, self.config.slip_road)
        self._signal_buf = {
            **{k: np.zeros((self.total,), dtype=np.bool_) for k in (
                'checked', 'finished', 'tile_stag', 'on_road', 'moved_forward', 'corner_curr',
                'corner_next', 'straight_curr', 'turning', 'braking', 'straight_aligned', 'brake_check',
            )},
            **{k: np.zeros((self.total,), dtype=NP_FLOAT) for k in ('phase_align', 'turn_align')},
        }
        self.ranking     = None
        self.friction    = np.full_like(self.vel_lin, self.config.friction_road)
        self.handling    = np.full_like(self.vel_lin, self.config.handling_road)
        self.slip        = np.full_like(self.vel_lin, self.config.slip_road)
        self.used_brake  = np.full((self.total,), False, dtype=bool)

        # LiDAR arrays (resized for new total)
        self.corner_entry_speed = np.full((self.total,), 0.0, dtype=NP_FLOAT)

        self.lidar_distances    = np.ones((self.total, self.lidar_angles), dtype=NP_FLOAT)
        self.lidar_surface_hits = np.full((self.total, self.lidar_angles), SURFACE_VOID, dtype=np.int8)
        self.surface            = np.zeros((self.total,), dtype=np.int8)

        # Race mode: reverse-layout starting grid behind / on the start tile
        self.draft_level = np.zeros((self.total,), dtype=NP_FLOAT)
        self._init_trail()
        self.bumped      = np.zeros((self.total,), dtype=bool)
        self.progress    = np.zeros((self.total,), dtype=NP_FLOAT)
        self.in_tile     = np.zeros((self.total,), dtype=NP_FLOAT)
        self._prev_rotation = np.zeros((self.total,), dtype=NP_FLOAT)
        self._prev_off   = np.zeros((self.total,), dtype=bool)
        self.shape_pos   = np.zeros((self.total,), dtype=NP_FLOAT)
        self.shape_neg   = np.zeros((self.total,), dtype=NP_FLOAT)
        self.target_speed = np.ones((self.total,), dtype=NP_FLOAT)   # safe speed ahead / max speed
        self._tr_pos     = np.zeros((self.total,), dtype=NP_FLOAT)
        self._tr_neg     = np.zeros((self.total,), dtype=NP_FLOAT)
        self._tr_edge    = np.ones((self.total,), dtype=NP_FLOAT)
        self.race_order  = None
        self.track       = TrackGeometry(self.grid)   # lap geometry also feeds the reward (progress / corners)
        if self.race.enabled:
            slots = StartingGrid(
                self.track, self.race.start_grid, self.width, self.height, self.race.max_racers
            ).layout(self.total)
            for i, slot in enumerate(slots):
                self.x[i], self.y[i], self.angle[i] = slot['x'], slot['y'], slot['angle']
                self.tile[i] = slot['tile']
                self.tile_idx[i] = self._tile_id[slot['tile']]
                # cars behind the start tile cross the line once before lap 1 starts
                self.laps_done[i] = slot['lap_offset']

        # (Re-)build cached mask numpy arrays from the freshly-built grid
        self._cache_lidar_arrays()
        if self.track is not None:
            self._cache_progress_table()

        self._setup_reward()
        self.players.reset(total)
        if self.race.enabled:
            self._update_race_order()   # needs players.active, so after the players reset

    def restart(self):
        # TODO: Maybe handle this under Numba wrapper
        restart_all  = self.crashed | self.finished
        restart_part = restart_all if not self.endless else self.crashed

        if restart_all.any():
            idx = np.flatnonzero(restart_part)
            if idx.size:
                # Respawn at the centre of the car's tile (or the start tile on a full restart)
                tid = np.full(idx.size, self._start_id, dtype=NP_INT) if self.full_restart else self.tile_idx[idx]
                self.x[idx] = self.t_center[tid, 0]
                self.y[idx] = self.t_center[tid, 1]
                ttype = self.t_type[tid]
                self.angle[idx] = self.t_rot[tid] + np.where(
                    ttype == Type.LEFT.value, 45.0, np.where(ttype == Type.RIGHT.value, -45.0, 0.0)
                )
                pend = idx[self.lap_pending[idx]]
                if pend.size:                          # respawn is past the line: count the armed lap
                    if not self.full_restart:
                        self.laps_done[pend] += 1
                    self.lap_pending[pend] = False
                self._update_regions()

            self.disp_x    [restart_part] = 0
            self.disp_y    [restart_part] = 0
            self.vel_lin   [restart_part] = 0
            self.vel_ang   [restart_part] = 0
            self.surface   [restart_part] = SURFACE_ROAD # assume road on respawn # TODO: Make it have a direct update method like the surface masks
            self.friction  [restart_part] = self.config.friction_road
            self.handling  [restart_part] = self.config.handling_road
            self.slip      [restart_part] = self.config.slip_road

            self.brake_steps[restart_part] = 0
            self.turn_steps [restart_part] = 0
            self.straight_steps[restart_part] = 0
            self.crashed    [restart_all]  = False
            self.started    [self.finished] = False
            self.finished   [restart_all]  = False

        self.players.restart()