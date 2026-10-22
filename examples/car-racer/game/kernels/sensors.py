"""Sensor kernels: surface lookup, LiDAR and the observation vector."""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from numpy import ndarray as Array

from ..constants import (
    L_BOUND, L_CURB, L_FINISH, L_GRASS, L_GRAVEL, L_ROAD, L_TRACK,
)
from ..util import grid, wrap180

__all__ = [
    "pack_layers", "update_surface", "update_lidar", "lidar_cars", "update_regions", "tile_beam", "get_state",
]


def pack_layers(
    road: Array, curb: Array, grass: Array, gravel: Array,
    track: Array, boundary: Array, finish: Array,
) -> Array:
    """Pack seven (W, H) bool maps into one C-contiguous uint8 (W, H) map."""
    layers = (
        road.astype(np.uint8) * L_ROAD | curb.astype(np.uint8) * L_CURB
        | grass.astype(np.uint8) * L_GRASS | gravel.astype(np.uint8) * L_GRAVEL
        | track.astype(np.uint8) * L_TRACK | boundary.astype(np.uint8) * L_BOUND
        | finish.astype(np.uint8) * L_FINISH
    )
    return np.ascontiguousarray(layers, dtype=np.uint8)


@njit(parallel=True, fastmath=True, cache=True)
def update_surface(layers: Array, x: Array, y: Array, surface: Array):
    """Surface code under each car center. Priority: road > curb > grass > gravel > void."""
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
def update_lidar(
    layers: Array, x: Array, y: Array, angle: Array, surface: Array,
    beam_offsets: Array, n_steps: int, step: float, max_dist: float,
    distances: Array, surface_hits: Array,
):
    """Ray-march every beam of every car.

    A beam stops at the first sample that leaves the car's home surface (and all inner
    layers), leaves the track, or lies on a tile boundary.

    Writes in place:
        distances     (N, B) float64, hit distance normalised to [0, 1]
        surface_hits  (N, B) int8, surface code at the hit point (4 = no hit)
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
        for d in range(n_steps):
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


@njit(parallel=True, fastmath=True, cache=True)
def lidar_cars(
    x: Array, y: Array, angle: Array, active: Array, beam_offsets: Array,
    max_dist: float, car_w: float, car_h: float, radius_scale: float,
    distances: Array, surface_hits: Array,
):
    """Shorten LiDAR beams that hit another active car.

    Each car is the two circles used by the collision model. The nearer of the
    wall hit and the car hit is kept, car hits are tagged SURFACE_CAR.
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
def update_regions(
    layers: Array, lut_off: Array, lut_n: Array, lut_id: Array, x: Array, y: Array, angle: Array,
    on_road: Array, on_curb: Array, on_grass: Array, on_gravel: Array,
    on_track: Array, on_finish: Array,
):
    """Surfaces touched by the car body.

    `lut_off[q, k]` holds the pixels of pygame's rotated car mask (0.01 degree steps,
    identical masks shared through `lut_id`) as offsets from the car center.
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


@njit(fastmath=True, cache=True)
def tile_beam(px, py, dx, dy, x0, x1, y0, y1):
    """Distance along (dx, dy) from (px, py) to the far boundary of the tile [x0,x1]x[y0,y1].

    Outside the tile one of two opposite beams goes negative and the other exceeds the diagonal.
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
def get_state(
    state: Array, x: Array, y: Array, angle: Array, vel_lin: Array, vel_ang: Array,
    tile_idx: Array, t_orient: Array, t_next: Array, t_check: Array, t_center: Array,
    lidar_distances: Array, surface: Array,
    half: float, tile_size: float, max_vel_lin: float, max_vel_ang: float,
    n_tiles: int, use_tile_lidar: bool, use_surface: bool, use_lidar: bool,
):
    """Fill the observation rows (features roughly in [-1, 1]):
        [dist x n_tiles, phase x n_tiles, vel_lin, vel_ang, heading,
         tile_lidar x4 (optional), surface (optional), lidar beams (optional)]

    dist_k     1 - distance to the exit midpoint of the k-th tile ahead / (half * (k + 1)), clipped
    phase_k    wrapped heading difference to the k-th tile's exit orientation / 180
    tile_lidar 4 beams at heading +0/+90/+180/+270 to the current tile boundary, / tile diagonal,
               mapped to [-1, 1] and not clipped
    """
    total, indices = grid((x.shape[0],))
    beams = lidar_distances.shape[1]
    base = 2 * n_tiles + 3
    diag = tile_size * math.sqrt(2.0)
    hs = tile_size * 0.5
    for i in prange(total):
        c  = indices[i]
        t  = tile_idx[c]
        # Distance to, and heading offset from, each upcoming tile exit
        for k in range(n_tiles):
            state[c, k] = min(max(1.0 - math.hypot(t_check[t, 0] - x[c], t_check[t, 1] - y[c]) / (half * (k + 1.0)), -1.0), 1.0)
            state[c, n_tiles + k] = wrap180(t_orient[t] - angle[c]) / 180.0
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
                d = tile_beam(x[c], y[c], -math.sin(a), -math.cos(a), cx - hs, cx + hs, cy - hs, cy + hs)
                state[c, col + k] = (d / diag) * 2.0 - 1.0
            col += 4
        if use_surface:
            # Surface code scaled to [-1, 1]
            state[c, col] = (float(surface[c]) / 4.0) * 2.0 - 1.0
            col += 1
        if use_lidar:
            for b in range(beams):
                state[c, col + b] = lidar_distances[c, b] * 2.0 - 1.0
