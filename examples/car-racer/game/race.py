"""
Race-mode helpers (CPU friendly, numpy only).

* TrackGeometry  - centre-line of every tile, used to place the grid and measure progress
* StartingGrid   - staggered grid laid out *backwards* from the start tile
* slipstream()   - pairwise cone test -> per-car draft level in [0, 1]
* collide()      - two-circle car bodies, positional + impulse response

Angle convention (matches Cars): heading vector = (-sin a, -cos a), a in degrees,
increasing `a` turns the car left on screen.
"""
from __future__ import annotations

import math
import warnings
import numpy as np

from numba import njit, prange
from numpy import ndarray as Array

from .config import StartGridConfig, SlipstreamConfig, CollisionConfig
from .functional import Type
from .grid import Grid
from .tile import Tile
from .util import grid


def heading_to_angle(dx: float, dy: float) -> float:
    """World direction (dx, dy) -> car angle in degrees."""
    return math.degrees(math.atan2(-dx, -dy)) % 360


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


# ======================================================================
# Track geometry
# ======================================================================

class TrackGeometry:
    """
    Every tile's driving line is the polyline entry-mid -> centre -> exit-mid, which is
    exactly the shape of the L-shaped road on corner tiles (a straight tile is the
    degenerate case).  Length of every tile line == tile size.
    """

    def __init__(self, grid: Grid):
        self.grid = grid
        self.size = float(grid.cell_size)
        self.next: dict[Tile, Tile] = dict(grid.continuity)
        self.prev: dict[Tile, Tile] = {b: a for a, b in self.next.items()}

        # lap index of every tile: start = 0, its successor = 1, ...
        self.lap_index: dict[Tile, int] = {}
        t = grid.start_cell
        for i in range(len(self.next)):
            self.lap_index[t] = i
            t = self.next[t]
        self.n_tiles = len(self.next)

        # (entry, centre, exit) per tile
        self.lines: dict[Tile, tuple[Array, Array, Array]] = {}
        for t in self.next:
            exit_ = np.array(t.center_check_abs, dtype=float)
            entry = np.array(self.prev[t].center_check_abs, dtype=float)  # exit of the tile before
            self.lines[t] = (entry, np.array(t.center_abs, dtype=float), exit_)

    # ---- queries --------------------------------------------------------
    def point(self, tile: Tile, dist: float, lateral: float = 0.0) -> tuple[float, float, float]:
        """(x, y, angle) at `dist` px from the tile entry; lateral > 0 is to the car's left.
        `dist` may exceed the tile size, in which case it continues into the next tile(s)."""
        while dist > self.size:
            dist -= self.size
            tile = self.next[tile]
        while dist < 0:
            tile = self.prev[tile]
            dist += self.size
        e, c, x = self.lines[tile]
        half = self.size / 2
        a, b, d = (e, c, dist) if dist <= half else (c, x, dist - half)
        v = b - a
        u = v / (np.hypot(*v) or 1.0)
        left = np.array([u[1], -u[0]])           # screen coords (y down): left of travel
        p = a + u * d + left * lateral
        ang = heading_to_angle(*u)
        # smooth the heading across the corner so cars on corner tiles are not stiffly square
        if tile.type in (Type.LEFT, Type.RIGHT):
            u1 = (c - e) / (np.hypot(*(c - e)) or 1.0)
            u2 = (x - c) / (np.hypot(*(x - c)) or 1.0)
            a1, a2 = heading_to_angle(*u1), heading_to_angle(*u2)
            w = np.clip((dist - (half - 0.25 * self.size)) / (0.5 * self.size), 0.0, 1.0)
            ang = (a1 + wrap180(a2 - a1) * w) % 360
        return float(p[0]), float(p[1]), float(ang)

    def dist_in_tile(self, tile: Tile, x: float, y: float) -> float:
        e, c, ex = self.lines[tile]
        p = np.array([x, y])
        best, best_d = 0.0, 1e18
        for a, b, off in ((e, c, 0.0), (c, ex, self.size / 2)):
            v = b - a
            L2 = float(v @ v) or 1.0
            t = float(np.clip((p - a) @ v / L2, 0.0, 1.0))
            d2 = float(np.sum((a + v * t - p) ** 2))
            if d2 < best_d:
                best_d, best = d2, off + t * (self.size / 2)
        return best

    def progress(self, tile: Tile, laps: int, x: float, y: float) -> float:
        """Distance (px) driven along the track, 0 = entry of the start tile."""
        return (laps * self.n_tiles + self.lap_index[tile]) * self.size + self.dist_in_tile(tile, x, y)


# ======================================================================
# Starting grid
# ======================================================================

class StartingGrid:
    """Places racers in staggered rows, start tile first, then the tiles behind it."""

    def __init__(self, track: TrackGeometry, cfg: StartGridConfig, car_w: float, car_h: float,
                 max_racers: int):
        self.track, self.cfg = track, cfg
        self.car_w, self.car_h = car_w, car_h
        self.max_racers = max_racers

    # ---- capacity -------------------------------------------------------
    def lanes(self, tile: Tile) -> int:
        if self.cfg.lanes is not None:
            return max(1, int(self.cfg.lanes))
        usable = tile.road_px * self.cfg.lane_spread + self.car_w * self.cfg.lane_gap
        return max(1, int(usable // (self.car_w * self.cfg.lane_gap)))

    def span(self, tile: Tile) -> tuple[float, float]:
        """(front, rear) distances back from the tile exit that cars may occupy.
        On the start tile the rear limit sits in front of the finish line, so the
        pole group lines up just past it and the next tile's group is behind it."""
        edge = self.track.size * self.cfg.edge_margin
        rear = self.track.size - edge
        if tile is self.track.grid.start_cell:
            rear = min(rear, self.track.size - tile.finish_line_back_offset - self.car_h)
        return edge, rear

    def usable_length(self, tile: Tile) -> float:
        front, rear = self.span(tile)
        return max(rear - front, 0.0)

    def tile_capacity(self, tile: Tile) -> int:
        rows = int(self.usable_length(tile) // (self.car_h * self.cfg.row_gap))
        return self.lanes(tile) * max(rows, 0)

    # ---- layout ---------------------------------------------------------
    def plan(self, total: int) -> list[int]:
        """Cars per tile (start tile first) for `total` racers."""
        if total > self.max_racers:
            raise ValueError(
                f"Race mode supports at most {self.max_racers} racers (race.max_racers), got {total}."
            )
        limit = max(1, min(self.cfg.max_grid_tiles, self.track.n_tiles // 2))
        wanted = list(self.cfg.cars_per_tile) if self.cfg.cars_per_tile else None

        counts, remaining, tile = [], total, self.track.grid.start_cell
        for k in range(limit):
            cap = self.tile_capacity(tile)
            want = cap if wanted is None else wanted[min(k, len(wanted) - 1)]
            if want > cap:
                warnings.warn(
                    f"start_grid.cars_per_tile[{k}]={want} does not fit on a {self.track.size:.0f}px tile "
                    f"with {self.car_h:.0f}px cars (row_gap={self.cfg.row_gap}); capped at {cap}."
                )
            take = max(0, min(want, cap, remaining))
            counts.append(take)
            remaining -= take
            tile = self.track.prev[tile]
            if remaining == 0:
                break
        if remaining > 0:
            raise ValueError(
                f"Cannot place {total} racers within max_grid_tiles={limit} "
                f"(placed {total - remaining}). Raise max_grid_tiles / cars_per_tile or lower the racer count."
            )
        return counts

    def layout(self, total: int) -> list[dict]:
        """One slot per racer (index 0 = pole): dict(x, y, angle, tile, lap_offset)."""
        slots = []
        tile = self.track.grid.start_cell
        lap_offset = 0
        for n in self.plan(total):
            lanes = min(self.lanes(tile), max(n, 1))
            rows = math.ceil(n / lanes)
            front, rear = self.span(tile)
            band = self.usable_length(tile) / rows
            spread = tile.road_px * self.cfg.lane_spread
            lat = np.linspace(spread / 2, -spread / 2, lanes) if lanes > 1 else np.zeros(1)
            for i in range(n):
                row, lane = divmod(i, lanes)
                # distance back from the tile exit, odd lanes staggered further back
                back = front + (row + 0.5) * band + (self.cfg.stagger * band if lane % 2 else 0.0)
                back = min(back, rear)
                x, y, ang = self.track.point(tile, self.track.size - back, float(lat[lane]))
                slots.append(dict(x=x, y=y, angle=ang, tile=tile, lap_offset=lap_offset))
            tile = self.track.prev[tile]
            lap_offset -= 1   # the tile behind the start must cross the line once before lap 1
        return slots


# ======================================================================
# Numba-accelerated helpers (same structure as backtest.trader / agent)
# ======================================================================

@njit(parallel=True, fastmath=True, cache=True)
def _slipstream(
    x: Array, y: Array, angle: Array, vel_norm: Array, active: Array, level: Array,
    out: Array,
    car_len: float, rng: float, cone: float, max_heading_diff: float, min_leader_speed: float,
    distance_falloff: float, angle_falloff: float, ramp: float, decay: float,
):
    """
    Per-follower draft update.  Every follower is independent, so numba
    parallelises across followers; the leader scan is a serial inner loop.
    """
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
    """
    Per-car contact response.  Each car accumulates the push-out, impulse and yaw
    torque from every neighbour (2 circles per car -> up to 4 circle pairs per
    neighbour), reading only pre-collision state so iterations stay independent.
    """
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

                    # How many of the 4 circle pairs overlap (impulse is shared between them)
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
        # disp_* hold the *negated* displacement of the car (position -= disp)
        disp_x[c] -= lat_x * lateral_transfer
        disp_y[c] -= lat_y * lateral_transfer
        if touched[c]:
            vel_ang[c] += dang[c]


# ======================================================================
# Slipstream
# ======================================================================

def slipstream(
    x: Array, y: Array, angle: Array, vel_norm: Array, active: Array,
    level: Array, car_len: float, cfg: SlipstreamConfig,
) -> Array:
    """
    Returns the new draft level (N,) in [0, 1].

    For every follower i and leader j:
        ahead  = (pos_j - pos_i) . heading_i            (leader must be in front)
        theta  = angle between heading_i and the line to the leader
        s      = (1 - dist/range)^a * (1 - theta/cone)^b * leader_speed
    The strongest wake wins, then the level eases towards it (ramp up / decay).
    """
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


# ======================================================================
# Slipstream trail (render helpers)
# ======================================================================

# Slipstream-trail constants. Module-level on purpose: numba freezes them at compile time (free
# at runtime), and they stay in this file so cache=True invalidates correctly when they change.
# Large primes used to mix (car, trail age, time) into a cheap, deterministic jitter hash.
# Same idea as spatial-hash primes; no RNG state, so it is safe inside prange and allocation free.
_HASH_CAR  = 73856093
_HASH_AGE  = 19349663
_HASH_TIME = 83492791
_HASH_MIX  = 40503        # odd 16-bit multiplier: decorrelates the 2nd jitter axis from the 1st
_HASH_MASK = 0xFFFF       # keep 16 bits -> value in [0, 65535]
_HASH_HALF = 32768.0      # 2**15: maps [0, 65535] to [-1, 1) after "/ _HASH_HALF - 1"
_JITTER_HOLD_FRAMES = 3   # particle jitter changes every N frames (shimmer, not per-frame noise)
_MIN_SPAN = 1             # guards the age division when only one point is drawn


@njit(parallel=True, fastmath=True, cache=True)
def _trail_push(
    tx: Array, ty: Array, ta: Array, head: Array, count: Array,
    x: Array, y: Array, angle: Array, active: Array, jump: float,
):
    """Append the current pose of every car to its ring buffer (O(N)).
    A teleport (respawn / reset) larger than `jump` px clears that car's history."""
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
    """Fill `out` (>= n, 2) with the streak polyline of car `c` (newest -> oldest), offset
    sideways by `side` px and behind the car by `rear` px along each stored heading.
    With spread > 0 points get a cheap hash jitter growing with age (particles).
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
            age = k / max(n - 1, _MIN_SPAN)   # 0 at the car, 1 at the tail: jitter grows with age
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


# ======================================================================
# Collisions
# ======================================================================

def collide(
    x: Array, y: Array, angle: Array, vel_lin: Array, vel_ang: Array,
    disp_x: Array, disp_y: Array, active: Array,
    car_w: float, car_h: float, max_vel: float, cfg: CollisionConfig,
) -> Array:
    """
    Resolves overlaps in place; returns a bool (N,) mask of cars that were hit this frame.

    Each car = 2 circles (front / rear).  Response (equal masses):
      * positional push-out along the contact normal
      * impulse along the normal  -> change of forward speed + a sideways slide (disp_x/y)
      * off-centre contact        -> small yaw kick on vel_ang
    """
    N = x.shape[0]
    touched = np.zeros(N, dtype=np.bool_)
    if N < 2:
        return touched

    # Broad phase: two cars can only touch when their centres are closer than one car length
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


# ======================================================================
# Demo / manual-test driver
# ======================================================================

class Autopilot:
    """Tiny pure-pursuit driver: aim a little way down the centre line, slow for corners."""

    def __init__(self, cars, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.cars = cars
        n = cars.total
        self.skill = rng.uniform(0.86, 0.98, n)                    # fraction of top speed it aims for
        self.lane = rng.uniform(-0.20, 0.20, n) * cars.grid.cell_size * 0.5
        self.closing = np.zeros(n, dtype=bool)

    def _pass_offset(self, i: int) -> float:
        """Pull out sideways when a slower car is right ahead (so a tow can become an overtake),
        and lift a little when closing in too fast."""
        c = self.cars
        rad = np.radians(c.angle[i])
        fx, fy = -np.sin(rad), -np.cos(rad)
        dx, dy = c.x - c.x[i], c.y - c.y[i]
        ahead = dx * fx + dy * fy
        lat = dx * fy - dy * fx                      # > 0: other car is to my left
        near = (ahead > 0) & (ahead < 3.0 * c.height) & (np.abs(lat) < 1.3 * c.width) & c.players.active
        near[i] = False
        self.closing[i] = near.any() and ahead[near].min() < 1.6 * c.height
        if not near.any():
            return 0.0
        j = np.flatnonzero(near)[np.argmin(ahead[near])]
        side = -1.0 if lat[j] > 0 else 1.0           # go round the other side
        return side * 0.30 * c.grid.cell_size * 0.5 * 0.9

    def act(self) -> np.ndarray:
        c, tr = self.cars, self.cars.track
        out = np.zeros((c.total, 2))
        for i in range(c.total):
            tile = c.tile[i]
            d = tr.dist_in_tile(tile, c.x[i], c.y[i])
            speed = c.vel_lin[i]
            look = max(0.40 * tr.size, 9.0 * speed)
            lane = self.lane[i] + self._pass_offset(i)
            tx, ty, _ = tr.point(tile, d + look, lane)
            err = wrap180(heading_to_angle(tx - c.x[i], ty - c.y[i]) - c.angle[i])
            out[i, 1] = np.clip(0.05 * err - 0.35 * c.vel_ang[i], -1, 1)

            # Corner ahead within the braking window?
            nxt = tr.next[tile]
            corner_here = tile.type.name in ('LEFT', 'RIGHT')
            corner_next = nxt.type.name in ('LEFT', 'RIGHT')
            target = self.skill[i]
            if corner_here or (corner_next and d > 0.35 * tr.size):
                target = min(target, 0.52)
            # never lift while in a tow
            if c.draft_level[i] > 0.1:
                target = 1.0 + 0.12
            if self.closing[i] and speed > 0.4 * c.max_vel_lin:   # (never at a standstill: that costs lives)
                target = min(target, speed / c.max_vel_lin - 0.02)   # feather the throttle
            out[i, 0] = np.clip((target - speed / c.max_vel_lin) * 6.0, -1, 1)
        return out