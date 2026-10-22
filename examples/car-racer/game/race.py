"""Race helpers: track center lines, starting grid layout and a demo autopilot."""
from __future__ import annotations

import math
import warnings

import numpy as np
from numpy import ndarray as Array

from . import kernels
from .config import SlipstreamConfig, StartGridConfig
from .constants import DRAFT_TREND_EPS, TRAIL_RESET_CARS
from .functional import Type
from .grid import Grid
from .tile import Tile
from .util import wrap180

__all__ = [
    "TrackGeometry", "StartingGrid", "RaceControl", "Slipstream", "RandomSlipstream",
    "Autopilot", "heading_to_angle",
]


def heading_to_angle(dx: float, dy: float) -> float:
    """World direction (dx, dy) to car angle in degrees."""
    return math.degrees(math.atan2(-dx, -dy)) % 360


class TrackGeometry:
    """Driving line of every tile: entry midpoint -> center -> exit midpoint (length = tile size)."""

    def __init__(self, grid: Grid):
        self.grid = grid
        self.size = float(grid.cell_size)
        self.next: dict[Tile, Tile] = dict(grid.continuity)
        self.prev: dict[Tile, Tile] = {b: a for a, b in self.next.items()}

        # Lap index of every tile: start = 0, its successor = 1, ...
        self.lap_index: dict[Tile, int] = {}
        t = grid.start_cell
        for i in range(len(self.next)):
            self.lap_index[t] = i
            t = self.next[t]
        self.n_tiles = len(self.next)

        # (entry, center, exit) per tile
        self.lines: dict[Tile, tuple[Array, Array, Array]] = {}
        for t in self.next:
            exit_ = np.array(t.center_check_abs, dtype=float)
            entry = np.array(self.prev[t].center_check_abs, dtype=float)
            self.lines[t] = (entry, np.array(t.center_abs, dtype=float), exit_)

    def point(self, tile: Tile, dist: float, lateral: float = 0.0) -> tuple[float, float, float]:
        """(x, y, angle) at `dist` px from the tile entry, `lateral` > 0 is to the left of travel.

        `dist` may run into the next tiles (or the previous ones when negative).
        """
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
        left = np.array([u[1], -u[0]])           # screen coordinates, y down
        p = a + u * d + left * lateral
        ang = heading_to_angle(*u)
        # Blend the heading across corner tiles
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


class StartingGrid:
    """Places racers in staggered rows on the start tile, then on the tiles behind it."""

    def __init__(self, track: TrackGeometry, cfg: StartGridConfig, car_w: float, car_h: float,
                 max_racers: int):
        self.track, self.cfg = track, cfg
        self.car_w, self.car_h = car_w, car_h
        self.max_racers = max_racers

    def lanes(self, tile: Tile) -> int:
        if self.cfg.lanes is not None:
            return max(1, int(self.cfg.lanes))
        usable = tile.road_px * self.cfg.lane_spread + self.car_w * self.cfg.lane_gap
        return max(1, int(usable // (self.car_w * self.cfg.lane_gap)))

    def span(self, tile: Tile) -> tuple[float, float]:
        """(front, rear) distances back from the tile exit that cars may occupy.

        On the start tile the rear limit stays in front of the finish line.
        """
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
                # Distance back from the tile exit, odd lanes staggered further back
                back = front + (row + 0.5) * band + (self.cfg.stagger * band if lane % 2 else 0.0)
                back = min(back, rear)
                x, y, ang = self.track.point(tile, self.track.size - back, float(lat[lane]))
                slots.append(dict(x=x, y=y, angle=ang, tile=tile, lap_offset=lap_offset))
            tile = self.track.prev[tile]
            lap_offset -= 1                             # cars behind the start cross the line once before lap 1
        return slots


class RaceControl:
    """Race order and finishing. Reads and writes the race arrays stored on `Cars`."""

    def update_order(self, cars, recompute: bool = True) -> None:
        """Live positions from the distance driven; finishers keep their finishing order."""
        if recompute:
            cars._compute_progress()
        key = np.where(cars.race_active, cars.progress, -np.inf)
        for rank, i in enumerate(cars.finish_order):
            if key[i] > -np.inf:
                key[i] = 1.0e15 - rank
        cars.race_order = np.argsort(-key, kind="stable")
        cars.ranking = cars.race_order

    def update_ranking(self, cars) -> None:
        """Per-frame ranking: race order in race mode, otherwise by score until most cars have lapped."""
        if cars.race_on:
            self.update_order(cars, recompute=False)
        elif np.count_nonzero(cars.laps_done >= 1) <= 3:
            cars.ranking = np.argsort(-cars.players.scores)

    def mark_finished(self, cars, laps_target: float) -> int:
        """Flag cars that completed `laps_target` laps. Returns how many cars have finished."""
        newly = (cars.laps_done >= laps_target) & ~cars.race_done & cars.race_active
        if newly.any():
            for i in sorted(np.flatnonzero(newly), key=lambda j: -cars.progress[j]):
                cars.race_done[i] = True
                cars.finish_frame[i] = cars.race_frame
                cars.finish_order.append(int(i))
        return len(cars.finish_order)


class Slipstream:
    """Drafting behind other cars: tow level, wear-off flag and the pose trails drawn as streaks."""

    def __init__(self, config: SlipstreamConfig):
        self.config = config

    def update(self, cars) -> bool:
        """Advance the draft level of every car. Returns True when drafting is active this frame."""
        ss = self.config
        cone = bool(cars.race_on and ss.enabled)
        drafting = bool(cone or ss.random_enabled)
        if not drafting:
            return False
        prev_level = cars.draft_level
        if cone:
            cars._draft_natural = kernels.slipstream(
                cars.x, cars.y, cars.angle, np.clip(cars.vel_lin / cars.max_vel_lin, 0.0, 1.0),
                cars.race_active, cars._draft_natural, cars.height, ss,
            )
        if ss.random_enabled:
            cars.random_ss.step(cars)
            cars.draft_level = np.maximum(cars._draft_natural, cars.random_ss.level)
        else:
            cars.draft_level = cars._draft_natural
        # True while the tow wears off, unchanged inside the dead band
        cars.draft_decay = np.where(
            cars.draft_level < prev_level - DRAFT_TREND_EPS, True,
            np.where(cars.draft_level > prev_level + DRAFT_TREND_EPS, False, cars.draft_decay),
        )
        return True

    def record_trail(self, cars) -> None:
        """Push the current poses into the trail ring buffers."""
        if cars.race_on and self.config.enabled:
            kernels.trail_push(
                cars.trail_x, cars.trail_y, cars.trail_a, cars.trail_head, cars.trail_count,
                cars.x, cars.y, cars.angle, cars.race_active, cars.height * TRAIL_RESET_CARS,
            )


class RandomSlipstream:
    """Random training tows: each car gets `random_events` tows per life, starting when it
    reaches a tile `random_min_tiles` to `random_max_tiles` ahead. Works outside race mode."""

    def __init__(self, config: SlipstreamConfig):
        self.config = config
        self.allocate(1)

    def allocate(self, total: int) -> None:
        self.tile = np.full(total, -1, dtype=np.int64)      # tile index that triggers the next tow
        self.left = np.zeros(total, dtype=np.int64)         # frames left in the current tow
        self.events = np.zeros(total, dtype=np.int64)       # tows already given this life
        self.level = np.zeros(total, dtype=np.float64)      # current draft level, 0..1

    def schedule(self, cars, idx: Array, reset_events: bool = False) -> None:
        """Pick the next trigger tile for the cars in `idx`."""
        ss = self.config
        if not ss.random_enabled or idx.size == 0 or not hasattr(cars, "t_next"):
            return
        if reset_events:
            self.events[idx] = 0
            self.left[idx] = 0
            self.level[idx] = 0.0
        ahead = cars.rng.integers(ss.random_min_tiles, ss.random_max_tiles + 1, size=idx.size)
        tid = cars.tile_idx[idx].copy()
        for step in range(ss.random_max_tiles):
            tid = np.where(step < ahead, cars.t_next[tid], tid)
        self.tile[idx] = np.where(self.events[idx] < ss.random_events, tid, -1)

    def step(self, cars) -> None:
        """Advance the tows by one frame."""
        ss = self.config
        active = self.left > 0
        hit = (~active) & (self.tile >= 0) & (cars.tile_idx == self.tile)
        if hit.any():
            idx = np.flatnonzero(hit)
            self.left[idx] = ss.random_frames
            self.events[idx] += 1
            self.schedule(cars, idx)
        target = np.where(self.left > 0, ss.random_level, 0.0)
        rate = np.where(target > self.level, ss.ramp, ss.decay)
        self.level += (target - self.level) * rate
        self.left = np.maximum(self.left - 1, 0)


class Autopilot:
    """Pure-pursuit driver for demos: aims down the center line and slows for corners."""

    def __init__(self, cars, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.cars = cars
        n = cars.total
        self.skill = rng.uniform(0.86, 0.98, n)                     # target fraction of top speed
        self.lane = rng.uniform(-0.20, 0.20, n) * cars.grid.cell_size * 0.5
        self.closing = np.zeros(n, dtype=bool)

    def _pass_offset(self, i: int) -> float:
        """Sideways offset around a slower car ahead. Sets `closing` when it is near."""
        c = self.cars
        rad = np.radians(c.angle[i])
        fx, fy = -np.sin(rad), -np.cos(rad)
        dx, dy = c.x - c.x[i], c.y - c.y[i]
        ahead = dx * fx + dy * fy
        lat = dx * fy - dy * fx                      # > 0: other car on the left
        near = (ahead > 0) & (ahead < 3.0 * c.height) & (np.abs(lat) < 1.3 * c.width) & c.players.active
        near[i] = False
        self.closing[i] = near.any() and ahead[near].min() < 1.6 * c.height
        if not near.any():
            return 0.0
        j = np.flatnonzero(near)[np.argmin(ahead[near])]
        side = -1.0 if lat[j] > 0 else 1.0
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

            # Slow down for corners
            nxt = tr.next[tile]
            corner_here = tile.type.name in ('LEFT', 'RIGHT')
            corner_next = nxt.type.name in ('LEFT', 'RIGHT')
            target = self.skill[i]
            if corner_here or (corner_next and d > 0.35 * tr.size):
                target = min(target, 0.52)
            # Never lift while in a tow
            if c.draft_level[i] > 0.1:
                target = 1.0 + 0.12
            if self.closing[i] and speed > 0.4 * c.max_vel_lin:     # never at a standstill, that costs lives
                target = min(target, speed / c.max_vel_lin - 0.02)
            out[i, 0] = np.clip((target - speed / c.max_vel_lin) * 6.0, -1, 1)
        return out