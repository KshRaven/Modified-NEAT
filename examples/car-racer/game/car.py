"""Car simulation: state arrays, physics update, sensors, rewards and race logic for all cars at once."""
from __future__ import annotations

from typing import Any

import numpy as np
import pygame
import pygame.surfarray
from numpy import ndarray as Array
from pygame import Surface # , Mask

from . import kernels
from .config import CarConfig, RaceConfig, RewardConfig, normalize_restrict_movement, overwrite_config
from .constants import (
    IMAGE_DIR, NP_FLOAT, NP_INT, SURFACE_ROAD, SURFACE_VOID,
)
from .functional import Position, Type
from .grid import Grid
from .player import Players
from .race import RaceControl, RandomSlipstream, Slipstream, StartingGrid, TrackGeometry
from .reward import RewardShaper
from .sprite import export_car_images, make_car
from .tile import Tile

__all__ = ["Cars"]


class Cars:
    """All cars of an episode, stored as flat arrays indexed by car."""

    def __init__(
        self,
        grid: Grid, players: Players, config: CarConfig | None = None,
        race: RaceConfig | None = None, reward: RewardConfig | None = None,
        **params: dict[str, Any],
    ):
        if config is None:
            config = CarConfig()
        overwrite_config(config, params)
        self.config = config
        self.race = race if race is not None else RaceConfig()
        self.reward = reward if reward is not None else RewardConfig()

        self.grid = grid
        self.players = players
        self.grid_mapping: dict[Position, Tile] = {tile.position: tile for tile in grid._cells.flatten()}
        self.rng = np.random.default_rng()

        # Footprint and body mask lookup
        self.bot_car_img = make_car((150, 150, 160), config.scale)
        export_car_images(IMAGE_DIR)
        self._car_lut_off, self._car_lut_n, self._car_lut_id = self._build_car_lut(self.bot_car_img)
        self.width, self.height = self.bot_car_img.get_size()
 
        # Observation
        self.use_lidar:      bool = config.use_lidar
        self.use_tile_lidar: bool = config.use_tile_lidar
        self.use_surface:    bool = config.use_surface
        self.state_tiles:    int  = max(1, int(config.state_tiles))
 
        # Driving
        self.max_vel_lin:   float = config.max_linear_velocity
        self.max_vel_ang:   float = config.max_angular_velocity
        self.acc_lin:       float = config.linear_acceleration
        self.acc_ang:       float = config.angular_acceleration
        self.coeff_reverse: float = config.reverse_coeff
        self.coeff_brake:   float = config.brake_coeff
        self.coeff_checker: float = config.checker_coeff
        self.toggle_reverse: bool = config.toggle_reverse
 
        # Episode rules
        self.min_brake_steps:    int   = config.min_brake_steps
        self.min_straight_steps: int   = config.min_straight_steps
        self.straight_alignment_tolerance: float = config.straight_alignment_tolerance
        self.max_approach_angle: float = config.max_approach_angle
        self.wall_damping:       float = config.wall_damping
        self.wall_align:         float = config.wall_align
        self.full_restart:       bool  = config.full_restart
        self.discrete:           bool  = config.discrete
        self.endless:            bool  = config.endless
        self.restrict_movement = config.restrict_movement
 
        # Surface coefficients, rows: friction / handling / slip, columns: road / curb / grass / gravel
        self.coef: Array = np.array([
            [config.friction_road, config.friction_curb, config.friction_grass, config.friction_gravel],
            [config.handling_road, config.handling_curb, config.handling_grass, config.handling_gravel],
            [config.slip_road,     config.slip_curb,     config.slip_grass,     config.slip_gravel],
        ], dtype=NP_FLOAT)
 
        # LiDAR
        self.lidar_angles:   int   = config.lidar_angles
        self.lidar_max_dist: float = (
            config.lidar_max_dist if config.lidar_max_dist is not None else float(grid.cell_size) * 2.5
        )
        self.lidar_step: int = config.lidar_step
        self.lidar_beam_offsets: Array = np.linspace(0.0, 360.0, self.lidar_angles, endpoint=False, dtype=NP_FLOAT)
        self._lidar_steps: int = len(np.arange(self.lidar_step, self.lidar_max_dist + self.lidar_step, self.lidar_step))
        self.layers: Array | None = None            # packed surface map, built in reset()
 
        # Tile tables, built in reset()
        self.tiles: list[Tile] = []
        self._tile_id: dict[Tile, int] = {}

        # Episode state
        self.total = 1
        self.init_x, self.init_y, self.init_angle = 0.0, 0.0, 0.0
        self.track: TrackGeometry | None = None
        self.race_laps: float = float("inf")
        self.usr_engaged = False
        self.debug_car: int | None = None
        self.labels: list[str] | None = None
        self.shaper = RewardShaper(self.reward, config)
        self.random_ss = RandomSlipstream(self.race.slipstream)
        self.slipstream = Slipstream(self.race.slipstream)
        self.control = RaceControl()
        self._allocate(self.total)
 
    @property
    def restrict_movement(self) -> int:
        return self._restrict_movement
 
    @restrict_movement.setter
    def restrict_movement(self, value: bool | int) -> None:
        self._restrict_movement = normalize_restrict_movement(value)
 
    @property
    def center(self) -> tuple[Array, Array]:
        return self.x, self.y
 
    @property
    def race_on(self) -> bool:
        return self.race.enabled and self.track is not None
 
    @property
    def race_active(self) -> Array:
        """Cars taking part in the race: cars with lives left, plus the human car in manual mode."""
        active = np.asarray(self.players.active).copy()
        if self.usr_engaged and active.size:
            active[-1] = True
        return active
 
    @property
    def finish_line_laps(self) -> bool:
        """True if laps only count when the finish line is crossed."""
        return bool(self.race.finish_line_laps if self.race_on else self.config.finish_line_laps)
 
    # ------------------------------------------------------------------
    # State allocation
    # ------------------------------------------------------------------
 
    def _allocate(self, total: int, start_tile: Tile | None = None, start_id: int = 0) -> None:
        """(Re)create every per-car array for `total` cars."""
        self.total = total
        cfg = self.config
        n = (total,)
 
        def flags(value: bool = False) -> Array:
            return np.full(n, value, dtype=bool)
 
        def ints(value: int = 0) -> Array:
            return np.full(n, value, dtype=NP_INT)
 
        def reals(value: float = 0.0) -> Array:
            return np.full(n, value, dtype=NP_FLOAT)
 
        # Pose and motion
        self.x, self.y, self.angle = reals(self.init_x), reals(self.init_y), reals(self.init_angle)
        self.disp_x, self.disp_y = reals(), reals()
        self.vel_lin, self.vel_ang = reals(), reals()
        self._vel_norm = reals()
        self.friction = reals(cfg.friction_road)
        self.handling = reals(cfg.handling_road)
        self.slip     = reals(cfg.slip_road)
 
        # Surface
        self.on_road, self.on_curb, self.on_grass, self.on_gravel = flags(), flags(), flags(), flags()
        self.on_track, self.on_finish = flags(True), flags()
        self.surface = np.full(n, SURFACE_ROAD, dtype=np.int8)
 
        # Tile, lap and crash tracking
        self.tile = np.full(n, start_tile, dtype=object)
        self.tile_idx = ints(start_id)
        self.crashed, self.finished, self.started = flags(), flags(), flags()
        self.lap_pending, self.tile_crash, self.used_brake = flags(), flags(), flags()
        self.laps_done, self.checks_done, self.crashes_done = ints(), ints(), ints()
        self.hiatus, self.tile_stag_steps = ints(), ints()
        self.brake_steps, self.turn_steps, self.straight_steps = ints(), ints(), ints()
        self.corner_entry_speed = reals()
        self._signal_buf = {
            **{k: flags() for k in (
                "checked", "finished", "tile_stag", "on_road", "moved_forward", "corner_curr",
                "corner_next", "straight_curr", "turning", "braking", "straight_aligned", "brake_check",
            )},
            **{k: reals() for k in ("phase_align", "turn_align")},
        }
 
        # Race
        self.race_done = flags()
        self.finish_order: list[int] = []
        self.finish_frame = ints(-1)
        self.race_frame = 0
        self.race_end_in: float | None = None
        self.race_order: Array | None = None
        self.ranking: Array | None = None
        self.bumped = flags()
        self.progress, self.in_tile = reals(), reals()
 
        self.shaper.allocate(total)
 
        # LiDAR
        self.lidar_distances = np.ones((total, self.lidar_angles), dtype=NP_FLOAT)
        self.lidar_surface_hits = np.full((total, self.lidar_angles), SURFACE_VOID, dtype=np.int8)
 
        # Slipstream: pose history (ring buffers) for the streaks, and the random tow
        cap = self.race.slipstream.trail_length
        self.trail_x, self.trail_y, self.trail_a = (np.zeros((total, cap), dtype=NP_FLOAT) for _ in range(3))
        self.trail_head, self.trail_count = ints(), ints()
        self.draft_level = reals()
        self.draft_decay = flags()
        self._draft_natural = reals()
        self.random_ss.allocate(total)
 
    def set_labels(self, labels: list[str] | None) -> None:
        """Per-car display text. `None` shows the car index."""
        self.labels = None if labels is None else [str(label) for label in labels]
 
    def label(self, idx: int) -> str | None:
        if self.labels is not None and 0 <= idx < len(self.labels):
            return self.labels[idx]
        return None
 
    # ------------------------------------------------------------------
    # Lookup tables
    # ------------------------------------------------------------------
 
    @staticmethod
    def _build_car_lut(img: Surface, steps: int = 36000) -> tuple[Array, Array, Array]:
        """Rotated body mask for every `360 / steps` degrees, as pixel offsets from the car center.
 
        Identical masks are stored once. Returns `(offsets, counts, ids)`, used by `kernels.sensors.update_regions`.
        """
        unique: dict[bytes, int] = {}
        points: list[Array] = []
        ids = np.zeros((steps,), dtype=NP_INT)
        for q in range(steps):
            rot = pygame.transform.rotate(img, q * 360.0 / steps)
            w, h = rot.get_size()
            px, py = np.nonzero(pygame.surfarray.array_alpha(rot) > 127)
            p = np.stack([px - w // 2, py - h // 2], axis=-1).astype(NP_INT)
            key = p.tobytes()
            if key not in unique:
                unique[key] = len(points)
                points.append(p)
            ids[q] = unique[key]
        offsets = np.zeros((len(points), max(1, max(len(p) for p in points)), 2), dtype=NP_INT)
        counts = np.zeros((len(points),), dtype=NP_INT)
        for u, p in enumerate(points):
            offsets[u, :len(p)] = p
            counts[u] = len(p)
        return offsets, counts, ids
 
    def _cache_layers(self) -> None:
        """Pack the grid surface masks into one uint8 map for the kernels."""
        if not self.grid.initialized:
            return
        gw, gh = self.grid.width, self.grid.height
 
        def to_bool(surf) -> Array:
            if surf is None:
                return np.zeros((gw, gh), dtype=bool)
            alpha = pygame.surfarray.pixels_alpha(surf)
            mask = alpha > 0
            del alpha                                   # releases the surface lock
            return mask
 
        road, curb = to_bool(self.grid._surface_road), to_bool(self.grid._surface_curb)
        grass, gravel = to_bool(self.grid._surface_grass), to_bool(self.grid._surface_gravel)
        track = (to_bool(self.grid._surface_track) if self.grid._surface_track is not None
                 else road | curb | grass | gravel)
        boundary = to_bool(self.grid._surface_boundary)
        self.layers = kernels.pack_layers(road, curb, grass, gravel, track, boundary, to_bool(self.grid._surface_finish))
 
    def _cache_tile_table(self) -> None:
        """Flatten tiles into arrays so the kernels never touch Python objects (row i = `self.tiles[i]`)."""
        continuity = self.grid.continuity
        tiles = list(continuity.keys())
        known = set(tiles)
        tiles += [t for t in self.grid_mapping.values() if not t.empty and t not in known]
        self.tiles = tiles
        self._tile_id = {t: i for i, t in enumerate(tiles)}
        self._start_id = self._tile_id[self.grid.start_cell]
 
        self.t_type   = np.array([t.type.value for t in tiles], dtype=NP_INT)
        self.t_orient = np.array([float(t.orientation) for t in tiles], dtype=NP_FLOAT)
        self.t_rot    = np.array([float(t.rotation) for t in tiles], dtype=NP_FLOAT)
        self.t_start  = self.t_type == Type.START.value
        self.t_turn   = np.where(
            self.t_type == Type.LEFT.value, 1.0, np.where(self.t_type == Type.RIGHT.value, -1.0, 0.0),
        ).astype(NP_FLOAT)
        self.t_next   = np.array([self._tile_id.get(continuity.get(t), i) for i, t in enumerate(tiles)], dtype=NP_INT)
        self.t_prev   = np.arange(len(tiles), dtype=NP_INT)
        self.t_prev[self.t_next] = np.arange(len(tiles), dtype=NP_INT)
        self.t_center = np.array([t.center_abs for t in tiles], dtype=NP_FLOAT)
        self.t_check  = np.array([t.center_check_abs for t in tiles], dtype=NP_FLOAT)
 
        cols = max(p[0] for p in self.grid_mapping) + 1
        rows = max(p[1] for p in self.grid_mapping) + 1
        self.tile_map = np.full((cols, rows), -1, dtype=NP_INT)
        for pos, tile in self.grid_mapping.items():
            if not tile.empty and tile in self._tile_id:
                self.tile_map[pos] = self._tile_id[tile]
 
    def _cache_progress_table(self) -> None:
        """Track center lines and lap index per tile for the progress kernel."""
        self.t_lap  = np.zeros((len(self.tiles),), dtype=NP_INT)
        self.t_line = np.zeros((len(self.tiles), 3, 2), dtype=NP_FLOAT)
        for tile, line in self.track.lines.items():
            i = self._tile_id[tile]
            self.t_lap[i]  = self.track.lap_index[tile]
            self.t_line[i] = np.stack(line)
 
    # ------------------------------------------------------------------
    # Sensors and location
    # ------------------------------------------------------------------
 
    def _update_surface(self) -> None:
        if self.layers is not None:
            kernels.update_surface(self.layers, self.x, self.y, self.surface)
 
    def _update_lidar(self) -> None:
        """Cast the LiDAR beams of every car. Writes `lidar_distances` and `lidar_surface_hits`."""
        if self.layers is None:
            return
        shape = (self.total, self.lidar_angles)
        if self.lidar_distances.shape != shape:
            self.lidar_distances = np.ones(shape, dtype=NP_FLOAT)
            self.lidar_surface_hits = np.full(shape, SURFACE_VOID, dtype=np.int8)
        kernels.update_lidar(
            self.layers, self.x, self.y, self.angle, self.surface,
            self.lidar_beam_offsets, self._lidar_steps, float(self.lidar_step), float(self.lidar_max_dist),
            self.lidar_distances, self.lidar_surface_hits,
        )
        if self.race_on and self.race.lidar_detect_cars and self.total > 1:
            kernels.lidar_cars(
                self.x, self.y, self.angle, self.race_active, self.lidar_beam_offsets,
                float(self.lidar_max_dist), float(self.width), float(self.height),
                float(self.race.collision.radius_scale),
                self.lidar_distances, self.lidar_surface_hits,
            )
 
    def _update_regions(self, index: int | None = None) -> None:
        """Refresh the on_road / on_curb / ... flags of the car bodies."""
        if self.grid.granularity != "cell" and self.layers is not None:
            kernels.update_regions(
                self.layers, self._car_lut_off, self._car_lut_n, self._car_lut_id, self.x, self.y, self.angle,
                self.on_road, self.on_curb, self.on_grass, self.on_gravel, self.on_track, self.on_finish,
            )
            return
 
        for i in (range(self.total) if index is None else [index]):
            rotated = pygame.transform.rotate(self.bot_car_img, self.angle[i].item())
            mask = pygame.mask.from_surface(rotated)
            x, y = rotated.get_rect(center=(self.x[i].item(), self.y[i].item())).topleft
            args = (x, y, self.width, self.height)
            self.on_road[i]   = self.grid.on_road(*args, mask=mask)
            self.on_track[i]  = self.grid.on_track(*args, mask=mask)
            self.on_gravel[i] = self.grid.on_gravel(*args, mask=mask)
            self.on_grass[i]  = self.grid.on_grass(*args, mask=mask)
            self.on_curb[i]   = self.grid.on_curb(*args, mask=mask)
            self.on_finish[i] = self.grid.on_finish(*args, mask=mask)
 
    def _update_location(self) -> None:
        prev_idx = self.tile_idx.copy()
        self._update_regions()
        kernels.update_location(
            self.tile_map, self.t_next, self.t_start, float(self.grid.cell_size),
            self.x, self.y, self.on_track, self.tile_idx,
            self.started, self.finished, self.crashed, self.tile_crash,
            self.laps_done, self.checks_done, self.tile_stag_steps,
            self.lap_pending, self.on_finish, bool(self.finish_line_laps), int(self.restrict_movement),
        )
        if self.race_done.any():                        # a finished car rolls slowly, that must not cost lives
            self.tile_stag_steps[self.race_done] = 0
        for i in np.flatnonzero(self.tile_idx != prev_idx):
            self.tile[i] = self.tiles[self.tile_idx[i]]
 
    def _compute_progress(self) -> None:
        kernels.progress(
            self.progress, self.in_tile, self.x, self.y, self.tile_idx, self.laps_done + self.lap_pending,
            self.t_lap, self.t_line, int(self.track.n_tiles), float(self.track.size),
        )
 
    def get_state(self) -> Array:
        """Observation of every car, shape (N, features).
 
        With n = `state_tiles`:
            [0, n)       dist       1 - distance to the exit midpoint of the k-th tile ahead (k = 0: current)
            [n, 2n)      phase      heading offset to the exit orientation of the k-th tile ahead
            2n           vel_lin    normalised linear speed
            2n + 1       vel_ang    normalised angular speed
            2n + 2       heading    absolute heading
        followed by, when enabled and in this order:
            use_tile_lidar (4)    beams to the current tile boundary at heading +0/+90/+180/+270
            use_surface    (1)    surface code under the car center
            use_lidar      (B)    normalised hit distance per beam
        """
        n = self.state_tiles
        features = (2 * n + 3 + (4 if self.use_tile_lidar else 0) + (1 if self.use_surface else 0)
                    + (self.lidar_angles if self.use_lidar else 0))
        state = np.empty((self.total, features), dtype=NP_FLOAT)
        kernels.get_state(
            state, self.x, self.y, self.angle, self.vel_lin, self.vel_ang,
            self.tile_idx, self.t_orient, self.t_next, self.t_check, self.t_center,
            self.lidar_distances, self.surface,
            float(np.sqrt((self.grid.cell_size ** 2) * 2) / 2), float(self.grid.cell_size),
            float(self.max_vel_lin), float(self.max_vel_ang), int(n),
            bool(self.use_tile_lidar), bool(self.use_surface), bool(self.use_lidar),
        )
        return state
 
    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
 
    def update(self, inputs: Array, verbose: bool = False) -> None:
        """Advance every car one frame. `inputs` has shape (N, 2): throttle / brake, steering."""
        if self.discrete:
            raise NotImplementedError()
 
        prev_checks, prev_laps = self.checks_done.copy(), self.laps_done.copy()
        prev_angle, prev_tile = self.angle.copy(), self.tile_idx.copy()
        prev_brake = self.brake_steps.copy()
        prev_x, prev_y = self.x.copy(), self.y.copy()
 
        inputs = np.asarray(inputs, dtype=NP_FLOAT)
        acceleration = np.ascontiguousarray(inputs[..., 0])
        rotation     = np.ascontiguousarray(inputs[..., 1])
        self.race_frame += 1
        if self.race_on and self.race_done.any():
            # After the flag a car keeps steering but brakes down to a slow roll (a standstill would cost lives)
            d = self.race_done
            crawl = 0.15 * self.max_vel_lin
            acceleration[d] = np.where(self.vel_lin[d] > crawl, -1.0, 0.05)
 
        # Steering and surface coefficients
        kernels.steer(
            rotation, self.vel_ang, self.on_gravel, self.on_grass, self.on_curb,
            self.friction, self.handling, self.slip,
            self.coef, float(self.max_vel_ang), float(self.acc_ang), float(self.coeff_checker),
        )
 
        ss = self.race.slipstream
        drafting = self.slipstream.update(self)
 
        # Gravel jitter is drawn here so the env RNG controls it
        noise = np.zeros(self.total, dtype=NP_FLOAT)
        n_gravel = int(np.count_nonzero(self.on_gravel))
        if n_gravel:
            amp = float(self.config.gravel_noise)
            noise[self.on_gravel] = self.rng.uniform(-amp, amp, size=n_gravel)
 
        kernels.drive(
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
 
        if self.race_on and self.race.collision.enabled:
            self.bumped = kernels.collide(
                self.x, self.y, self.angle, self.vel_lin, self.vel_ang, self.disp_x, self.disp_y,
                self.race_active, self.width, self.height, self.max_vel_lin, self.race.collision,
            )
 
        if 1 <= self.restrict_movement <= 2:
            kernels.walls(
                self.layers, self.tile_map, self.t_next, self.t_prev, float(self.grid.cell_size),
                self.x, self.y, self.angle, prev_x, prev_y, prev_angle,
                self.vel_lin, self.vel_ang, self.disp_x, self.disp_y, self.tile_idx,
                float(self.width) * 0.45, float(self.height) * 0.45,
                bool(self.restrict_movement == 2), float(self.wall_damping), float(self.wall_align),
            )
 
        self.slipstream.record_trail(self)
 
        # Location and sensors
        self._update_location()
        self._update_surface()
        self._update_lidar()
        prev_progress = self.progress.copy()
        self._compute_progress()
 
        # Reward signals, counters and crash flags
        sg = self._signal_buf
        kernels.signals(
            sg["checked"], sg["finished"], sg["tile_stag"], sg["on_road"],
            sg["moved_forward"], sg["corner_curr"], sg["corner_next"], sg["straight_curr"],
            sg["phase_align"], sg["turning"], sg["braking"], sg["straight_aligned"],
            sg["brake_check"], sg["turn_align"],
            self.turn_steps, self.brake_steps, self.straight_steps, self.used_brake,
            self.corner_entry_speed, self.crashed, self.tile_crash, self.crashes_done, self.hiatus,
            self.angle, prev_angle, self.tile_idx, prev_tile,
            self.checks_done, prev_checks, self.laps_done, prev_laps,
            self.tile_stag_steps, prev_brake, self.vel_lin, vel_norm,
            self.on_road, self.on_curb, self.on_grass, self.on_gravel, self.on_track,
            acceleration, rotation,
            self.t_type, self.t_orient, self.t_next,
            int(self.players.max_hiatus), float(self.straight_alignment_tolerance),
            int(self.restrict_movement), bool(self.race_on), float(self.max_approach_angle),
        )
 
        self.shaper.update(self, sg, rotation, vel_norm, prev_progress)
 
        self.players.update(
            sg["checked"], sg["finished"], self.crashed, self.hiatus, sg["moved_forward"], sg["tile_stag"],
            sg["corner_curr"], sg["straight_curr"], vel_norm, sg["on_road"],
            shape_pos=self.shaper.shape_pos, shape_neg=self.shaper.shape_neg,
            crash_scale=self.shaper.crash_scale,
            verbose=verbose,
        )
 
        self.control.update_ranking(self)
 
        self.restart()
        self._compute_progress()                        # respawned cars must not register a progress jump
 
    def update_on_event(self, verbose: bool = False, bot_inputs: Array | None = None) -> None:
        """Human drives the last car with WASD. `bot_inputs` (N, 2) optionally drives the others."""
        if self.discrete:
            raise NotImplementedError()
        keys = pygame.key.get_pressed()
        inputs = np.full((self.total, 2), 0.0, dtype=NP_FLOAT)
        if bot_inputs is not None:
            inputs[:-1] = bot_inputs[:-1]
        moving = self.vel_lin[-1].item() > 0
        if moving:
            inputs[-1, 0] = -1                          # coast down by braking
 
        if   keys[pygame.K_a]: inputs[-1, 1] = 1
        elif keys[pygame.K_d]: inputs[-1, 1] = -1
        else:                  self.vel_ang[-1] = 0
        if keys[pygame.K_w]:
            inputs[-1, 0] = 1
        if keys[pygame.K_s]:
            if moving:
                inputs[-1, 0] = -1 * self.coeff_brake
            if (self.toggle_reverse or self.race_on) and not moving:
                inputs[-1, 0] = -1 * self.coeff_reverse
        self.update(inputs, verbose=verbose)
        
    # ------------------------------------------------------------------
    # Reset and restart
    # ------------------------------------------------------------------
 
    def reset(self, total: int) -> None:
        """Start a new episode with `total` cars on the current grid."""
        if not self.grid.initialized:
            raise RuntimeError("Grid must be initialized.")
 
        self.shaper.grade_episode(self.checks_done, getattr(self.players, "deaths", None))
        self.grid_mapping = {tile.position: tile for tile in self.grid._cells.flatten()}
        self._cache_tile_table()
        self.init_x, self.init_y = self.grid.start_cell.center_abs
        self.init_angle = self.grid.start_cell.rotation
        self._allocate(total, self.grid.start_cell, self._start_id)
 
        self.track = TrackGeometry(self.grid)
        if self.race.enabled:
            slots = StartingGrid(
                self.track, self.race.start_grid, self.width, self.height, self.race.max_racers,
            ).layout(self.total)
            for i, slot in enumerate(slots):
                self.x[i], self.y[i], self.angle[i] = slot["x"], slot["y"], slot["angle"]
                self.tile[i] = slot["tile"]
                self.tile_idx[i] = self._tile_id[slot["tile"]]
                self.laps_done[i] = slot["lap_offset"]  # cars behind the start tile cross the line once first
 
        self._cache_layers()
        self._cache_progress_table()
        self.shaper.setup(self)
        self.players.reset(total)
        self.random_ss.schedule(self, np.arange(self.total), reset_events=True)
        if self.race.enabled:
            self.control.update_order(self)                   # needs players.active, so after the players reset
 
    def restart(self) -> None:
        """Respawn crashed cars (and finished ones unless `endless`)."""
        restart_all = self.crashed | self.finished
        if not restart_all.any():
            return
        part = restart_all if not self.endless else self.crashed
 
        idx = np.flatnonzero(part)
        if idx.size:
            tid = np.full(idx.size, self._start_id, dtype=NP_INT) if self.full_restart else self.tile_idx[idx]
            self.x[idx] = self.t_center[tid, 0]
            self.y[idx] = self.t_center[tid, 1]
            ttype = self.t_type[tid]
            self.angle[idx] = self.t_rot[tid] + np.where(
                ttype == Type.LEFT.value, 45.0, np.where(ttype == Type.RIGHT.value, -45.0, 0.0),
            )
            pending = idx[self.lap_pending[idx]]
            if pending.size:                            # the respawn is past the line: count the armed lap
                if not self.full_restart:
                    self.laps_done[pending] += 1
                self.lap_pending[pending] = False
            self._update_regions()
 
        self.disp_x[part] = 0
        self.disp_y[part] = 0
        self.vel_lin[part] = 0
        self.vel_ang[part] = 0
        self.surface[part] = SURFACE_ROAD
        self.friction[part] = self.config.friction_road
        self.handling[part] = self.config.handling_road
        self.slip[part] = self.config.slip_road
 
        self.brake_steps[part] = 0
        self.turn_steps[part] = 0
        self.straight_steps[part] = 0
        self.random_ss.schedule(self, np.flatnonzero(part), reset_events=True)
        self.crashed[restart_all] = False
        self.started[self.finished] = False
        self.finished[restart_all] = False