import pygame
import pygame.surfarray
import numpy as np

from numpy import ndarray as Array
from pygame import Surface, Mask
from pygame.font import Font
from typing import Optional, Any

from .config import CarConfig, overwrite_config
from .functional import Type, Position
from .constants import NP_INT, NP_FLOAT, IMAGE_DIR, COLOR_LIDAR, COLOR_CURB_PRI, COLOR_CURB_SEC
from .util import scale_image, draw_arrow
from .grid import Grid
from .tile import Tile
from .player import Players


# Surface integer codes (shared by Cars.surface and lidar_surface_hits)
SURFACE_ROAD   = np.int8(0)
SURFACE_CURB   = np.int8(1)
SURFACE_GRASS  = np.int8(2)
SURFACE_GRAVEL = np.int8(3)
SURFACE_VOID   = np.int8(4)  # out-of-bounds / off-track


class Cars:
    BOT_CAR_IMG = pygame.image.load(f"{IMAGE_DIR}/purple-car.png")
    USR_CAR_IMG = pygame.image.load(f"{IMAGE_DIR}/green-car.png")
    POS2_CAR_IMG = pygame.image.load(f"{IMAGE_DIR}/white-car.png")
    POS3_CAR_IMG = pygame.image.load(f"{IMAGE_DIR}/red-car.png")

    def __init__(
        self, 
        grid: Grid, players: Players, config: CarConfig | None = None, 
        **params: dict[str, Any]
    ):
        if config is None:
            config = CarConfig()

        overwrite_config(config, params)
        self.config = config
        
        self.grid = grid
        self.players = players
        self.grid_mapping: dict[Position, Tile] = {tile.position: tile for tile in self.grid._cells.flatten()}
        self.reverse_mapping: dict[Tile, Position] = {tile: position for position, tile in self.grid_mapping.items()}

        self.bot_car_img = scale_image(self.BOT_CAR_IMG, config.scale)
        self.usr_car_img = scale_image(self.USR_CAR_IMG, config.scale)
        self.pos2_car_img = scale_image(self.POS2_CAR_IMG, config.scale)
        self.pos3_car_img = scale_image(self.POS3_CAR_IMG, config.scale)
        self.mask = pygame.mask.from_surface(self.bot_car_img)
        self.width, self.height = self.bot_car_img.get_size()
        self.total = 1

        self.init_x, self.init_y, self.init_angle = 0.0, 0.0, 0.0
        
        # Physics and control configuration
        self.use_lidar: bool    = config.use_lidar
        self.max_vel_lin: float = config.max_linear_velocity
        self.max_vel_ang: float = config.max_angular_velocity
        self.acc_lin: float     = config.linear_acceleration
        self.vel_ang: float     = config.angular_velocity
        self.coeff_reverse: float = config.reverse_coeff
        self.coeff_brake: float = config.brake_coeff
        self.min_brake_steps: int = config.min_brake_steps
        self.cutoff: float     = config.cutoff
        self.toggle_reverse: bool = config.toggle_reverse
        self.full_restart: bool = config.full_restart
        self.restrict_movement: bool = config.restrict_movement
        self.frames_per_tile: int = config.frames_per_tile
        self.discrete: bool     = config.discrete
        self.endless: bool      = config.endless

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
        # lidar_angle: number of evenly-spaced beams cast from the car centre.
        #   Beams are relative to the car's own heading (0° = forward,
        #   90° = right, 180° = behind, 270° = left).
        # lidar_max_dist: maximum ray length in pixels (default = 2.5 tiles).
        # lidar_step: sampling stride in pixels; smaller = more accurate but slower.
        self.lidar_angle: int   = config.lidar_angle
        self.lidar_max_dist: float = config.lidar_max_dist if config.lidar_max_dist is not None else float(self.grid.cell_size) * 2.5
        self.lidar_step: int    = config.lidar_step
        self._lidar_n_steps: int = max(1, int(self.lidar_max_dist / self.lidar_step))

        # Fixed, car-relative beam offsets (degrees, evenly spaced over 360°)
        self._lidar_beam_offsets: Array = np.linspace(
            0.0, 360.0, self.lidar_angle, endpoint=False, dtype=NP_FLOAT
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
        self.angle      = np.full((self.total,), self.init_angle, dtype=NP_FLOAT)
        self.crashed    = np.full((self.total,), False, dtype=bool)
        self.finished   = np.full((self.total,), False, dtype=bool)
        self.started    = np.full((self.total,), False, dtype=bool)
        self.laps_done  = np.full((self.total,), 0, dtype=NP_INT)
        self.checks_done = np.full((self.total,), 0, dtype=NP_INT)
        self.crashes_done = np.full((self.total,), 0, dtype=NP_INT)
        self.hiatus     = np.full((self.total,), 0, dtype=NP_INT)
        self.tile_stag_steps = np.full((self.total,), 0, dtype=NP_INT)
        self.brake_steps = np.full((self.total,), 0, dtype=NP_INT)
        self.turn_steps = np.full((self.total,), 0, dtype=NP_INT)
        self.on_road    = np.full((self.total,), False, dtype=bool)
        self.on_curb    = np.full((self.total,), False, dtype=bool)
        self.on_grass   = np.full((self.total,), False, dtype=bool)
        self.on_gravel  = np.full((self.total,), False, dtype=bool)
        self.on_track   = np.full((self.total,), True, dtype=bool)
        self.on_finish  = np.full((self.total,), False, dtype=bool)
        self.tile: Array[tuple[int], Tile] = np.full((self.total,), None, dtype=object)
        self.friction: Array | None = np.full_like(self.vel_lin, 1.0)
        self.handling: Array | None = np.full_like(self.vel_lin, 1.0)
        self.slip: Array | None = np.full_like(self.vel_lin, 0.0)
        self.ranking: Array | None = None
        self.used_brake = np.full((self.total,), False, dtype=bool)

        # Corner-entry speed: recorded once when the car first enters a corner
        # tile; held constant until the car moves to the next tile.  Used by
        # Players.update() to award the entry-speed bonus.
        self.corner_entry_speed: Array = np.full((self.total,), 0.0, dtype=NP_FLOAT)

        # LiDAR output arrays
        # lidar_distances     (N, lidar_angle) float32, each beam's hit distance [0, 1]
        # lidar_surface_hits  (N, lidar_angle) int8, surface code at hit point
        # surface             (N,) int8, dominant surface under car centre
        self.lidar_distances:    Array = np.ones(
            (self.total, self.lidar_angle), dtype=NP_FLOAT
        )
        self.lidar_surface_hits: Array = np.full(
            (self.total, self.lidar_angle), SURFACE_VOID, dtype=np.int8
        )
        self.surface: Array = np.zeros((self.total,), dtype=np.int8)  # default: road

        self.usr_engaged = False
        self.font: Optional[Font] = None

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
            alpha = pygame.surfarray.array_alpha(surf)   # (W, H) uint8
            return alpha > 0

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

    def _update_surface(self) -> None:
        """
        Update self.surface for every car using centre-point pixel lookup.

        Priority (highest to lowest): road → curb → grass → gravel / void.
        Shape: (N,) int8.
        """
        if self._lidar_road_arr is None:
            return

        gw, gh = self._lidar_gw, self._lidar_gh
        xi = np.clip(self.x.astype(np.int32), 0, gw - 1)
        yi = np.clip(self.y.astype(np.int32), 0, gh - 1)

        # Start with worst surface; upgrade inward
        self.surface[:] = SURFACE_VOID
        self.surface[self._lidar_gravel_arr[xi, yi]] = SURFACE_GRAVEL
        self.surface[self._lidar_grass_arr [xi, yi]] = SURFACE_GRASS
        self.surface[self._lidar_curb_arr  [xi, yi]] = SURFACE_CURB
        self.surface[self._lidar_road_arr  [xi, yi]] = SURFACE_ROAD

    def _update_lidar(self) -> None:
        """
        Cast `lidar_angle` rays from each active car's centre and record the
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
        # TODO: The lidar beams seem to penetrate the track boundary when it is 1 pixel thick. Fix
        if self._lidar_road_arr is None:
            return

        gw, gh = self._lidar_gw, self._lidar_gh
        N  = self.total
        B  = self.lidar_angle
        st = self.lidar_step
        md = self.lidar_max_dist

        # ── Beam directions ────────────────────────────────────────────────
        # Convention: angle 0 = north (up), increases clockwise.
        # Pygame y increases downward, so:  x += -sin(a),  y += -cos(a)
        angles_rad = np.radians(
            self.angle[:, None] + self._lidar_beam_offsets[None, :]
        )  # (N, B)
        dx = (-np.sin(angles_rad)).astype(NP_FLOAT)   # (N, B)
        dy = (-np.cos(angles_rad)).astype(NP_FLOAT)   # (N, B)

        # ── Sample positions ───────────────────────────────────────────────
        t = np.arange(st, md + st, st, dtype=NP_FLOAT)   # (D,)
        D = t.shape[0]

        # (N, B, D) world-space sample coordinates
        sx = self.x[:, None, None] + dx[:, :, None] * t[None, None, :]
        sy = self.y[:, None, None] + dy[:, :, None] * t[None, None, :]

        sx_i = sx.astype(np.int32)   # (N, B, D)
        sy_i = sy.astype(np.int32)

        in_bounds  = (sx_i >= 0) & (sx_i < gw) & (sy_i >= 0) & (sy_i < gh)
        sx_s = np.clip(sx_i, 0, gw - 1)
        sy_s = np.clip(sy_i, 0, gh - 1)

        flat_x = sx_s.ravel()   # (N*B*D,)
        flat_y = sy_s.ravel()

        # ── Surface presence at every sample ──────────────────────────────
        # Each array is True where that surface exists, False elsewhere.
        # out-of-bounds samples are treated as False on every surface.
        on_road_flat   = self._lidar_road_arr  [flat_x, flat_y]
        on_curb_flat   = self._lidar_curb_arr  [flat_x, flat_y]
        on_grass_flat  = self._lidar_grass_arr [flat_x, flat_y]
        on_gravel_flat = self._lidar_gravel_arr[flat_x, flat_y]
        on_track_flat  = self._lidar_track_arr [flat_x, flat_y]

        # Apply in-bounds mask so off-grid pixels never appear "on" anything
        ib_flat = in_bounds.ravel()
        on_road_flat   &= ib_flat
        on_curb_flat   &= ib_flat
        on_grass_flat  &= ib_flat
        on_gravel_flat &= ib_flat
        on_track_flat  &= ib_flat

        # Reshape to (N, B, D)
        on_road   = on_road_flat  .reshape(N, B, D)
        on_curb   = on_curb_flat  .reshape(N, B, D)
        on_grass  = on_grass_flat .reshape(N, B, D)
        on_gravel = on_gravel_flat.reshape(N, B, D)
        on_track  = on_track_flat .reshape(N, B, D)

        # ── Tile boundary constraint ──────────────────────────────────────
        # Extract boundary pixels at every sample location.
        # Beams must not cross tile boundaries to avoid spurious surface
        # detections from adjacent tiles oriented in different directions.
        on_boundary_flat = self._lidar_boundary_arr[flat_x, flat_y] & ib_flat
        on_boundary = on_boundary_flat.reshape(N, B, D)   # (N, B, D)

        # ── Per-car home-surface level ─────────────────────────────────────
        # Derived from self.surface (N,) int8 which was updated this frame by
        # _update_surface().  We promote "void" to "gravel" so that a car that
        # has just left the track still measures to the track boundary.
        home = np.where(self.surface == SURFACE_VOID, SURFACE_GRAVEL,
                        self.surface)  # (N,) clamped to [0..3]

        # Broadcast to (N, 1, 1) for vectorised comparisons below
        home_nb = home[:, None, None]

        # ── "Still on home surface" predicate at each sample step ─────────
        # A sample is "inside" if it is on the home layer OR any layer
        # strictly inner (lower code) than home.
        #
        #   home=road(0)   → sample must be on road
        #   home=curb(1)   → sample must be on road OR curb
        #   home=grass(2)  → sample must be on road, curb, OR grass
        #   home=gravel(3) → sample must be on any track pixel
        #
        # We build this as a 4-level selector.  Using np.where chains is fast
        # and avoids any Python loop over surface levels.
        inside = np.where(
            home_nb == SURFACE_ROAD,
            on_road,
            np.where(
                home_nb == SURFACE_CURB,
                on_road | on_curb,
                np.where(
                    home_nb == SURFACE_GRASS,
                    on_road | on_curb | on_grass,
                    on_track   # home=gravel(3): stop at track boundary
                )
            )
        )  # (N, B, D) bool — True while beam is still inside home surface

        # ── Track boundary: hard clamp ────────────────────────────────────
        # Regardless of home surface, a beam must never leave the track or
        # cross a tile boundary. For non-gravel beams the track clamp seldom
        # matters (the surface edge comes first), but it is the sole guard
        # against U-turn bleed. Boundary clamping prevents beams from crossing
        # tile edges and detecting surfaces from adjacent tiles in different
        # directions.
        #
        # We find the last on-track step for each beam so the endpoint stays
        # just inside the outermost surface pixel (the "inside" logic above
        # already guarantees this for gravel beams; the clamp is a safety
        # net for the others).
        #
        # Implementation: we apply the track and boundary clamps AFTER finding
        # the primary hit by building a "clamped inside" mask.
        inside_and_on_track = inside & on_track & ~on_boundary   # (N, B, D)

        # ── First hit: first step where the beam exits the home surface ───
        hit_mask  = ~inside_and_on_track          # (N, B, D) True at exit
        no_hit    = ~hit_mask.any(axis=2)          # (N, B) beam never exits
        first_hit = np.argmax(hit_mask, axis=2)    # (N, B) index of first exit

        # For the gravel/track case we want to report the last INSIDE step
        # (i.e. the pixel just before the boundary), not the first outside
        # step.  We compute this as (first_hit - 1) clamped to ≥ 0, and
        # apply it only where home == gravel.
        last_inside = np.maximum(first_hit - 1, 0)   # (N, B)
        is_gravel_home = (home == SURFACE_GRAVEL)     # (N,)
        reported_idx = np.where(
            is_gravel_home[:, None] & ~no_hit,
            last_inside,
            first_hit
        )  # (N, B)

        # Convert sample index to pixel distance
        distances = (reported_idx.astype(NP_FLOAT) + 1.0) * st   # (N, B) pixels
        distances[no_hit] = md   # beam never exited → max distance

        self.lidar_distances = distances / md   # normalise to [0, 1]

        # ── Surface at hit point ───────────────────────────────────────────
        # Evaluate which surface layer is present at the pixel where each
        # beam stops.  Priority: outermost wins (gravel > grass > curb > road)
        # so the agent knows WHAT it is about to cross onto.
        # For no-hit beams we report SURFACE_VOID ("open, nothing hit").
        hit_x = (self.x[:, None] + dx * distances).astype(np.int32)  # (N, B)
        hit_y = (self.y[:, None] + dy * distances).astype(np.int32)

        ib = (hit_x >= 0) & (hit_x < gw) & (hit_y >= 0) & (hit_y < gh)  # (N, B)
        hx = np.clip(hit_x, 0, gw - 1)
        hy = np.clip(hit_y, 0, gh - 1)
        fx_h, fy_h = hx.ravel(), hy.ravel()
        ib_h = ib.ravel()

        # Start with void, then assign outermost-wins (road last so it can
        # only overwrite when the beam genuinely stops on road, which happens
        # e.g. when the car is on curb/grass and the beam points inward).
        surf_hits = np.full((N * B,), SURFACE_VOID, dtype=np.int8)
        surf_hits[ib_h & self._lidar_road_arr  [fx_h, fy_h]] = SURFACE_ROAD
        surf_hits[ib_h & self._lidar_curb_arr  [fx_h, fy_h]] = SURFACE_CURB
        surf_hits[ib_h & self._lidar_grass_arr [fx_h, fy_h]] = SURFACE_GRASS
        surf_hits[ib_h & self._lidar_gravel_arr[fx_h, fy_h]] = SURFACE_GRAVEL

        # no-hit beams always get SURFACE_VOID regardless of where they land
        no_hit_flat = no_hit.ravel()
        surf_hits[no_hit_flat] = SURFACE_VOID

        self.lidar_surface_hits = surf_hits.reshape(N, B)

    # ======================================================================
    # Location / collision tracking
    # ======================================================================

    def _update_regions(self, index: int):
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
        for idx in range(self.total):
            self._update_regions(idx)

            if self.on_track[idx]:
                tile_key = (
                    int(abs(self.x[idx].item()) / self.grid.cell_size),
                    int(abs(self.y[idx].item()) / self.grid.cell_size)
                )
                if tile_key in self.grid_mapping:
                    new_tile: Tile | None = self.grid_mapping.get(tile_key)
                    if new_tile is not None and not new_tile.empty:
                        old_tile: Tile = self.tile[idx]
                        exp_tile = self.grid.continuity[old_tile]
                        if new_tile == old_tile or new_tile == exp_tile:
                            if new_tile != old_tile:
                                if old_tile.type == Type.START:
                                    self.started[idx] |= True
                                if new_tile.type == Type.START:
                                    self.finished[idx] = True
                                    self.laps_done[idx] += 1
                                self.checks_done[idx] += 1
                                self.tile_stag_steps[idx] = 0
                            else:
                                self.tile_stag_steps[idx] += 1
                            self.tile[idx] = new_tile
                        else:
                            self.crashed[idx] |= True

    # ======================================================================
    # Main update
    # ======================================================================

    def update(self, inputs: Array, verbose=False):
        # ------------------------------------------------------------------
        # Cache the previous states / counts
        # ------------------------------------------------------------------
        _checks_done    = self.checks_done.copy()
        _laps_done      = self.laps_done.copy()
        _angle          = self.angle.copy()
        _tile           = self.tile.copy()
        _tile_angle_curr = np.array([t.orientation for t in _tile])
        _tile_angle_next = np.array([self.grid.continuity[t].orientation for t in _tile])
        _phase_curr     = -(_angle - _tile_angle_curr)
        _phase_curr     = (_phase_curr + 180) % 360 - 180 # wrap to [-180, 180]
        _phase_next     = -(_angle - _tile_angle_next)
        _phase_next     = (_phase_next + 180) % 360 - 180
        _brake_steps    = self.brake_steps.copy()
        _disp_x         = self.disp_x.copy()
        _disp_y         = self.disp_y.copy()
        _friction       = self.friction.copy()
        _slip           = self.slip.copy()

        if not self.discrete:
            # ------------------------------------------------------------------
            # 0. Inputs
            # ------------------------------------------------------------------
            # inputs = np.clip(inputs, -1.0, +1.0)
            # accelerate, brake, left, right = np.transpose(inputs, (-1, -2))
            # acceleration = accelerate - brake
            # rotation = left - right
            # inputs = np.clip(inputs, -1.0, +1.0)
            acceleration, rotation = np.transpose(inputs, (-1, -2))

            # ------------------------------------------------------------------
            # 1. Per-car surface coefficients
            # ------------------------------------------------------------------
            self.friction = np.where(
                self.on_gravel, self.friction_gravel,
                np.where(
                    self.on_grass, self.friction_grass,
                    np.where(self.on_curb, self.friction_curb, self.friction_road)
                )
            )
            self.handling = np.where(
                self.on_gravel, self.handling_gravel,
                np.where(
                    self.on_grass, self.handling_grass,
                    np.where(self.on_curb, self.handling_curb, self.handling_road)
                )
            )
            self.slip = np.where(
                self.on_gravel, self.slip_gravel,
                np.where(
                    self.on_grass, self.slip_grass,
                    np.where(self.on_curb, self.slip_curb, self.slip_road)
                )
            )

            # ------------------------------------------------------------------
            # 2. Rotate (steering degraded on loose/slippery surfaces)
            # ------------------------------------------------------------------
            self.angle[:] += self.vel_ang * rotation * self.handling
            self.angle[:] %= 360
            self.angle[self.on_gravel] += np.random.uniform(
                -2, 2, size=np.count_nonzero(self.on_gravel)
            )

            # ------------------------------------------------------------------
            # 3. Accelerate / brake
            # ------------------------------------------------------------------
            self.vel_lin[:] = np.clip(
                self.vel_lin + (self.acc_lin * acceleration),
                -self.max_vel_lin if self.toggle_reverse else 0.0,
                +self.max_vel_lin,
            )

            # ------------------------------------------------------------------
            # 4. Surface friction (passive velocity decay)
            # ------------------------------------------------------------------
            self.vel_lin[:] *= self.friction

        else:
            raise NotImplementedError()

        # Normalised linear speed [0, 1]
        vel_norm = np.clip(self.vel_lin / self.max_vel_lin, -1.0, 1.0).astype(NP_FLOAT)

        # Move
        radians = np.radians(self.angle)
        self.disp_x = np.clip(
            (np.sin(radians) * self.vel_lin) 
            + (_disp_x * _slip * np.abs(vel_norm) * _friction ** 3)
            , -self.max_vel_lin, self.max_vel_lin
        )
        self.disp_y = np.clip(
            (np.cos(radians) * self.vel_lin) 
            + (_disp_y * self.slip * np.abs(vel_norm) * _friction ** 3)
            , -self.max_vel_lin, self.max_vel_lin
        )
        self.x -= self.disp_x
        self.y -= self.disp_y

        # ------------------------------------------------------------------
        # Update location, surface flags, and LiDAR
        # ------------------------------------------------------------------
        self._update_location()
        self._update_surface()   # update self.surface from centre-point lookup
        self._update_lidar()     # cast beams; update lidar_distances / lidar_surface_hits

        # # Normalize
        # acceleration /= 2
        # rotation /= 2

        # ------------------------------------------------------------------
        # Derived signals for reward shaping
        # ------------------------------------------------------------------
        
        tile_angle_curr = np.array([t.orientation for t in self.tile])
        tile_angle_next = np.array([self.grid.continuity[t].orientation for t in self.tile])
        phase_curr = -(self.angle - tile_angle_curr)
        phase_curr = (phase_curr + 180) % 360 - 180 # wrap to [-180, 180]
        phase_next = -(self.angle - tile_angle_next)
        phase_next = (phase_next + 180) % 360 - 180

        turned_in_curr = np.abs(phase_curr) < np.abs(_phase_curr)
        turned_in_next = np.abs(phase_next) < np.abs(_phase_next)

        # Heading alignment: cosine of (car angle − tile rotation).
        # 1  = perfectly aligned with tile direction
        # 0  = perpendicular (90° off)
        # -1 = going the wrong way
        phase_align = np.cos(np.radians(phase_curr)).astype(NP_FLOAT) # (N,)

        # Forward / crash detection
        offset = np.array([
            +45 if t.type == Type.LEFT else -45 if t.type == Type.RIGHT else 0
            for t in self.tile
        ], dtype=NP_FLOAT)
        left  = +135 + offset
        right = -135 + offset
        moved_forward = (self.vel_lin > 0) & (phase_curr < left) & (phase_curr > right)
        out_of_bounds = ~self.on_track

        # Tile-type masks
        tile_type_curr   = np.array([t.type.value for t in self.tile])
        corner_tile_curr = np.isin(tile_type_curr, [Type.LEFT.value, Type.RIGHT.value])
        tile_type_next   = np.array([self.grid.continuity[t].type.value for t in self.tile])
        corner_tile_next = np.isin(tile_type_next, [Type.LEFT.value, Type.RIGHT.value])

        # Checkpoint / lap flags
        checked   = self.checks_done > _checks_done
        finished  = self.laps_done > _laps_done
        tile_stag = self.tile_stag_steps > self.players.max_hiatus

        # Location
        on_road = self.on_road & (~self.on_curb) & (~self.on_grass) & (~self.on_gravel) & self.on_track

        # Steps
        delayed_reset = checked | out_of_bounds
        turning = ((turned_in_curr & corner_tile_curr) | (turned_in_next & corner_tile_next)) & on_road
        # turning = turned_in_curr | turned_in_next
        self.turn_steps[turning] += 1
        self.turn_steps[delayed_reset] = 0
        braking = acceleration < 0 
        self.brake_steps[braking]  += 1
        self.brake_steps[~braking]  = 0
        self.used_brake |= braking
        
        # Check if brakes were used on checkpoint
        brake_check = (
            (
                checked & (
                    ((_brake_steps > 0) & corner_tile_curr) 
                    | (~corner_tile_curr)
                )
            ) 
            | (~checked)
        )

        # ------------------------------------------------------------------
        # Turn-direction alignment signal
        # ------------------------------------------------------------------
        # For each car on a corner tile, compute whether the car is actually
        # steering in the geometrically required direction.
        #
        # The required turn sign:  LEFT tile → car must turn left  → rotation
        # input should be positive (self.vel_ang * rotation adds to angle, and
        # in pygame-angle convention increasing angle is clockwise = rightward).
        # Looking at the update() rotation code:
        #   self.angle += vel_ang * rotation * handling
        # A positive `rotation` input increases angle (clockwise = RIGHT turn).
        # A LEFT tile therefore requires a NEGATIVE rotation input.
        # So:  LEFT tile → required_sign = -1,  RIGHT tile → required_sign = +1.
        #
        # `rotation` is the raw agent action ∈ [-1, +1].
        # turn_align = rotation * required_sign, clamped to [-1, +1].
        required_turn_sign = np.array([
            -1.0 if t.type == Type.LEFT else 1.0 if t.type == Type.RIGHT else 0.0
            for t in self.tile
        ], dtype=NP_FLOAT)
        # rotation was unpacked at the top of update() as a (N,) array
        turn_align = np.clip(rotation * required_turn_sign, -1.0, 1.0).astype(NP_FLOAT)
        # Zero out for non-corner tiles so the signal is clean
        turn_align[~corner_tile_curr] = 0.0

        # ------------------------------------------------------------------
        # Corner-entry speed — snapshot vel_norm when entering a corner
        # ------------------------------------------------------------------
        # `checked & corner_tile` is True the first frame the car crosses
        # into a new corner tile (checked fires on tile transition, and
        # corner_tile now reflects the *new* tile after the transition).
        just_entered_corner = checked & corner_tile_curr
        self.corner_entry_speed[just_entered_corner] = vel_norm[just_entered_corner]
        # For non-corner tiles, keep the value at current vel_norm so the
        # Players code always receives a valid float in [0,1].
        self.corner_entry_speed[~corner_tile_curr] = vel_norm[~corner_tile_curr]
        
        # ------------------------------------------------------------------
        # 
        # ------------------------------------------------------------------
        eliminate = out_of_bounds
        if self.restrict_movement:
            eliminate |= ~moved_forward
        # if True:
        #     eliminate |= ~brake_check
        self.crashed[eliminate] |= True

        self.crashes_done[self.crashed] += 1
        self.hiatus[~moved_forward] += 1
        self.hiatus[moved_forward]   = 0

        # ------------------------------------------------------------------
        # Player (reward) update
        # ------------------------------------------------------------------
        self.players.update(
            checked, finished, 
            self.crashed.copy(), self.hiatus.copy(),
            moved_forward, tile_stag,
            corner_tile_curr, corner_tile_next,
            vel_norm, phase_align,
            on_road,
            turning, self.turn_steps.copy(), self.min_brake_steps,
            braking, self.brake_steps.copy(), self.min_brake_steps, brake_check,
            turn_align,
            self.corner_entry_speed.copy(),
            None, verbose,
        )

        if np.count_nonzero(self.laps_done >= 1) <= 3:
            self.ranking = np.argsort(-self.players.scores)

        self.restart()

    # ======================================================================
    # Event-driven (human) update
    # ======================================================================

    def update_on_event(self, verbose=False):
        keys = pygame.key.get_pressed()

        if not self.discrete:
            inputs = np.full((self.total, 2), 0.0, dtype=NP_FLOAT)
            moving = self.vel_lin[-1].item() > 0
            if moving: inputs[-1, 0] = -1

            if keys[pygame.K_a]: inputs[-1, 1] =  1
            if keys[pygame.K_d]: inputs[-1, 1] = -1
            if keys[pygame.K_w]: inputs[-1, 0] =  1
            if keys[pygame.K_s]:
                if moving: inputs[-1, 0] = -1 * self.coeff_brake
                if self.toggle_reverse and not moving: inputs[-1, 0] = -1 * self.coeff_reverse

            self.update(inputs, verbose=verbose)
        else:
            raise NotImplementedError()

    # ======================================================================
    # State representations
    # ======================================================================

    def get_state(self) -> Array:
        """
        Return the observation vector for all cars.

        Parameters
        ----------
        use_lidar : bool
            False (default) → 9-feature geometric state (backward-compatible).
            True            → geometric state + LiDAR distances + surface code,
                              shape (N, 9 + lidar_angle + 1).

        Default (geometric) features
        -----------------------------
        0  dist_check          normalised distance to current checkpoint
        1  dist_check_next     normalised distance to next checkpoint
        2  dist_center         normalised distance to tile centre
        3  vel_lin             normalised linear speed
        4  phase               heading offset from current tile (normalised)
        5  phase_next          heading offset from next tile
        6  angle               absolute car heading (normalised)
        7  friction            normalised surface friction coefficient
        8  handling            normalised surface handling coefficient

        LiDAR extension (use_lidar=True)
        ---------------------------------
        9 … 9+B-1  lidar_distances[b]  per-beam normalised road-edge distance
        9+B        surface / 3         dominant surface under car centre [0, 1]
        """
        tile_next = [self.grid.continuity[t] for t in self.tile]
        tile_next_1 = [self.grid.continuity[self.grid.continuity[t]] for t in self.tile]
        tile_center      = np.array([list(tile.center_abs) for tile in self.tile])
        x_t, y_t         = np.transpose(tile_center)
        tile_check_curr  = np.array([list(tile.center_check_abs) for tile in self.tile])
        x_c0, y_c0       = np.transpose(tile_check_curr)
        tile_check_next  = np.array([list(tile.center_check_abs) for tile in tile_next])
        x_c1, y_c1       = np.transpose(tile_check_next)
        tile_check_next_1  = np.array([list(tile.center_check_abs) for tile in tile_next_1])
        x_c2, y_c2       = np.transpose(tile_check_next)

        a_r  = np.array([tile.rotation for tile in self.tile])
        a_t0 = np.array([tile.orientation for tile in self.tile]) # angle [0, 360]
        a_t1 = np.array([tile.orientation for tile in tile_next]) # 000d=UP, 090d=LEFT, 270d=RIGHT
        a_t2 = np.array([tile.orientation for tile in tile_next_1]) # 000d=UP, 090d=LEFT, 270d=RIGHT

        x_p, y_p, a_p = self.x.copy(), self.y.copy(), self.angle.copy()

        def _signed_diff(a: Array, b: Array):
            """Signed angular difference (a - b) wrapped to [-180, 180]."""
            return (a - b + 180) % 360 - 180

        def _normalize_dist(d: Array, scale: float):
            return d / scale # - 1
        
        # --- distances (normalized, offset to [... (-1), 1]) ---
        half = np.sqrt((self.grid.cell_size ** 2) * 2) / 2
        distances = 1.0 - np.stack([
            _normalize_dist(np.hypot(x_c0 - x_p, y_c0 - y_p), half),
            _normalize_dist(np.hypot(x_c1 - x_p, y_c1 - y_p), half * 2),
            _normalize_dist(np.hypot(x_c2 - x_p, y_c2 - y_p), half * 3),
            _normalize_dist(np.hypot(x_t  - x_p, y_t  - y_p), half / 2),
        ], axis=-1)

        # --- phases: how far the player's heading deviates from each tile ---
        phases = np.stack([
            _signed_diff(a_t0, a_p),
            _signed_diff(a_t1, a_p),
            _signed_diff(a_t2, a_p),
            _signed_diff(a_r , a_p),
        ], axis=-1) / 180.0

        # --- vehicle state ---
        linear_velocity = self.vel_lin / self.max_vel_lin
        # if not self.toggle_reverse: (linear_velocity * 2) - 1 # NOTE: Disabled because of confusion when reverse is enabled post-training
        # friction = (self.friction - self.friction_min) / (self.friction_max - self.friction_min)
        # handling = (self.handling - self.handling_min) / (self.handling_max - self.handling_min)
        vehicle = np.stack([
            linear_velocity,
            (self.angle - 180) / (- 180.0), # Ensures 1.0=UP, 0.5=LEFT and -0.5=RIGHT
            # friction,
            # handling, # TODO: Might replace with surface
        ], axis=-1)

        state = np.concatenate([
            distances,
            phases,
            vehicle,
        ], axis=-1) # (players, features)

        if not self.use_lidar:
            return state

        # ── LiDAR extension ───────────────────────────────────────────────
        # lidar_distances: (N, B) already normalised [0, 1]
        lidar = (self.lidar_distances * 2) - 1
        # surface encoded as scalar in [0, 1]  (0=road … 1=void)
        surface = ((self.surface.astype(NP_FLOAT) / float(SURFACE_VOID))[:, None] * 2) - 1  # (N, 1)
        state =  np.concatenate([state, lidar, surface], axis=-1) # (N, 9+B+1)
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

    def render(self, surface: Surface, verbose: int | bool = False):
        if self.font is None:
            self.font = pygame.font.Font(None, 24) # , self.font_size)

        ranking = (self.ranking if self.ranking is not None else np.argsort(-self.players.fitness))[:3]
        for idx, (x, y, angle, active) in enumerate(zip(*self.center, self.angle, self.players.active)):
            if active or (self.usr_engaged and idx == self.total - 1):
                ranked = idx in ranking
                if not ranked:
                    img = self.bot_car_img
                else:
                    if idx == ranking[0]:
                        img = self.usr_car_img
                    elif self.total >= 2 and idx == ranking[1]:
                        img = self.pos2_car_img
                    else:
                        img = self.pos3_car_img
                if self.usr_engaged:
                    if idx == self.total - 1:
                        img = self.usr_car_img
                    else:
                        if ranked:
                            if img == self.usr_car_img:
                                img = self.pos2_car_img
                            elif img == self.pos2_car_img:
                                img = self.pos3_car_img
                            else:
                                img = self.bot_car_img

                img_center_point = (x.item(), y.item())
                rot_img  = pygame.transform.rotate(img, angle.item())
                rot_rect = rot_img.get_rect(center=img_center_point)
                surface.blit(rot_img, rot_rect.topleft)

                best_idx = ranking[0] if not self.usr_engaged else self.total - 1
                if idx == best_idx:
                    position = (x, y)

                    # Render debugging text
                    self._render_debugging(surface, idx)

                    if verbose:
                        # Render directions
                        tile_curr = self.tile[idx]
                        tile_next = self.grid.continuity[tile_curr]
                        tile_next_1 = self.grid.continuity[tile_next]
                        draw_arrow(surface, tile_curr.respawn_point, tile_curr.center_check_abs, color=(102, 0, 153))
                        draw_arrow(surface, tile_next.respawn_point, tile_next.center_check_abs, color=(255, 255, 255))
                        draw_arrow(surface, tile_next_1.respawn_point, tile_next_1.center_check_abs, color=(255, 255, 255))
            
                        # Render Checkpoint midpoint
                        draw_arrow(surface, position, tile_curr.center_check_abs, color=(255, 0, 0))
                        draw_arrow(surface, position, tile_next.center_check_abs, color=(0, 0, 255))
                        draw_arrow(surface, position, tile_next_1.center_check_abs, color=(0, 255, 0))

                    # Draw the LiDAR system
                    if verbose and self.use_lidar:
                        self._render_lidar(surface, best_idx, x.item(), y.item())

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

        # Render status text
        text_surface = self.font.render(status_text, True, status_color)
        text_rect = text_surface.get_rect(topleft=(10, 10))
        surface.blit(text_surface, text_rect)
        
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
        pos_surface = self.font.render(pos_text, True, (200, 200, 200))
        pos_rect = pos_surface.get_rect(topleft=(10, 40))
        surface.blit(pos_surface, pos_rect)

        # Render car status text
        score_text = (
            f"Checks: {self.checks_done[index]} "
            f"Laps: {self.laps_done[index]} "
            f"Crashes: {self.crashes_done[index]} "
        )
        text_surface = self.font.render(score_text, True, (200, 200, 200))
        text_rect = text_surface.get_rect(topleft=(10, 70))
        surface.blit(text_surface, text_rect)

        # Render player status text
        score_text = (
            f"Hiatus: {self.hiatus[index]} "
            f"Reward: {self.players.get_reward()[index]:.1f} "
            f"Lives: {self.players.lives[index]} "
            f"Score: {self.players.true_scores[index]} "
        )
        text_surface = self.font.render(score_text, True, (200, 200, 200))
        text_rect = text_surface.get_rect(topleft=(10, 100))
        surface.blit(text_surface, text_rect)

        # Render player status text
        score_text = (
            f"Brake: {self.brake_steps[index]} "
            f"Turn: {self.turn_steps[index]} "
        )
        text_surface = self.font.render(score_text, True, (200, 200, 200))
        text_rect = text_surface.get_rect(topleft=(10, 130))
        surface.blit(text_surface, text_rect)

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
        }
        
        car_angle = self.angle[car_idx]
        distances = self.lidar_distances[car_idx]  # (lidar_angle,)
        surface_hits = self.lidar_surface_hits[car_idx]  # (lidar_angle,)
        
        # Draw each LiDAR beam
        for beam_idx in range(self.lidar_angle):
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

        self.init_x, self.init_y = self.grid.start_cell.center_abs
        self.init_angle = self.grid.start_cell.rotation
        self.total = total

        self.x           = np.full((self.total,), self.init_x, dtype=NP_FLOAT)
        self.y           = np.full((self.total,), self.init_y, dtype=NP_FLOAT)
        self.disp_x      = np.full((self.total,), 0.0, dtype=NP_FLOAT)
        self.disp_y      = np.full((self.total,), 0.0, dtype=NP_FLOAT)
        self.vel_lin     = np.full((self.total,), 0.0, dtype=NP_FLOAT)
        self.angle       = np.full((self.total,), self.init_angle, dtype=NP_FLOAT)
        self.crashed     = np.full((self.total,), False, dtype=bool)
        self.finished    = np.full((self.total,), False, dtype=bool)
        self.started     = np.full((self.total,), False, dtype=bool)
        self.laps_done   = np.full((self.total,), 0, dtype=NP_INT)
        self.checks_done = np.full((self.total,), 0, dtype=NP_INT)
        self.crashes_done= np.full((self.total,), 0, dtype=NP_INT)
        self.hiatus      = np.full((self.total,), 0, dtype=NP_INT)
        self.tile_stag_steps = np.full((self.total,), 0, dtype=NP_INT)
        self.brake_steps = np.full((self.total,), 0, dtype=NP_INT)
        self.turn_steps  = np.full((self.total,), 0, dtype=NP_INT)
        self.on_road     = np.full((self.total,), False, dtype=bool)
        self.on_curb     = np.full((self.total,), False, dtype=bool)
        self.on_grass    = np.full((self.total,), False, dtype=bool)
        self.on_gravel   = np.full((self.total,), False, dtype=bool)
        self.on_track    = np.full((self.total,), True, dtype=bool)
        self.on_finish   = np.full((self.total,), False, dtype=bool)
        self.tile        = np.full((self.total,), self.grid.start_cell, dtype=object)
        self.ranking     = None
        self.friction    = np.full_like(self.vel_lin, 1.0)
        self.handling    = np.full_like(self.vel_lin, 1.0)
        self.used_brake  = np.full((self.total,), False, dtype=bool)

        # LiDAR arrays (resized for new total)
        self.corner_entry_speed = np.full((self.total,), 0.0, dtype=NP_FLOAT)

        self.lidar_distances    = np.ones((self.total, self.lidar_angle), dtype=NP_FLOAT)
        self.lidar_surface_hits = np.full((self.total, self.lidar_angle), SURFACE_VOID, dtype=np.int8)
        self.surface            = np.zeros((self.total,), dtype=np.int8)

        # (Re-)build cached mask numpy arrays from the freshly-built grid
        self._cache_lidar_arrays()

        self.players.reset(total)

    def restart(self):
        restart_all  = self.crashed | self.finished
        restart_part = restart_all if not self.endless else self.crashed

        if restart_all.any():
            for idx, (tile, reset) in enumerate(zip(self.tile, restart_part)):
                if reset:
                    try:
                        if self.full_restart: tile = self.grid.start_cell
                        x, y = tile.center_abs
                        self.x[idx], self.y[idx] = x, y
                        self.angle[idx] = tile.rotation + (
                            45 if tile.type == Type.LEFT else -45 if tile.type == Type.RIGHT else 0
                        )
                    except AttributeError as e:
                        print(self.grid._cells)
                        print(self.tile)
                        raise e
                    
                    self._update_regions(idx)

            self.disp_x    [restart_part] = 0
            self.disp_y    [restart_part] = 0
            self.vel_lin   [restart_part] = 0
            self.surface   [restart_part] = SURFACE_ROAD # assume road on respawn # TODO: Make it have a direct update method like the surface masks

            self.brake_steps[restart_part] = 0
            self.turn_steps [restart_part] = 0
            self.crashed    [restart_all]  = False
            self.started    [self.finished] = False
            self.finished   [restart_all]  = False

        self.players.restart()