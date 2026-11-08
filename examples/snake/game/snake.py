"""Snake state, movement and sensing for every player (the `Cars` of this environment)."""
from __future__ import annotations

import numpy as np
from numpy import ndarray as Array

from . import kernels
from .config import SnakeConfig
from .constants import NP_FLOAT, NP_INT
from .grid import Grid
from .player import Players
from .reward import RewardShaper

__all__ = ["Snakes"]


class Snakes:
    """All snakes as arrays. Each snake is a ring buffer of cells (see `kernels.movement`)."""

    def __init__(self, grid: Grid, players: Players, config: SnakeConfig | None = None, shaper: RewardShaper | None = None) -> None:
        self.grid = grid
        self.players = players
        self.config = config if config is not None else SnakeConfig()
        self.shaper = shaper if shaper is not None else RewardShaper(players.reward, players.max_hiatus, grid)
        self.init_length = int(self.config.init_length)
        # Without wrapping, the centred spawn must leave enough room on either side of the
        # selected heading. With wrapping, the body must fit without revisiting a cell.
        # (The road reward modes spawn along the ring road instead and only need the body to be shorter than the board.)
        if players.reward.mode != "basic":
            fits = self.init_length < grid.cell_total
        elif grid.wrap:
            fits = self.init_length <= min(grid.shape)
        else:
            fits = self.init_length - 1 <= min(grid.shape) // 2
        if not fits:
            rule = "fit without overlapping after wrap" if grid.wrap else "fit from the centred spawn without crossing a wall"
            raise ValueError(
                f"init_length={self.init_length} does not {rule} on a {grid.width}x{grid.height} board"
            )
        self.ray_length = int(self.config.ray_length or max(grid.shape))
        state_type = self.config.state_type
        if state_type == "auto":
            state_type = "ego"
        if state_type not in ("vector", "ego", "strategy", "grid"):
            raise ValueError(f"state_type must be 'auto', 'ego', 'vector', 'strategy' or 'grid', got {self.config.state_type!r}")
        self.state_type = state_type
        self.convolutional = state_type == "grid"
        self.strategic = state_type == "strategy"
        self.egocentric = state_type == "ego"
        self.ego_road = bool(self.config.ego_road) and players.reward.mode != "basic"      # give the ego state the road geometry (needs a road reward mode)
        self.shaper.want_scan = self.strategic          # the strategy state reads the move grades
        self.total = 0
        self.frame = 0
        self.labels: list[str] | None = None                # optional display name per snake (HUD)
        self.allocate(1)

    # ------------------------------------------------------------------
    # Allocation / reset
    # ------------------------------------------------------------------

    def allocate(self, total: int) -> None:
        n, cap = (int(total),), self.grid.cell_total
        self.total = int(total)
        self.body_x = np.zeros((self.total, cap), dtype=NP_INT)
        self.body_y = np.zeros((self.total, cap), dtype=NP_INT)
        self.head_idx = np.zeros(n, dtype=NP_INT)
        self.length = np.zeros(n, dtype=NP_INT)
        self.direction = np.zeros(n, dtype=NP_INT)
        self.hiatus = np.zeros(n, dtype=NP_INT)             # frames since the last meal
        self.action_count = np.zeros((self.total, 3), dtype=NP_INT)
        # Per-frame signals
        self.ate = np.zeros(n, dtype=bool)
        self.hit_wall = np.zeros(n, dtype=bool)
        self.hit_self = np.zeros(n, dtype=bool)
        self.completed = np.zeros(n, dtype=bool)
        self.low_usage = np.zeros(n, dtype=bool)
        self.dist = np.zeros(n, dtype=NP_FLOAT)             # food distance in cells
        self.food_dx = np.zeros(n, dtype=NP_FLOAT)
        self.food_dy = np.zeros(n, dtype=NP_FLOAT)
        self.obstacle = np.zeros((self.total, 3), dtype=NP_INT)
        self.food_ray = np.zeros((self.total, 3), dtype=NP_INT)

    def reset(self, total: int) -> None:
        """New episode with `total` snakes."""
        if self.frame > 0:                                  # grade the finished episode: survive first, then fast
            self.shaper.grade_episode(self.players.true_scores, self.players.deaths)
        self.grid.allocate(total)
        self.allocate(total)
        self.players.reset(total)
        self.shaper.allocate(total)                         # also picks the road of this episode
        self.frame = 0
        self._spawn(np.ones(self.total, dtype=bool))
        self.grid.place_food()
        self._measure()
        self.shaper.prev_dist[:] = self.dist
        self.shaper.scan(self, self.players.active & ~self.players.completed)

    def _spawn(self, mask: Array) -> None:
        if self.shaper.has_road:                            # tail on a random road square, body along the road
            start = self.shaper.setup_roads(mask, self.grid.rng)
            kernels.spawn_road(
                self.grid.cells, self.body_x, self.body_y, self.head_idx, self.length, self.direction,
                self.hiatus, self.action_count, mask, start, self.shaper.succ, self.init_length, self.grid.width,
            )
            return
        init = self.config.init_direction
        dirs = (self.grid.rng.integers(0, 4, self.total) if init is None
                else np.full(self.total, int(init) % 4)).astype(NP_INT)
        kernels.spawn(
            self.grid.cells, self.body_x, self.body_y, self.head_idx, self.length, self.direction,
            self.hiatus, self.action_count, mask, dirs, self.init_length, self.grid.wrap,
        )

    def restart(self, mask: Array) -> None:
        """Respawn the masked snakes with fresh food."""
        if not mask.any():
            return
        self.grid.food_x[mask] = -1
        self._spawn(mask)
        self.grid.place_food(mask)

    def _measure(self) -> None:
        g = self.grid
        kernels.food_distance(
            self.body_x, self.body_y, self.head_idx, g.food_x, g.food_y,
            g.width, g.height, g.wrap, self.dist, self.food_dx, self.food_dy,
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    @staticmethod
    def to_actions(actions: Array) -> Array:
        """Accept (N,) ints, (N, 1) ints or (N, 3) scores -> (N,) ints in {0, 1, 2}."""
        actions = np.asarray(actions)
        if actions.ndim == 2:
            if actions.shape[1] == 3:
                actions = actions.argmax(axis=-1)
            elif actions.shape[1] == 1:
                actions = actions[:, 0]
            else:
                raise ValueError(
                    f"Expected action shape (N,), (N, 1), or (N, 3), got {actions.shape}"
                )
        if actions.ndim != 1:
            raise ValueError(f"Unsupported action shape '{actions.shape}'")
        if not np.issubdtype(actions.dtype, np.number) or not np.all(np.isfinite(actions)):
            raise ValueError("Actions must contain finite numeric values")
        return np.clip(actions, 0, 2).astype(NP_INT)

    def update(self, actions: Array, verbose: bool = False) -> Array:
        """Advance every snake one cell, update lives / fitness and respawn eliminated snakes.

        Returns the mask of snakes that lost a life this frame.
        """
        g, pl = self.grid, self.players
        actions = self.to_actions(actions)
        if actions.shape[0] != self.total:
            raise ValueError(f"Expected {self.total} actions, got {actions.shape[0]}")
        active = pl.active & ~pl.completed
        self.shaper.pre_step(actions, active, self.head_idx)    # grades of the chosen moves, tree swaps applied

        kernels.step(
            g.cells, self.body_x, self.body_y, self.head_idx, self.length, self.direction,
            self.hiatus, self.action_count, actions, active, g.wrap,
            self.ate, self.hit_wall, self.hit_self,
        )

        # Food: replace what was eaten, and optionally move food nobody reached in time
        move = self.ate.copy()
        timeout = g.config.food_timeout
        if timeout:
            move |= active & ~self.ate & (self.hiatus > 0) & (self.hiatus % int(timeout) == 0)
        g.place_food(move)

        self._measure()
        self.completed = active & (self.length >= g.cell_total)
        if self.config.var_thresh is not None:
            kernels.low_usage(self.action_count, float(self.config.var_thresh), 20, self.low_usage)
        else:
            self.low_usage[:] = False

        died = self.hit_wall | self.hit_self | (self.hiatus >= pl.hiatus_limit)
        self.shaper.max_hiatus = pl.max_hiatus             # follows runtime changes of players.max_hiatus
        self.shaper.update(self.dist, self.ate, move, died, self.hiatus, self.low_usage, active, self.length)
        eliminated = pl.update(
            self.ate, self.hit_wall, self.hit_self, self.completed, self.hiatus, self.length,
            self.shaper.shape_pos, self.shape_neg, verbose=verbose, crash_scale=self.shaper.crash_scale,
        )

        # Respawn snakes that still have lives, then measure from the new position
        respawn = eliminated & pl.active
        self.restart(respawn)
        pl.restart(respawn)
        self._measure()
        self.shaper.prev_dist[:] = self.dist
        self.frame += 1
        self.shaper.scan(self, pl.active & ~pl.completed)     # grades of the next moves (reward + state)
        return eliminated

    @property
    def shape_neg(self) -> Array:
        return self.shaper.shape_neg

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------

    def cast_rays(self) -> None:
        g = self.grid
        kernels.cast_rays(
            g.cells, self.body_x, self.body_y, self.head_idx, self.direction,
            g.wrap, self.ray_length, self.obstacle, self.food_ray,
        )

    def get_state(self) -> Array:
        """State per snake. 'vector': (N, 12) features, see `kernels.sensors.get_state`.
        'ego': (N, 13) or (N, 8) egocentric, sensing + road positions, no move verdicts, see `kernels.sensors.ego_state`.
        'strategy': (N, 20) egocentric, ORACLE-FED: food bearing, rays, per-move grade / road gain / room / tail access, road gap; see `kernels.sensors.strategy_state`.
        'grid': (N, 3, W, H) image: body age, food, next-move cells, see `kernels.sensors.grid_state`."""
        g = self.grid
        if self.convolutional:
            out = np.zeros((self.total, kernels.GRID_CHANNELS, g.width, g.height), dtype=NP_FLOAT)
            kernels.grid_state(
                out, self.body_x, self.body_y, self.head_idx, self.length, self.direction, g.food_x, g.food_y,
                self.players.active, g.width, g.height, bool(g.wrap),
            )
            return out
        self.cast_rays()
        if self.egocentric:
            use_road = self.ego_road and self.shaper.has_road
            n = kernels.N_FEATURES_EGO if self.ego_road else kernels.N_FEATURES_EGO_PLAIN
            ego = np.zeros((self.total, n), dtype=NP_FLOAT)
            kernels.ego_state(
                ego, self.body_x, self.body_y, self.head_idx, self.length, self.direction, g.food_x, g.food_y,
                self.obstacle, self.shaper.rank, self.players.active, g.width, g.height, bool(g.wrap),
                self.ray_length, use_road,
            )
            return ego
        out = np.zeros((self.total, kernels.N_FEATURES), dtype=NP_FLOAT)
        kernels.get_state(
            out, self.direction, self.length, self.dist, self.food_dx, self.food_dy,
            self.obstacle, self.food_ray, g.width, g.height, g.wrap, self.ray_length,
        )
        if not self.strategic:
            return out
        sh = self.shaper
        full = np.zeros((self.total, kernels.N_FEATURES_STRATEGY), dtype=NP_FLOAT)
        kernels.strategy_state(
            full, self.direction, self.length, self.dist, self.food_dx, self.food_dy, self.obstacle,
            sh.kind, sh.gain, sh.space, sh.tail_ok, sh.phi, sh.synced, g.width, g.height, g.wrap, self.ray_length,
        )
        return full

    # ------------------------------------------------------------------
    # Accessors for rendering
    # ------------------------------------------------------------------

    def set_labels(self, labels: list[str] | None) -> None:
        self.labels = list(labels) if labels is not None else None

    def label(self, index: int) -> str | None:
        if self.labels is not None and 0 <= index < len(self.labels):
            return self.labels[index]
        return None

    def head(self, index: int) -> tuple[int, int]:
        i = self.head_idx[index]
        return int(self.body_x[index, i]), int(self.body_y[index, i])

    def body(self, index: int) -> Array:
        """(length, 2) cells of one snake, head first."""
        cap = self.body_x.shape[1]
        idx = (self.head_idx[index] - np.arange(self.length[index])) % cap
        return np.stack([self.body_x[index, idx], self.body_y[index, idx]], axis=-1)