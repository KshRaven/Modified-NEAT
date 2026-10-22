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
        self.shaper = shaper if shaper is not None else RewardShaper(players.reward, players.max_hiatus)
        self.init_length = int(self.config.init_length)
        if min(grid.shape) <= self.init_length * 2:
            raise ValueError(f"init_length {self.init_length} does not fit a {grid.width}x{grid.height} board")
        self.ray_length = int(self.config.ray_length or max(grid.shape))
        self.convolutional = self.config.state_type == "grid"
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
        self.grid.allocate(total)
        self.allocate(total)
        self.players.reset(total)
        self.shaper.allocate(total)
        self.frame = 0
        self._spawn(np.ones(self.total, dtype=bool))
        self.grid.place_food()
        self._measure()
        self.shaper.prev_dist[:] = self.dist

    def _spawn(self, mask: Array) -> None:
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
            actions = actions.argmax(axis=-1) if actions.shape[1] == 3 else actions[:, 0]
        if actions.ndim != 1:
            raise ValueError(f"Unsupported action shape '{actions.shape}'")
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

        died = self.hit_wall | self.hit_self | (self.hiatus >= pl.max_hiatus)
        self.shaper.max_hiatus = int(pl.max_hiatus)        # follows runtime changes of players.max_hiatus
        self.shaper.update(self.dist, move, died, self.hiatus, self.low_usage, active)   # moved food = no progress term
        eliminated = pl.update(
            self.ate, self.hit_wall, self.hit_self, self.completed, self.hiatus, self.length,
            self.shaper.shape_pos, self.shape_neg, verbose=verbose,
        )

        # Respawn snakes that still have lives, then measure from the new position
        respawn = eliminated & pl.active
        self.restart(respawn)
        pl.restart(respawn)
        self._measure()
        self.shaper.prev_dist[:] = self.dist
        self.frame += 1
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
        'grid': (N, 1, W, H) normalized board."""
        g = self.grid
        if self.convolutional:
            out = np.zeros((self.total, 1, g.width, g.height), dtype=NP_FLOAT)
            kernels.grid_state(g.cells, out)
            return out
        self.cast_rays()
        out = np.zeros((self.total, kernels.N_FEATURES), dtype=NP_FLOAT)
        kernels.get_state(
            out, self.direction, self.length, self.dist, self.food_dx, self.food_dy,
            self.obstacle, self.food_ray, g.width, g.height, g.wrap, self.ray_length,
        )
        return out

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