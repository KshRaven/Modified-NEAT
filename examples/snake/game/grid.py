"""The board: occupancy map of every player and food placement."""
from __future__ import annotations

import numpy as np
from numpy import ndarray as Array

from . import kernels
from .config import GridConfig
from .constants import CELL_EMPTY, NP_CELL, NP_INT

__all__ = ["Grid"]


class Grid:
    """One (width x height) board per player, stored as a single (players, W, H) int8 array.

    Cell codes are in `constants` (CELL_EMPTY / FOOD / BODY / HEAD). With `wrap` the boundaries
    are not walls: a snake leaving one side appears on the opposite side.
    """

    def __init__(self, config: GridConfig | None = None) -> None:
        self.config = config if config is not None else GridConfig()
        self.width, self.height = (int(v) for v in self.config.grid_size)
        if self.width < 3 or self.height < 3:
            raise ValueError(f"Board too small: {self.config.grid_size}")
        self.rng = np.random.default_rng()      # Env-private RNG (food, random spawns); seeded by Game.seed
        self.total = 0
        self.allocate(1)

    @property
    def wrap(self) -> bool:
        return bool(self.config.wrap)

    @property
    def shape(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def cell_total(self) -> int:
        return self.width * self.height

    @property
    def max_distance(self) -> int:
        """Largest food distance (in cells) the board allows."""
        if self.wrap:
            return self.width // 2 + self.height // 2
        return self.width - 1 + self.height - 1

    def allocate(self, total: int) -> None:
        """Per-player arrays, recreated on every reset."""
        self.total = int(total)
        self.cells = np.full((self.total, self.width, self.height), CELL_EMPTY, dtype=NP_CELL)
        self.food_x = np.full(self.total, -1, dtype=NP_INT)
        self.food_y = np.full(self.total, -1, dtype=NP_INT)

    def place_food(self, mask: Array | None = None) -> None:
        """New food for the masked players (all when None)."""
        if mask is None:
            mask = np.ones(self.total, dtype=bool)
        if not mask.any():
            return
        kernels.place_food(self.cells, self.food_x, self.food_y, mask, self.rng.random(self.total))

    @property
    def free_total(self) -> Array:
        """Empty cells per player."""
        return np.count_nonzero(self.cells == CELL_EMPTY, axis=(1, 2))
