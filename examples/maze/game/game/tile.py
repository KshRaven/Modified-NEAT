from typing import Any

from pygame import Surface, draw

from .config import TileConfig, overwrite_config
from .functional import Position, Direction, CellType, ALL_DIRECTIONS
from .constants import COLOR_GRID_LINE


class Cell:
    """A single maze cell: 4 walls + a semantic type (start/goal/trap/empty)."""

    def __init__(
        self, position: Position, size: int,
        config: TileConfig | None = None, **params: dict[str, Any],
    ):
        if config is None:
            config = TileConfig()

        overwrite_config(config, params)
        self.config = config

        self.position: Position = position  # (row, col) — the only coordinate that matters
        self.size = int(size)
        self.position_abs: tuple[int, int] = (position[1] * size, position[0] * size)  # (x, y) pixels, for drawing only
        self.center_abs: tuple[float, float] = (
            self.position_abs[0] + size / 2.0, self.position_abs[1] + size / 2.0,
        )

        # Every wall starts up; the generator carves passages by clearing these.
        self.walls: dict[Direction, bool] = {d: True for d in ALL_DIRECTIONS}
        self.visited: bool = False  # generation bookkeeping (recursive backtracker)

        self.type: CellType = CellType.EMPTY

        self.floor_color = config.floor_color
        self.wall_color = config.wall_color
        self.start_color = config.start_color
        self.goal_color = config.goal_color
        self.trap_color = config.trap_color
        self.wall_thickness = config.wall_thickness

    @property
    def wall_count(self) -> int:
        return sum(1 for present in self.walls.values() if present)

    @property
    def is_dead_end(self) -> bool:
        return self.wall_count == 3

    def render(self, surface: Surface) -> None:
        x, y = self.position_abs
        s = self.size

        color = self.floor_color
        if self.type == CellType.START:
            color = self.start_color
        elif self.type == CellType.GOAL:
            color = self.goal_color
        elif self.type == CellType.TRAP:
            color = self.trap_color

        # Simple flat colored square — no curves, no rotation.
        draw.rect(surface, color, (x, y, s, s))
        draw.rect(surface, COLOR_GRID_LINE, (x, y, s, s), 1)

        t = self.wall_thickness
        if self.walls[Direction.UP]:
            draw.rect(surface, self.wall_color, (x, y, s, t))
        if self.walls[Direction.DOWN]:
            draw.rect(surface, self.wall_color, (x, y + s - t, s, t))
        if self.walls[Direction.LEFT]:
            draw.rect(surface, self.wall_color, (x, y, t, s))
        if self.walls[Direction.RIGHT]:
            draw.rect(surface, self.wall_color, (x + s - t, y, t, s))

    def __repr__(self) -> str:
        return f"Cell(pos={self.position}, type={self.type.name}, walls={self.wall_count})"
