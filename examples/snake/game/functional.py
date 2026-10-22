"""Enums, type aliases and direction helpers."""
from __future__ import annotations

from enum import Enum

__all__ = [
    "Position", "Color", "Direction", "Action",
    "turn_direction", "get_next_position", "wrap_position", "action_towards",
]

Position = tuple[int, int]
Color    = tuple[int, int, int]


class Direction(Enum):
    """Travel direction, counter-clockwise from right (screen coordinates, y grows down)."""
    RIGHT = 0
    UP    = 1
    LEFT  = 2
    DOWN  = 3


class Action(Enum):
    """Relative action of the snake."""
    LEFT     = 0
    STRAIGHT = 1
    RIGHT    = 2


_DX = (1, 0, -1, 0)
_DY = (0, -1, 0, 1)


def turn_direction(current: int, action: int) -> int:
    """Direction after `action` (0 left = counter-clockwise, 1 straight, 2 right = clockwise)."""
    return (int(current) + 1 - int(action)) % 4


def get_next_position(current: Position, direction: int, wrap: bool = False, size: Position | None = None) -> Position:
    """Cell reached by one step in `direction`. With `wrap` the result is taken modulo `size`."""
    x, y = current
    x, y = x + _DX[int(direction) % 4], y + _DY[int(direction) % 4]
    return wrap_position((x, y), size) if wrap and size is not None else (x, y)


def wrap_position(position: Position, size: Position) -> Position:
    """Fold a cell back into the board (cross a boundary and appear on the opposite side)."""
    return position[0] % size[0], position[1] % size[1]


def action_towards(current: int, wanted: int) -> int:
    """Relative action that turns `current` into `wanted` (a reversal falls back to straight)."""
    diff = (int(wanted) - int(current)) % 4
    return {0: 1, 1: 0, 3: 2}.get(diff, 1)
