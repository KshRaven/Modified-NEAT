"""Track enums, type aliases and tile direction helpers."""
from __future__ import annotations

from enum import Enum
from math import cos, sin, pi

__all__ = [
    "Position", "Color",
    "Type", "Rotation", "Direction",
    "get_next_direction", "get_next_position", "direction_to_rotation", "get_next_rotation",
]

Position = tuple[int, int]
Color    = tuple[int, int, int]


class Type(Enum):
    """Track tile type."""
    START    = 0
    STRAIGHT = 1
    LEFT     = 2
    RIGHT    = 3


class Rotation(Enum):
    """Tile rotation in degrees. 0 faces right, clockwise positive."""
    DEG_0   = 0
    DEG_90  = 90
    DEG_180 = 180
    DEG_270 = 270


class Direction(Enum):
    """Travel direction on the grid, counter-clockwise from right."""
    RIGHT = 0
    UP    = 1
    LEFT  = 2
    DOWN  = 3


def get_next_direction(current: Direction, turn: Type) -> Direction:
    """Direction after passing a tile of type `turn`."""
    if turn in (Type.STRAIGHT, Type.START):
        return current
    if turn == Type.LEFT:
        return Direction((current.value + 1) % 4)
    if turn == Type.RIGHT:
        return Direction((current.value - 1) % 4)
    raise ValueError(f"Invalid turn type: {turn}")


def get_next_position(current: Position, direction: Direction) -> Position:
    """Grid cell reached by moving one step in `direction` (screen coordinates, y grows down)."""
    if direction not in Direction:
        raise ValueError(f"Invalid direction: {direction}")
    x, y = current
    angle = direction.value * 0.5 * pi
    return x + round(cos(angle)), y - round(sin(angle))


def direction_to_rotation(direction: Direction) -> Rotation:
    """Rotation matching a travel direction."""
    if direction not in Direction:
        raise ValueError(f"Invalid direction: {direction}")
    return Rotation(direction.value * 90)


def get_next_rotation(current: Rotation, turn: Type) -> Rotation:
    """Tile rotation after a turn: LEFT adds 90 degrees, RIGHT subtracts 90."""
    if turn in (Type.STRAIGHT, Type.START):
        return current
    if turn == Type.LEFT:
        return Rotation((current.value + 90) % 360)
    if turn == Type.RIGHT:
        return Rotation((current.value - 90) % 360)
    raise ValueError(f"Invalid turn type: {turn}")