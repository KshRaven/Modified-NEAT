"""
Functional module for the Maze game.
Enums, type aliases, and pure helper functions for maze generation & movement.
Movement is always ONE discrete cell -> an adjacent cell (e.g. (1,'A') -> (2,'A')).
There is no continuous/pixel movement anywhere in the underlying simulation.
"""

from enum import Enum

Position = tuple[int, int]   # (row, col) grid-cell coordinate — the ONLY position representation used internally
Color = tuple[int, int, int]


# ============================================================================
# ENUMS
# ============================================================================

class Direction(Enum):
    """Discrete movement direction. Value == discrete action index (0-3)."""
    UP    = 0
    RIGHT = 1
    DOWN  = 2
    LEFT  = 3


class CellType(Enum):
    """Semantic role of a maze cell."""
    EMPTY = 0
    START = 1
    GOAL  = 2
    TRAP  = 3


ALL_DIRECTIONS: tuple[Direction, ...] = (
    Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT,
)

# Row/Col displacement for each direction (row grows downward, like screen-space)
_DELTA: dict[Direction, Position] = {
    Direction.UP:    (-1, 0),
    Direction.RIGHT: (0, 1),
    Direction.DOWN:  (1, 0),
    Direction.LEFT:  (0, -1),
}

_OPPOSITE: dict[Direction, Direction] = {
    Direction.UP:    Direction.DOWN,
    Direction.DOWN:  Direction.UP,
    Direction.LEFT:  Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


def get_delta(direction: Direction) -> Position:
    return _DELTA[direction]


def get_next_position(current: Position, direction: Direction) -> Position:
    dr, dc = _DELTA[direction]
    r, c = current
    return (r + dr, c + dc)


def opposite(direction: Direction) -> Direction:
    return _OPPOSITE[direction]


def vector_to_direction(dx: float, dy: float, cutoff: float = 0.35) -> Direction | None:
    """
    Snap a continuous 2D action vector (+x = right/col+, +y = down/row+) to the
    dominant cardinal Direction. This is how continuous actions still only ever
    produce a single discrete grid-cell step. Returns None (no move this step)
    if the vector doesn't clear `cutoff` magnitude on its dominant axis.
    """
    ax, ay = abs(dx), abs(dy)
    if ax < cutoff and ay < cutoff:
        return None
    if ax >= ay:
        return Direction.RIGHT if dx > 0 else Direction.LEFT
    else:
        return Direction.DOWN if dy > 0 else Direction.UP


def direction_to_vector(direction: Direction) -> Position:
    return _DELTA[direction]
