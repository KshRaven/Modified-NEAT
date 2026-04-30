"""
Functional module for Car Racer game.
Contains enums, type aliases, and helper functions for track generation.
"""

from enum import Enum
from math import cos, sin, pi
from typing import Tuple
from collections import deque
from random import choice, randint, shuffle, seed as set_seed

import numpy as np


# ============================================================================
# TYPE ALIASES
# ============================================================================

Position = tuple[int, int]
Color = tuple[int, int, int]

# ============================================================================
# ENUMS
# ============================================================================

class Type(Enum):
    """Track segment type."""
    START = 0
    STRAIGHT = 1
    LEFT = 2
    RIGHT = 3


class Rotation(Enum):
    """Tile rotation angle."""
    DEG_0 = 0
    DEG_90 = 90
    DEG_180 = 180
    DEG_270 = 270


class Direction(Enum):
    """Movement direction."""
    RIGHT   = 0
    UP      = 1
    LEFT    = 2
    DOWN    = 3


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_next_direction(current: Direction, turn: Type) -> Direction:
    """
    Get the next direction based on the current direction and turn type.
    Directions are assumed to be in the order: RIGHT (0), UP (1), LEFT (2), DOWN (3).
    Turn types are STRAIGHT (0), LEFT (1), RIGHT (2).
    """
    if turn in [Type.STRAIGHT, Type.START]:
        return current
    elif turn == Type.LEFT:
        return Direction((current.value + 1) % 4)
    elif turn == Type.RIGHT:
        return Direction((current.value - 1) % 4)
    else:
        raise ValueError(f"Invalid turn type: {turn}")


def get_next_position(current: Position, direction: Direction) -> Position:
    """
    Get the next position based on the current position and direction.
    Directions are assumed to be in the order: RIGHT (0), UP (1), LEFT (2), DOWN (3).
    """
    x, y = current
    if direction in Direction:
        value = direction.value
        x_disp = round(cos(value * 0.5 * pi))
        y_disp = round(sin(value * 0.5 * pi))
        return (x + x_disp, y + (-y_disp))  # Invert y-axis for screen coordinates
    else:
        raise ValueError(f"Invalid direction: {direction}")


def direction_to_rotation(direction: Direction) -> Rotation:
    """
    Convert a direction to a rotation.
    Directions are assumed to be in the order: RIGHT (0), UP (1), LEFT (2), DOWN (3).
    Rotations are assumed to be in the order: DEG_0 (0), DEG_90 (90), DEG_180 (180), DEG_270 (270).
    """
    if direction in Direction:
        return Rotation(direction.value * 90)
    else:
        raise ValueError(f"Invalid direction: {direction}")


def get_next_rotation(current: Rotation, turn: Type) -> Rotation:
    """
    Assuming all tiles initially face up, but DEG_0 is positive x-axis (right), and rotation is clockwise:
     - STRAIGHT and START keep the same rotation
     - LEFT turns rotate counter-clockwise by 90 degrees
     - RIGHT turns rotate clockwise by 90 degrees
    """
    if turn in [Type.STRAIGHT, Type.START]:
        return current
    elif turn == Type.LEFT:
        return Rotation((current.value + 90) % 360)
    elif turn == Type.RIGHT:
        return Rotation((current.value - 90) % 360)
    else:
        raise ValueError(f"Invalid turn type: {turn}")
