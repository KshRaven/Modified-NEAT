"""Enums, type aliases and small helpers."""
from __future__ import annotations

from enum import Enum

__all__ = ["Position", "Color", "Action", "pipe_center", "clamp"]

Position = tuple[float, float]
Color    = tuple[int, int, int]


class Action(Enum):
    """Action of a bird."""
    NOOP = 0
    FLAP = 1


def pipe_center(gap_top: float, gap_bot: float) -> float:
    """Vertical centre of a pipe gap."""
    return 0.5 * (gap_top + gap_bot)


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value