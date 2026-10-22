"""Pygame drawing helpers and small Numba utilities."""
from __future__ import annotations

import math

import numpy as np
import pygame
from numba import njit, types
from numba.extending import overload
from pygame import Surface, draw, transform

from .functional import Color, Position

__all__ = [
    "scale_image", "blit_rotate_center", "draw_sector", "draw_arrow", "draw_panel",
    "render_text", "clear_text_cache",
    "grid", "sign", "wrap180",
]


# ----------------------------------------------------------------------------
# Pygame helpers
# ----------------------------------------------------------------------------

def scale_image(img: Surface, factor: float) -> Surface:
    """Scale `img` by `factor` (1.0 = unchanged)."""
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return transform.scale(img, size)


def blit_rotate_center(surface: Surface, image: Surface, top_left: tuple[int, int], angle: float) -> None:
    """Blit `image` rotated by `angle` degrees around its own center."""
    rotated_image = transform.rotate(image, angle)
    rect = rotated_image.get_rect(center=image.get_rect(topleft=top_left).center)
    surface.blit(rotated_image, rect.topleft)


def draw_sector(
    surface: Surface, color: Color, center: tuple[int, int],
    radius: float, start_angle: float, end_angle: float,
) -> None:
    """Draw a filled pie slice. Angles are in radians."""
    span = end_angle - start_angle
    count = max(5, int(span * radius / 10))
    points = [center]
    for i in range(count + 1):
        a = start_angle + span * i / count
        points.append((center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)))
    draw.polygon(surface, color, points)


def draw_arrow(
    surface: Surface, start: Position, end: Position,
    width: int = 3, head_length: float = 10, head_angle: float = 30,
    color: Color = (0, 0, 0),
) -> None:
    """Draw a line from `start` to `end` with a triangular head at `end`."""
    draw.line(surface, color, start, end, width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    left  = angle + math.radians(180 - head_angle)
    right = angle - math.radians(180 - head_angle)
    draw.polygon(surface, color, [
        end,
        (end[0] + head_length * math.cos(left),  end[1] + head_length * math.sin(left)),
        (end[0] + head_length * math.cos(right), end[1] + head_length * math.sin(right)),
    ])


_PANEL_CACHE: dict[tuple, Surface] = {}
_TEXT_CACHE:  dict[tuple, Surface] = {}


def draw_panel(
    surface: Surface, rect: pygame.Rect, color: Color = (0, 0, 0),
    alpha: int = 120, radius: int = 6,
) -> None:
    """Blit a translucent rounded rectangle. The panel surface is cached per size, color and alpha."""
    if alpha <= 0 or rect.w <= 0 or rect.h <= 0:
        return
    key = (rect.w, rect.h, tuple(color), int(alpha), radius)
    panel = _PANEL_CACHE.get(key)
    if panel is None:
        if len(_PANEL_CACHE) > 64:
            _PANEL_CACHE.clear()
        panel = Surface((rect.w, rect.h), pygame.SRCALPHA)
        draw.rect(panel, (*color, int(alpha)), panel.get_rect(), border_radius=radius)
        _PANEL_CACHE[key] = panel
    surface.blit(panel, rect.topleft)


def render_text(font: pygame.font.Font, text: str, antialias: bool, color: tuple) -> Surface:
    """`font.render` with a cache. Call `clear_text_cache` when fonts are recreated."""
    key = (id(font), text, bool(antialias), tuple(color))
    surf = _TEXT_CACHE.get(key)
    if surf is None:
        if len(_TEXT_CACHE) > 512:
            _TEXT_CACHE.clear()
        surf = _TEXT_CACHE[key] = font.render(text, antialias, color)
    return surf


def clear_text_cache() -> None:
    _TEXT_CACHE.clear()


# ----------------------------------------------------------------------------
# Numba helpers
# ----------------------------------------------------------------------------

@njit(cache=True)
def sign(v: float) -> int:
    if v > 0.0: return 1
    if v < 0.0: return -1
    return 0


@njit(cache=True)
def wrap180(a: float) -> float:
    """Wrap an angle in degrees to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def grid(dims: tuple[int, ...]) -> tuple[int, list[int | tuple[int, ...]]]:
    """Flat iteration space over `dims` (1 to 3 dimensions) for `prange` loops.

    Returns `(total, indices)`. `dims` must be a tuple so the Numba overload
    below can specialise on its length.
    """
    if len(dims) <= 0 or len(dims) > 3:
        raise ValueError("Unsupported number of dims")
    return math.prod(dims), list(np.ndindex(*dims))


@overload(grid)
def _grid_overload(dims: tuple[int, ...]):
    if not isinstance(dims, (types.UniTuple, types.Tuple)):
        return None

    ndim = len(dims)
    if ndim <= 0 or ndim > 3:
        raise TypeError("Unsupported number of dims (grid supports 1, 2 or 3)")

    if ndim == 1:
        def impl(dims: tuple[int, ...]):
            n0 = dims[0]
            return n0, [i for i in range(n0)]
        return impl

    if ndim == 2:
        def impl(dims: tuple[int, ...]):
            n0, n1 = dims[0], dims[1]
            return n0 * n1, [(i, j) for i in range(n0) for j in range(n1)]
        return impl

    def impl(dims: tuple[int, ...]):
        n0, n1, n2 = dims[0], dims[1], dims[2]
        return n0 * n1 * n2, [(i, j, k) for i in range(n0) for j in range(n1) for k in range(n2)]
    return impl