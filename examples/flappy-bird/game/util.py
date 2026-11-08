"""Pygame drawing helpers."""
from __future__ import annotations

import math

import pygame
from pygame import Surface

from .functional import Color

__all__ = ["render_text", "clear_text_cache", "draw_panel", "shade", "mix", "draw_dashed_line"]

_TEXT_CACHE: dict[tuple, Surface] = {}


def clear_text_cache() -> None:
    _TEXT_CACHE.clear()


def render_text(font: pygame.font.Font, text: str, color: Color = (235, 235, 235)) -> Surface:
    """Render `text`, cached by (font, text, color)."""
    key = (id(font), text, color)
    surf = _TEXT_CACHE.get(key)
    if surf is None:
        if len(_TEXT_CACHE) > 512:
            _TEXT_CACHE.clear()
        surf = _TEXT_CACHE[key] = font.render(text, True, color)
    return surf


def draw_panel(screen: Surface, rect: pygame.Rect, color=(0, 0, 0), alpha: int = 150) -> None:
    """Translucent rounded panel behind HUD text."""
    panel = Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(panel, (*color, alpha), panel.get_rect(), border_radius=6)
    screen.blit(panel, rect.topleft)


def shade(color: Color, k: float) -> Color:
    """Multiply a color by `k` (clamped)."""
    return tuple(max(0, min(255, int(v * k))) for v in color[:3])


def mix(a: Color, b: Color, t: float) -> Color:
    """Linear blend of two colors (t = 0 -> a, t = 1 -> b)."""
    return tuple(max(0, min(255, int(a[i] + (b[i] - a[i]) * t))) for i in range(3))


def draw_dashed_line(screen: Surface, color: Color, start, end, dash: int = 8, width: int = 1) -> None:
    """Dashed line between two points."""
    x0, y0 = start
    x1, y1 = end
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-6:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    d = 0.0
    while d < length:
        e = min(d + dash, length)
        pygame.draw.line(screen, color, (x0 + ux * d, y0 + uy * d), (x0 + ux * e, y0 + uy * e), width)
        d += dash * 2