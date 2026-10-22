"""Pygame drawing helpers."""
from __future__ import annotations

import pygame
from pygame import Surface

from .functional import Color

__all__ = ["render_text", "clear_text_cache", "draw_panel", "shade", "draw_dashed_rect"]

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


def draw_dashed_rect(screen: Surface, color: Color, rect: pygame.Rect, dash: int = 8, width: int = 2) -> None:
    """Dashed rectangle outline (used for the boundary when the snake can wrap)."""
    x0, y0, x1, y1 = rect.left, rect.top, rect.right - 1, rect.bottom - 1
    for x in range(x0, x1, dash * 2):
        pygame.draw.line(screen, color, (x, y0), (min(x + dash, x1), y0), width)
        pygame.draw.line(screen, color, (x, y1), (min(x + dash, x1), y1), width)
    for y in range(y0, y1, dash * 2):
        pygame.draw.line(screen, color, (x0, y), (x0, min(y + dash, y1)), width)
        pygame.draw.line(screen, color, (x1, y), (x1, min(y + dash, y1)), width)
