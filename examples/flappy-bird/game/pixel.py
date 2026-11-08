"""Optional retro (pixel art) renderer. Visual only, never touches the simulation.

The world is drawn on a small canvas (one canvas pixel = `pixel_size` screen pixels) with hard-edged shapes and a
reduced palette, then scaled up once with nearest-neighbour. The HUD and the debug panel stay at full resolution.
"""
from __future__ import annotations

import pygame
from pygame import Surface

from .renderer import quantize

__all__ = ["PixelRenderer"]


class PixelRenderer:
    def __init__(self, config) -> None:
        self.config = config
        self._small: Surface | None = None      # low-resolution canvas
        self._up: Surface | None = None         # upscaled canvas

    @property
    def px(self) -> int:
        return max(1, int(self.config.pixel_size))

    @property
    def bits(self) -> int:
        b = int(self.config.pixel_color_bits)
        return b if 1 <= b < 8 else 0

    def canvas_size(self, world_w: int, world_h: int) -> tuple[int, int]:
        """Canvas size in canvas pixels: a whole number, so every canvas pixel is square on screen."""
        s = float(self.config.scale) / self.px
        return max(1, round(world_w * s)), max(1, round(world_h * s))

    def board_size(self, world_w: int, world_h: int) -> tuple[int, int]:
        cw, ch = self.canvas_size(world_w, world_h)
        return cw * self.px, ch * self.px

    def draw(self, win, index: int, level: int, left: int, top: int) -> None:
        """Draw the world onto `win.screen`, its area starting at (`left`, `top`)."""
        w = win.world
        px = self.px
        cw, ch = self.canvas_size(w.width, w.height)
        if self._small is None or self._small.get_size() != (cw, ch):
            self._small = Surface((cw, ch)).convert()
            self._up = None
        self._small.fill(quantize(win.bg_color, self.bits))
        win.renderer.draw_scene(self._small, 0, 0, float(self.config.scale) / px, index, level, bits=self.bits, hard=True)

        size = (cw * px, ch * px)
        if self._up is None or self._up.get_size() != size:
            self._up = Surface(size).convert()
        pygame.transform.scale(self._small, size, self._up)         # single hard upscale
        win.screen.blit(self._up, (left, top))
