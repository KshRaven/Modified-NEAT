"""Optional retro (pixel art) renderer. Visual only, never touches the simulation.

The board is drawn on a small canvas (one canvas pixel = `pixel_size` screen pixels) with hard-edged shapes and
a reduced palette, then scaled up once with nearest-neighbour. The HUD and the debug panel stay at full resolution.
"""
from __future__ import annotations

from pygame import Surface
import pygame

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

    def effective_cell(self) -> int:
        """Cell size in screen pixels: a whole number of canvas pixels, so every canvas pixel is square."""
        px = self.px
        return max(2 * px, (int(self.config.cell_size) // px) * px)

    def draw(self, win, index: int, level: int, top: int) -> None:
        """Draw the board of snake `index` onto `win.screen`, its area starting `top` pixels from the top."""
        px, cell = self.px, win.cell
        cp = max(1, cell // px)                                     # canvas pixels per cell
        g, m = win.grid, win.margin
        cw, ch = (g.width + 2 * m) * cp, (g.height + 2 * m) * cp
        if self._small is None or self._small.get_size() != (cw, ch):
            self._small = Surface((cw, ch)).convert()
            self._up = None
        self._small.fill(quantize(win.bg_color, self.bits))
        win.renderer.draw_scene(self._small, index, m * cp, m * cp, cp, level, bits=self.bits, hard=True)

        size = (cw * px, ch * px)
        if self._up is None or self._up.get_size() != size:
            self._up = Surface(size).convert()
        pygame.transform.scale(self._small, size, self._up)         # single hard upscale
        win.screen.blit(self._up, (0, top))
