"""Pygame rendering of one snake's board: grid, boundary, food, snake, sensor rays, HUD and debug panel.

The scene is drawn by `draw_scene` for any cell size, so the pixel (retro) renderer can reuse it on a small
canvas with `hard=True` (no anti-aliased or rounded shapes) and a reduced palette (`bits`).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pygame
from pygame import Surface

from .constants import (
    COLOR_BODY, COLOR_BODY_ALT, COLOR_EMPTY, COLOR_EMPTY_ALT, COLOR_FOOD, COLOR_FOOD_LEAF, COLOR_HEAD,
    COLOR_RAY, COLOR_RAY_FOOD, COLOR_TEXT, COLOR_WALL, COLOR_WRAP, DIR_DX, DIR_DY,
)
from .util import draw_dashed_rect, draw_panel, render_text, shade

if TYPE_CHECKING:
    from .config import WindowConfig
    from .snake import Snakes

__all__ = ["SnakeRenderer", "quantize"]

HUD_BG = (14, 16, 22)
HUD_DIM = (140, 145, 155)
_REL = (1, 0, 3)            # ray r (left, straight, right) -> direction offset (matches kernels.cast_rays)


def quantize(color, bits: int) -> tuple[int, int, int]:
    """Posterise a color to `bits` bits per channel (0 or >= 8 = unchanged)."""
    if not 1 <= bits < 8:
        return int(color[0]), int(color[1]), int(color[2])
    sh = 8 - bits
    half = (1 << sh) >> 1
    return tuple(min(255, ((int(v) >> sh) << sh) | half) for v in color[:3])


class SnakeRenderer:
    """Draws a `Snakes` instance. Holds all display state (fonts, the cached board)."""

    def __init__(self, snakes: Snakes, config: WindowConfig) -> None:
        self.snakes = snakes
        self.config = config
        self.font: pygame.font.Font | None = None
        self.small: pygame.font.Font | None = None
        self._board: Surface | None = None
        self._board_key: tuple | None = None

    # ------------------------------------------------------------------
    # Fonts / layout
    # ------------------------------------------------------------------

    def set_fonts(self, size: int) -> None:
        self.font = pygame.font.Font(None, size)
        self.small = pygame.font.Font(None, max(12, int(size * 0.8)))

    def close(self) -> None:
        self.font = self.small = None
        self._board = self._board_key = None

    def hud_height(self) -> int:
        lines = 2 if self.config.show_controls_hint else 1
        line = self.font.get_linesize() if self.font else self.config.font_size
        return lines * line + 10

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    @staticmethod
    def _c(color, bits: int):
        return quantize(color, bits) if bits else tuple(color[:3])

    def draw_scene(
        self, surface: Surface, index: int, bx: int, by: int, cell: int,
        level: int = 0, bits: int = 0, hard: bool = False,
    ) -> None:
        """Draw snake `index`'s board with its top-left cell at (bx, by). `level` >= 1 adds the sensor rays."""
        self._draw_board(surface, bx, by, cell, bits)
        self._draw_food(surface, index, bx, by, cell, bits, hard)
        self._draw_snake(surface, index, bx, by, cell, bits, hard)
        if level >= 1:
            self._draw_rays(surface, index, bx, by, cell, bits, hard)

    def _draw_board(self, surface: Surface, bx: int, by: int, cell: int, bits: int) -> None:
        g = self.snakes.grid
        pad = max(1, cell // 8)
        key = (g.width, g.height, g.wrap, cell, bits, self.config.grid_lines)
        if self._board is None or key != self._board_key:
            w, h = g.width * cell, g.height * cell
            board = Surface((w + 2 * pad, h + 2 * pad))
            board.fill(self._c(self.config.background_color, bits))
            area = Surface((w, h))
            area.fill(self._c(COLOR_EMPTY, bits))
            if self.config.grid_lines:
                alt = self._c(COLOR_EMPTY_ALT, bits)
                for x in range(g.width):
                    for y in range(g.height):
                        if (x + y) & 1:
                            area.fill(alt, (x * cell, y * cell, cell, cell))
            board.blit(area, (pad, pad))
            rect = pygame.Rect(0, 0, w + 2 * pad, h + 2 * pad)
            if g.wrap:      # dashed: the boundary can be crossed
                draw_dashed_rect(board, self._c(COLOR_WRAP, bits), rect, dash=max(2, cell // 2), width=pad)
            else:           # solid: the boundary kills
                pygame.draw.rect(board, self._c(COLOR_WALL, bits), rect, pad)
            self._board, self._board_key = board, key
        surface.blit(self._board, (bx - pad, by - pad))

    def _draw_food(self, surface: Surface, index: int, bx: int, by: int, cell: int, bits: int, hard: bool) -> None:
        g = self.snakes.grid
        fx, fy = int(g.food_x[index]), int(g.food_y[index])
        if fx < 0:
            return
        x0, y0 = bx + fx * cell, by + fy * cell
        red, leaf = self._c(COLOR_FOOD, bits), self._c(COLOR_FOOD_LEAF, bits)
        if hard:
            gap = 1 if cell >= 4 else 0
            surface.fill(red, (x0 + gap, y0 + gap + (1 if cell >= 6 else 0), cell - 2 * gap, cell - 2 * gap - (1 if cell >= 6 else 0)))
            surface.fill(leaf, (x0 + cell // 2, y0, max(1, cell // 4), max(1, cell // 4)))
            return
        cx, cy = x0 + cell / 2, y0 + cell / 2 + cell * 0.04
        pygame.draw.circle(surface, red, (cx, cy), cell * 0.38)
        pygame.draw.circle(surface, leaf, (cx + cell * 0.14, cy - cell * 0.38), max(1.0, cell * 0.12))

    def _draw_snake(self, surface: Surface, index: int, bx: int, by: int, cell: int, bits: int, hard: bool) -> None:
        sn = self.snakes
        body = sn.body(index)
        length = len(body)
        gap = max(1, cell // 12) if not hard else (1 if cell >= 4 else 0)
        size = cell - 2 * gap
        alive = bool(sn.players.active[index])
        radius = 0 if hard else max(2, cell // 4)
        for k in range(length - 1, -1, -1):
            x, y = int(body[k, 0]), int(body[k, 1])
            if k == 0:
                color = COLOR_HEAD
            else:
                color = shade(COLOR_BODY if k & 1 else COLOR_BODY_ALT, 1.0 - 0.4 * k / max(length, 1))
            if not alive:
                color = shade(color, 0.45)
            rect = (bx + x * cell + gap, by + y * cell + gap, size, size)
            if hard:
                surface.fill(self._c(color, bits), rect)
            else:
                pygame.draw.rect(surface, self._c(color, bits), rect, border_radius=radius)

        # Eyes on the head
        x, y = int(body[0, 0]), int(body[0, 1])
        d = int(sn.direction[index])
        dx, dy = int(DIR_DX[d]), int(DIR_DY[d])
        cx, cy = bx + x * cell + cell / 2, by + y * cell + cell / 2
        eye = self._c((10, 12, 16), bits)
        for side in (-1, 1):
            ex = cx + dx * cell * 0.18 + (-dy) * side * cell * 0.2
            ey = cy + dy * cell * 0.18 + dx * side * cell * 0.2
            if hard:
                if cell >= 6:
                    surface.fill(eye, (round(ex), round(ey), 1, 1))
            else:
                pygame.draw.circle(surface, eye, (ex, ey), max(1.0, cell * 0.09))

    def _draw_rays(self, surface: Surface, index: int, bx: int, by: int, cell: int, bits: int, hard: bool) -> None:
        sn, g = self.snakes, self.snakes.grid
        sn.cast_rays()
        hx, hy = sn.head(index)
        d0 = int(sn.direction[index])
        ray_color, food_color = self._c(COLOR_RAY, bits), self._c(COLOR_RAY_FOOD, bits)
        dot = max(1, cell // 8)
        for r in range(3):
            d = (d0 + _REL[r]) % 4
            dx, dy = int(DIR_DX[d]), int(DIR_DY[d])
            blocked = int(sn.obstacle[index, r])
            seen = int(sn.food_ray[index, r])
            reach = min(blocked, sn.ray_length)
            for s in range(1, reach + 1):
                x, y = hx + dx * s, hy + dy * s
                if g.wrap:
                    x, y = x % g.width, y % g.height
                elif not (0 <= x < g.width and 0 <= y < g.height):
                    break                                   # the wall: nothing to draw beyond the board
                px, py = bx + x * cell + cell // 2, by + y * cell + cell // 2
                hit = s == blocked
                color = food_color if s == seen else ray_color
                if hard:
                    size = dot * (2 if hit else 1)
                    surface.fill(color, (px - size // 2, py - size // 2, size, size))
                else:
                    pygame.draw.circle(surface, color, (px, py), dot * (2 if hit else 1))

    # ------------------------------------------------------------------
    # HUD / debug
    # ------------------------------------------------------------------

    def render_hud(self, surface: Surface, index: int, extra: list[str] | None = None) -> None:
        """Top bar: stats of the shown snake, boundary mode and the controls hint."""
        if self.font is None:
            return
        sn, pl, g = self.snakes, self.snakes.players, self.snakes.grid
        width = surface.get_width()
        bar = pygame.Rect(0, 0, width, self.hud_height())
        surface.fill(HUD_BG, bar)

        name = sn.label(index)
        who = f"Snake {name}" if name else f"Snake {index + 1}/{sn.total}"
        left = (f"{who}   Score {int(pl.scores[index])}   Best {int(pl.true_scores[index])}   "
                f"Lives {int(pl.lives[index])}   Length {int(sn.length[index])}")
        if not pl.active[index]:
            left += "   OUT"
        elif pl.completed[index]:
            left += "   COMPLETE"
        surface.blit(render_text(self.font, left, COLOR_TEXT), (8, 4))

        right = "  ".join([f"Fitness {pl.scaled_fitness[index]:.1f}", "WRAP" if g.wrap else "WALLS", *(extra or [])])
        text = render_text(self.font, right, COLOR_WRAP if g.wrap else COLOR_WALL)
        surface.blit(text, (width - text.get_width() - 8, 4))

        if self.config.show_controls_hint:
            hint = "Space pause   V debug   Tab/1-9 snake   P pixel   Shift +/- speed   Esc end"
            surface.blit(render_text(self.small, hint, HUD_DIM), (8, 4 + self.font.get_linesize()))

    def render_debug(self, surface: Surface, index: int, x: int, y: int) -> None:
        """Panel with the numbers the agent sees (verbose level 2+)."""
        if self.small is None:
            return
        sn = self.snakes
        sn.cast_rays()
        pl = sn.players
        lines = [
            f"dir {int(sn.direction[index])}  hiatus {int(sn.hiatus[index])}/{int(pl.max_hiatus)}",
            f"food dist {sn.dist[index]:.1f}  d=({sn.food_dx[index]:+.0f}, {sn.food_dy[index]:+.0f})",
            f"obstacle L/S/R {' '.join(str(int(v)) for v in sn.obstacle[index])}",
            f"food ray L/S/R {' '.join(str(int(v)) for v in sn.food_ray[index])}",
            f"actions L/S/R {' '.join(str(int(v)) for v in sn.action_count[index])}" + ("  LOW USAGE" if sn.low_usage[index] else ""),
            f"shape +{sn.shaper.shape_pos[index]:.2f} -{sn.shaper.shape_neg[index]:.2f}",
        ]
        line_h = self.small.get_linesize()
        width = max(self.small.size(s)[0] for s in lines) + 12
        draw_panel(surface, pygame.Rect(x, y, width, line_h * len(lines) + 8))
        for i, text in enumerate(lines):
            surface.blit(render_text(self.small, text, COLOR_TEXT), (x + 6, y + 4 + i * line_h))
