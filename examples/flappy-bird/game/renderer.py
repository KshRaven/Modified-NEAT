"""Pygame rendering of the world: sky, pipes, floor, birds, debug lines, HUD and debug panel.

The scene is drawn by `draw_scene` for any scale (screen pixels per world pixel), so the pixel (retro) renderer can
reuse it on a small canvas with `hard=True` (no anti-aliased shapes, banded sky) and a reduced palette (`bits`).
Everything is drawn from raw geometry and small character grids (see sprite.py, scenery.py); there are no image files.
Shapes are square-like pixel art scaled with nearest-neighbour, so they stay crisp with or without the pixel renderer.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pygame
from pygame import Surface

from .constants import (
    COLOR_CLOUD, COLOR_FLOOR, COLOR_FLOOR_ALT, COLOR_FLOOR_EDGE, COLOR_GAP, COLOR_HILL, COLOR_HITBOX, COLOR_PIPE,
    COLOR_PIPE_DARK, COLOR_PIPE_LIGHT, COLOR_RAY, COLOR_SKY_BOTTOM, COLOR_SKY_TOP, COLOR_TEXT,
)
from .scenery import CELL, LAYERS, make_cloud, make_floor_tile, make_layer, make_pipe_cap, make_pipe_strip
from .sprite import ANIMATIONS, HALO_COLORS, make_bird, make_halo, rainbow
from .util import draw_dashed_line, draw_panel, mix, render_text, shade

if TYPE_CHECKING:
    from .bird import Birds
    from .config import WindowConfig

__all__ = ["FlappyRenderer", "quantize"]

HUD_BG = (14, 16, 22)
HUD_DIM = (140, 145, 155)
BUCKETS = 32                    # distinct bird hues (sprites are cached per hue)
CAP_H = 32                      # pipe cap height in world pixels
CAP_OVER = 8                    # how far the cap sticks out on each side
TILE_W = 48                     # floor stripe period in world pixels (a multiple of 6 cells)


def quantize(color, bits: int) -> tuple[int, int, int]:
    """Posterise a color to `bits` bits per channel (0 or >= 8 = unchanged)."""
    if not 1 <= bits < 8:
        return int(color[0]), int(color[1]), int(color[2])
    sh = 8 - bits
    half = (1 << sh) >> 1
    return tuple(min(255, ((int(v) >> sh) << sh) | half) for v in color[:3])


class FlappyRenderer:
    """Draws a `Birds` instance and its world. Holds all display state (fonts, cached backgrounds and sprites)."""

    def __init__(self, birds: Birds, config: WindowConfig) -> None:
        self.birds = birds
        self.config = config
        self.font: pygame.font.Font | None = None
        self.small: pygame.font.Font | None = None
        self._bg: Surface | None = None
        self._bg_key: tuple | None = None
        self._floor: Surface | None = None
        self._floor_key: tuple | None = None
        self._sprites: dict[tuple, Surface] = {}
        self._rotated: dict[tuple, Surface] = {}
        self._halos: dict[tuple, Surface] = {}
        self._scenery: dict[tuple, Surface] = {}     # scaled pixel-art layers, clouds and floor tile

    # ------------------------------------------------------------------
    # Fonts / layout
    # ------------------------------------------------------------------

    def set_fonts(self, size: int) -> None:
        self.font = pygame.font.Font(None, size)
        self.small = pygame.font.Font(None, max(12, int(size * 0.8)))

    def close(self) -> None:
        self.font = self.small = None
        self._bg = self._bg_key = self._floor = self._floor_key = None
        self._sprites.clear()
        self._rotated.clear()
        self._halos.clear()
        self._scenery.clear()

    def hud_height(self) -> int:
        lines = 2 if self.config.show_controls_hint else 1
        line = self.font.get_linesize() if self.font else self.config.font_size
        return lines * line + 10

    @staticmethod
    def _c(color, bits: int):
        return quantize(color, bits) if bits else tuple(color[:3])

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def draw_scene(
        self, surface: Surface, ox: int, oy: int, s: float, shown: int,
        level: int = 0, bits: int = 0, hard: bool = False,
    ) -> None:
        """Draw the world with its top-left corner at (ox, oy), `s` screen pixels per world pixel.
        `shown` is the focused bird; `level` >= 1 adds the gap lines and its hitbox."""
        w = self.birds.world
        clip = surface.get_clip()
        surface.set_clip(pygame.Rect(ox, oy, max(1, round(w.width * s)), max(1, round(w.height * s))).clip(clip))
        self._draw_background(surface, ox, oy, s, bits, hard)
        self._draw_pipes(surface, ox, oy, s, bits, hard)
        self._draw_floor(surface, ox, oy, s, bits, hard)
        self._draw_birds(surface, ox, oy, s, shown, bits, hard)
        self._draw_draft(surface, ox, oy, s, bits)
        if level >= 1:
            self._draw_debug(surface, ox, oy, s, shown, bits, hard)
        surface.set_clip(clip)

    @staticmethod
    def _rect(ox: int, oy: int, s: float, x: float, y: float, w: float, h: float) -> pygame.Rect:
        x0, y0 = round(ox + x * s), round(oy + y * s)
        x1, y1 = round(ox + (x + w) * s), round(oy + (y + h) * s)
        return pygame.Rect(x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    def _scaled(self, key: tuple, make, cell_px: float) -> Surface:
        """A pixel-art surface (one pixel per design cell) scaled with nearest-neighbour; cached."""
        img = self._scenery.get(key)
        if img is None:
            small = make()
            size = (max(1, round(small.get_width() * cell_px)), max(1, round(small.get_height() * cell_px)))
            img = self._scenery[key] = pygame.transform.scale(small, size)
        return img

    def _scroll_layer(self, surface: Surface, ox: int, oy: int, s: float, bits: int, name: str, distance: float) -> None:
        """Blit a tileable layer, bottom on the floor line, scrolling at its parallax factor."""
        w = self.birds.world
        cw, ch, factor = LAYERS[name]
        cell_px = CELL * s
        img = self._scaled((name, round(s, 4), bits), lambda: make_layer(name, bits), cell_px)
        tw = img.get_width()
        top = oy + round(w.floor_y * s) - img.get_height()
        x = ox - (distance * factor * s) % tw
        end = ox + round(w.width * s)
        while x < end:
            surface.blit(img, (round(x), top))
            x += tw

    def _draw_background(self, surface: Surface, ox: int, oy: int, s: float, bits: int, hard: bool) -> None:
        w = self.birds.world
        key = (w.width, w.height, w.floor_y, round(s, 4), bits, hard)
        if self._bg is None or key != self._bg_key:
            bw, bh = max(1, round(w.width * s)), max(1, round(w.height * s))
            bg = Surface((bw, bh))
            top, bot = self._c(COLOR_SKY_TOP, bits), self._c(COLOR_SKY_BOTTOM, bits)
            fy = max(1, round(w.floor_y * s))
            bands = 6                                               # stepped sky: flat colour bands
            for i in range(bands):
                y0, y1 = fy * i // bands, fy * (i + 1) // bands
                bg.fill(mix(top, bot, i / (bands - 1)), (0, y0, bw, max(1, y1 - y0)))
            bg.fill(bot, (0, fy, bw, bh - fy))
            self._bg, self._bg_key = bg, key
        surface.blit(self._bg, (ox, oy))
        # Floating clouds drift at a fraction of the scroll speed
        cell_px = CELL * s
        span = w.width + 240
        for i in range(5):
            cx = (i * 230 + 90 - w.distance * 0.15) % span - 120
            cy = 70 + 55 * ((i * 37) % 5)
            kind = i % 2
            img = self._scaled(("cloud", kind, round(s, 4), bits), lambda: make_cloud(kind, bits), cell_px)
            surface.blit(img, (round(ox + cx * s), round(oy + cy * s)))
        for name in ("band", "city", "green"):                      # horizon band, skyline, trees and bushes
            self._scroll_layer(surface, ox, oy, s, bits, name, w.distance)

    def _pipe_img(self, key: tuple, make, size: tuple[int, int]) -> Surface:
        img = self._scenery.get(key)
        if img is None:
            if len(self._scenery) > 2048:
                self._scenery.clear()
            img = self._scenery[key] = pygame.transform.scale(make(), size)
        return img

    def _draw_pipes(self, surface: Surface, ox: int, oy: int, s: float, bits: int, hard: bool) -> None:
        w = self.birds.world
        pw = w.pw
        wpx = max(2, round(pw * s))
        cap_w = max(3, round((pw + 2 * CAP_OVER) * s))
        cap_h = max(3, round(CAP_H * s))
        cells = max(5, round(pw / CELL))
        cap_cells = max(7, round((pw + 2 * CAP_OVER) / CELL))
        strip = self._pipe_img(("pstrip", wpx, bits), lambda: make_pipe_strip(cells, bits), (wpx, 1))
        cap = self._pipe_img(("pcap", cap_w, cap_h, bits), lambda: make_pipe_cap(cap_cells, max(5, round(CAP_H / CELL)), bits), (cap_w, cap_h))
        for k in range(w.count):
            x = float(w.px[k])
            if x > w.width + CAP_OVER or x + pw < -CAP_OVER:
                continue
            gt, gb = float(w.gap_top[k]), float(w.gap_bot[k])
            x0 = round(ox + x * s)
            for y0, y1, cap_y in ((0.0, gt, gt - CAP_H), (gb, w.floor_y, gb)):
                if y1 <= y0:
                    continue
                top, bot = round(oy + y0 * s), round(oy + y1 * s)
                if bot > top:
                    surface.blit(pygame.transform.scale(strip, (wpx, bot - top)), (x0, top))
                surface.blit(cap, (round(ox + (x - CAP_OVER) * s), round(oy + cap_y * s)))
            if w.vdir[k] != 0:                                    # moving gap: a small arrow on the cap
                cx = ox + (x + 0.5 * pw) * s
                cy = oy + 0.5 * (gt + gb) * s
                dy = -1 if w.vdir[k] < 0 else 1
                a = max(2, 10 * s)
                pygame.draw.polygon(surface, self._c(COLOR_GAP, bits), [(cx, cy + dy * a), (cx - a, cy - dy * a * 0.5), (cx + a, cy - dy * a * 0.5)])

    def _draw_floor(self, surface: Surface, ox: int, oy: int, s: float, bits: int, hard: bool) -> None:
        w = self.birds.world
        fh = w.height - w.floor_y
        cells_w, cells_h = TILE_W // CELL, max(5, math.ceil(fh / CELL))
        img = self._scaled(("floor", cells_h, round(s, 4), bits), lambda: make_floor_tile(cells_w, cells_h, bits), CELL * s)
        tw = img.get_width()
        top = oy + round(w.floor_y * s)
        x = ox - (w.distance * s) % tw
        end = ox + round(w.width * s)
        while x < end:
            surface.blit(img, (round(x), top))
            x += tw

    # ------------------------------------------------------------------
    # Birds
    # ------------------------------------------------------------------

    def _sprite(self, bucket: int, frame: int, s: float, bits: int, hard: bool) -> Surface:
        key = (bucket, frame, round(s, 4), bits, hard)
        spr = self._sprites.get(key)
        if spr is None:
            color = self._c(rainbow(bucket, BUCKETS), bits)
            spr = self._sprites[key] = make_bird(color, frame, (self.birds.bw, self.birds.bh), s, hard)
        return spr

    def _rotate(self, spr: Surface, key: tuple, tilt: int) -> Surface:
        k = (*key, tilt)
        img = self._rotated.get(k)
        if img is None:
            if len(self._rotated) > 4096:
                self._rotated.clear()
            img = self._rotated[k] = pygame.transform.rotate(spr, tilt)
        return img

    def _halo(self, rank: int, radius: int, bits: int, hard: bool) -> Surface:
        key = (rank, radius, bits, hard)
        halo = self._halos.get(key)
        if halo is None:
            halo = self._halos[key] = make_halo(self._c(HALO_COLORS[rank % 3], bits), radius, hard)
        return halo

    def drawn(self, shown: int) -> list[int]:
        """Birds to draw: all that are active (or the best `max_drawn`), the shown one included."""
        b, pl = self.birds, self.birds.players
        limit = self.config.max_drawn
        if limit is None:
            idx = [int(i) for i in range(b.total) if pl.active[i]]
        else:
            idx = b.ranking(int(limit))
        if 0 <= shown < b.total and pl.active[shown] and shown not in idx:
            idx.append(shown)
        return idx

    def _draw_birds(self, surface: Surface, ox: int, oy: int, s: float, shown: int, bits: int, hard: bool) -> None:
        b = self.birds
        w = b.world
        total = max(b.total, 1)
        top = b.ranking(max(0, int(self.config.halos)))
        order = [i for i in self.drawn(shown) if i not in top and i != shown]
        order += [i for i in reversed(top) if i != shown]
        if shown in self.drawn(shown):
            order.append(shown)
        frame_clock = b.frame // max(1, int(b.config.anime_time))
        radius = max(6, round(0.62 * max(b.bw, b.bh) * s))
        for i in order:
            cx, cy = ox + (b.x[i] + 0.5 * b.bw) * s, oy + (b.y[i] + 0.5 * b.bh) * s
            if i in top:                                           # rank halo under the bird
                halo = self._halo(top.index(i), radius, bits, hard)
                surface.blit(halo, halo.get_rect(center=(round(cx), round(cy))))
            tilt = float(b.tilt[i])
            frame = 1 if tilt <= -80 else ANIMATIONS[(frame_clock + i) % len(ANIMATIONS)]   # nose dive: wings stay level
            bucket = min(BUCKETS - 1, i * BUCKETS // total) if total > BUCKETS else (i * (BUCKETS - 1) // max(total - 1, 1) if total > 1 else 0)
            spr = self._sprite(bucket, frame, s, bits, hard)
            img = self._rotate(spr, (bucket, frame, round(s, 4), bits, hard), int(round(tilt / 5.0)) * 5)
            surface.blit(img, img.get_rect(center=(round(cx), round(cy))))
            if i == shown:
                pygame.draw.rect(surface, self._c(COLOR_GAP, bits), pygame.Rect(round(cx) - radius, round(cy) - radius, 2 * radius, 2 * radius), max(1, round(2 * s)))

    def _draw_draft(self, surface: Surface, ox: int, oy: int, s: float, bits: int) -> None:
        """Wind streaks and a direction arrow. They fade in while the draft is announced (so a player can time the
        jump) and are at full strength once it blows. Blocky rectangles only, so it suits the pixel renderer too."""
        w = self.birds.world
        if w.draft_phase == w.IDLE:
            return
        blow = w.draft_phase == w.BLOW
        prog = w.draft_progress
        horiz = w.draft_axis == w.AXIS_X
        sign = w.draft_dir
        W, F = float(w.width), float(w.floor_y)
        color = self._c((255, 255, 255) if blow else (255, 226, 120), bits)
        count = 24 if blow else max(2, int(4 + 14 * prog))
        length, thick = 46.0, max(1, round(3 * s))
        speed = 14.0 if blow else 6.0
        clock = float(w.draft_clock)
        for i in range(count):
            lane = (i * 97 + 31) % 211 / 211.0
            phase = (i * 53) % 101 / 101.0
            if horiz:
                x = (phase * (W + length) + sign * speed * clock) % (W + length) - length
                y = 20.0 + lane * (F - 40.0)
                r = self._rect(ox, oy, s, x, y, length, 3.0)
            else:
                y = (phase * (F + length) - sign * speed * clock) % (F + length) - length
                x = 10.0 + lane * (W - 20.0)
                r = self._rect(ox, oy, s, x, y, 3.0, length)
            surface.fill(color, r)
        # arrow near the top, blinking while announced, solid while blowing
        if blow or (w.draft_clock // 5) % 2 == 0:
            a = 16.0
            cx, cy = ox + 0.5 * W * s, oy + 44.0 * s
            if horiz:
                tip = [(cx + sign * a * s, cy), (cx - sign * a * 0.6 * s, cy - a * s), (cx - sign * a * 0.6 * s, cy + a * s)]
            else:
                tip = [(cx, cy - sign * a * s), (cx - a * s, cy + sign * a * 0.6 * s), (cx + a * s, cy + sign * a * 0.6 * s)]
            fill = self._c((255, 255, 255) if blow else (255, 214, 64), bits)
            pygame.draw.polygon(surface, fill, tip)
            pygame.draw.polygon(surface, self._c((84, 56, 71), bits), tip, max(1, round(2 * s)))

    def _draw_debug(self, surface: Surface, ox: int, oy: int, s: float, shown: int, bits: int, hard: bool) -> None:
        b = self.birds
        w = b.world
        if not (0 <= shown < b.total) or not b.players.active[shown]:
            return
        cx, cy = b.center(shown)
        t = w.t1
        mark = float(w.px[t]) + w.pass_off
        gt, gb = float(w.gap_top[t]), float(w.gap_bot[t])
        ray = self._c(COLOR_RAY, bits)
        width = max(1, round(2 * s))
        p0 = (ox + cx * s, oy + cy * s)
        pygame.draw.line(surface, ray, p0, (ox + mark * s, oy + gt * s), width)
        pygame.draw.line(surface, ray, p0, (ox + mark * s, oy + gb * s), width)
        draw_dashed_line(surface, self._c(COLOR_GAP, bits), (ox + mark * s, oy + gt * s), (ox + mark * s, oy + gb * s), max(3, round(8 * s)), width)
        mid = (ox + mark * s, oy + 0.5 * (gt + gb) * s)
        pygame.draw.line(surface, self._c(COLOR_GAP, bits), p0, mid, 1)
        # hitbox ellipse
        sc = float(b.config.hitbox_scale)
        r = pygame.Rect(0, 0, max(2, round(b.bw * sc * s)), max(2, round(b.bh * sc * s)))
        r.center = (round(p0[0]), round(p0[1]))
        pygame.draw.ellipse(surface, self._c(COLOR_HITBOX, bits), r, max(1, round(2 * s)))

    # ------------------------------------------------------------------
    # HUD / debug
    # ------------------------------------------------------------------

    def render_hud(self, surface: Surface, index: int, extra: list[str] | None = None) -> None:
        """Top bar: stats of the shown bird and the controls hint."""
        if self.font is None:
            return
        b, pl = self.birds, self.birds.players
        width = surface.get_width()
        bar = pygame.Rect(0, 0, width, self.hud_height())
        surface.fill(HUD_BG, bar)

        name = b.label(index)
        who = f"Bird {name}" if name else f"Bird {index + 1}/{b.total}"
        goal = f"/{b.goal}" if b.goal else ""
        left = (f"{who}   Score {int(pl.scores[index])}{goal}   Best {int(pl.true_scores[index])}   "
                f"Lives {int(pl.lives[index])}   Alive {pl.active_total}/{b.total}")
        if not pl.active[index]:
            left += "   OUT"
        elif pl.completed[index]:
            left += "   COMPLETE"
        surface.blit(render_text(self.font, left, COLOR_TEXT), (8, 4))

        right = "  ".join([f"Fitness {pl.scaled_fitness[index]:.1f}", b.shaper.mode_name.upper(), *(extra or [])])
        text = render_text(self.font, right, COLOR_PIPE_LIGHT)
        surface.blit(text, (width - text.get_width() - 8, 4))

        if self.config.show_controls_hint:
            hint = "Space pause   V debug   Tab/1-9 bird   P pixel   Shift +/- speed   Esc end"
            surface.blit(render_text(self.small, hint, HUD_DIM), (8, 4 + self.font.get_linesize()))

    def render_debug(self, surface: Surface, index: int, x: int, y: int) -> None:
        """Panel with the numbers the agent sees (verbose level 2+)."""
        if self.small is None:
            return
        b, pl = self.birds, self.birds.players
        w = b.world
        t = w.t1
        lines = [
            f"y {b.y[index]:.0f}  vel {b.vel[index]:.1f}  disp {b.disp[index]:+.1f}  tilt {b.tilt[index]:.0f}",
            f"hiatus {int(b.hiatus[index])}/{'inf' if pl.max_hiatus is None else int(pl.max_hiatus)}  since flap {int(b.tick[index])}",
            f"gap {w.gap_top[t]:.0f}..{w.gap_bot[t]:.0f}  err {b.err[index]:.0f}px  next pipe {b.dist_x:+.0f}px",
            f"actions idle/flap {int(b.action_count[index, 0])}/{int(b.action_count[index, 1])}" + ("  LOW USAGE" if b.low_usage[index] else ""),
            f"shape +{b.shaper.shape_pos[index]:.2f} -{b.shaper.shape_neg[index]:.2f}  skill {b.shaper.skill:.2f}",
        ]
        line_h = self.small.get_linesize()
        width = max(self.small.size(s)[0] for s in lines) + 12
        draw_panel(surface, pygame.Rect(x, y, width, line_h * len(lines) + 8))
        for i, text in enumerate(lines):
            surface.blit(render_text(self.small, text, COLOR_TEXT), (x + 6, y + 4 + i * line_h))