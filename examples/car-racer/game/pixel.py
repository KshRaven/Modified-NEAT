"""
Pixel mode: a separate, optional, visual-only renderer for the retro (NES / arcade) look.

It never touches the simulation. Instead of filtering a finished full-resolution frame, it draws the
scene directly onto a small canvas (one canvas pixel = `pixel_size` screen pixels) with hard-edged
primitives and then scales that canvas up once with nearest-neighbour:

    track   : posterised once per track (cached), then cropped + shrunk to the canvas each frame
    cars    : sprites shrunk to a few pixels, alpha cut to 0/255, palette reduced, rotated without smoothing
    streaks : square dots in three tones (bright head, base, dithered dark tail), no alpha blending
    rings   : 1-pixel rank circles; rank labels and the HUD stay sharp at full resolution

No per-frame surfarray work: the only numpy pixel edits run on the cached track (once) and on tiny
sprites (once per cache miss), each wrapped so a failure just skips the palette reduction.
"""
from __future__ import annotations

import numpy as np
import pygame
from pygame import Surface

from .car import DRAFT_VISIBLE, STREAK_REAR_CARS
from .race import trail_points
from .sprite import HALO_COLORS
from .util import draw_panel


class PixelRenderer:
    def __init__(self, config) -> None:
        self.config = config
        self._small: Surface | None = None       # low-resolution canvas
        self._up: Surface | None = None          # nearest-neighbour upscaled copy, blitted to the screen
        self._track_src: Surface | None = None   # the window's track surface this cache was built from
        self._track: Surface | None = None       # posterised copy of it
        self._sprites: dict = {}                 # id(full sprite) -> (full sprite, shrunk pixel sprite)
        self._sprite_k = 0.0

    # ------------------------------------------------------------------ palette helpers
    def _bits(self) -> int:
        b = int(self.config.pixel_color_bits)
        return b if 1 <= b < 8 else 0

    def _q(self, c) -> tuple[int, int, int]:
        """Posterise one colour to the palette."""
        b = self._bits()
        if not b:
            return (int(c[0]), int(c[1]), int(c[2]))
        sh = 8 - b
        half = (1 << sh) >> 1
        return tuple(min(255, ((int(v) >> sh) << sh) | half) for v in c[:3])

    def _posterise(self, surf: Surface) -> Surface:
        """Palette-reduce a whole surface once (copy). Falls back to the untouched surface on any error."""
        b = self._bits()
        if not b:
            return surf
        try:
            sh = 8 - b
            arr = pygame.surfarray.array3d(surf)               # a copy: nothing stays locked
            arr = ((arr >> sh) << sh) | ((1 << sh) >> 1)
            out = pygame.surfarray.make_surface(arr.astype(np.uint8))
            return out.convert()
        except Exception:
            return surf

    def _prep_sprite(self, img: Surface, k: float) -> Surface:
        """Shrink a car sprite to canvas size, hard alpha edge, reduced palette."""
        w, h = img.get_size()
        size = (max(2, round(w * k)), max(2, round(h * k)))
        small = pygame.transform.smoothscale(img, size) if img.get_bitsize() == 32 else pygame.transform.scale(img, size)
        small = small.convert_alpha()
        try:
            alpha = pygame.surfarray.pixels_alpha(small)
            alpha[:] = np.where(alpha >= 110, 255, 0).astype(np.uint8)
            del alpha
            b = self._bits()
            if b:
                rgb = pygame.surfarray.pixels3d(small)
                sh = 8 - b
                rgb[:] = ((rgb >> sh) << sh) | ((1 << sh) >> 1)
                del rgb
        except Exception:
            pass
        return small

    # ------------------------------------------------------------------ drawing
    def draw(self, win, level: int, left: float, top: float, scale: float) -> None:
        """
        Draw the whole game layer (track, streaks, rings, cars, labels) onto win.screen.
        (left, top) is the world point at the screen's top-left, `scale` = screen pixels per world pixel.
        """
        cfg, cars, screen = self.config, win.cars, win.screen
        px = max(1, int(cfg.pixel_size))
        sw, sh = screen.get_size()
        cw, ch = max(1, -(-sw // px)), max(1, -(-sh // px))       # canvas size (rounded up)
        k = scale / px                                             # world pixels -> canvas pixels
        if self._small is None or self._small.get_size() != (cw, ch):
            self._small = Surface((cw, ch)).convert()
            self._up = None
        small = self._small
        small.fill(self._q(win.bg_color))

        self._blit_track(win, small, left, top, k, cw, ch)

        # keep the bookkeeping that the normal render does
        cars._frame += 1
        cars._screen_size = screen.get_size()
        cars._debug_level = level
        if cars.font is None:
            cars.font = pygame.font.Font(None, 17)
        if cars.label_font is None:
            cars.label_font = pygame.font.Font(None, 15)
        halo_r = int(max(cars.width, cars.height) * 0.8)
        cars._halo_r = halo_r

        if cars.race_on:
            self._streaks(cars, small, left, top, k)

        ranking = [int(i) for i in (cars.ranking if cars.ranking is not None
                                     else np.argsort(-cars.players.fitness))[:3]]
        best_idx = ranking[0] if not cars.usr_engaged and ranking else (cars.total - 1)
        if cars.debug_car is not None and 0 <= cars.debug_car < cars.total:
            best_idx = cars.debug_car
        cars._debug_idx = best_idx

        if abs(k - self._sprite_k) > 1e-6:                         # zoom changed: shrunk sprites are stale
            self._sprites.clear()
            self._sprite_k = k
        ring_r = max(3, round(halo_r * k))
        tags = []
        for idx, (x, y, angle, active) in enumerate(zip(*cars.center, cars.angle, cars.players.active)):
            if not (active or (cars.usr_engaged and idx == cars.total - 1)):
                continue
            wx, wy = x.item(), y.item()
            sx, sy = (wx - left) * k, (wy - top) * k
            if sx < -ring_r * 2 or sy < -ring_r * 2 or sx > cw + ring_r * 2 or sy > ch + ring_r * 2:
                continue
            rank = ranking.index(idx) if idx in ranking else -1
            ix, iy = int(round(sx)), int(round(sy))
            if rank >= 0:
                pygame.draw.circle(small, self._q(HALO_COLORS[rank]), (ix, iy), ring_r, 1)
                tags.append((rank, idx, (wx - left) * scale, (wy - top) * scale))
            img = cars._sprite(idx, cars._steer_angle(idx))
            ent = self._sprites.get(id(img))
            if ent is None or ent[0] is not img:
                if len(self._sprites) > 512:
                    self._sprites.clear()
                ent = self._sprites[id(img)] = (img, self._prep_sprite(img, k))
            rot = pygame.transform.rotate(ent[1], angle.item())     # nearest-neighbour: stays pixel-art
            small.blit(rot, rot.get_rect(center=(ix, iy)))
            if cars.usr_engaged and idx == cars.total - 1:
                pygame.draw.circle(small, (255, 255, 255), (ix, iy), ring_r, 1)

        # one hard upscale to the screen
        size = (cw * px, ch * px)
        if self._up is None or self._up.get_size() != size:
            self._up = Surface(size).convert()
        pygame.transform.scale(small, size, self._up)
        screen.blit(self._up, (0, 0))

        self._labels(cars, screen, tags, halo_r * scale)

    def _blit_track(self, win, small: Surface, left: float, top: float, k: float, cw: int, ch: int) -> None:
        src = win._track_surface()
        if src is not self._track_src or self._track is None:
            self._track_src, self._track = src, self._posterise(src)
        gw, gh = win.world_size
        x0, y0 = max(left, 0.0), max(top, 0.0)
        x1, y1 = min(left + cw / k, gw), min(top + ch / k, gh)
        sx0, sy0 = int(np.floor(x0)), int(np.floor(y0))
        sx1, sy1 = min(int(np.ceil(x1)), gw), min(int(np.ceil(y1)), gh)
        if sx1 <= sx0 or sy1 <= sy0:
            return
        crop = self._track.subsurface(pygame.Rect(sx0, sy0, sx1 - sx0, sy1 - sy0))
        size = (max(1, round((sx1 - sx0) * k)), max(1, round((sy1 - sy0) * k)))
        dest = (round((sx0 - left) * k), round((sy0 - top) * k))
        try:
            if crop.get_bitsize() >= 24:
                part = pygame.transform.smoothscale(crop, size)   # averaging keeps thin kerbs from shimmering
            else:
                part = pygame.transform.scale(crop, size)
        except (ValueError, pygame.error):
            part = pygame.transform.scale(crop, size)
        small.blit(part, dest)

    def _streaks(self, cars, small: Surface, left: float, top: float, k: float) -> None:
        """Slipstream as hard square dots: bright 2x2 head, base-colour body, dithered dark tail."""
        ss = cars.race.slipstream
        if not ss.enabled:
            return
        cap = ss.trail_length
        side_px = cars.width * ss.side_offset
        for i in np.flatnonzero(cars.draft_level > DRAFT_VISIBLE):
            if not cars.players.active[i]:
                continue
            lvl = float(cars.draft_level[i])
            n = max(2, int(np.ceil(lvl * cap)))
            base = ss.color_decay if cars._draft_decay[i] else ss.color_gain
            head = self._q(tuple(min(255, (c + 255) // 2) for c in base))
            body = self._q(base)
            tail = self._q(tuple(int(c * 0.55) for c in base))
            for side in (-1.0, 1.0):
                m = trail_points(
                    cars._trail_x, cars._trail_y, cars._trail_a, cars._trail_head, cars._trail_count,
                    i, n, cars.height * STREAK_REAR_CARS, side * side_px, 1, 0.0, cars._frame, cars._trail_buf,
                )
                if m < 2:
                    continue
                pts = cars._trail_buf[:m]
                xs = np.floor((pts[:, 0] - left) * k).astype(int)
                ys = np.floor((pts[:, 1] - top) * k).astype(int)
                last = None
                for j in range(m):
                    cell = (xs[j], ys[j])
                    if cell == last:
                        continue                                  # many points land on one canvas pixel
                    last = cell
                    t = j / (m - 1)
                    if t < 0.3:
                        small.fill(head, (cell[0], cell[1], 2, 2))
                    elif t < 0.7:
                        small.fill(body, (cell[0], cell[1], 1, 1))
                    elif (cell[0] + cell[1]) & 1 == 0:            # checkerboard dither fades the tail
                        small.fill(tail, (cell[0], cell[1], 1, 1))

    def _labels(self, cars, screen: Surface, tags: list, ring_screen_r: float) -> None:
        """Rank labels at full resolution, above each ringed car (same look as the normal render)."""
        sw, sh = screen.get_size()
        for rank, idx, cx, cy in tags:
            text = cars.label(idx)
            if text is None:
                text = str(idx)
            txt = cars._text(f"{rank + 1}  {text}", (240, 240, 240))
            w, h = txt.get_width() + 8, txt.get_height() + 4
            ty = cy - ring_screen_r - 6 - h
            if ty < 2:
                ty = cy + ring_screen_r + 6
            rect = pygame.Rect(0, 0, w, h)
            rect.centerx = int(min(max(cx, w // 2 + 2), sw - w // 2 - 2))
            rect.y = int(ty)
            color = HALO_COLORS[rank]
            draw_panel(screen, rect, cars.config.overlay_color, 170, radius=4)
            pygame.draw.rect(screen, color, rect, width=1, border_radius=4)
            screen.blit(txt, (rect.x + 4, rect.y + 2))