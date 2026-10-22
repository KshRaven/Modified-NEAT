"""Pygame rendering of the cars, slipstream streaks, debug overlays and race HUD."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pygame
from pygame import Surface

from .constants import (
    COLOR_CURB_PRI, COLOR_CURB_SEC, COLOR_GRASS, COLOR_GRAVEL, COLOR_LIDAR,
    DRAFT_TOWER_MARK, DRAFT_VISIBLE, NP_FLOAT, STREAK_REAR_CARS,
    SURFACE_CAR, SURFACE_CURB, SURFACE_GRASS, SURFACE_GRAVEL, SURFACE_ROAD, SURFACE_VOID,
)
from .kernels import trail_points
from .sprite import HALO_COLORS, WHEEL_PAD, make_car, make_halo, rainbow
from .util import draw_arrow, draw_panel, render_text

if TYPE_CHECKING:
    from .car import Cars

__all__ = ["CarRenderer", "flame_color"]

LIDAR_COLORS = {
    int(SURFACE_ROAD):   (255, 255, 255),
    int(SURFACE_CURB):   COLOR_CURB_PRI,
    int(SURFACE_GRASS):  COLOR_GRASS,
    int(SURFACE_GRAVEL): COLOR_GRAVEL,
    int(SURFACE_VOID):   COLOR_LIDAR,
    int(SURFACE_CAR):    (255, 80, 220),
}


def flame_color(base, hot, heat: float) -> tuple[int, int, int]:
    """Blend from a dark ember of `base` (heat 0) through `base` (0.5) to `hot` (1)."""
    if heat >= 0.5:
        u, a, b = (heat - 0.5) * 2.0, base, hot
    else:
        u, a, b = heat * 2.0, tuple(0.25 * v for v in base), base
    return (int(a[0] + (b[0] - a[0]) * u), int(a[1] + (b[1] - a[1]) * u), int(a[2] + (b[2] - a[2]) * u))


class CarRenderer:
    """Draws a `Cars` instance. Holds all display state (fonts, sprite caches, overlays)."""

    def __init__(self, cars: Cars) -> None:
        self.cars = cars
        self.font: pygame.font.Font | None = None
        self.label_font: pygame.font.Font | None = None
        self.frame = 0
        self.screen_size = (0, 0)
        self.debug_idx: int | None = None
        self.debug_level = 0
        self.halo_r = 0
        self._halos: list[Surface] | None = None
        self._sprites: dict[tuple[int, int, int], Surface] = {}
        self._text_cache: dict[tuple, Surface] = {}
        self._trail_overlay: Surface | None = None
        self._trail_buf: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_fonts(self, size: int) -> None:
        self.font = pygame.font.Font(None, size)
        self.label_font = pygame.font.Font(None, max(8, size - 2))

    def close(self) -> None:
        """Drop fonts and cached surfaces (required before `pygame.quit`)."""
        self.font = None
        self.label_font = None
        self._text_cache.clear()

    def begin_frame(self, size: tuple[int, int], level: int) -> None:
        """Per-frame bookkeeping shared by the normal and the pixel renderer."""
        self.frame += 1
        self.screen_size = size
        self.debug_level = level
        if self.font is None:
            self.font = pygame.font.Font(None, 17)
        if self.label_font is None:
            self.label_font = pygame.font.Font(None, 15)
        cars = self.cars
        self.halo_r = int(max(cars.width, cars.height) * 0.8)
        if self._halos is None:
            self._halos = [make_halo(c, self.halo_r) for c in HALO_COLORS]

    def leaders(self) -> tuple[list[int], int]:
        """Top 3 car indices and the car shown by the debug overlays."""
        cars = self.cars
        order = cars.ranking if cars.ranking is not None else np.argsort(-cars.players.fitness)
        ranking = [int(i) for i in order[:3]]
        best = ranking[0] if not cars.usr_engaged and ranking else cars.total - 1
        if cars.debug_car is not None and 0 <= cars.debug_car < cars.total:
            best = cars.debug_car
        self.debug_idx = best
        return ranking, best

    # ------------------------------------------------------------------
    # Cached surfaces
    # ------------------------------------------------------------------

    def steer_angle(self, idx: int) -> float:
        """Visual front wheel angle in degrees (positive = left). The lock shrinks with speed."""
        cars, cfg = self.cars, self.cars.config
        turn = float(np.clip(cars.vel_ang[idx] / max(cars.max_vel_ang, 1e-6), -1.0, 1.0))
        speed = float(np.clip(abs(cars.vel_lin[idx]) / max(cars.max_vel_lin, 1e-6), 0.0, 1.0))
        lock = cfg.wheel_lock_low_speed + (cfg.wheel_lock_top_speed - cfg.wheel_lock_low_speed) * speed
        return turn * lock

    def sprite(self, idx: int, steer: float = 0.0) -> Surface:
        cars = self.cars
        step = max(float(cars.config.wheel_angle_step), 0.5)
        bucket = int(round(steer / step))
        key = (idx, cars.total, bucket)
        img = self._sprites.get(key)
        if img is None:
            if len(self._sprites) > 4096:
                self._sprites.clear()
            img = self._sprites[key] = make_car(
                rainbow(idx, cars.total), cars.config.scale, steer=bucket * step, pad=WHEEL_PAD,
            )
        return img

    def trail_buffer(self, capacity: int) -> np.ndarray:
        """Scratch polyline buffer for `trail_points`."""
        if self._trail_buf is None or len(self._trail_buf) != capacity:
            self._trail_buf = np.empty((capacity, 2), dtype=NP_FLOAT)
        return self._trail_buf

    def text(self, text: str, color) -> Surface:
        key = (text, tuple(color))
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) > 256:
                self._text_cache.clear()
            surf = self._text_cache[key] = self.label_font.render(text, True, color)
        return surf

    # ------------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------------

    def render(self, surface: Surface, verbose: int | bool = False, hud: bool = True, debug_panel: bool = True) -> None:
        """Draw streaks, cars, halos and rank labels onto `surface`.

        verbose: 0 nothing, 1 stats panel, 2 + LiDAR beams, 3 + tile exit arrows (True = 3).
        The overlays describe `cars.debug_car`, or the leader when it is None.
        debug_panel=False leaves the stats panel to `render_debug`, hud=False leaves the
        leaderboard to `render_hud`, so a camera can draw both on the screen.
        """
        cars = self.cars
        level = 3 if verbose is True else int(verbose or 0)
        self.begin_frame(surface.get_size(), level)
        self._render_streaks(surface)

        ranking, best_idx = self.leaders()
        tags = []
        for idx, (x, y, angle, active) in enumerate(zip(*cars.center, cars.angle, cars.players.active)):
            if not (active or (cars.usr_engaged and idx == cars.total - 1)):
                continue
            cx, cy = x.item(), y.item()
            rank = ranking.index(idx) if idx in ranking else -1

            if rank >= 0:
                halo = self._halos[rank]
                surface.blit(halo, halo.get_rect(center=(cx, cy)))

            img = pygame.transform.rotate(self.sprite(idx, self.steer_angle(idx)), angle.item())
            surface.blit(img, img.get_rect(center=(cx, cy)))

            if cars.usr_engaged and idx == cars.total - 1:
                pygame.draw.circle(surface, (255, 255, 255), (int(cx), int(cy)), self.halo_r, width=1)
            if rank >= 0:
                tags.append((rank, idx, cx, cy))
            if idx == best_idx:
                if debug_panel and level >= 1:
                    self._render_stats(surface, idx)
                if cars.use_lidar and level >= 2:
                    self._render_lidar(surface, idx, cx, cy)
                if level >= 3:
                    self._render_tile_arrows(surface, idx, (x, y))

        self.render_labels(surface, tags)
        if hud:
            self.render_hud(surface)

    def render_labels(
        self, surface: Surface, tags: list[tuple[int, int, float, float]],
        radius: float | None = None, connector: bool = True,
    ) -> None:
        """Rank labels above the ringed cars. `tags` holds (rank, car index, x, y).

        `radius` is the ring radius on `surface` (default `halo_r`), `connector` draws the line to the ring.
        """
        radius = self.halo_r if radius is None else radius
        sw, sh = surface.get_size()
        for rank, idx, cx, cy in tags:
            label = self.cars.label(idx)
            color = HALO_COLORS[rank]
            txt = self.text(f"{rank + 1}  {idx if label is None else label}", (240, 240, 240))
            w, h = txt.get_width() + 8, txt.get_height() + 4
            ty = cy - radius - 6 - h
            if ty < 2:                                  # near the top edge: place below the ring
                ty = cy + radius + 6
                anchor = (cx, cy + radius)
            else:
                anchor = (cx, cy - radius)
            rect = pygame.Rect(0, 0, w, h)
            rect.centerx = int(min(max(cx, w // 2 + 2), sw - w // 2 - 2))
            rect.y = int(ty)
            if connector:
                pygame.draw.line(surface, color, anchor, (rect.centerx, rect.bottom if ty < cy else rect.top), 1)
            draw_panel(surface, rect, self.cars.config.overlay_color, 170, radius=4)
            pygame.draw.rect(surface, color, rect, width=1, border_radius=4)
            surface.blit(txt, (rect.x + 4, rect.y + 2))

    def render_debug(self, surface: Surface) -> None:
        """Stats panel of the debug car (verbose level >= 1)."""
        idx = self.debug_idx
        if self.debug_level >= 1 and idx is not None and 0 <= idx < self.cars.total:
            self._render_stats(surface, idx)

    def render_hud(self, surface: Surface) -> None:
        """Race leaderboard and driver panel."""
        cars = self.cars
        if not cars.race_on or self.font is None or cars.race_order is None:
            return
        ss = cars.race.slipstream
        overlay, alpha = cars.config.overlay_color, cars.config.overlay_alpha

        done = max(0, int(cars.laps_done[int(cars.race_order[0])]))
        if math.isfinite(cars.race_laps):
            head = f"Laps {min(done, int(cars.race_laps))}/{int(cars.race_laps)}"
        else:
            head = f"Laps {done}"
        if cars.race_end_in is not None:
            head += f"  ends {cars.race_end_in:.1f}s"
        rows = [render_text(self.font, head, True, (255, 215, 0))]
        for row, idx in enumerate(cars.race_order[:5]):
            name = cars.label(int(idx))
            tag = f"{row + 1} {int(idx)}" + (f" {name}" if name is not None else "")
            tag += " FIN" if cars.race_done[idx] else (" *" if cars.draft_level[idx] > DRAFT_TOWER_MARK else "")
            rows.append(render_text(self.font, tag, True, (230, 230, 230)))

        pad = 6
        w = max(r.get_width() for r in rows) + 2 * pad
        rect = pygame.Rect(surface.get_width() - w - 4, 4, w, len(rows) * 16 + 2 * pad - 4)
        draw_panel(surface, rect, overlay, alpha)
        for row, txt in enumerate(rows):
            surface.blit(txt, (rect.x + pad, rect.y + pad + row * 16))

        if cars.usr_engaged:
            me = cars.total - 1
            pos = int(np.flatnonzero(cars.race_order == me)[0]) + 1
            color = ss.color_decay if cars.draft_decay[me] else ss.color_gain
            txt = render_text(self.font, f"P{pos}/{len(cars.race_order)}  Draft {cars.draft_level[me] * 100:3.0f}%", True, color)
            rect = pygame.Rect(4, surface.get_height() - (txt.get_height() + 12), txt.get_width() + 12, txt.get_height() + 8)
            draw_panel(surface, rect, overlay, alpha)
            surface.blit(txt, (rect.x + 6, rect.y + 4))

    # ------------------------------------------------------------------
    # Slipstream streaks
    # ------------------------------------------------------------------

    def _render_streaks(self, surface: Surface) -> None:
        cars = self.cars
        ss = cars.race.slipstream
        if not (cars.race_on and ss.enabled):
            return
        cap = ss.trail_length
        buf = self.trail_buffer(cap)
        step = ss.particle_step if ss.particles else 1
        spread = ss.particle_spread if ss.particles else 0.0
        side_px = cars.width * ss.side_offset
        dirty = None
        for i in np.flatnonzero(cars.draft_level > DRAFT_VISIBLE):
            if not cars.players.active[i]:
                continue
            n = max(2, int(math.ceil(float(cars.draft_level[i]) * cap)))
            color = ss.color_decay if cars.draft_decay[i] else ss.color_gain
            for side in (-1.0, 1.0):
                m = trail_points(
                    cars.trail_x, cars.trail_y, cars.trail_a, cars.trail_head, cars.trail_count,
                    i, n, cars.height * STREAK_REAR_CARS, side * side_px, step, spread, self.frame, buf,
                )
                if m < 2:
                    continue
                frac = min(max((n - 2) / max(cap - 2, 1), 0.0), 1.0)
                tail_alpha = ss.alpha_tail_short + (ss.alpha_tail_long - ss.alpha_tail_short) * frac
                dirty = self._draw_trail(buf[:m], color, ss, tail_alpha, dirty)

        if dirty is not None:                           # composite the translucent layer once
            dirty = dirty.clip(surface.get_rect())
            surface.blit(self._trail_overlay, dirty, dirty)
            self._trail_overlay.fill((0, 0, 0, 0), dirty)

    def _draw_trail(self, pts: np.ndarray, color, ss, tail_alpha: float, dirty):
        """Draw one streak (newest to oldest point) on the shared alpha layer.

        Opacity falls linearly from `ss.alpha_head` to `tail_alpha`. Returns the updated dirty rect.
        """
        if self._trail_overlay is None or self._trail_overlay.get_size() != self.screen_size:
            self._trail_overlay = Surface(self.screen_size, pygame.SRCALPHA)
        ov = self._trail_overlay
        m = len(pts)
        span = max(m - 1, 1)
        pad = int(max(ss.flame_width, ss.particle_spread, 3.0)) + 3
        lo_x, lo_y = float(pts[:, 0].min()), float(pts[:, 1].min())
        hi_x, hi_y = float(pts[:, 0].max()), float(pts[:, 1].max())
        box = pygame.Rect(int(lo_x) - pad, int(lo_y) - pad, int(hi_x - lo_x) + 2 * pad, int(hi_y - lo_y) + 2 * pad)

        def alpha_at(k: int) -> float:
            return ss.alpha_head + (tail_alpha - ss.alpha_head) * (k / span)

        if ss.flame:
            layers = ss.flame_layers
            for layer in range(layers):                 # outer glow first, hot core last
                core = layer / max(layers - 1, 1)
                heat_layer = 0.35 + 0.65 * core
                for k in range(m):
                    age = k / span
                    fade = alpha_at(k) * (0.55 + 0.45 * core)
                    if ss.flame_flicker > 0.0:
                        fade *= 1.0 + ss.flame_flicker * 0.5 * math.sin(self.frame * 0.9 + k * 1.7 + layer * 2.1)
                    alpha = int(255 * min(max(fade, 0.0), 1.0))
                    if alpha <= 0:
                        continue
                    rgb = flame_color(color, ss.flame_hot, heat_layer * (1.0 - 0.65 * age))
                    radius = max(1.0, 0.5 * ss.flame_width * (1.0 - ss.flame_taper * age) * (1.0 - 0.6 * core))
                    pygame.draw.circle(ov, (*rgb, alpha), (int(pts[k, 0]), int(pts[k, 1])), int(round(radius)))
        elif ss.particles:
            for k in range(m):
                rgba = (*color, int(255 * min(max(alpha_at(k), 0.0), 1.0)))
                px, py = int(pts[k, 0]), int(pts[k, 1])
                if k < m // 2:
                    pygame.draw.circle(ov, rgba, (px, py), 2)
                else:
                    ov.fill(rgba, (px, py, 1, 1))
        else:
            for k in range(m - 1):
                rgba = (*color, int(255 * min(max(alpha_at(k), 0.0), 1.0)))
                pygame.draw.line(ov, rgba, (pts[k, 0], pts[k, 1]), (pts[k + 1, 0], pts[k + 1, 1]), 1)
        return box if dirty is None else dirty.union(box)

    # ------------------------------------------------------------------
    # Debug overlays
    # ------------------------------------------------------------------

    def _render_stats(self, surface: Surface, index: int) -> None:
        """Surface, pose, progress and reward stats of one car."""
        cars = self.cars
        if surface is None or self.font is None:
            return

        x, y, vel_lin, angle = (float(a[index]) for a in (cars.x, cars.y, cars.vel_lin, cars.angle))
        tile = cars.tile[index]
        tile_next = cars.grid.continuity[tile]
        phase = -(angle - tile.orientation)
        phase = (phase + 180) % 360 - 180
        phase_next = -(angle - tile_next.orientation)
        phase_next = (phase_next + 180) % 360 - 180

        touching = [name for name, flag in (
            ("road", cars.on_road), ("curb", cars.on_curb), ("grass", cars.on_grass),
            ("gravel", cars.on_gravel), ("finish", cars.on_finish),
        ) if flag[index]]
        if touching:
            status, status_color = f"Surface: {', '.join(touching).upper()}", COLOR_CURB_SEC
        else:
            status, status_color = "Surface: OFF TRACK", (255, 255, 255)

        gray = (200, 200, 200)
        entries = [
            (status, status_color),
            (f"Pos: ({x:.1f}, {y:.1f}) Vel: {vel_lin:.1f} Angle: {angle:.1f} "
             f"Rot: {tile.orientation:.1f} PhC: {phase:.1f} PhN: {phase_next:.1f} ", gray),
            (f"Checks: {cars.checks_done[index]} Laps: {cars.laps_done[index]} "
             f"Crashes: {cars.crashes_done[index]} ", gray),
            (f"Hiatus: {cars.hiatus[index]} Reward: {cars.players.get_reward()[index]:.1f} "
             f"Lives: {cars.players.lives[index]} Score: {cars.players.true_scores[index]} ", gray),
            (f"Brake: {cars.brake_steps[index]} Turn: {cars.turn_steps[index]} ", gray),
        ]
        lines = [(render_text(self.font, text, True, color), (10, 10 + 18 * row))
                 for row, (text, color) in enumerate(entries)]

        pad = 6
        panel = pygame.Rect(
            10 - pad, 10 - pad,
            max(t.get_width() for t, _ in lines) + 2 * pad,
            max(p[1] + t.get_height() for t, p in lines) - 10 + 2 * pad,
        )
        draw_panel(surface, panel, cars.config.overlay_color, cars.config.overlay_alpha)
        for text_surface, pos in lines:
            surface.blit(text_surface, pos)

    def _render_lidar(self, surface: Surface, idx: int, car_x: float, car_y: float) -> None:
        """Beams of one car, colored by the surface they hit."""
        cars = self.cars
        for beam in range(cars.lidar_angles):
            angle = np.radians(cars.angle[idx] + cars.lidar_beam_offsets[beam])
            reach = cars.lidar_distances[idx, beam] * cars.lidar_max_dist
            end = (car_x - np.sin(angle) * reach, car_y - np.cos(angle) * reach)
            color = LIDAR_COLORS.get(int(cars.lidar_surface_hits[idx, beam]), COLOR_LIDAR)
            pygame.draw.line(surface, color, (car_x, car_y), end, width=1)
            pygame.draw.circle(surface, color, (int(end[0]), int(end[1])), radius=2)
        pygame.draw.circle(surface, COLOR_LIDAR, (int(car_x), int(car_y)), radius=3, width=2)

    def _render_tile_arrows(self, surface: Surface, idx: int, position) -> None:
        """Arrows from the respawn point and from the car to the exit of the current and next two tiles."""
        cars = self.cars
        tile = cars.tile[idx]
        tile_1 = cars.grid.continuity[tile]
        tile_2 = cars.grid.continuity[tile_1]
        draw_arrow(surface, tile.respawn_point, tile.center_check_abs, color=(102, 0, 153))
        draw_arrow(surface, tile_1.respawn_point, tile_1.center_check_abs, color=(255, 255, 255))
        draw_arrow(surface, tile_2.respawn_point, tile_2.center_check_abs, color=(255, 255, 255))
        draw_arrow(surface, position, tile.center_check_abs, color=(255, 0, 0))
        draw_arrow(surface, position, tile_1.center_check_abs, color=(0, 0, 255))
        draw_arrow(surface, position, tile_2.center_check_abs, color=(0, 255, 0))
