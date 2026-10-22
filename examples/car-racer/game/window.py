"""Pygame window: canvas, camera, keyboard controls, overlays and the race countdown."""
from __future__ import annotations

from typing import Any

import numpy as np
import pygame
from pygame import Surface

from .car import Cars
from .config import RaceConfig, WindowConfig, overwrite_config
from .grid import Grid
from .pixel import PixelRenderer
from .player import Players
from .renderer import CarRenderer
from .util import clear_text_cache, draw_panel, render_text

__all__ = ["Window"]


class Window:
    FPS_STEPS = (5, 10, 15, 20, 30, 45, 60, 90, 120, 144, 240, None)   # frame rate cap ladder, None = uncapped

    def __init__(
        self,
        grid: Grid, players: Players, cars: Cars,
        config: WindowConfig | None = None,
        race: RaceConfig | None = None,
        **params: dict[str, Any],
    ) -> None:
        if config is None:
            config = WindowConfig()

        overwrite_config(config, params)
        self.config = config
        self.race = race if race is not None else RaceConfig()

        self.grid = grid
        self.players = players
        self.cars = cars
        self.car_renderer = CarRenderer(cars)
        self.width, self.height = self.grid.width, self.grid.height
        self.offset = 0
        self.title = config.title
        self.screen: Surface | None = None
        self._pixel = PixelRenderer(config)
        self.clock: pygame.time.Clock | None = None
        self.running = False
        self.fps = config.fps
        self.font: pygame.font.Font | None = None
        self.bg_color = config.background_color
        self.font_size = max(8, round(config.font_size * config.font_scale))
        self.font_scale = config.font_scale

        # Cinematic mode: fixed canvas, a camera looks at the grid ("world")
        self.cinematic = bool(config.cinematic)
        self.world: Surface | None = None
        self.world_size = (self.grid.width, self.grid.height)
        self._canvas: tuple[int, int] | None = None

        # Camera. `scale` is screen pixels per world pixel.
        self.cam_x, self.cam_y = self.world_size[0] / 2, self.world_size[1] / 2
        self.scale = 1.0
        self.auto_fit = True                # frame the leading cars
        self.focus: int | None = None       # car being followed
        self._focus_rank = 0
        self._focus_slot: int | None = None # followed race position (0 = leader)
        self._zoom_bias = 1.0               # user zoom on top of auto-fit / focus framing
        self._free_scale = 1.0              # zoom when neither auto-fit nor focus is active
        self._fit_grid = False              # whole grid in view
        self._snap = True
        self._rng = np.random.default_rng()

        # Run control, set by the keyboard and read by the loop that owns the episode
        self.paused = False
        self.quit_requested = False         # window closed
        self.end_requested = False          # Esc pressed
        self._verbose: bool | int = False
        self.verbose_level = int(config.verbose_level)
        self._verbose_touched = False       # set once V was pressed

        # The track is drawn once per grid and reused
        self._track_cache: Surface | None = None
        self._track_sig = None

    def initialize(self) -> None:
        pygame.init()
        pygame.display.set_caption(self.title)
        if self.cinematic:
            current = pygame.display.get_surface()
            if self.config.resizable and current is not None:
                self._canvas = current.get_size()           # keep the size the user dragged to
            elif self._canvas is None:
                self._canvas = self._initial_canvas()
            flags = pygame.RESIZABLE if self.config.resizable else 0
            self.screen = pygame.display.set_mode(self._canvas, flags)
            self.width, self.height = self.screen.get_size()
            self.world = Surface(self.world_size).convert()
        else:
            self.screen = pygame.display.set_mode((self.width + self.offset, self.height + self.offset))
        self.clock = pygame.time.Clock()
        clear_text_cache()
        self.font = pygame.font.Font(None, self.font_size)
        self.car_renderer.set_fonts(self.font_size)
        self.running = True

    def show_message(self, text: str) -> None:
        """Paint a centered message so the window is not black during slow setup."""
        if self.screen is None or self.font is None:
            return
        pygame.event.pump()
        self.screen.fill(self.bg_color)
        txt = pygame.font.Font(None, max(24, int(self.screen.get_height() * 0.06))).render(text, True, (200, 200, 200))
        self.screen.blit(txt, txt.get_rect(center=self.screen.get_rect().center))
        pygame.display.flip()

    def _initial_canvas(self) -> tuple[int, int]:
        """`cinematic_size`, shrunk when it does not fit on the desktop."""
        w, h = self.config.cinematic_size
        try:
            dw, dh = pygame.display.get_desktop_sizes()[0]
            w, h = min(w, dw), min(h, max(240, dh - 90))    # room for the title bar and taskbar
        except (pygame.error, AttributeError, IndexError):
            pass
        return max(320, int(w)), max(240, int(h))

    def close(self) -> None:
        self.running = False
        clear_text_cache()
        self.font = None                    # fonts must be released before pygame.quit()
        self.car_renderer.close()
        pygame.quit()

    def render(self, verbose: bool | int = False) -> None:
        if self.screen is None:
            raise RuntimeError("Call initialize() before render().")
        self._verbose = verbose
        self._compose(verbose)
        pygame.display.flip()

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.config.precise_fps:
            self.clock.tick_busy_loop(self.fps or 0)    # exact cap, busy-waits on one core
        else:
            self.clock.tick(self.fps or 0)              # 0 = uncapped

    def _track_surface(self) -> Surface:
        """The drawn track, rebuilt when the grid changes."""
        sig = (self.world_size, tuple(id(seg) for row in self.grid.cells for seg in row))
        if self._track_cache is None or sig != self._track_sig:
            surf = Surface(self.world_size)
            surf.fill(self.bg_color)
            self.grid.render(surf, verbose=False)
            self._track_cache, self._track_sig = surf.convert(), sig
        return self._track_cache

    def _compose(self, verbose: bool | int = False, hud: bool = True, paused_overlay: bool = True) -> None:
        """Draw one frame onto the screen (no flip, no clock tick)."""
        level = max(self.verbose_level, 3 if verbose is True else int(verbose or 0))
        self.cars.debug_car = self.focus if self.cinematic else None
        verbose = level
        pixel = self._pixel_active() and level < 2      # debug views use the normal renderer
        if not self.cinematic:
            if pixel:
                self._pixel.draw(self, level, 0.0, 0.0, 1.0)
                if hud:
                    self.car_renderer.render_debug(self.screen)
                    self.car_renderer.render_hud(self.screen)
            else:
                self.screen.blit(self._track_surface(), (0, 0))
                self.car_renderer.render(self.screen, verbose=verbose, hud=hud)
            if hud:
                self._draw_stats()
            if self.paused and paused_overlay:
                self._draw_paused()
            return

        self.screen = pygame.display.get_surface() or self.screen   # follow a resized window
        self.width, self.height = self.screen.get_size()
        self._update_camera()
        left, top, vw, vh = self._view_rect()
        if pixel:
            self._pixel.draw(self, level, left, top, self.scale)
        else:
            # Redraw only the visible part of the track
            crop = pygame.Rect(int(left) - 2, int(top) - 2, int(vw) + 6, int(vh) + 6).clip(self.world.get_rect())
            self.world.blit(self._track_surface(), crop.topleft, crop)
            self.car_renderer.render(self.world, verbose=verbose, hud=False, debug_panel=False)
            self._blit_camera()
        if hud:
            self.car_renderer.render_debug(self.screen)
            self.car_renderer.render_hud(self.screen)
            self._draw_stats()
            self._draw_hint()
        if self.paused and paused_overlay:
            self._draw_paused()

    # Keyboard and run control

    def poll_events(self) -> list:
        """Handle the window shortcuts and return all pending pygame events. Blocks while paused.

        Sets `quit_requested` (window closed) and `end_requested` (Esc), read with `consume_stop()`.
        """
        if self.screen is None:
            return []
        events = pygame.event.get()
        for event in events:
            self._handle_event(event)
        while self.paused and not (self.quit_requested or self.end_requested):
            self._compose(self._verbose)
            pygame.display.flip()
            pygame.time.Clock().tick(30)
            for event in pygame.event.get():
                self._handle_event(event)
        return events

    def consume_stop(self) -> bool:
        """True once after the window was closed or Esc was pressed. Clears the request."""
        stop = self.quit_requested or self.end_requested
        self.quit_requested = self.end_requested = False
        self.paused = False
        return stop

    def _handle_event(self, event) -> None:
        if event.type == pygame.QUIT:
            self.quit_requested = True
        elif event.type == pygame.MOUSEWHEEL and self.cinematic:
            self._zoom(self.config.zoom_step ** event.y)
        elif event.type == pygame.KEYDOWN:
            key = event.key
            if key == pygame.K_ESCAPE:
                self.end_requested = True
            elif key == pygame.K_SPACE:
                self.paused = not self.paused
            elif key == pygame.K_v:
                self.verbose_level = (self.verbose_level + 1) % 4
                self._verbose_touched = True
            elif key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS) \
                    and getattr(event, "mod", 0) & pygame.KMOD_SHIFT:
                self._change_fps(-1 if key in (pygame.K_MINUS, pygame.K_KP_MINUS) else +1)
            elif not self.cinematic:
                return                                      # camera keys need cinematic mode
            elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self._zoom(1.0 / self.config.zoom_step)
            elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self._zoom(self.config.zoom_step)
            elif key == pygame.K_l:
                self._fit_grid, self.auto_fit, self.focus, self._focus_slot = True, False, None, None
            elif key in (pygame.K_0, pygame.K_KP0):
                self._toggle_auto_fit()
            elif key == pygame.K_TAB:
                self._focus_random()
            elif pygame.K_1 <= key <= pygame.K_9:
                self._focus_position(key - pygame.K_1)
            elif pygame.K_KP1 <= key <= pygame.K_KP9:
                self._focus_position(key - pygame.K_KP1)

    def _change_fps(self, direction: int) -> None:
        """Move the frame rate cap one step down (-1) or up (+1) on `FPS_STEPS`."""
        steps = self.FPS_STEPS
        if self.fps is None:
            i = len(steps) - 1
        else:
            i = min(range(len(steps) - 1), key=lambda k: abs(steps[k] - self.fps))
        self.fps = steps[min(max(i + direction, 0), len(steps) - 1)]

    def _zoom(self, factor: float) -> None:
        if self._fit_grid:
            self._fit_grid, self._free_scale = False, self.scale
        if self.auto_fit or self.focus is not None:
            self._zoom_bias *= factor
        else:
            self._free_scale *= factor

    def _toggle_auto_fit(self) -> None:
        self._fit_grid = False
        if self.auto_fit:                                   # freeze the camera
            self.auto_fit = False
            self._free_scale = self.scale
        else:
            self.auto_fit, self.focus, self._zoom_bias, self._focus_slot = True, None, 1.0, None

    def _focus_position(self, position: int) -> None:
        """Follow the car in race position `position` (0 = leader)."""
        alive = self._alive()
        order = [i for i in self._order() if alive[i]]
        if position < len(order):
            self._focus_car(order[position])
            self._focus_slot = position

    def _focus_car(self, idx: int) -> None:
        if 0 <= idx < self.cars.total and self._alive()[idx]:
            self._fit_grid = False
            self._focus_slot = None
            self.focus, self.auto_fit = idx, False
            self._focus_rank = self._rank_of(idx)

    def _focus_random(self) -> None:
        alive = np.flatnonzero(self._alive())
        choices = alive[alive != self.focus] if alive.size > 1 else alive
        if choices.size:
            self._focus_car(int(self._rng.choice(choices)))

    def reset_camera(self) -> None:
        """New episode: follow P1, or the human car in a manual run. Clears stale requests."""
        self.paused = self.quit_requested = self.end_requested = False
        self._zoom_bias = 1.0
        self._fit_grid = False
        self._focus_slot = None
        self._track_cache = None
        if self.screen is not None and pygame.display.get_init() and pygame.display.get_surface() is not None:
            self._track_surface()
        self._snap = True
        if self.cars.usr_engaged and self.cars.total > 0:
            if not self._verbose_touched:
                self.verbose_level = max(self.verbose_level, 1)
            self.focus, self.auto_fit = self.cars.total - 1, False
            self._focus_rank = self._rank_of(self.focus)
        else:
            self.focus, self.auto_fit = None, False
            self._focus_slot = 0

    # Camera

    def _alive(self) -> np.ndarray:
        return np.asarray(self.cars.race_active, dtype=bool)

    def _order(self) -> list[int]:
        order = self.cars.ranking
        if order is None:
            order = np.argsort(-self.players.fitness)
        return [int(i) for i in order]

    def _rank_of(self, idx: int) -> int:
        order = self._order()
        return order.index(idx) if idx in order else 0

    def _fit_scale(self) -> float:
        """Scale at which the whole grid fits the canvas (minimum zoom)."""
        w, h = self.screen.get_size()
        return min(w / self.world_size[0], h / self.world_size[1])

    def _update_camera(self) -> None:
        cfg = self.config
        w, h = self.screen.get_size()
        lo = self._fit_scale()
        hi = max(cfg.zoom_max, lo)
        car = float(max(self.cars.width, self.cars.height))
        alive = self._alive()

        # Following a race position: the current holder is the focus
        if self._focus_slot is not None:
            order = [i for i in self._order() if alive[i]]
            if order:
                self.focus = order[min(self._focus_slot, len(order) - 1)]
                self._focus_rank = self._rank_of(self.focus)
            else:
                self.focus, self._focus_slot, self.auto_fit = None, None, True

        # An eliminated car hands the camera to the next one in the standings
        if self.focus is not None and not alive[self.focus]:
            order = self._order()
            after = [i for i in order[self._focus_rank + 1:] if alive[i]]
            before = [i for i in reversed(order[:self._focus_rank]) if alive[i]]
            nxt = (after or before or [None])[0]
            if nxt is None:
                self.focus, self.auto_fit = None, True
            else:
                self.focus = nxt
                self._focus_rank = self._rank_of(nxt)
        elif self.focus is not None and self._focus_slot is None:
            self._focus_rank = self._rank_of(self.focus)

        tx, ty, ts = self.cam_x, self.cam_y, self.scale
        if self._fit_grid:
            tx, ty, ts = self.world_size[0] / 2, self.world_size[1] / 2, lo
        elif self.focus is not None:
            tx, ty = float(self.cars.x[self.focus]), float(self.cars.y[self.focus])
            ts = min(w, h) / (cfg.focus_view_cars * car) * self._zoom_bias
        elif self.auto_fit:
            lead = [i for i in self._order() if alive[i]][:cfg.fit_cars]
            if lead:
                xs, ys = self.cars.x[lead], self.cars.y[lead]
                margin = 2 * cfg.fit_margin_cars * car
                bw = max(float(xs.max() - xs.min()) + margin, cfg.min_view_cars * car)
                bh = max(float(ys.max() - ys.min()) + margin, cfg.min_view_cars * car)
                tx, ty = float(xs.min() + xs.max()) / 2, float(ys.min() + ys.max()) / 2
                ts = min(w / bw, h / bh) * self._zoom_bias
            else:
                tx, ty, ts = self.world_size[0] / 2, self.world_size[1] / 2, lo
        else:
            ts = self._free_scale

        raw = ts
        ts = min(max(ts, lo), hi)
        if self._fit_grid:
            pass
        elif self.focus is not None or self.auto_fit:
            if raw > 0:
                self._zoom_bias *= ts / raw      # keeps zoom keys from winding up at a limit
        else:
            self._free_scale = ts

        k = 1.0 if (self._snap or self._fit_grid) else cfg.camera_smoothing
        self._snap = False
        self.cam_x += (tx - self.cam_x) * k
        self.cam_y += (ty - self.cam_y) * k
        self.scale = float(np.exp(np.log(self.scale) + (np.log(ts) - np.log(self.scale)) * k))
        self.scale = min(max(self.scale, lo), hi)

    # Pixel mode

    def _pixel_active(self) -> bool:
        """True when `config.pixel_mode` ("off", "on", "race") applies to this frame."""
        mode = self.config.pixel_mode
        return mode == "on" or (mode == "race" and self.cars.race.enabled)

    def _view_rect(self) -> tuple[float, float, float, float]:
        """World rectangle shown on the canvas: (left, top, width, height)."""
        w, h = self.screen.get_size()
        gw, gh = self.world_size
        vw, vh = w / self.scale, h / self.scale
        # Clamp the view to the grid, center it when the grid is smaller
        left = (gw - vw) / 2 if vw >= gw else min(max(self.cam_x - vw / 2, 0.0), gw - vw)
        top = (gh - vh) / 2 if vh >= gh else min(max(self.cam_y - vh / 2, 0.0), gh - vh)
        return left, top, vw, vh

    def _blit_camera(self) -> None:
        left, top, vw, vh = self._view_rect()
        gw, gh = self.world_size
        x0, y0 = max(left, 0.0), max(top, 0.0)
        x1, y1 = min(left + vw, gw), min(top + vh, gh)
        sx0, sy0 = int(np.floor(x0)), int(np.floor(y0))
        sx1, sy1 = min(int(np.ceil(x1)), gw), min(int(np.ceil(y1)), gh)
        if sx1 <= sx0 or sy1 <= sy0:
            self.screen.fill(self.bg_color)
            return
        src = self.world.subsurface(pygame.Rect(sx0, sy0, sx1 - sx0, sy1 - sy0))
        size = (max(1, round((sx1 - sx0) * self.scale)), max(1, round((sy1 - sy0) * self.scale)))
        dest = (round((sx0 - left) * self.scale), round((sy0 - top) * self.scale))
        if dest != (0, 0) or size[0] < self.screen.get_width() or size[1] < self.screen.get_height():
            self.screen.fill(self.bg_color)
        if size == src.get_size():
            self.screen.blit(src, dest)
            return
        scale = pygame.transform.smoothscale if self.config.smooth_scaling else pygame.transform.scale
        if dest == (0, 0) and size == self.screen.get_size():
            try:
                scale(src, size, self.screen)
                return
            except (ValueError, pygame.error):
                pass                                        # pixel formats differ
        self.screen.blit(scale(src, size), dest)

    # Overlays

    def _panel_text(self, text: str, anchor: str, y_pad: int = 4) -> None:
        if self.font is None:
            return
        txt = render_text(self.font, text, True, (230, 230, 230))
        rect = pygame.Rect(0, 0, txt.get_width() + 12, txt.get_height() + 8)
        sw, sh = self.screen.get_size()
        rect.bottom = sh - y_pad
        if anchor == "right":
            rect.right = sw - 4
        else:
            rect.centerx = sw // 2
        draw_panel(self.screen, rect, self.cars.config.overlay_color, self.cars.config.overlay_alpha)
        self.screen.blit(txt, (rect.x + 6, rect.y + 4))

    def _draw_stats(self) -> None:
        """Top center: measured frame rate against the cap, and the episode frame count."""
        if self.font is None:
            return
        measured = self.clock.get_fps() if self.clock is not None else 0.0
        cap = f"{self.fps}" if self.fps else "max"
        frames = int(np.max(self.players.frames_done)) if len(self.players.frames_done) else 0
        txt = render_text(self.font, f"FPS {measured:.0f}/{cap}   Frame {frames}", True, (230, 230, 230))
        rect = pygame.Rect(0, 4, txt.get_width() + 12, txt.get_height() + 8)
        rect.centerx = self.screen.get_width() // 2
        draw_panel(self.screen, rect, self.cars.config.overlay_color, self.cars.config.overlay_alpha)
        self.screen.blit(txt, (rect.x + 6, rect.y + 4))

    def _draw_hint(self) -> None:
        if not self.config.show_controls_hint:
            return
        if self.focus is not None:
            label = self.cars.label(self.focus)
            mode = f"P{self._rank_of(self.focus) + 1} car {self.focus}" + (f" ({label})" if label is not None else "")
        elif self._fit_grid:
            mode = "Grid"
        else:
            mode = f"Top {self.config.fit_cars}" if self.auto_fit else "Free"
        verbose = ("off", "stats", "+lidar", "+tiles")[self.verbose_level]
        self._panel_text(
            f"{mode} | x{self.scale:.2f} | -/+ zoom  0 top{self.config.fit_cars}  L grid  1-9 position  Tab random  "
            f"\nV info:{verbose}  Space pause  Shift+-/+ fps  Esc end", "right"
        )

    def _draw_paused(self) -> None:
        """PAUSED banner in the middle of the window."""
        w, h = self.screen.get_size()
        big = pygame.font.Font(None, max(32, int(h * 0.14)))
        txt = big.render("PAUSED", True, (255, 255, 255))
        rect = txt.get_rect(center=(w // 2, h // 2))
        draw_panel(self.screen, rect.inflate(int(h * 0.08), int(h * 0.05)), (0, 0, 0), 170, 12)
        self.screen.blit(txt, rect)

    def _draw_scene(self) -> None:
        self._compose(False, hud=False, paused_overlay=False)

    def countdown(self) -> bool:
        """Blocking 3-2-1-GO sequence over the frozen starting grid.

        Returns False if the window was closed meanwhile (QUIT is re-posted for the caller).
        """
        cfg = self.race.countdown
        if self.screen is None:
            self.initialize()
        clock = pygame.time.Clock()
        clock.tick()
        w, h = self.screen.get_size()
        big = pygame.font.Font(None, int(h * 0.22))
        small = pygame.font.Font(None, int(h * 0.06))
        steps = cfg.steps
        total_s = steps * cfg.step_seconds + cfg.go_seconds
        light_r = max(8, int(h * 0.035))

        t = 0.0                                             # stands still while paused
        while True:
            if t >= total_s:
                return True
            for event in pygame.event.get():
                self._handle_event(event)
            if self.quit_requested:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                self.running = False
                return False
            if self.end_requested:
                return True

            self._draw_scene()
            go = t >= steps * cfg.step_seconds
            step = min(int(t // cfg.step_seconds), steps - 1)
            local = (t - (steps * cfg.step_seconds if go else step * cfg.step_seconds)) / (
                cfg.go_seconds if go else cfg.step_seconds
            )

            # Dim backdrop, fades out during GO
            veil = pygame.Surface((w, h), pygame.SRCALPHA)
            veil.fill((0, 0, 0, int(cfg.dim * (1.0 - local) if go else cfg.dim)))
            self.screen.blit(veil, (0, 0))

            # Start lights
            gap = light_r * 3
            x0 = w // 2 - gap * (steps - 1) // 2
            for i in range(steps):
                lit = go or i <= step
                color = (40, 220, 80) if go else (230, 30, 30) if lit else (60, 25, 25)
                pos = (x0 + i * gap, int(h * 0.07))
                pygame.draw.circle(self.screen, (15, 15, 15), pos, light_r + 4)
                pygame.draw.circle(self.screen, color, pos, light_r)

            # Big number that starts large and settles
            label = cfg.text_go if go else str(steps - step)
            color = (60, 235, 100) if go else (255, 255, 255)
            scale = 1.0 + 0.35 * (1.0 - min(local * 4.0, 1.0)) ** 2
            text = big.render(label, True, color)
            text = pygame.transform.rotozoom(text, 0, scale)
            alpha = int(255 * (1.0 - local ** 3)) if go else 255
            text.set_alpha(alpha)
            self.screen.blit(text, text.get_rect(center=(w // 2, int(h * 0.26))))
            if not go and step == 0:
                hint = small.render("Get ready", True, (200, 200, 200))
                self.screen.blit(hint, hint.get_rect(center=(w // 2, int(h * 0.40))))

            if self.paused:
                self._draw_paused()
            pygame.display.flip()
            dt = min(clock.tick(60) / 1000.0, 0.1)
            if not self.paused:
                t += dt