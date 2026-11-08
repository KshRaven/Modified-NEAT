"""Pygame window: canvas, layout, keyboard controls and overlays. Drawing itself lives in renderer.py / pixel.py."""
from __future__ import annotations

from typing import Any

import numpy as np
import pygame
from pygame import Surface

from .bird import Birds
from .config import WindowConfig, overwrite_config
from .pixel import PixelRenderer
from .player import Players
from .renderer import FlappyRenderer
from .util import clear_text_cache, draw_panel, render_text
from .world import World

__all__ = ["Window"]

MIN_WIDTH = 640     # keeps the HUD readable on small worlds


class Window:
    FPS_STEPS = (10, 15, 20, 30, 45, 60, 90, 120, 144, 240, None)   # frame rate cap ladder, None = uncapped

    def __init__(
        self,
        world: World, players: Players, birds: Birds,
        config: WindowConfig | None = None,
        **params: dict[str, Any],
    ) -> None:
        if config is None:
            config = WindowConfig()

        overwrite_config(config, params)
        self.config = config

        self.world = world
        self.players = players
        self.birds = birds
        self.renderer = FlappyRenderer(birds, config)
        self._pixel = PixelRenderer(config)

        self.title = config.title
        self.bg_color = config.background_color
        self.font_size = int(config.font_size)
        self.screen: Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None
        self.running = False
        self.fps = config.fps
        self.width = self.height = 0

        # Which bird is shown (None = the best one)
        self.focus: int | None = None
        self._rng = np.random.default_rng()

        # Run control, set by the keyboard and read by the loop that owns the episode
        self.manual = False                 # manual play: Space flaps, Enter pauses
        self.message: str | None = None     # centred banner (manual play: start / game over)
        self.paused = False
        self.quit_requested = False         # window closed
        self.end_requested = False          # Esc pressed
        self._verbose: bool | int = False
        self.verbose_level = int(config.verbose_level)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    @property
    def pixel(self) -> bool:
        return bool(self.config.pixel)

    @property
    def scale(self) -> float:
        return max(0.1, float(self.config.scale))

    @property
    def board_size(self) -> tuple[int, int]:
        w = self.world
        if self.pixel:
            return self._pixel.board_size(w.width, w.height)
        return max(1, round(w.width * self.scale)), max(1, round(w.height * self.scale))

    def _layout(self) -> tuple[int, int, int, int]:
        """(screen width, screen height, board left, board top)."""
        bw, bh = self.board_size
        hud = self.renderer.hud_height()
        width = max(bw, MIN_WIDTH)
        return width, hud + bh, (width - bw) // 2, hud

    def _apply_size(self) -> None:
        self.width, self.height, _, _ = self._layout()
        self.screen = pygame.display.set_mode((self.width, self.height))

    def initialize(self) -> None:
        pygame.init()
        pygame.display.set_caption(self.title)
        self.font = pygame.font.Font(None, self.font_size)
        self.renderer.set_fonts(self.font_size)
        self._apply_size()
        self.clock = pygame.time.Clock()
        clear_text_cache()
        self.running = True

    def close(self) -> None:
        self.running = False
        clear_text_cache()
        self.font = None                    # fonts must be released before pygame.quit()
        self.renderer.close()
        self.screen = None
        pygame.quit()

    def reset_view(self) -> None:
        """New episode: show the best bird, clear stale requests."""
        self.paused = self.quit_requested = self.end_requested = False
        self.focus = None
        if self.screen is not None and pygame.display.get_init() and pygame.display.get_surface() is not None:
            if (self.width, self.height) != self._layout()[:2]:
                self._apply_size()

    # ------------------------------------------------------------------
    # Shown bird
    # ------------------------------------------------------------------

    def shown(self) -> int:
        """Index of the bird on screen: the focused one, else the best active one."""
        total = self.birds.total
        if self.focus is not None and 0 <= self.focus < total:
            return self.focus
        return self.players.best_index if self.players.active.any() else 0

    def _focus_random(self) -> None:
        alive = np.flatnonzero(self.players.active)
        shown = self.shown()
        choices = alive[alive != shown] if alive.size > 1 else alive
        if choices.size:
            self.focus = int(self._rng.choice(choices))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

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

    def _compose(self, verbose: bool | int = False, hud: bool = True, paused_overlay: bool = True) -> None:
        """Draw one frame onto the screen (no flip, no clock tick)."""
        level = max(self.verbose_level, 2 if verbose is True else int(verbose or 0))
        index = self.shown()
        _, _, left, top = self._layout()
        self.screen.fill(self.bg_color)

        if self.pixel:
            self._pixel.draw(self, index, level, left, top)
        else:
            self.renderer.draw_scene(self.screen, left, top, self.scale, index, level)

        if hud:
            extra = [f"{self.fps} fps" if self.fps else "max fps"]
            if self.world.draft_label:
                extra.insert(0, self.world.draft_label)
            self.renderer.render_hud(self.screen, index, extra)
            if level >= 2:
                self.renderer.render_debug(self.screen, index, left + 6, top + 6)
        if self.message and not self.paused:
            self._draw_banner(self.message)
        if self.paused and paused_overlay:
            self._draw_paused()

    def _draw_banner(self, message: str) -> None:
        text = render_text(self.font, message, (255, 255, 255))
        rect = text.get_rect(center=(self.width // 2, self.height // 2 + self.renderer.hud_height() // 2))
        draw_panel(self.screen, rect.inflate(24, 16), alpha=170)
        self.screen.blit(text, rect)

    def _draw_paused(self) -> None:
        key = "Enter" if self.manual else "Space"
        text = render_text(self.font, f"PAUSED  ({key} to resume)", (255, 255, 255))
        rect = text.get_rect(center=(self.width // 2, self.height // 2 + self.renderer.hud_height() // 2))
        draw_panel(self.screen, rect.inflate(24, 16), alpha=170)
        self.screen.blit(text, rect)

    # ------------------------------------------------------------------
    # Keyboard and run control
    # ------------------------------------------------------------------

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
        elif event.type == pygame.KEYDOWN:
            key = event.key
            if key == pygame.K_ESCAPE:
                self.end_requested = True
            elif key == (pygame.K_RETURN if self.manual else pygame.K_SPACE):
                self.paused = not self.paused
            elif key == pygame.K_v:
                self.verbose_level = (self.verbose_level + 1) % 3
            elif key == pygame.K_p:
                self.config.pixel = not self.config.pixel
                self._apply_size()
            elif key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS) \
                    and getattr(event, "mod", 0) & pygame.KMOD_SHIFT:
                self._change_fps(-1 if key in (pygame.K_MINUS, pygame.K_KP_MINUS) else +1)
            elif key == pygame.K_TAB:
                self._focus_random()
            elif key in (pygame.K_0, pygame.K_KP0):
                self.focus = None
            elif pygame.K_1 <= key <= pygame.K_9:
                self._focus_index(key - pygame.K_1)
            elif pygame.K_KP1 <= key <= pygame.K_KP9:
                self._focus_index(key - pygame.K_KP1)

    def _focus_index(self, index: int) -> None:
        if 0 <= index < self.birds.total:
            self.focus = index

    def _change_fps(self, direction: int) -> None:
        """Move the frame rate cap one step down (-1) or up (+1) on `FPS_STEPS`."""
        steps = self.FPS_STEPS
        if self.fps is None:
            i = len(steps) - 1
        else:
            i = min(range(len(steps) - 1), key=lambda k: abs(steps[k] - self.fps))
        self.fps = steps[min(max(i + direction, 0), len(steps) - 1)]