import pygame
import numpy as np

from typing import Optional, Any
from pygame import Surface

from .config import WindowConfig, RaceConfig, overwrite_config
from .functional import Color        
from .constants import COLOR_CURB_SEC
from .tile import Tile
from .grid import Grid
from .car import Cars
from .player import Players


class Window:
    def __init__(
            self, 
            grid: Grid, players: Players, cars: Cars,
            config: WindowConfig | None = None,
            race: RaceConfig | None = None,
            **params: dict[str, Any]
    ) -> None:
        # Require config to be provided
        if config is None:
            config = WindowConfig()

        overwrite_config(config, params)
        self.config = config
        self.race = race if race is not None else RaceConfig()
        
        self.grid = grid
        self.players = players
        self.cars = cars
        self.width, self.height = self.grid.width, self.grid.height
        self.offset = 0
        self.title = config.title
        self.screen: Optional[Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.running = False
        self.fps = config.fps
        self.font: Optional[pygame.font.Font] = None
        self.bg_color = config.background_color
        self.font_size = max(8, round(config.font_size * config.font_scale))
        self.font_scale = config.font_scale

    def initialize(self) -> None:
        pygame.init()
        pygame.display.set_caption(self.title)
        self.screen = pygame.display.set_mode((self.width+self.offset, self.height+self.offset))
        if self.fps is not None:
            self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, self.font_size)
        self.cars.font = pygame.font.Font(None, self.font_size)
        self.cars.label_font = pygame.font.Font(None, max(8, self.font_size - 2))
        self.running = True

    def close(self) -> None:
        self.running = False
        self.font = None # NOTE: Not resetting/closing the font and surfaces leads to segmentation error/ PyGame window freezing
        self.cars.font = None
        self.cars.label_font = None
        self.cars._label_cache.clear()
        pygame.quit()

    def render(self, verbose: bool | int = False) -> None:
        if self.screen is None:
            raise RuntimeError("Call initialize() before render().")

        self.screen.fill(self.bg_color)
        # space = Surface((self.width, self.height), pygame.SRCALPHA)
        self.grid.render(self.screen, verbose=False)
        self.cars.render(self.screen, verbose=verbose)
        # self.screen.blit(space, (self.offset, self.offset))

        pygame.display.flip()

        if self.clock is not None:
            self.clock.tick(self.fps)
    
    # ------------------------------------------------------------------
    # Race start countdown
    # ------------------------------------------------------------------

    def _draw_scene(self) -> None:
        self.screen.fill(self.bg_color)
        self.grid.render(self.screen, verbose=False)
        self.cars.render(self.screen, verbose=False)

    def countdown(self) -> bool:
        """
        Blocking 3-2-1-GO! sequence drawn over the (frozen) starting grid.
        Returns False if the window was closed meanwhile (the QUIT event is re-posted
        so the caller's normal event loop still sees it).
        """
        cfg = self.race.countdown
        if self.screen is None:
            self.initialize()
        clock = self.clock or pygame.time.Clock()
        w, h = self.screen.get_size()
        big = pygame.font.Font(None, int(h * 0.22))
        small = pygame.font.Font(None, int(h * 0.06))
        steps = cfg.steps
        total_s = steps * cfg.step_seconds + cfg.go_seconds
        light_r = max(8, int(h * 0.035))

        start = pygame.time.get_ticks()
        while True:
            t = (pygame.time.get_ticks() - start) / 1000.0
            if t >= total_s:
                return True
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                    self.running = False
                    return False

            self._draw_scene()
            go = t >= steps * cfg.step_seconds
            step = min(int(t // cfg.step_seconds), steps - 1)
            local = (t - (steps * cfg.step_seconds if go else step * cfg.step_seconds)) / (
                cfg.go_seconds if go else cfg.step_seconds
            )

            # dim backdrop, fading out during GO!
            veil = pygame.Surface((w, h), pygame.SRCALPHA)
            veil.fill((0, 0, 0, int(cfg.dim * (1.0 - local) if go else cfg.dim)))
            self.screen.blit(veil, (0, 0))

            # start lights: one more red each step, all green on GO!
            gap = light_r * 3
            x0 = w // 2 - gap * (steps - 1) // 2
            for i in range(steps):
                lit = go or i <= step
                color = (40, 220, 80) if go else (230, 30, 30) if lit else (60, 25, 25)
                pos = (x0 + i * gap, int(h * 0.07))
                pygame.draw.circle(self.screen, (15, 15, 15), pos, light_r + 4)
                pygame.draw.circle(self.screen, color, pos, light_r)

            # big number with a punchy "pop" (starts large, settles)
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

            pygame.display.flip()
            clock.tick(60)