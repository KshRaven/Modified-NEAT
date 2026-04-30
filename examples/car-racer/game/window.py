import pygame
import numpy as np

from typing import Optional, Any
from pygame import Surface

from .config import WindowConfig, overwrite_config
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
            **params: dict[str, Any]
    ) -> None:
        # Require config to be provided
        if config is None:
            config = WindowConfig()

        overwrite_config(config, params)
        self.config = config
        
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
        self.font_size = config.font_size

    def initialize(self) -> None:
        pygame.init()
        pygame.display.set_caption(self.title)
        self.screen = pygame.display.set_mode((self.width+self.offset, self.height+self.offset))
        if self.fps is not None:
            self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, self.font_size)
        self.cars.font = pygame.font.Font(None, self.font_size)
        self.running = True

    def close(self) -> None:
        self.running = False
        self.font = None # NOTE: Not resetting/closing the font and surfaces leads to segmentation error/ PyGame window freezing
        self.cars.font = None
        pygame.quit()

    def render(self) -> None:
        if self.screen is None:
            raise RuntimeError("Call initialize() before render().")

        self.screen.fill(self.bg_color)
        # space = Surface((self.width, self.height), pygame.SRCALPHA)
        self.grid.render(self.screen, verbose=False)
        self.cars.render(self.screen, verbose=False)
        # self.screen.blit(space, (self.offset, self.offset))

        pygame.display.flip()

        if self.clock is not None:
            self.clock.tick(self.fps)
    