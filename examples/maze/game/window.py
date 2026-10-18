from typing import Any, Optional

import pygame
from pygame import Surface

from .config import WindowConfig, overwrite_config
from .grid import Grid
from .functional import Direction
from .agent import (
    Agents, LIDAR_WALL, LIDAR_PATH, LIDAR_ALTERNATE, LIDAR_CHECKPOINT,
    LIDAR_START, LIDAR_GOAL, LIDAR_TRAP, LIDAR_DEAD_END,
)
from .player import Players
from .constants import (
    COLOR_PLAYER_BOT, COLOR_PLAYER_RANK1, COLOR_PLAYER_RANK2, COLOR_PLAYER_RANK3,
    COLOR_PATH_BOT, COLOR_PATH_RANK1, COLOR_PATH_RANK2, COLOR_PATH_RANK3, COLOR_TEXT,
    COLOR_BEST_PATH, COLOR_LIDAR_WALL, COLOR_LIDAR_PATH, COLOR_LIDAR_CHECKPOINT,
    COLOR_LIDAR_START, COLOR_LIDAR_GOAL, COLOR_LIDAR_TRAP, COLOR_LIDAR_DEAD_END,
)


class Window:
    def __init__(
        self, grid: Grid, players: Players, agents: Agents,
        config: WindowConfig | None = None, **params: dict[str, Any],
    ) -> None:
        if config is None:
            config = WindowConfig()
        overwrite_config(config, params)
        self.config = config

        self.grid = grid
        self.players = players
        self.agents = agents
        self.width, self.height = self.grid.width, self.grid.height

        self.title = config.title
        self.fps = config.fps
        self.bg_color = config.background_color
        self.font_size = config.font_size
        self.draw_path = config.draw_path
        self.draw_best_path = config.draw_best_path
        self.best_path_color = config.best_path_color or COLOR_BEST_PATH
        self.best_path_width = max(1, int(config.best_path_width))
        self.draw_lidar = config.draw_lidar
        self.lidar_width = max(1, int(config.lidar_width))
        self.blob_radius = max(2, int(self.grid.cell_size * config.blob_radius_ratio))

        self.screen: Optional[Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        self.running = False

    def initialize(self) -> None:
        pygame.init()
        pygame.display.set_caption(self.title)
        self.screen = pygame.display.set_mode((self.width, self.height))
        if self.fps is not None:
            self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, self.font_size)
        self.running = True

    def close(self) -> None:
        self.running = False
        self.font = None
        pygame.quit()

    def _rank_colors(self):
        """Map genome index -> (blob_color, path_color) using fitness ranking; top-3 stand out."""
        ranking = self.players.ranking()
        colors = {}
        top_blob = [COLOR_PLAYER_RANK1, COLOR_PLAYER_RANK2, COLOR_PLAYER_RANK3]
        top_path = [COLOR_PATH_RANK1, COLOR_PATH_RANK2, COLOR_PATH_RANK3]
        for rank, idx in enumerate(ranking[:3]):
            colors[int(idx)] = (top_blob[rank], top_path[rank])
        return colors

    def _best_lidar_player(self) -> int | None:
        """Return the highest-fitness unfinished player with a life remaining."""
        eligible = (self.players.lives > 0) & (~self.players.completed)
        for index in self.players.ranking():
            index = int(index)
            if eligible[index]:
                return index
        return None

    def _render_lidar(self) -> None:
        player = self._best_lidar_player()
        if player is None:
            return

        row, col = int(self.agents.row[player]), int(self.agents.col[player])
        origin = self.grid.cell((row, col)).center_abs
        type_colors = {
            LIDAR_WALL: COLOR_LIDAR_WALL,
            LIDAR_PATH: COLOR_LIDAR_PATH,
            LIDAR_ALTERNATE: COLOR_LIDAR_PATH,
            LIDAR_CHECKPOINT: COLOR_LIDAR_CHECKPOINT,
            LIDAR_START: COLOR_LIDAR_START,
            LIDAR_GOAL: COLOR_LIDAR_GOAL,
            LIDAR_TRAP: COLOR_LIDAR_TRAP,
            LIDAR_DEAD_END: COLOR_LIDAR_DEAD_END,
        }
        deltas = ((-1, 0), (0, 1), (1, 0), (0, -1))
        directions = tuple(Direction)
        for (dr, dc), direction in zip(deltas, directions):
            distance, target_type = self.agents._cast_lidar(row, col, direction)
            if distance == 0:
                end = (origin[0] + dc * self.grid.cell_size / 2, origin[1] + dr * self.grid.cell_size / 2)
            else:
                target = self.grid.cell((row + dr * distance, col + dc * distance)).center_abs
                end = target
            color = type_colors.get(target_type, COLOR_LIDAR_PATH)
            pygame.draw.line(
                self.screen, color, (int(origin[0]), int(origin[1])),
                (int(end[0]), int(end[1])), self.lidar_width,
            )
            pygame.draw.circle(self.screen, color, (int(end[0]), int(end[1])), self.lidar_width + 1)

    def render(self) -> None:
        if self.screen is None:
            raise RuntimeError("Call initialize() before render().")

        self.screen.fill(self.bg_color)
        self.grid.render(self.screen)

        if self.draw_best_path and len(self.grid.best_path) >= 2:
            best_points = [self.grid.cell(pos).center_abs for pos in self.grid.best_path]
            pygame.draw.lines(
                self.screen,
                self.best_path_color,
                False,
                best_points,
                self.best_path_width,
            )
            marker_radius = max(2, self.best_path_width // 2)
            for checkpoint in self.grid.checkpoint_order:
                cx, cy = self.grid.cell(checkpoint).center_abs
                pygame.draw.circle(
                    self.screen,
                    self.best_path_color,
                    (int(cx), int(cy)),
                    marker_radius,
                )

        if self.draw_lidar:
            self._render_lidar()

        top3 = self._rank_colors()

        if self.draw_path:
            for i in range(self.agents.total):
                if not self.players.active[i]:
                    continue
                _, path_color = top3.get(i, (COLOR_PLAYER_BOT, COLOR_PATH_BOT))
                path = self.agents.paths[i]
                if len(path) >= 2:
                    pts = [self.grid.cell(p).center_abs for p in path]
                    pygame.draw.lines(self.screen, path_color, False, pts, 1)

        for i in range(self.agents.total):
            if not self.players.active[i]:
                continue
            blob_color, _ = top3.get(i, (COLOR_PLAYER_BOT, COLOR_PATH_BOT))
            pos = (int(self.agents.row[i]), int(self.agents.col[i]))
            cx, cy = self.grid.cell(pos).center_abs
            pygame.draw.circle(self.screen, blob_color, (int(cx), int(cy)), self.blob_radius)

        self._render_hud()

        pygame.display.flip()
        if self.clock is not None:
            self.clock.tick(self.fps)

    def _render_hud(self) -> None:
        if self.font is None:
            return
        text = (
            f"Active: {self.players.active_total}/{self.agents.total}  "
            f"Best score: {int(self.players.true_scores.max()) if self.agents.total else 0}"
        )
        surf = self.font.render(text, True, COLOR_TEXT)
        self.screen.blit(surf, (6, 4))
