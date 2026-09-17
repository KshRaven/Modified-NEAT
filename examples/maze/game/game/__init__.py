"""
Maze Runner Game Environment Package

Same organization/conventions as the Car-Racer environment package.

- functional : Enums, types, and pure helper functions for grid movement
- constants  : Colors and numpy dtypes
- config     : Configuration classes (the 4 modes + return_grid + reset_on_death live in AgentConfig)
- tile       : Cell class — a single maze grid cell
- grid       : Grid class — maze generation, goal/trap placement, BFS distances
- player     : Players class — vectorized lives/score/fitness bookkeeping
- agent      : Agents class — the dots: grid-cell movement + observations for all 4 modes
- window     : Window class — pygame rendering (flat squares, path trails, ranked blobs)
- env        : Game(gymnasium.Env) — reset/step/render/run
"""

from .functional import (
    Direction, CellType, Position, Color,
    get_delta, get_next_position, opposite, vector_to_direction, direction_to_vector,
    ALL_DIRECTIONS,
)

from .constants import (
    COLOR_BACKGROUND, COLOR_FLOOR, COLOR_WALL, COLOR_START, COLOR_GOAL, COLOR_TRAP,
    COLOR_PLAYER_BOT, COLOR_PLAYER_RANK1, COLOR_PLAYER_RANK2, COLOR_PLAYER_RANK3,
)

from .config import (
    Configuration, TileConfig, GridConfig, GenConfig, AgentConfig, WindowConfig,
    PlayerConfig, EnvironmentConfig, overwrite_config,
)

from .tile import Cell
from .grid import Grid
from .player import Players
from .agent import Agents
from .window import Window
from .env import Game

__all__ = [
    'Direction', 'CellType', 'Position', 'Color',
    'get_delta', 'get_next_position', 'opposite', 'vector_to_direction', 'direction_to_vector',
    'ALL_DIRECTIONS',

    'COLOR_BACKGROUND', 'COLOR_FLOOR', 'COLOR_WALL', 'COLOR_START', 'COLOR_GOAL', 'COLOR_TRAP',
    'COLOR_PLAYER_BOT', 'COLOR_PLAYER_RANK1', 'COLOR_PLAYER_RANK2', 'COLOR_PLAYER_RANK3',

    'Configuration', 'TileConfig', 'GridConfig', 'GenConfig', 'AgentConfig', 'WindowConfig',
    'PlayerConfig', 'EnvironmentConfig', 'overwrite_config',

    'Cell', 'Grid', 'Players', 'Agents', 'Window', 'Game',
]
