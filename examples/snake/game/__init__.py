"""
Snake Game Environment Package

Organized game components (same layout as the Car-Racer package):
- functional: Enums, types and direction helpers
- constants: dtypes, cell codes, colors
- config: Configuration classes
- util: Utility functions for rendering and drawing
- kernels: Numba accelerated movement, sensors and reward shaping
- grid: Board occupancy and food placement
- snake: Snake state, movement and sensing
- player: Lives, score and fitness
- reward: Reward shaping
- renderer / pixel: Board drawing and the optional retro look
- window: Window, keyboard controls and overlays
- env: Gymnasium Environment wrapper
"""

from .functional import (
    Position, Color, Direction, Action,
    turn_direction, get_next_position, wrap_position, action_towards,
)
from .constants import (
    COLOR_EMPTY, COLOR_EMPTY_ALT, COLOR_WALL, COLOR_WRAP, COLOR_FOOD, COLOR_FOOD_LEAF,
    COLOR_HEAD, COLOR_BODY, COLOR_BODY_ALT, COLOR_RAY, COLOR_RAY_FOOD, COLOR_TEXT,
)
from .config import (
    Configuration, GridConfig, SnakeConfig, RewardConfig, WindowConfig,
    GenConfig, PlayerConfig, EnvironmentConfig,
)
from .util import render_text, draw_panel, shade, draw_dashed_rect
from .grid import Grid
from .snake import Snakes
from .player import Players
from .reward import RewardShaper
from .renderer import SnakeRenderer
from .pixel import PixelRenderer
from .window import Window
from .env import Game

__all__ = [
    # Enums and types
    'Position', 'Color', 'Direction', 'Action',
    # Helper functions
    'turn_direction', 'get_next_position', 'wrap_position', 'action_towards',
    # Configuration
    'Configuration', 'GridConfig', 'SnakeConfig', 'RewardConfig', 'WindowConfig',
    'GenConfig', 'PlayerConfig', 'EnvironmentConfig',
    # Colors
    'COLOR_EMPTY', 'COLOR_EMPTY_ALT', 'COLOR_WALL', 'COLOR_WRAP', 'COLOR_FOOD', 'COLOR_FOOD_LEAF',
    'COLOR_HEAD', 'COLOR_BODY', 'COLOR_BODY_ALT', 'COLOR_RAY', 'COLOR_RAY_FOOD', 'COLOR_TEXT',
    # Utilities
    'render_text', 'draw_panel', 'shade', 'draw_dashed_rect',
    # Board and players
    'Grid', 'Snakes', 'Players', 'RewardShaper',
    # Rendering
    'SnakeRenderer', 'PixelRenderer', 'Window',
    # Environment
    'Game',
]