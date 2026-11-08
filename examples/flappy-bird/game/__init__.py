"""
Flappy Bird Game Environment Package

Organized game components (same layout as the Car-Racer and Snake packages):
- functional: Enums, types and small helpers
- constants: dtypes, action codes, colors
- config: Configuration classes
- util: Utility functions for rendering and drawing
- kernels: Numba accelerated physics, pipe scrolling, sensors and reward shaping
- world: Pipe ring, scroll distance and the Env-private RNG
- bird: Bird state, flight, collision and sensing
- player: Lives, score and fitness
- reward: Reward shaping and the skill curriculum
- sprite / scenery / renderer / pixel: Pixel-art bird, city / trees / clouds / floor, world drawing and the optional retro look
- window: Window, keyboard controls and overlays
- env: Gymnasium Environment wrapper
"""

from .functional import Position, Color, Action, pipe_center, clamp
from .constants import (
    ACT_NOOP, ACT_FLAP,
    COLOR_SKY_TOP, COLOR_SKY_BOTTOM, COLOR_HILL, COLOR_CLOUD, COLOR_PIPE, COLOR_PIPE_DARK, COLOR_PIPE_LIGHT,
    COLOR_FLOOR, COLOR_FLOOR_ALT, COLOR_FLOOR_EDGE, COLOR_RAY, COLOR_GAP, COLOR_HITBOX, COLOR_TEXT,
)
from .config import (
    Configuration, WorldConfig, BirdConfig, RewardConfig, WindowConfig,
    GenConfig, PlayerConfig, EnvironmentConfig,
)
from .util import render_text, draw_panel, shade, mix, draw_dashed_line
from .world import World
from .bird import Birds
from .player import Players
from .reward import RewardShaper
from .scenery import CELL, make_layer, make_cloud, make_floor_tile, make_pipe_strip, make_pipe_cap
from .sprite import rainbow, make_bird, make_halo, export_bird_images
from .renderer import FlappyRenderer
from .pixel import PixelRenderer
from .window import Window
from .env import Game

__all__ = [
    # Enums and types
    'Position', 'Color', 'Action',
    # Helper functions
    'pipe_center', 'clamp',
    # Configuration
    'Configuration', 'WorldConfig', 'BirdConfig', 'RewardConfig', 'WindowConfig', 'DraftConfig',
    'GenConfig', 'PlayerConfig', 'EnvironmentConfig',
    # Constants
    'ACT_NOOP', 'ACT_FLAP',
    'COLOR_SKY_TOP', 'COLOR_SKY_BOTTOM', 'COLOR_HILL', 'COLOR_CLOUD', 'COLOR_PIPE', 'COLOR_PIPE_DARK', 'COLOR_PIPE_LIGHT',
    'COLOR_FLOOR', 'COLOR_FLOOR_ALT', 'COLOR_FLOOR_EDGE', 'COLOR_RAY', 'COLOR_GAP', 'COLOR_HITBOX', 'COLOR_TEXT',
    # Utilities
    'render_text', 'draw_panel', 'shade', 'mix', 'draw_dashed_line',
    # World, birds and players
    'World', 'Birds', 'Players', 'RewardShaper',
    # Rendering
    'CELL', 'make_layer', 'make_cloud', 'make_floor_tile', 'make_pipe_strip', 'make_pipe_cap',
    'rainbow', 'make_bird', 'make_halo', 'export_bird_images', 'FlappyRenderer', 'PixelRenderer', 'Window',
    # Environment
    'Game',
]