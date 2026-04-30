"""
Car Racer Game Environment Package

Organized game components:
- functional: Enums, types, and helper functions for track generation
- config: Configuration classes
- util: Utility functions for rendering and drawing
- tiles: Tile and Grid classes for track management
- cars: Car classes for agents/players
- window: Window and rendering classes
- env: Gymnasium Environment wrapper
"""

# Core functionality and types
from .functional import (
    Type, Direction, Rotation, Position, Color,
    get_next_direction, get_next_position,
    direction_to_rotation, get_next_rotation
)

# Constants
from .constants import (
    COLOR_ROAD, COLOR_GRASS, COLOR_CURB_PRI, COLOR_CURB_SEC,
    COLOR_START_SEC, COLOR_START_PRI, COLOR_GRAVEL, COLOR_EMPTY, COLOR_LIDAR,
)

# Configuration
from .config import (
    Configuration, CarConfig, GridConfig, TileConfig, WindowConfig, 
    EnvironmentConfig, GenConfig, PlayerConfig
)

# Utilities
from .util import scale_image, blit_rotate_center, draw_sector, clip_rotated_surface

# Track system
from .tile import Tile
from .grid import Grid
from .car import Cars
from .window import Window
from .player import Players

# Environment
from .env import Game

__all__ = [
    # Enums and types
    'Type', 'Direction', 'Rotation', 'Position', 'Color',
    
    # Helper functions
    'get_next_direction', 'get_next_position',
    'direction_to_rotation', 'get_next_rotation',
    
    # Configuration
    'Configuration', 'CarConfig', 'GridConfig', 'TileConfig', 'WindowConfig', 
    'EnvironmentConfig', 'GenConfig', 'PlayerConfig',
    
    # Colors
    'COLOR_ROAD', 'COLOR_GRASS', 'COLOR_CURB_PRI', 'COLOR_CURB_SEC',
    'COLOR_START_SEC', 'COLOR_START_PRI', 'COLOR_GRAVEL', 'COLOR_EMPTY', 'COLOR_LIDAR',
    
    # Utilities
    'scale_image', 'blit_rotate_center', 'draw_sector', 'clip_rotated_surface',
    
    # Track system
    'Tile', 'Grid', 'Cars', 'Window', 'Players',
    
    # Environment
    'Game',
]
