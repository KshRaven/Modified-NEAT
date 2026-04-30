import os
import json
import warnings

from typing import Any

from .constants import (
    COLOR_ROAD, COLOR_GRASS, COLOR_CURB_PRI, COLOR_CURB_SEC, COLOR_START_SEC, 
    COLOR_START_PRI, COLOR_GRAVEL, COLOR_EMPTY
)


class Configuration:
    """Base configuration class for loading parameters from file or dict."""
    
    def __init__(self, *, params: dict[str, Any] | None = None, file: str = None) -> None:
        # Load from json file provided or parameter dictionary
        if file is not None:
            if params is not None:
                warnings.warn(
                    "Overloading parameters with file input. "
                    "File input takes precedence over provided params."
                )
            params = self.load(file)
        
        if params is None:
            params = dict()
        
        self.params = params

    def load(self, file: str) -> dict[str, Any]:
        config_type = self.__class__.__name__
        # Check if file exists
        if not os.path.isfile(file):
            raise FileNotFoundError(f"{config_type} parameter file not found: {file}")
        
        with open(file, 'r') as f:
            print(f"Loaded {config_type} parameters from file: {file}")
            return json.load(f)
        
    def save(self, file: str) -> bool:
        # TODO: Implement
        pass


class TileConfig(Configuration):
    """Tile rendering configuration."""
    
    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)
        
        # Surface ratios
        self.gravel: float = self.params.get("gravel", 0.050)
        self.grass: float = self.params.get("grass", 0.125)
        self.curb: float = self.params.get("curb", 0.125)
        self.curb_pattern_splits: int = abs(int(self.params.get("curb_pattern_splits", 8)))
        self.start_checker_size: int | None = self.params.get("start_checker_size", None)
        self.finish_line_back_offset_ratio: float = self.params.get("finish_line_back_offset_ratio", 0.10)
        
        # Surface colors
        self.road_color: tuple[int, int, int] = tuple(self.params.get("road_color", COLOR_ROAD))
        self.grass_color: tuple[int, int, int] = tuple(self.params.get("grass_color", COLOR_GRASS))
        self.curb_red: tuple[int, int, int] = tuple(self.params.get("curb_red", COLOR_CURB_PRI))
        self.curb_white: tuple[int, int, int] = tuple(self.params.get("curb_white", COLOR_CURB_SEC))
        self.gravel_color: tuple[int, int, int] = tuple(self.params.get("gravel_color", COLOR_GRAVEL))


class GridConfig(Configuration):
    """Grid configuration for track layout."""
    
    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)
        
        self.grid_size: tuple[int, int] = tuple(self.params.get("grid_size", (5, 5)))
        self.cell_size: int = self.params.get("cell_size", 100)
        self.granularity: str = self.params.get("granularity", "pixels")  # 'pixels' or 'cell'
        
        self.tile: TileConfig = TileConfig(params=self.params.get("tile", {}))


class CarConfig(Configuration):
    """Car/Vehicle physics and control configuration."""
    
    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)
        
        # Movement physics
        self.max_linear_velocity: float = self.params.get("max_linear_velocity", 7.5)
        self.max_angular_velocity: float | None = self.params.get("max_angular_velocity", None)
        self.linear_acceleration: float = self.params.get("linear_acceleration", 0.25)
        self.angular_velocity: float = self.params.get("angular_velocity", 7.5)
        
        # Control coefficients
        self.reverse_coeff: float = self.params.get("reverse_coeff", 0.50)
        self.brake_coeff: float = self.params.get("brake_coeff", 1.00)
        
        # Visual/behavioral
        self.scale: float = self.params.get("car_scale", 0.40)
        self.use_lidar: bool = self.params.get("use_lidar", True)
        
        # Control settings
        self.toggle_reverse: bool = self.params.get("toggle_reverse", False)
        self.restrict_movement: bool = self.params.get("restrict_movement", True)
        self.discrete: bool = self.params.get("discrete", False)
        self.full_restart: bool = self.params.get("full_restart", False)
        
        # Thresholds and limits
        self.min_brake_steps: int = self.params.get("min_brake_steps", 5)
        self.cutoff: float = self.params.get("cutoff", 0.5)
        self.frames_per_tile: int = self.params.get("frames_per_tile", 100)
        
        # Surface friction: velocity multiplier applied every frame (1.0 = no decay)
        self.friction_road:   float = self.params.get("friction_road",   1.00)
        self.friction_curb:   float = self.params.get("friction_curb",   0.98)
        self.friction_grass:  float = self.params.get("friction_grass",  0.96)
        self.friction_gravel: float = self.params.get("friction_gravel", 0.94)
        
        # Surface handling: angular velocity multiplier (1.0 = full control)
        self.handling_road:   float = self.params.get("handling_road",   1.00)
        self.handling_curb:   float = self.params.get("handling_curb",   0.90)
        self.handling_grass:  float = self.params.get("handling_grass",  0.75)
        self.handling_gravel: float = self.params.get("handling_gravel", 0.90)
        
        # Surface slip: percentage of previous displacement of current frames displacement (0.0 = none)
        self.slip_road:   float = self.params.get("slip_road",   0.10)
        self.slip_curb:   float = self.params.get("slip_curb",   0.25)
        self.slip_grass:  float = self.params.get("slip_grass",  0.75)
        self.slip_gravel: float = self.params.get("slip_gravel", 0.50)
        
        # LiDAR configuration
        self.lidar_angle: int = self.params.get("lidar_angle", 8)
        self.lidar_max_dist: float | None = self.params.get("lidar_max_dist", None)  # If None, uses grid.cell_size * 2.5
        self.lidar_step: int = self.params.get("lidar_step", 4)
        
        # Additional behavior flags
        self.endless: bool = self.params.get("endless", True)


class WindowConfig(Configuration):
    """Window/Display configuration."""
    
    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)
        
        self.title: str = self.params.get("title", "Car Racer")
        self.fps: int | None = self.params.get("fps", 60)
        self.background_color: tuple[int, int, int] = tuple(self.params.get("background_color", (20, 20, 20)))
        self.font_size: int = self.params.get("font_size", 24)


class GenConfig(Configuration):
    """Track generation configuration."""
    
    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)
        
        # Generation parameters
        self.seed: int | None = self.params.get("seed", None)
        self.max_attempts: int = self.params.get("max_attempts", 1000)
        self.min_tiles: int = self.params.get("min_tiles", 0)


class PlayerConfig(Configuration):
    """Player initialization and behavior configuration."""
    
    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)
        
        self.lives: int = self.params.get("lives", 3)
        self.max_hiatus: int = self.params.get("max_hiatus", 100)
        self.max_frames: int = self.params.get("max_frames", 1000)


class EnvironmentConfig(Configuration):
    """Environment configuration for RacerGame (Gym-like environment)."""
    
    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)
        
        # Track layout
        self.grid_size: tuple[int, int] = tuple(self.params.get("grid_size", (8, 8)))
        self.cell_size: int = self.params.get("cell_size", 100)
        self.layout: list[str] | None = self.params.get("layout", None)
        
        # Track generation
        self.max_attempts: int = self.params.get("max_attempts", 1000)
        self.min_tiles: int = self.params.get("min_tiles", 0)
        self.seed: int | None = self.params.get("seed", None)
        
        # Gameplay
        self.lives: int = self.params.get("lives", 3)
        self.max_hiatus: int = self.params.get("max_hiatus", 100)
        self.max_frames: int = self.params.get("max_frames", 1000)
        self.min_laps: int = self.params.get("min_laps", 3)
        
        # Nested configurations
        self.tile: TileConfig = TileConfig(params=self.params.get("tile", {}))
        self.grid: GridConfig = GridConfig(params=self.params.get("grid", {}))
        self.car: CarConfig = CarConfig(params=self.params.get("car", {}))
        self.window: WindowConfig = WindowConfig(params=self.params.get("window", {}))
        self.generation: GenConfig = GenConfig(params=self.params.get("generation", {})) # NOTE: parameters used for track generation
        self.player: PlayerConfig = PlayerConfig(params=self.params.get("player", {})) # NOTE: parameters used to initalize Players class


def overwrite_config(config: Configuration, params: dict[str, Any]):
    for attr, value in params.items():
        if hasattr(config, attr):
            setattr(config, attr, value)
            