from __future__ import annotations

import os
import json
import warnings

from typing import Any

__all__ = [
    "Configuration",
    "GridConfig",
    "SnakeConfig",
    "RewardConfig",
    "WindowConfig",
    "GenConfig",
    "PlayerConfig",
    "EnvironmentConfig",
    "overwrite_config",
]


class Configuration:
    """Base configuration class for loading parameters from file or dict."""

    def __init__(self, *, params: dict[str, Any] | None = None, file: str = None) -> None:
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
        if not os.path.isfile(file):
            raise FileNotFoundError(f"{config_type} parameter file not found: {file}")

        with open(file, 'r') as f:
            print(f"Loaded {config_type} parameters from file: {file}")
            return json.load(f)

    def to_dict(self) -> dict[str, Any]:
        """Current settings as a JSON-serializable dict; nested configs become nested dicts."""
        out: dict[str, Any] = {}
        for key, value in vars(self).items():
            if key == "params" or key.startswith("_"):
                continue
            if isinstance(value, Configuration):
                out[key] = value.to_dict()
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                continue
            out[key] = value
        return out

    def save(self, file: str) -> bool:
        """Write the current settings to a JSON file. Returns True on success."""
        directory = os.path.dirname(os.path.abspath(file))
        os.makedirs(directory, exist_ok=True)
        with open(file, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"Saved parameters to file: {file}")
        return True


class GridConfig(Configuration):
    """Board configuration."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        # Playable board as (columns, rows) of cells. Normally set through the top-level grid_size.
        self.grid_size: tuple[int, int] = tuple(self.params.get("grid_size", (10, 10)))
        # Optional mode: crossing a boundary makes the snake appear on the opposite side instead of dying.
        # Normally set through the top-level wrap.
        self.wrap: bool = bool(self.params.get("wrap", False))
        # Frames after which the food jumps to a new cell when not eaten (null = food stays put).
        self.food_timeout: int | None = self.params.get("food_timeout", None)


class SnakeConfig(Configuration):
    """Snake spawn, movement and sensor configuration."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        # Starting direction: 0 right, 1 up, 2 left, 3 down, null = random on every (re)spawn
        self.init_direction: int | None = self.params.get("init_direction", 1)
        # Starting length in cells (head included)
        self.init_length: int = max(1, int(self.params.get("init_length", 3)))
        # An action the snake uses less than this share of its most used action counts as "low action
        # usage" (spiralling / looping). null disables the check.
        self.var_thresh: float | None = self.params.get("var_thresh", 0.05)
        # State layout: 'vector' (feature vector per snake) or 'grid' (normalized board, 1 channel)
        self.state_type: str = self.params.get("state_type", "vector")
        # Sensor ray length in cells (null = longest board side)
        self.ray_length: int | None = self.params.get("ray_length", None)


class RewardConfig(Configuration):
    """Reward shaping weights. Per-frame fitness lives in player.py, the shaping terms in reward.py."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        # Dense shaping on/off. Off = only food, death and completion events
        self.shaping: bool = bool(self.params.get("shaping", True))
        # Event rewards (scaled by lives left / deaths so far, see player.py)
        self.food_reward: float = self.params.get("food_reward", 200.0)
        self.death_penalty: float = self.params.get("death_penalty", 50.0)
        self.complete_reward: float = self.params.get("complete_reward", 100000.0)
        # Dense terms (shaping only)
        self.closer_weight: float = self.params.get("closer_weight", 1.0)      # moved closer to the food
        self.away_weight: float = self.params.get("away_weight", 1.1)          # moved away from the food
        self.time_penalty: float = self.params.get("time_penalty", 0.01)       # flat cost per frame
        self.hiatus_weight: float = self.params.get("hiatus_weight", 0.5)      # cost growing with frames since food
        self.loop_weight: float = self.params.get("loop_weight", 0.01)         # low action usage (spiralling)
        # Added to every living snake's reward each frame, and multiplier on the reward the learner sees
        self.reward_offset: float = self.params.get("reward_offset", 0.0)
        self.reward_scale: float = self.params.get("reward_scale", 0.01)


class WindowConfig(Configuration):
    """Display configuration."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.title: str = self.params.get("title", "Snake Environment")
        self.cell_size: int = int(self.params.get("cell_size", 24))        # pixels per board cell
        self.margin: int = int(self.params.get("margin", 1))               # cells of border drawn around the board
        self.fps: int | None = self.params.get("fps", 30)                  # frame cap, null = uncapped
        self.precise_fps: bool = bool(self.params.get("precise_fps", True))
        self.font_size: int = int(self.params.get("font_size", 24))
        self.background_color: tuple[int, int, int] = tuple(self.params.get("background_color", (10, 12, 16)))
        self.grid_lines: bool = bool(self.params.get("grid_lines", True))  # checkerboard shading of the board
        self.verbose_level: int = int(self.params.get("verbose_level", 0)) # 0 off, 1 sensor rays, 2 + debug panel
        self.show_controls_hint: bool = bool(self.params.get("show_controls_hint", True))
        # Optional retro look: scene drawn on a small canvas and scaled up with nearest-neighbour
        self.pixel: bool = bool(self.params.get("pixel", False))
        self.pixel_size: int = int(self.params.get("pixel_size", 3))       # screen pixels per canvas pixel
        self.pixel_color_bits: int = int(self.params.get("pixel_color_bits", 3))  # bits per channel (8 = off)


class GenConfig(Configuration):
    """Randomness configuration."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.seed: int | None = self.params.get("seed", None)   # null = different every run


class PlayerConfig(Configuration):
    """Player initialization and behavior configuration."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.lives: int = self.params.get("lives", 3)              # respawns a snake gets before it is out
        self.max_hiatus: int = self.params.get("max_hiatus", 100)  # frames without eating before a life is lost
        self.max_frames: int = self.params.get("max_frames", 1000) # episode length in frames


class EnvironmentConfig(Configuration):
    """
    Settings for the whole environment (`Game`). Every key is optional: a missing key uses the default
    shown here. Load from a JSON file with `EnvironmentConfig(file=...)` or pass `params={...}`.
    Top-level keys below can also be overridden as keyword arguments of `Game(...)`.

    Quick guide to what people usually change:
      * Bigger or smaller board ........ grid_size
      * Cross boundaries ............... wrap = true
      * Longer or shorter episodes ..... max_frames
      * More forgiving or stricter ..... lives, max_hiatus
      * Same run every time ............ seed
      * Retro look ..................... window.pixel = true

    Nested sections (each is a JSON object with the same name): grid, snake, window, generation,
    player, reward.
    """

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        # ---- Nested sections (built first so that a value given only inside a section is not lost) ----
        self.grid: GridConfig = GridConfig(params=self.params.get("grid", {}))
        self.snake: SnakeConfig = SnakeConfig(params=self.params.get("snake", {}))
        self.window: WindowConfig = WindowConfig(params=self.params.get("window", {}))
        self.generation: GenConfig = GenConfig(params=self.params.get("generation", {}))
        self.player: PlayerConfig = PlayerConfig(params=self.params.get("player", {}))
        self.reward: RewardConfig = RewardConfig(params=self.params.get("reward", {}))

        # ---- Board ----
        # A top-level key wins over the same key inside its section; otherwise the section value is used.
        self.grid_size: tuple[int, int] = tuple(self.params.get("grid_size", self.grid.grid_size))
        # Boundary crossing: false = walls kill, true = leave one side and appear on the opposite one
        self.wrap: bool = bool(self.params.get("wrap", self.grid.wrap))
        self.seed: int | None = self.params.get("seed", self.generation.seed)

        # ---- Episode rules ----
        self.lives: int = self.params.get("lives", self.player.lives)
        self.max_hiatus: int = self.params.get("max_hiatus", self.player.max_hiatus)
        self.max_frames: int = self.params.get("max_frames", self.player.max_frames)
        # Multiplier applied to the state returned by reset() and step()
        self.state_scale: float = self.params.get("state_scale", 1.0)


def overwrite_config(config: Configuration, params: dict[str, Any]):
    for attr, value in params.items():
        if attr != "params" and hasattr(config, attr):
            setattr(config, attr, value)