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
    "normalize_hiatus",
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
        # State layout: 'ego' (egocentric: food / tail bearing + distance, 3 rays, length, optionally the road offsets;
        # no feature judges a move), 'vector' (12 compass-dependent features), 'strategy' (20 features that include the
        # road solver's verdict on every move - the oracle, kept only for comparison), 'grid' (the board as a 3-channel
        # image: body age, food, next-move cells; no oracle) or 'auto' (= 'ego')
        self.state_type: str = self.params.get("state_type", "auto")
        # 'ego' state only: append the road geometry (road offset of the three neighbour cells, the food and the tail,
        # 13 inputs). False = pure sensing, 8 inputs (bearing / distance of food and tail, 3 rays, length).
        self.ego_road: bool = self.params.get("ego_road", True)
        # Sensor ray length in cells (null = longest board side)
        self.ray_length: int | None = self.params.get("ray_length", None)


class RewardConfig(Configuration):
    """
    Reward shaping. Per-frame fitness lives in player.py, the shaping terms in reward.py.

    `mode` picks what the dense shaping imitates:

      basic      food-distance shaping only (closer / away / time / hiatus / looping). The original reward.
      dhcr       "Dynamic Hamiltonian Cycle Repair, tier 1". The board has a one-way ring road through every
                 cell. A snake whose body stays in road order can never trap itself, and it may hop to a free
                 neighbour that is further along the road but not past the food (a shortcut). The reward
                 grades every move as road / shortcut / slow / unsafe from exactly that rule.
      cell_tree  The ring road is the outline of a spanning tree over 2x2 cells, and a move off the road is
                 safe only if a tree swap makes it a road link while the body stays one piece. Legal swaps
                 are really applied, so the road bends around the snake, like the test solver.
                 Needs an even width and height (>= 4).
      staged     dhcr while the snake is still learning, cell_tree once `skill` passes `staged_switch`.

    Road modes use a skill curriculum (like the car racer): survive first (stay in road order, never box
    yourself in, heavy crash cost), then go fast (shortcuts are credited more and more as `skill` rises).
    A board a road mode cannot use (odd cells, or odd sides for cell_tree) falls back to `basic` with a warning.
    """

    MODES = ("basic", "dhcr", "cell_tree", "staged")

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.mode: str = str(self.params.get("mode", "staged")).lower()
        if self.mode not in self.MODES:
            raise ValueError(f"reward mode must be one of {self.MODES}, got {self.mode!r}")

        # Dense shaping on/off. Off = only food, death and completion events
        self.shaping: bool = bool(self.params.get("shaping", True))
        # Event rewards (scaled by lives left / deaths so far, see player.py)
        self.food_reward: float = self.params.get("food_reward", 200.0)
        self.death_penalty: float = self.params.get("death_penalty", 50.0)
        self.complete_reward: float = self.params.get("complete_reward", 100000.0)
        # Dense terms shared by every mode
        self.closer_weight: float = self.params.get("closer_weight", 1.0)      # moved closer to the food
        self.away_weight: float = self.params.get("away_weight", 1.1)          # moved away from the food
        self.time_penalty: float = self.params.get("time_penalty", 0.01)       # flat cost per frame
        self.hiatus_weight: float = self.params.get("hiatus_weight", 0.5)      # cost growing with frames since food
        self.loop_weight: float = self.params.get("loop_weight", 0.01)         # low action usage (spiralling)
        # Added to every living snake's reward each frame, and multiplier on the reward the learner sees
        self.reward_offset: float = self.params.get("reward_offset", 0.0)
        self.reward_scale: float = self.params.get("reward_scale", 0.01)

        # ---- Road modes (dhcr / cell_tree / staged) ----
        # Per safe frame on the road (survive-first). Fades out as the skill rises.
        self.road_weight: float = self.params.get("road_weight", 1.0)
        # Credit per cell of road distance to the food removed this frame (a plain road step removes 1)
        self.progress_weight: float = self.params.get("progress_weight", 2.0)
        # A shortcut removes more than 1 cell: the extra cells are credited at this multiple x skill
        self.shortcut_mult: float = self.params.get("shortcut_mult", 1.5)
        # Largest road distance a single bad move is charged for (cells). Overshooting the food costs a lap.
        self.slow_cap: float = self.params.get("slow_cap", 20.0)
        # One-off cost of the move that breaks road order (cell_tree: an illegal swap, multiplied by the crash
        # scale, see below), and a mild per-frame cost while the body is out of road order.
        self.unsafe_weight: float = self.params.get("unsafe_weight", 30.0)
        self.desync_weight: float = self.params.get("desync_weight", 0.3)      # per frame, NOT crash-scaled
        # Share of the unsafe cost paid back when the body is in road order again (road order = a potential)
        self.resync_refund: float = self.params.get("resync_refund", 0.9)
        # Per living frame, x (1 - skill), fading with the frames since the last meal. Keeps living worth more than
        # dying while the snake has no skill (see kernels/reward.py).
        self.alive_weight: float = self.params.get("alive_weight", 1.0)
        # Cost of a move into a pocket the snake cannot get out of (flood fill: less room than its length and
        # no way to the tail). Also applies to `basic` when the strategy state is on.
        self.trap_weight: float = self.params.get("trap_weight", 10.0)
        # Driving style of the dhcr "mods": NASCAR = cost for reversing the turn direction (keeps the free
        # space in big pieces), apple-fear = cost for swinging off the road into the food on the last step.
        self.nascar_weight: float = self.params.get("nascar_weight", 0.3)
        self.fear_weight: float = self.params.get("fear_weight", 3.0)
        self.fear_min_frac: float = self.params.get("fear_min_frac", 0.25)   # board share filled before it counts
        # dhcr hops are credited up to this board share (the original solver switches to repair after it)
        self.shortcut_max_frac: float = self.params.get("shortcut_max_frac", 0.5)
        # cell_tree: also try neighbours that are not closer to the food (slower, rarely useful)
        self.tree_any_direction: bool = bool(self.params.get("tree_any_direction", False))
        # New random spanning tree and a random start on the road for every (re)spawn (false = comb tree, fixed start)
        self.random_road: bool = bool(self.params.get("random_road", True))
        # `staged`: skill above which the road is switched from dhcr to cell_tree
        self.staged_switch: float = self.params.get("staged_switch", 0.6)
        # The road needs up to cells-1 moves per meal: raise max_hiatus to at least the cell count
        self.auto_hiatus: bool = bool(self.params.get("auto_hiatus", True))

        # ---- Skill curriculum (road modes), graded from the finished episode on every reset ----
        self.curriculum: bool = bool(self.params.get("curriculum", True))
        self.skill_start: float = self.params.get("skill_start", 0.0)
        self.skill_rate: float = self.params.get("skill_rate", 0.15)           # how fast the skill follows the quality
        self.skill_food_per_death: float = self.params.get("skill_food_per_death", 3.0)  # meals one death is worth
        self.skill_lo: float = self.params.get("skill_lo", 0.3)                # quality at which skill starts to grow
        self.skill_hi: float = self.params.get("skill_hi", 0.9)                # quality at which skill is full
        # Death, unsafe and trap costs are multiplied by 1 + early_crash_mult * (1 - skill)
        self.early_crash_mult: float = self.params.get("early_crash_mult", 3.0)


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
        self.max_hiatus: int | None = normalize_hiatus(self.params.get("max_hiatus", 100))  # frames without eating before a life is lost, null = never
        self.max_frames: int = self.params.get("max_frames", 2000) # episode length in frames


class EnvironmentConfig(Configuration):
    """
    Settings for the whole environment (`Game`). Every key is optional: a missing key uses the default
    shown here. Load from a JSON file with `EnvironmentConfig(file=...)` or pass `params={...}`.
    Top-level keys below can also be overridden as keyword arguments of `Game(...)`.

    Quick guide to what people usually change:
      * Bigger or smaller board ........ grid_size
      * Cross boundaries ............... wrap = true
      * Longer or shorter episodes ..... max_frames
      * More forgiving or stricter ..... lives, max_hiatus (null = no starvation limit)
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
        self.max_hiatus: int | None = normalize_hiatus(self.params.get("max_hiatus", self.player.max_hiatus))
        self.max_frames: int = self.params.get("max_frames", self.player.max_frames)
        # Multiplier applied to the state returned by reset() and step()
        self.state_scale: float = self.params.get("state_scale", 1.0)


def normalize_hiatus(value: Any) -> int | None:
    """Frames without food before a life is lost. None, 0, a negative number or infinity mean no limit (None)."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    if value == float("inf") or value <= 0:
        return None
    return int(value)


def overwrite_config(config: Configuration, params: dict[str, Any]):
    for attr, value in params.items():
        if attr != "params" and hasattr(config, attr):
            setattr(config, attr, value)