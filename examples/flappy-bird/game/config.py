from __future__ import annotations

import os
import json
import warnings

from typing import Any

__all__ = [
    "Configuration",
    "WorldConfig",
    "BirdConfig",
    "RewardConfig",
    "WindowConfig",
    "DraftConfig",
    "GenConfig",
    "PlayerConfig",
    "EnvironmentConfig",
    "overwrite_config",
    "normalize_hiatus",
    "normalize_seq_len",
]


def _pair(value: Any, default: Any) -> tuple:
    """int | (a, b) -> (a, b)."""
    if value is None:
        value = default
    if isinstance(value, (list, tuple)):
        return (value[0], value[0]) if len(value) == 1 else (value[0], value[1])
    return (value, value)


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


class WorldConfig(Configuration):
    """World geometry: screen size, floor, scrolling and pipe generation. Units are world pixels."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.width: int = int(self.params.get("width", 800))                # world width
        self.height: int = int(self.params.get("height", 650))              # world height (floor included)
        self.floor: int = int(self.params.get("floor", 100))                # floor height from the bottom
        self.velocity: float = float(self.params.get("velocity", 6.0))      # scroll speed, pixels per frame
        self.spawn_width: int = int(self.params.get("spawn_width", 200))    # horizontal space between two pipes
        self.pipe_width: int = int(self.params.get("pipe_width", 100))
        # Margins between the ceiling / floor and the highest / lowest gap edge: int or (lower, upper)
        self.gap_offset: tuple[int, int] = _pair(self.params.get("gap_offset"), 100)
        # Gap size drawn per pipe from this range: int or (min, max)
        self.gap_size: tuple[int, int] = _pair(self.params.get("gap_size"), 200)
        # Vertical pipe movement: speed in pixels per frame (0 = pipes never move), share of moving pipes
        # Master switch for vertical pipe movement (False = every pipe stays put, whatever pipe_speed_y says)
        self.pipe_movement: bool = bool(self.params.get("pipe_movement", True))
        self.pipe_speed_y: float = float(self.params.get("pipe_speed_y", 3.0))
        self.moving_chance: float = float(self.params.get("moving_chance", 0.5))
        # x of the first pipe's left edge (null = 3 * bird init_x + spawn_width, like the original)
        self.first_pipe_x: float | None = self.params.get("first_pipe_x", None)


class BirdConfig(Configuration):
    """Bird spawn, flight physics, hitbox and sensor configuration."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.init_x: float = float(self.params.get("init_x", 100.0))
        self.init_y: float = float(self.params.get("init_y", 300.0))        # top edge of the bird sprite
        self.bird_size: tuple[int, int] = tuple(self.params.get("bird_size", (60, 40)))   # (w, h)
        # Hitbox = ellipse inscribed in the sprite box, scaled by this factor (1 = full box)
        self.hitbox_scale: float = float(self.params.get("hitbox_scale", 0.85))
        # Flight, per frame. After a flap the displacement is vel * t + fall_coeff * t^2 (t = frames since the flap)
        self.fall_coeff: float = float(self.params.get("fall_coeff", 1.75))
        self.terminal_vel: float = float(self.params.get("terminal_vel", 20.0))
        self.jump_vel: float = float(self.params.get("jump_vel", 10.0))     # initial upward velocity of a flap
        self.jump_offset: float = float(self.params.get("jump_offset", 7.5))  # extra pixels while rising
        self.max_rotation: float = float(self.params.get("max_rotation", 25.0))
        self.ang_vel: float = float(self.params.get("ang_vel", 30.0))
        self.anime_time: int = int(self.params.get("anime_time", 4))        # frames per wing animation step
        # A pipe counts as passed when the bird is this share of the pipe width beyond the pipe's left edge
        self.pass_threshold: float = float(self.params.get("pass_threshold", 0.5))
        # Continuous actions flap when the score is above this (tanh outputs: 0 = even odds)
        self.action_threshold: float = float(self.params.get("action_threshold", 0.5))
        # A flap / no-flap share below this fraction of the most used one counts as "low action usage"
        # (never flapping or spamming). null disables the check.
        self.var_thresh: float | None = self.params.get("var_thresh", 0.05)
        # State layout: 'basic' (the original 3: y and the gap edges), 'full' (the original 5, or 7 when pipes move
        # vertically), 'ego' (10 features relative to the next gap) or 'auto' (= 'full')
        self.state_type: str = self.params.get("state_type", "auto")


class RewardConfig(Configuration):
    """
    Reward shaping. Per-frame fitness lives in player.py, the shaping terms in reward.py.

    `mode` picks what the dense shaping imitates:

      basic   alignment shaping only: moving the bird closer to / away from the centre of the next gap, a flat
              time cost, a cost growing with frames since the last pipe, a cost for low flap usage.
      gap     basic + the safe corridor: staying between the gap edges while a pipe is near is credited, leaving
              them is charged, hugging the ceiling / floor and wasted flaps cost a little.
      staged  basic while the bird is still learning, gap once `skill` passes `staged_switch`.

    Both use a skill curriculum (like the car racer): survive first (living bonus, heavy crash cost), then be
    precise (alignment and corridor terms count more as `skill` rises).
    """

    MODES = ("basic", "gap", "staged")

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.mode: str = str(self.params.get("mode", "basic")).lower()
        if self.mode not in self.MODES:
            raise ValueError(f"reward mode must be one of {self.MODES}, got {self.mode!r}")

        # Dense shaping on/off. Off = only pipe, death and completion events
        self.shaping: bool = bool(self.params.get("shaping", False))
        # Event rewards (scaled by lives left / deaths so far, see player.py)
        self.pass_reward: float = self.params.get("pass_reward", 200.0)
        self.death_penalty: float = self.params.get("death_penalty", 50.0)
        self.complete_reward: float = self.params.get("complete_reward", 10000.0)
        # Dense terms shared by every mode (distances in units of `distance_unit` pixels)
        self.distance_unit: float = self.params.get("distance_unit", 10.0)
        self.closer_weight: float = self.params.get("closer_weight", 0.5)       # moved closer to the gap centre
        self.away_weight: float = self.params.get("away_weight", 0.55)          # moved away from it
        self.time_penalty: float = self.params.get("time_penalty", 0.01)        # flat cost per frame
        self.hiatus_weight: float = self.params.get("hiatus_weight", 0.5)       # cost growing with frames since a pipe
        self.loop_weight: float = self.params.get("loop_weight", 0.05)          # low flap usage
        self.alive_weight: float = self.params.get("alive_weight", 1.0)         # per living frame x (1 - skill)
        # Dying far from the gap costs more: this x (distance to the gap centre in units), capped by slow_cap
        self.death_dist_weight: float = self.params.get("death_dist_weight", 1.0)
        self.slow_cap: float = self.params.get("slow_cap", 30.0)
        # Added to every living bird's reward each frame, and multiplier on the reward the learner sees
        self.reward_offset: float = self.params.get("reward_offset", 0.0)
        self.reward_scale: float = self.params.get("reward_scale", 0.01)

        # ---- Gap mode ----
        self.approach_dist: float = self.params.get("approach_dist", 400.0)     # pixels in front of a pipe where the corridor counts
        self.corridor_margin: float = self.params.get("corridor_margin", 12.0)  # pixels kept clear of the gap edges
        self.corridor_weight: float = self.params.get("corridor_weight", 0.5)   # per frame inside the corridor
        self.outside_weight: float = self.params.get("outside_weight", 0.3)     # per unit outside it
        self.edge_weight: float = self.params.get("edge_weight", 1.0)           # hugging the ceiling / floor
        self.edge_frac: float = self.params.get("edge_frac", 0.08)              # share of the height that counts as close
        self.flap_weight: float = self.params.get("flap_weight", 0.3)           # a flap while already above the corridor
        # `staged`: skill above which the corridor terms are switched on
        self.staged_switch: float = self.params.get("staged_switch", 0.5)
        # The hiatus must allow one pipe: raise max_hiatus to at least this many frames per pipe spacing
        self.auto_hiatus: bool = bool(self.params.get("auto_hiatus", True))

        # ---- Skill curriculum, graded from the finished episode on every reset ----
        self.curriculum: bool = bool(self.params.get("curriculum", True))
        self.skill_start: float = self.params.get("skill_start", 0.0)
        self.skill_rate: float = self.params.get("skill_rate", 0.15)            # how fast the skill follows the quality
        self.skill_pipes_per_death: float = self.params.get("skill_pipes_per_death", 3.0)  # pipes one death is worth
        self.skill_lo: float = self.params.get("skill_lo", 0.3)                 # quality at which skill starts to grow
        self.skill_hi: float = self.params.get("skill_hi", 0.9)                 # quality at which skill is full
        # Death costs are multiplied by 1 + early_crash_mult * (1 - skill)
        self.early_crash_mult: float = self.params.get("early_crash_mult", 3.0)


class WindowConfig(Configuration):
    """Display configuration."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.title: str = self.params.get("title", "Flappy Bird Environment")
        self.scale: float = float(self.params.get("scale", 1.0))           # screen pixels per world pixel
        self.fps: int | None = self.params.get("fps", 30)                  # frame cap, null = uncapped
        self.precise_fps: bool = bool(self.params.get("precise_fps", True))
        self.font_size: int = int(self.params.get("font_size", 24))
        self.background_color: tuple[int, int, int] = tuple(self.params.get("background_color", (10, 12, 16)))
        self.verbose_level: int = int(self.params.get("verbose_level", 0))  # 0 off, 1 gap lines + hitbox, 2 + debug panel
        self.show_controls_hint: bool = bool(self.params.get("show_controls_hint", True))
        self.halos: int = int(self.params.get("halos", 3))                 # rank halos under the top birds
        self.max_drawn: int | None = self.params.get("max_drawn", None)    # draw only the best N birds (null = all)
        # Optional retro look: scene drawn on a small canvas and scaled up with nearest-neighbour
        self.pixel: bool = bool(self.params.get("pixel", True))
        self.pixel_size: int = int(self.params.get("pixel_size", 3))       # screen pixels per canvas pixel
        self.pixel_color_bits: int = int(self.params.get("pixel_color_bits", 4))  # bits per channel (8 = off)


class DraftConfig(Configuration):
    """
    Drafts: random gusts of wind that change the birds' displacement. At most ONE draft is blowing (or about to
    blow) at any time, and it is either an x draft or a y draft.

      x draft  forward: the birds cover more ground per frame (the world scrolls faster); backward: less.
      y draft  up: flaps lift more and falls are slower; down: flaps are weaker and falls are faster.

    Every draft is announced `draft_warning` frames before it takes effect (wind streaks and an arrow are drawn),
    so a human player can time the jump. Strengths are fractions: 0.3 = 30 % more / less displacement.
    """

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.draft_x: bool = bool(self.params.get("draft_x", True))                # allow x drafts
        self.draft_y: bool = bool(self.params.get("draft_y", True))                # allow y drafts
        self.draft_x_strength: float = float(self.params.get("draft_x_strength", 0.35))
        self.draft_y_strength: float = float(self.params.get("draft_y_strength", 0.30))
        # Chance per frame that a draft is announced once the cooldown is over
        self.draft_chance: float = float(self.params.get("draft_chance", 0.01))
        self.draft_cooldown: int = int(self.params.get("draft_cooldown", 90))       # calm frames after a draft (and at the start)
        self.draft_duration: tuple[int, int] = _pair(self.params.get("draft_duration"), (60, 150))  # frames: int or (min, max)
        self.draft_warning: int = int(self.params.get("draft_warning", 40))         # frames of warning before it blows

    @property
    def enabled(self) -> bool:
        return (self.draft_x or self.draft_y) and self.draft_chance > 0.0


class GenConfig(Configuration):
    """Randomness configuration."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.seed: int | None = self.params.get("seed", None)   # null = different every run


class PlayerConfig(Configuration):
    """Player initialization and behavior configuration."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.lives: int = self.params.get("lives", 10)               # respawns a bird gets before it is out
        # Frames without passing a pipe before a life is lost, null = never
        self.max_hiatus: int | None = normalize_hiatus(self.params.get("max_hiatus", 300))
        self.max_frames: int = self.params.get("max_frames", 2000)  # episode length in frames
        # Pipes a bird has to pass in one life to complete the episode, null = endless
        self.goal: int | None = self.params.get("goal", 20)


class EnvironmentConfig(Configuration):
    """
    Settings for the whole environment (`Game`). Every key is optional: a missing key uses the default
    shown here. Load from a JSON file with `EnvironmentConfig(file=...)` or pass `params={...}`.
    Any key of any section can also be overridden as a keyword argument of `Game(...)`.

    Quick guide to what people usually change:
      * Bigger or smaller world ........ width, height
      * Harder pipes ................... gap_size, spawn_width, velocity, pipe_speed_y
      * Longer or shorter episodes ..... max_frames, goal (null = endless)
      * More forgiving or stricter ..... lives, max_hiatus (null = no limit)
      * Same run every time ............ seed
      * Retro look ..................... window.pixel = true

    Nested sections (each is a JSON object with the same name): world, bird, window, generation, player, reward, draft.
    """

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        # ---- Nested sections (built first so that a value given only inside a section is not lost) ----
        self.world: WorldConfig = WorldConfig(params=self.params.get("world", {}))
        self.bird: BirdConfig = BirdConfig(params=self.params.get("bird", {}))
        self.window: WindowConfig = WindowConfig(params=self.params.get("window", {}))
        self.generation: GenConfig = GenConfig(params=self.params.get("generation", {}))
        self.player: PlayerConfig = PlayerConfig(params=self.params.get("player", {}))
        self.reward: RewardConfig = RewardConfig(params=self.params.get("reward", {}))
        self.draft: DraftConfig = DraftConfig(params=self.params.get("draft", {}))

        # ---- Shared keys: a top-level key wins over the same key inside its section ----
        self.seed: int | None = self.params.get("seed", self.generation.seed)
        self.lives: int = self.params.get("lives", self.player.lives)
        self.max_hiatus: int | None = normalize_hiatus(self.params.get("max_hiatus", self.player.max_hiatus))
        self.max_frames: int = self.params.get("max_frames", self.player.max_frames)
        self.goal: int | None = self.params.get("goal", self.player.goal)
        # Multiplier applied to the state returned by reset() and step()
        self.state_scale: float = self.params.get("state_scale", 1.0)
        # State layout: None / 0 -> instance states (N, features); an integer -> sequential states (N, seq_len, features),
        # oldest frame first, newest last (the history of a bird is refilled with its first state after a respawn)
        self.seq_len: int | None = normalize_seq_len(self.params.get("seq_len", None))


def normalize_hiatus(value: Any) -> int | None:
    """Frames without a pipe before a life is lost. None, 0, a negative number or infinity mean no limit (None)."""
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
            if attr in ("gap_offset", "gap_size", "draft_duration"):
                value = _pair(value, None)
            setattr(config, attr, value)


def normalize_seq_len(value: Any) -> int | None:
    """Sequence length of the states. None, 0 or a negative number mean instance states (None)."""
    if value is None or isinstance(value, bool):
        return None
    value = int(value)
    return value if value > 0 else None