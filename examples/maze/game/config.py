import os
import json
import warnings

from typing import Any

from .constants import (
    COLOR_FLOOR, COLOR_WALL, COLOR_START, COLOR_GOAL, COLOR_TRAP, COLOR_BACKGROUND,
)


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

    def save(self, file: str) -> bool:
        # TODO: Implement
        pass


class TileConfig(Configuration):
    """Cell rendering configuration. Flat colored squares only — no rects with curves."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.floor_color: tuple[int, int, int]  = tuple(self.params.get("floor_color", COLOR_FLOOR))
        self.wall_color: tuple[int, int, int]   = tuple(self.params.get("wall_color", COLOR_WALL))
        self.start_color: tuple[int, int, int]  = tuple(self.params.get("start_color", COLOR_START))
        self.goal_color: tuple[int, int, int]   = tuple(self.params.get("goal_color", COLOR_GOAL))
        self.trap_color: tuple[int, int, int]   = tuple(self.params.get("trap_color", COLOR_TRAP))
        self.wall_thickness: int                = self.params.get("wall_thickness", 4)


class GridConfig(Configuration):
    """Grid configuration for maze layout. A simple Cols x Rows grid used internally as one."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.grid_size: tuple[int, int] = tuple(self.params.get("grid_size", (16, 16)))  # (cols, rows)
        self.cell_size: int = self.params.get("cell_size", 40)
        self.static: bool = self.params.get("static", False)

        self.tile: TileConfig = TileConfig(params=self.params.get("tile", {}))


class GenConfig(Configuration):
    """Maze generation configuration."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.seed: int | None           = self.params.get("seed", None)
        # Fraction of eligible dead-ends (excluding start/goal) converted into traps
        self.trap_ratio: float          = self.params.get("trap_ratio", 0.35)
        # Goal must land within this central fraction of the grid (0.34 -> middle third)
        self.goal_region_ratio: float   = self.params.get("goal_region_ratio", 0.34)
        # Extra passages carved after the perfect maze, to add loops/branches (0 = pure tree maze)
        self.braid_ratio: float         = self.params.get("braid_ratio", 0.08)
        # Number of reward checkpoints placed evenly along the shortest start->goal path
        self.checkpoints: int           = self.params.get("checkpoints", 10)


class AgentConfig(Configuration):
    """
    Agent (the 'dot' / genome) configuration.

    ── The 4 major modes ──────────────────────────────────────────────────
    Modes are NOT an integer switch anywhere in EnvironmentConfig. They are
    set here, directly, via two independent booleans that combine into 4
    modes:

        discrete_states   | discrete_actions  | meaning
        ------------------+--------------------+----------------------------
        True              | True               | tabular grid-world: int
                          |                    | cell/wall observation,
                          |                    | 1-of-4 discrete action
        True              | False              | int observation, continuous
                          |                    | 2D action vector (snapped
                          |                    | to a cardinal step)
        False             | True               | continuous [-1,1] features,
                          |                    | 1-of-4 discrete action
        False             | False              | continuous [-1,1] features,
                          |                    | continuous 2D action vector

    `discrete_states` is forwarded into GridConfig/Grid so that Grid knows
    which dtype/encoding to build its `return_grid` array in.
    `discrete_actions` is only consumed internally by Agents.update() when
    interpreting the raw action array.

    ── return_grid ─────────────────────────────────────────────────────────
    When True, Agents.get_state() ignores the compact feature vector and
    instead returns the ENTIRE grid array, once per genome/player (with that
    genome's own position baked in), for convolutional/embedding models:
      - discrete_states=False -> float32 array in [-1, 1] per cell/channel
      - discrete_states=True  -> int array of small integer codes per cell,
        suitable for nn.Embedding lookups.

    ── reset_on_death ──────────────────────────────────────────────────────
    When True (default), a genome that loses a life (hits a trap) restarts
    from the maze's start cell. When False, it keeps playing from wherever
    it died (still loses the life / lives budget).
    """

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)
        
        self.seq_len: int | None = self.params.get("seq_len", None)

        # ---- The 2 flags that combine into the 4 major modes --------------
        self.discrete_states: bool  = self.params.get("discrete_states", False)
        self.discrete_actions: bool = self.params.get("discrete_actions", False)

        # ---- New modes / behavior flags ------------------------------------
        self.return_grid: bool    = self.params.get("return_grid", False)
        self.reset_on_death: bool = self.params.get("reset_on_death", False)

        # ---- Continuous-action interpretation -------------------------------
        # Minimum magnitude (per axis, post dominant-axis selection) a continuous
        # action vector needs to clear before it triggers a discrete cell step.
        self.action_cutoff: float = self.params.get("action_cutoff", 0.35)

        # ---- Sensors (used for the continuous compact feature vector) -------
        # Ray-cast distance-to-wall readings, in the 8 principal directions,
        # measured in whole cells (this is a grid, so "sensors" are just
        # wall-distance counts, not pixel raycasts).
        self.sensor_count: int     = self.params.get("sensor_count", 8)
        self.sensor_max_cells: int = self.params.get("sensor_max_cells", 6)

        # ---- Behavior flags ---------------------------------------------------
        self.wall_bump_ends_episode: bool = self.params.get("wall_bump_ends_episode", False)


class WindowConfig(Configuration):
    """Window/Display configuration."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.title: str         = self.params.get("title", "Maze Runner")
        self.fps: int | None    = self.params.get("fps", 60)
        self.background_color: tuple[int, int, int] = tuple(self.params.get("background_color", COLOR_BACKGROUND))
        self.font_size: int     = self.params.get("font_size", 20)
        self.draw_path: bool    = self.params.get("draw_path", True)
        self.path_length: int   = self.params.get("path_length", 200)  # trail length, in cells
        self.blob_radius_ratio: float = self.params.get("blob_radius_ratio", 0.28)  # of cell_size


class RewardConfig(Configuration):
    """
    All fitness-shaping coefficients used by Players._update_fitness(). Nothing that
    affects the reward signal should be a bare numeric literal in player.py — it
    should be read from here, so reward shaping can be tuned/swept without touching code.
    """

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        # ── Potential-based progress reward ─────────────────────────────
        # Reward per cell of new-best distance improvement is scaled by (lives + 1).
        self.progress_multiplier: float = self.params.get("progress_multiplier", 5.0)

        # ── Per-step penalty (encourages speed) ─────────────────────────
        self.step_penalty: float = self.params.get("step_penalty", 0.2)

        # ── Wall-bump penalty ────────────────────────────────────────────
        self.wall_bump_penalty: float = self.params.get("wall_bump_penalty", 1.0)

        # ── Stagnation (hiatus limit reached) ───────────────────────────
        self.stagnation_penalty: float = self.params.get("stagnation_penalty", 50.0)

        # ── Trap penalty, scaled by (deaths + 1) ────────────────────────
        self.trap_penalty: float = self.params.get("trap_penalty", 100.0)

        # ── Goal reward ──────────────────────────────────────────────────
        # Base + (time_bonus * weight), both multiplied by (lives + 1).
        self.goal_base_reward: float = self.params.get("goal_base_reward", 1000.0)
        self.goal_time_bonus_weight: float = self.params.get("goal_time_bonus_weight", 1000.0)
        # frames_done / goal_time_bonus_frame_norm sets how fast the time bonus decays
        self.goal_time_bonus_frame_norm: float = self.params.get("goal_time_bonus_frame_norm", 1000.0)
        self.goal_time_bonus_min: float = self.params.get("goal_time_bonus_min", 0.1)
        self.goal_time_bonus_max: float = self.params.get("goal_time_bonus_max", 1.0)

        # ── Checkpoint reward ────────────────────────────────────────────
        # Given once per genome, the first time it touches each checkpoint cell
        # along the best path. Scaled by (lives + 1), like the progress reward.
        self.checkpoint_reward: float = self.params.get("checkpoint_reward", 150.0)

        # ── Goal-hold reward ─────────────────────────────────────────────
        # A player that already reached the goal stops moving, but the episode
        # may keep running (see EnvironmentConfig.first_winners) until enough
        # other players finish too. Every step, every step_penalty/etc is
        # skipped for that player and this constant reward is added instead,
        # so genomes that finish EARLIER don't end up with a lower return than
        # ones that finish later just because they racked up more idle
        # step-penalty frames while waiting for the episode to end.
        self.goal_hold_reward: float = self.params.get("goal_hold_reward", 5.0)

        # ── Score bookkeeping (separate from fitness; see Players.scores) ──
        self.score_goal: int = self.params.get("score_goal", 10)
        self.score_trap: int = self.params.get("score_trap", -1)


class PlayerConfig(Configuration):
    """Player bookkeeping/reward configuration."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        self.lives: int = self.params.get("lives", 3)
        self.max_hiatus: int = self.params.get("max_hiatus", 50)   # steps w/o progress toward goal
        self.max_frames: int = self.params.get("max_frames", 1000)  # steps == frames (1 step = 1 cell decision)

        self.reward: RewardConfig = RewardConfig(params=self.params.get("reward", {}))


class EnvironmentConfig(Configuration):
    """Environment configuration for MazeGame (Gym-like environment). No mode integer here."""

    def __init__(self, *, params: dict[str, Any] = None, file: str = None):
        super().__init__(params=params, file=file)

        # # Layout — arbitrary grid size
        # self.grid_size: tuple[int, int] = tuple(self.params.get("grid_size", (10, 10)))
        # self.cell_size: int     = self.params.get("cell_size", 32)

        # Generation
        self.seed: int | None   = self.params.get("seed", None)
        self.trap_ratio: float  = self.params.get("trap_ratio", 0.35)
        self.goal_region_ratio: float = self.params.get("goal_region_ratio", 0.34)
        self.braid_ratio: float = self.params.get("braid_ratio", 0.15)
        self.checkpoints: int   = self.params.get("checkpoints", 10)

        # Gameplay
        self.lives: int         = self.params.get("lives", 5)
        self.max_hiatus: int    = self.params.get("max_hiatus", 20)
        self.max_frames: int    = self.params.get("max_frames", 500)
        # Episode doesn't terminate the instant ONE player reaches the goal —
        # it keeps running until `first_winners` players have reached it (or
        # everyone is disqualified/timed out). A finished player stops moving
        # (see Players.active) but keeps receiving `reward.goal_hold_reward`
        # each remaining step.
        self.first_winners: int = self.params.get("first_winners", 10)

        # Nested configurations
        self.tile: TileConfig   = TileConfig(params=self.params.get("tile", {}))
        self.grid: GridConfig   = GridConfig(params=self.params.get("grid", {}))
        self.generation: GenConfig = GenConfig(params=self.params.get("generation", {}))
        self.player: PlayerConfig = PlayerConfig(params=self.params.get("player", {}))
        self.window: WindowConfig = WindowConfig(params=self.params.get("window", {}))
        self.agent: AgentConfig = AgentConfig(params=self.params.get("agent", {}))


def overwrite_config(config: Configuration, params: dict[str, Any]):
    for attr, value in params.items():
        if hasattr(config, attr):
            setattr(config, attr, value)
