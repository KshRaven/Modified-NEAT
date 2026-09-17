import numpy as np
import pygame

from gymnasium import Env, spaces
from gymnasium.core import RenderFrame
from numpy import ndarray as Array
from typing import Any

try:
    from torch import Tensor
except ImportError:  # pragma: no cover - torch optional
    Tensor = None

from .constants import NP_INT, NP_FLOAT
from .config import EnvironmentConfig, overwrite_config
from .grid import Grid
from .player import Players
from .agent import Agents
from .window import Window
from .functional import Direction


class Game(Env):
    """
    Maze-Runner Gymnasium environment. Same package layout/usage pattern as the
    Car-Racer environment (`.reset()`, `.step()`, `.render()`, `.run()`), but the
    underlying world is a plain Cols x Rows grid: every genome/player occupies a
    single (row, col) cell and moves exactly one cell per step.

    The 4 modes + return_grid + reset_on_death live ONLY in `config.agent`
    (AgentConfig) — there is no integer `mode` anywhere in EnvironmentConfig.
    """

    def __init__(
        self,
        render_mode: str | None = "human",
        config: EnvironmentConfig | None = None,
        **params: dict[str, Any],
    ):
        if config is None:
            config = EnvironmentConfig()

        overwrite_config(config, params)
        overwrite_config(config.grid, vars(config))
        overwrite_config(config.player, vars(config))
        overwrite_config(config.generation, vars(config))
        self.config = config

        self.grid = Grid(config=config.grid, tile_config=config.tile)
        self.grid.randomize(
            seed=config.seed, trap_ratio=config.trap_ratio,
            goal_region_ratio=config.goal_region_ratio, braid_ratio=config.braid_ratio,
            checkpoints=config.checkpoints,
        )

        self.players = Players(config.lives, config.max_hiatus)
        self.agents = Agents(self.grid, self.players, config=config.agent)
        self.agents.reset(1)

        self.window = Window(self.grid, self.players, self.agents, config=config.window)

        self.previous_rewards: Array | None = None
        self.rewards: Array | None = None

        self.render_mode = render_mode
        self.max_frames = config.max_frames
        self.terminated = False

        self.update_spaces()

    # ------------------------------------------------------------------
    def update_spaces(self) -> None:
        """Rebuild observation_space / action_space for the current agent count.

        discrete channels (categorical, embedding-index data) get a MultiDiscrete sized to the *real* vocab —
        agents.max_options for observations, 4 (directions) for actions —
        rather than an arbitrary Box range. Continuous channels keep using
        Box over their true value range ([-1, 1] everywhere here).
        """
        total = self.agents.total
        state = self.agents.get_state()

        if self.agents.discrete_states:
            vocab = int(self.agents.max_options)
            self.observation_space = spaces.MultiDiscrete(
                np.full(state.shape, vocab, dtype=NP_INT)
            )
        else:
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=state.shape, dtype=NP_FLOAT)

        if self.agents.discrete_actions:
            self.action_space = spaces.MultiDiscrete(np.full((total,), 4, dtype=NP_INT))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(total, 2), dtype=NP_FLOAT)

    # ------------------------------------------------------------------
    def reset(self, **params) -> tuple[Array, dict[str, Any]]:
        """
        :kwarg seed: optional(int) - Seed for the random maze generator.
        :kwarg keys: list(int) | int - Keys used to create genomes/players.
        """
        self.terminated = False

        if self.render_mode != 'human':
            self.window.close()
        else:
            self.window.initialize()

        seed = params.get('seed', self.config.seed)
        keys: list[int] | int | None = params.get('keys')
        if keys is None:
            raise ValueError("Must pass a list of keys or a keys-total int to reset the environment")
        elif isinstance(keys, int):
            keys = list(range(keys))
        total = len(keys)

        if not self.grid.static:
            self.grid.randomize(
                seed=seed, trap_ratio=self.config.trap_ratio,
                goal_region_ratio=self.config.goal_region_ratio, braid_ratio=self.config.braid_ratio,
                checkpoints=self.config.checkpoints,
            )
        self.agents.reset(total)
        self.update_spaces()

        self.rewards = self.players.get_reward()
        self.previous_rewards = self.rewards.copy()

        return self.agents.get_state(), {}

    def step(self, actions: Array) -> tuple[Array, Array, bool, bool, dict[str, Any]]:
        if self.terminated:
            raise RuntimeError("Environment is terminated!")

        if Tensor is not None and isinstance(actions, Tensor):
            actions = actions.cpu().numpy()

        self.agents.update(actions)

        states = self.agents.get_state()

        self.previous_rewards = self.rewards.copy()
        self.rewards = self.players.get_reward()
        rewards = np.expand_dims(self.rewards, -1)

        completed_count = int(np.count_nonzero(self.players.completed))
        # Episode doesn't end the instant one player reaches the goal — it waits
        # for `first_winners` players to finish (a finished player stops moving,
        # see Players.active/restart, but keeps accruing goal_hold_reward).
        enough_winners = completed_count >= max(1, self.config.first_winners)
        all_disqualified = bool(np.all(~self.players.active))
        timed_out = bool(np.any(self.players.frames_done >= self.max_frames))
        done = enough_winners or all_disqualified or timed_out

        if self.render_mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

        if done:
            self.terminated = True

        return states, rewards, done, done, {
            'completed': enough_winners,
            'completed_count': completed_count,
            'disqualified': all_disqualified,
            'time_out': timed_out,
        }

    def render(self, **options) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            return None
        if self.render_mode == 'human':
            if self.window.screen is None:
                self.window.initialize()
            self.window.render()
        else:
            if self.window.screen is not None:
                self.window.close()

    # ------------------------------------------------------------------
    # Manual play (testing)
    # ------------------------------------------------------------------
    def run(self, total: int = 3):
        self.render_mode = 'human'
        self.window.fps = 15  # slower, since every frame = one discrete cell decision
        self.window.initialize()
        self.agents.reset(total)

        key_to_direction = {
            pygame.K_UP: Direction.UP, pygame.K_w: Direction.UP,
            pygame.K_RIGHT: Direction.RIGHT, pygame.K_d: Direction.RIGHT,
            pygame.K_DOWN: Direction.DOWN, pygame.K_s: Direction.DOWN,
            pygame.K_LEFT: Direction.LEFT, pygame.K_a: Direction.LEFT,
        }

        print("Starting manual run! Arrow keys / WASD move genome 0. N = new maze. Q = quit.")
        try:
            while self.window.running:
                move_dir = None
                quit_now = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_now = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_n:
                            self.grid.randomize(
                                seed=None, trap_ratio=self.config.trap_ratio,
                                goal_region_ratio=self.config.goal_region_ratio,
                                braid_ratio=self.config.braid_ratio,
                                checkpoints=self.config.checkpoints,
                            )
                            self.agents.reset(total)
                        elif event.key == pygame.K_q:
                            quit_now = True
                        elif event.key in key_to_direction:
                            move_dir = key_to_direction[event.key]

                if quit_now:
                    break

                actions = np.zeros((total, 2), dtype=np.float32) if not self.agents.discrete_actions \
                    else np.zeros((total,), dtype=np.int64)
                if move_dir is not None:
                    if self.agents.discrete_actions:
                        actions[0] = move_dir.value
                    else:
                        dr, dc = {
                            Direction.UP: (-1, 0), Direction.DOWN: (1, 0),
                            Direction.LEFT: (0, -1), Direction.RIGHT: (0, 1),
                        }[move_dir]
                        actions[0] = [dc, dr]
                    self.agents.update(actions)

                if np.all(self.players.lives <= 0):
                    self.grid.randomize(
                        seed=None, trap_ratio=self.config.trap_ratio,
                        goal_region_ratio=self.config.goal_region_ratio,
                        braid_ratio=self.config.braid_ratio,
                        checkpoints=self.config.checkpoints,
                    )
                    self.agents.reset(total)

                self.window.render()
        finally:
            self.window.close()

        self.render_mode = None
        self.window.fps = None
        self.window.clock = None


__all__ = ['Game']
