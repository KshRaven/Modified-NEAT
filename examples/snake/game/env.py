"""Top-level Gymnasium environment of the Snake game."""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np
import pygame
from gymnasium import Env, spaces
from gymnasium.core import RenderFrame
from numpy import ndarray as Array

try:
    from torch import Tensor
except ImportError:
    class Tensor: pass

from .config import EnvironmentConfig, overwrite_config
from .functional import action_towards
from .grid import Grid
from .player import Players
from .snake import Snakes
from .window import Window

__all__ = ["Game"]

_DICT_KEYS_WARNED = False


class Game(Env):
    """
    Vectorised Snake environment: one board and one snake per key, all stepped together.

    Typical use::

        env = Game(render_mode=None, grid_size=(20, 20), wrap=True)
        states, _ = env.reset(keys=100)
        states, rewards, _, done, info = env.step(actions)      # actions: (N,) ints or (N, 3) scores

    Actions are relative: 0 turn left, 1 straight, 2 turn right. With ``wrap=True`` a snake leaving one
    side of the board appears on the opposite side instead of dying.
    """

    def __init__(
            self,
            render_mode: str | None = "human",
            config: EnvironmentConfig | None = None,
            **params: dict[str, Any]
    ):
        if config is None:
            config = EnvironmentConfig()

        # Keyword overrides: top-level keys first, then the keys of the sections (e.g. pixel=True, init_length=4)
        overwrite_config(config, params)
        for section in (config.grid, config.snake, config.window, config.reward):
            overwrite_config(section, params)
        # Top-level values are the single source of truth for the shared keys
        overwrite_config(config.grid, vars(config))
        overwrite_config(config.player, vars(config))
        overwrite_config(config.generation, vars(config))
        self.config = config

        self.grid = Grid(config=config.grid)
        self.grid.rng = np.random.default_rng(config.seed)      # Env-private RNG, None -> OS entropy
        self._active_seed: int | None = config.seed

        self.players = Players(config.lives, config.max_hiatus, config.reward)
        self.snakes = Snakes(self.grid, self.players, config=config.snake)
        self.snakes.reset(1)
        self.window = Window(self.grid, self.players, self.snakes, config=config.window)

        self.previous_rewards: Array | None = None
        self.rewards: Array | None = None

        self.render_mode = render_mode
        self.max_frames = config.max_frames
        self._set_spaces(1)
        self.terminated = False

    # ------------------------------------------------------------------
    # Seeding / modes
    # ------------------------------------------------------------------

    def seed(self, seed: int | None = None, force: bool = False) -> None:
        """
        Seed the Env-private ``numpy.random.Generator`` (food and random spawns). The global ``random`` /
        ``np.random`` / torch state is never touched.

        * integer: restarts the generator only when it differs from the seed in use, so repeating the same
          value keeps the stream running (each reset gives the next reproducible game). ``force=True`` restarts.
        * ``None``: re-seeds from OS entropy (non-reproducible).
        """
        if seed is not None:
            seed = int(seed)
            if not force and seed == self._active_seed:
                return
        self._active_seed = seed
        self.grid.rng = np.random.default_rng(seed)

    @property
    def rng(self) -> np.random.Generator:
        """The Env-private NumPy generator."""
        return self.grid.rng

    def set_wrap(self, enabled: bool) -> bool:
        """Enable/disable boundary crossing. Call it before ``reset()`` (it changes the rules and the sensors).
        Returns the previous setting."""
        previous = bool(self.grid.config.wrap)
        self.config.wrap = self.config.grid.wrap = bool(enabled)
        return previous

    @contextmanager
    def wrap_mode(self, enabled: bool = True):
        """`with env.wrap_mode(): env.reset(...)` - wrap for the block, restored afterwards."""
        previous = self.set_wrap(enabled)
        try:
            yield self
        finally:
            self.set_wrap(previous)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_key(key: Any) -> str:
        if isinstance(key, (tuple, list, set, frozenset)):
            return "/".join(Game._fmt_key(k) for k in key)       # grouped key (123, 678) -> "123/678"
        return str(key)

    @staticmethod
    def _labels_from(mapping: dict | list | tuple | None) -> list[str] | None:
        """Snake index -> label. A list gives the label by position, a dict {key: index} by its value."""
        if mapping is None:
            return None
        if isinstance(mapping, dict):
            size = max((int(i) for i in mapping.values()), default=-1) + 1
            groups: list[list[str]] = [[] for _ in range(size)]
            for key, index in mapping.items():
                groups[int(index)].append(Game._fmt_key(key))
            return [",".join(g) if g else str(i) for i, g in enumerate(groups)]
        return [Game._fmt_key(k) for k in mapping]

    def _state_shape(self) -> tuple[int, ...]:
        return tuple(self.snakes.get_state().shape[1:])

    def _set_spaces(self, total: int) -> None:
        shape = self._state_shape()
        bound = abs(self.config.state_scale)
        self.observation_space = spaces.Box(low=-bound, high=bound, shape=(total, *shape), dtype=np.float64)
        self.action_space = spaces.Box(low=0, high=1, shape=(total, 3), dtype=np.float64)   # scores, argmax = action

    def _state(self) -> Array:
        return self.snakes.get_state() * self.config.state_scale

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, **params) -> tuple[Array, dict[str, Any]]:
        """
        Resets the entire game.
        :kwarg seed: optional(int) - Seeds this Env's private RNG and overrides ``config.seed``. An integer only
                     restarts the RNG when it differs from the seed already in use; None re-seeds from OS entropy.
        :kwarg keys: list(key) | int - One snake per key, in order. A list is also shown as the snake labels.
        :kwarg mapping: optional(dict[key, index]) - Alternative to ``keys``: keys are taken from ``mapping.keys()``.
        At least one of ``keys`` / ``mapping`` is required.
        """
        self.terminated = False

        if self.render_mode != "human":
            self.window.close()
        elif self.window.screen is None or not self.window.running:
            self.window.initialize()

        seed = params.get("seed")
        if seed is None:
            seed = self.config.seed
        keys: list | int | dict | None = params.get("keys")
        mapping: dict | list | tuple | None = params.get("mapping")
        if isinstance(keys, dict):
            global _DICT_KEYS_WARNED
            if not _DICT_KEYS_WARNED:
                _DICT_KEYS_WARNED = True
                warnings.warn("reset(keys=<dict>) is treated as a mapping; pass keys=list(mapping.keys()) "
                              "or mapping=<dict> instead.", stacklevel=2)
            if mapping is None:
                mapping = keys
            keys = None
        if keys is None and mapping is not None:
            keys = list(mapping.keys()) if isinstance(mapping, dict) else list(mapping)
        if keys is None:
            raise ValueError("Must input a list of keys or a total number of keys to reset the environment")
        if isinstance(keys, int):
            keys = list(range(keys))
            labels = None                                       # plain indices
        else:
            labels = self._labels_from(mapping if mapping is not None else list(keys))
        if isinstance(params.get("keys"), int) and mapping is not None:
            labels = self._labels_from(mapping)

        total = len(keys)
        self.seed(seed)
        self.snakes.reset(total)
        self.snakes.set_labels(labels)
        self._set_spaces(total)

        self.rewards = self.players.get_reward()
        self.previous_rewards = self.rewards.copy()
        self.window.reset_view()
        return self._state(), {}

    def step(self, actions: Array | Tensor) -> tuple[Array, Array, bool, bool, dict[str, Any]]:
        if self.terminated:
            raise RuntimeError("Environment is terminated!")

        if isinstance(actions, Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.ndim not in (1, 2):
            raise ValueError(f"Unsupported number of dimension '{actions.ndim}'")

        eliminated = self.snakes.update(actions)                # move, respawn, lives, fitness
        states = self._state()

        self.previous_rewards = self.rewards.copy()
        self.rewards = self.players.get_reward()
        rewards = np.expand_dims(self.rewards, -1)              # shape(players, 1)

        # Check env status
        any_complete = bool(np.any(self.players.completed))     # a snake filled the whole board
        all_disqualified = not bool(np.any(self.players.active))
        timed_out = bool(np.any(self.players.frames_done >= self.max_frames))
        done = any_complete or all_disqualified or timed_out

        if self.render_mode == "human":
            self.window.poll_events()                           # Space (pause), V, Tab, fps keys; blocks while paused
            if self.window.consume_stop():                      # window closed or Esc: end the episode
                done = True
        if done:
            self.terminated = True

        return states, rewards, done, done, {
            "completed": any_complete,
            "disqualified": all_disqualified,
            "time_out": timed_out,
            "eliminated": eliminated,
        }

    def render(self, **options) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            if self.window.screen is None or not self.window.running:
                self.window.initialize()
            self.window.render(options.get("verbose", False))
        elif self.window.screen is not None:
            self.window.close()                                 # only 'human' rendering is supported
        return None

    def close(self) -> None:
        if self.window.screen is not None:
            self.window.close()

    # ------------------------------------------------------------------
    # Manual play
    # ------------------------------------------------------------------

    _KEYS = {
        pygame.K_RIGHT: 0, pygame.K_d: 0,
        pygame.K_UP: 1, pygame.K_w: 1,
        pygame.K_LEFT: 2, pygame.K_a: 2,
        pygame.K_DOWN: 3, pygame.K_s: 3,
    }

    def run(self, fps: int = 10) -> None:
        """Manual play. Arrows / WASD steer (absolute directions), T toggles boundary wrap and restarts,
        N new game, Q quit, plus the window keys (Space pause, V info, P pixel, Shift +/- speed, Esc end)."""
        saved = (self.render_mode, self.window.fps)
        self.render_mode = "human"
        self.window.fps = fps
        self.window.initialize()

        def new_game() -> int:
            self.reset(keys=1)
            return int(self.snakes.direction[0])

        print("Starting run!  (arrows/WASD steer, T wrap, N new game, Q quit)")
        try:
            wanted = new_game()
            while self.window.running:
                done = restart = False
                for event in self.window.poll_events():
                    if event.type != pygame.KEYDOWN:
                        continue
                    if event.key in self._KEYS:
                        wanted = self._KEYS[event.key]
                    elif event.key == pygame.K_t:
                        self.set_wrap(not self.grid.wrap)
                        restart = True
                    elif event.key == pygame.K_n:
                        restart = True
                    elif event.key == pygame.K_q:
                        done = True

                if done or self.window.consume_stop():
                    break

                action = action_towards(int(self.snakes.direction[0]), wanted)
                self.snakes.update(np.array([action]))
                pl = self.players
                print(f"\rscore={pl.scores[0]} best={pl.true_scores[0]} lives={pl.lives[0]} "
                      f"length={self.snakes.length[0]} wrap={self.grid.wrap}   ", end="")

                if restart or pl.completed.any() or not pl.active.any():
                    wanted = new_game()
                self.window.render()
        finally:
            print("\nRun has ended!")
            self.window.close()
            self.render_mode, self.window.fps = saved
            self.window.clock = None