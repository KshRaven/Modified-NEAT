"""Top-level Gymnasium environment of the Flappy Bird game."""
from __future__ import annotations

import warnings
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

from .bird import Birds
from .config import EnvironmentConfig, normalize_hiatus, normalize_seq_len, overwrite_config
from .player import Players
from .window import Window
from .world import World

__all__ = ["Game"]

_DICT_KEYS_WARNED = False


class Game(Env):
    """
    Vectorised Flappy Bird environment: one level (the same pipes) and one bird per key, all stepped together.

    States are instances ``(N, features)`` when ``seq_len=None`` (default) and sequences ``(N, seq_len, features)``
    (oldest frame first) when ``seq_len`` is an integer.

    Typical use::

        env = Game(render_mode=None, goal=20, gap_size=(180, 220), seq_len=None)
        states, _ = env.reset(keys=100)
        states, rewards, _, done, info = env.step(actions)      # actions: (N,) / (N, 1) scores or (N,) bools

    A bird flaps when its score is above ``bird.action_threshold`` (0 for tanh outputs); (N, 2) scores use the
    argmax and integer actions flap on 1. Birds are removed when they hit a pipe or leave the playfield and
    respawn while they have lives left.
    """

    def __init__(
            self,
            render_mode: str | None = "human",
            config: EnvironmentConfig | None = None,
            **params: dict[str, Any]
    ):
        if config is None:
            config = EnvironmentConfig()

        # Keyword overrides: top-level keys first, then the keys of the sections (e.g. pixel=True, gap_size=(150, 200))
        overwrite_config(config, params)
        for section in (config.world, config.bird, config.window, config.reward, config.draft):
            overwrite_config(section, params)
        # Top-level values are the single source of truth for the shared keys
        overwrite_config(config.player, vars(config))
        overwrite_config(config.generation, vars(config))
        config.seq_len = normalize_seq_len(config.seq_len)
        self.config = config
        self.seq_len: int | None = config.seq_len               # None = instance states, int = sequential states
        self._history: Array | None = None                      # (N, seq_len, features), oldest first

        self.world = World(config.world, config.bird, config.draft)
        self.world.rng = np.random.default_rng(config.seed)     # Env-private RNG, None -> OS entropy
        self._active_seed: int | None = config.seed

        max_hiatus = normalize_hiatus(config.max_hiatus)        # None = a bird never starves
        if max_hiatus is not None and config.reward.auto_hiatus:
            max_hiatus = max(max_hiatus, self._first_pipe_frames() + 10)    # the first pipe has to be reachable
        self.players = Players(config.lives, max_hiatus, config.reward)
        self.birds = Birds(self.world, self.players, config.bird, goal=config.goal)
        self.birds.reset(1)
        self.window = Window(self.world, self.players, self.birds, config=config.window)

        self.previous_rewards: Array | None = None
        self.rewards: Array | None = None

        self.render_mode = render_mode
        self.max_frames = config.max_frames
        self._set_spaces(1)
        self.terminated = False

    def _first_pipe_frames(self) -> int:
        """Frames until the first pipe's pass point reaches the birds."""
        w, c = self.world, self.config.world
        first = c.first_pipe_x if c.first_pipe_x is not None else 3.0 * w.bird_x + c.spawn_width
        return int(np.ceil(max(first + w.pass_off - w.bird_x, 0.0) / max(c.velocity, 1e-6)))

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed(self, seed: int | None = None, force: bool = False) -> None:
        """
        Seed the Env-private ``numpy.random.Generator`` (the pipes). The global ``random`` / ``np.random`` /
        torch state is never touched.

        * integer: restarts the generator only when it differs from the seed in use, so repeating the same
          value keeps the stream running (each reset gives the next reproducible level). ``force=True`` restarts.
        * ``None``: re-seeds from OS entropy (non-reproducible).
        """
        if seed is not None:
            seed = int(seed)
            if not force and seed == self._active_seed:
                return
        self._active_seed = seed
        self.world.rng = np.random.default_rng(seed)

    @property
    def rng(self) -> np.random.Generator:
        """The Env-private NumPy generator."""
        return self.world.rng

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
        """Bird index -> label. A list gives the label by position, a dict {key: index} by its value."""
        if mapping is None:
            return None
        if isinstance(mapping, dict):
            size = max((int(i) for i in mapping.values()), default=-1) + 1
            groups: list[list[str]] = [[] for _ in range(size)]
            for key, index in mapping.items():
                groups[int(index)].append(Game._fmt_key(key))
            return [",".join(g) if g else str(i) for i, g in enumerate(groups)]
        return [Game._fmt_key(k) for k in mapping]

    @property
    def sequential(self) -> bool:
        return self.seq_len is not None

    def _state_shape(self) -> tuple[int, ...]:
        if self.seq_len is None:
            return (self.birds.features,)
        return (self.seq_len, self.birds.features)

    def _set_spaces(self, total: int) -> None:
        shape = self._state_shape()
        bound = max(abs(self.config.state_scale), 1e-9) * 3.0
        self.observation_space = spaces.Box(low=-bound, high=bound, shape=(total, *shape), dtype=np.float64)
        self.action_space = spaces.Box(low=-1, high=1, shape=(total, 1), dtype=np.float64)   # score > threshold = flap

    def _state(self, reset: bool = False, respawned: Array | None = None) -> Array:
        """Current state: (N, features) for instance states, (N, seq_len, features) for sequential states."""
        state = self.birds.get_state() * self.config.state_scale
        if self.seq_len is None:
            return state
        if reset or self._history is None or self._history.shape[0] != state.shape[0]:
            self._history = np.repeat(state[:, None, :], self.seq_len, axis=1)      # start: first state repeated
        else:
            self._history[:, :-1] = self._history[:, 1:]                           # drop the oldest frame
            self._history[:, -1] = state
            if respawned is not None and respawned.any():                          # new life: no memory of the old one
                self._history[respawned] = state[respawned][:, None, :]
        return self._history.copy()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, **params) -> tuple[Array, dict[str, Any]]:
        """
        Resets the entire game.
        :kwarg seed: optional(int) - Seeds this Env's private RNG and overrides ``config.seed``. An integer only
                     restarts the RNG when it differs from the seed already in use; None re-seeds from OS entropy.
        :kwarg keys: list(key) | int - One bird per key, in order. A list is also shown as the bird labels.
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
        if isinstance(keys, (int, np.integer)):
            if int(keys) < 1:
                raise ValueError("keys must specify at least one bird")
            keys = list(range(int(keys)))
            labels = None                                       # plain indices
        else:
            labels = self._labels_from(mapping if mapping is not None else list(keys))
        if isinstance(params.get("keys"), (int, np.integer)) and mapping is not None:
            labels = self._labels_from(mapping)

        total = len(keys)
        if total < 1: raise ValueError("Must reset with at least one bird")
        self.seed(seed)
        self.birds.reset(total)
        self.birds.set_labels(labels)
        self._set_spaces(total)

        self.rewards = self.players.get_reward()
        self.previous_rewards = self.rewards.copy()
        self.window.reset_view()
        return self._state(reset=True), {}

    def step(self, actions: Array | Tensor) -> tuple[Array, Array, bool, bool, dict[str, Any]]:
        if self.terminated:
            raise RuntimeError("Environment is terminated!")

        if isinstance(actions, Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.ndim not in (1, 2):
            raise ValueError(f"Unsupported number of dimension '{actions.ndim}'")

        eliminated = self.birds.update(actions)                 # move, collide, respawn, lives, fitness
        states = self._state(respawned=eliminated & self.players.active)

        self.previous_rewards = self.rewards.copy()
        self.rewards = self.players.get_reward()
        rewards = np.expand_dims(self.rewards, -1)              # shape(players, 1)

        # Check env status
        any_complete = bool(np.any(self.players.completed))     # a bird reached the goal
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
            "passed": self.birds.passed.copy(),
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

    _FLAP_KEYS = (pygame.K_SPACE, pygame.K_UP, pygame.K_w)

    def run(self, fps: int = 60) -> None:
        """Manual play. Space / Up / W flap, N new game, Q quit, Enter pauses, plus the window keys
        (V info, P pixel, Shift +/- speed, Esc end)."""
        saved = (self.render_mode, self.window.fps)
        self.render_mode = "human"
        self.window.fps = fps
        self.window.manual = True
        self.window.initialize()

        def new_game() -> None:
            self.reset(keys=1)
            self.window.message = "Press Space to flap"

        print("Starting run!  (Space/Up/W flap, N new game, Enter pause, Q quit)")
        try:
            new_game()
            started = False
            while self.window.running:
                flap = restart = done = False
                for event in self.window.poll_events():
                    if event.type != pygame.KEYDOWN:
                        continue
                    if event.key in self._FLAP_KEYS:
                        flap = True
                    elif event.key == pygame.K_n:
                        restart = True
                    elif event.key == pygame.K_q:
                        done = True

                if done or self.window.consume_stop():
                    break
                pl = self.players
                over = bool(pl.completed.any() or not pl.active.any())
                if restart or (over and flap):
                    new_game()
                    started = False
                    over = False
                if flap and not over:
                    started = True
                    self.window.message = None
                if started and not over:
                    self.birds.update(np.array([flap]))
                    print(f"\rscore={pl.scores[0]} best={pl.true_scores[0]} lives={pl.lives[0]}   ", end="")
                    if pl.completed.any():
                        self.window.message = "GOAL REACHED - Space to play again"
                    elif not pl.active.any():
                        self.window.message = "GAME OVER - Space to play again"
                self.window.render()
        finally:
            print("\nRun has ended!")
            self.window.message = None
            self.window.manual = False
            self.window.close()
            self.render_mode, self.window.fps = saved
            self.window.clock = None