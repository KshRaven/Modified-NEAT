"""Top-level Gymnasium environment of the Signal Tracker."""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import Env, spaces
from numpy import ndarray as Array

try:
    from torch import Tensor
except ImportError:
    class Tensor: pass

from .agent import Agents
from .constants import DECOY
from .config import EnvironmentConfig, normalize_seq_len, overwrite_config
from .player import Players
from .kernels import sensors as ksensors
from .renderer import SLIDERS, render_line
from .window import Window
from .world import World

__all__ = ["Game"]


class Game(Env):
    """
    Vectorised Signal Tracker: one noisy stream, one cursor per key, all stepped together.

    A hidden target drifts slowly in [-1, 1]. Each frame shows [cursor, cue, tag]: on CLEAN frames the cue is the target
    (+ small noise) and the tag is high; on DECOY frames (arriving in bursts, ``decoy_run``) the cue is uniform noise and
    the tag is low. A model that only sees the current frame is wrong whenever the frame is a decoy; a sequential model
    has to attend to the recent clean frames and ignore the decoys (attention noise: between instances of the
    sequence). ``drift`` adds a slowly varying component shared by all frames (common-mode noise, what differential
    attention subtracts); ``obs_noise`` adds plain per-feature noise.

    States are ``(N, features)`` when ``seq_len=None`` and ``(N, seq_len, features)`` (oldest first) otherwise.
    Actions: (N,) / (N, 1) scores (tanh), (N, 2) argmax or integer 0 / 1; cursor moves ``speed * action``.
    Reward per step: ``1 - min(|cursor - target|, 1)``. The episode ends after ``horizon`` steps.

        env = Game(render_mode=None, horizon=200, seq_len=32)
        states, _ = env.reset(keys=50)
        states, rewards, _, done, info = env.step(actions)
    """

    def __init__(self, render_mode: str | None = None, config: EnvironmentConfig | None = None, fps: int | None = 30,
                 **params: Any):
        config = config or EnvironmentConfig()
        overwrite_config(config, params)
        for section in (config.stream, config.agent, config.reward):
            overwrite_config(section, params)
        config.seq_len = normalize_seq_len(config.seq_len)
        self.config = config
        self.seq_len = config.seq_len
        self.horizon = int(config.horizon)
        self.world = World(config.stream)
        self.world.rng = np.random.default_rng(config.seed)
        self._active_seed = config.seed
        self.agents = Agents(self.world, config.agent, config.stream.obs_noise)
        self.players = Players()
        self.render_mode = render_mode      # 'human' = pygame window, 'ansi' = one text line per render, None = off
        self.window = Window(fps=fps)
        self.window.speed = float(config.agent.speed)
        self.window.set_params(self.params())
        self._window_kinds = np.zeros(0, bool)
        self._history: Array | None = None
        self._frame: Array | None = None
        self.focus: Array | None = None     # optional bool mask: which cursors count as 'best' (compare mode excludes baselines)
        self.terminated = False
        self._set_spaces(1)

    # ---------------------------------------------------------------- misc
    def seed(self, seed: int | None = None, force: bool = False) -> None:
        if seed is not None and not force and int(seed) == self._active_seed:
            return
        self._active_seed = None if seed is None else int(seed)
        self.world.rng = np.random.default_rng(self._active_seed)

    # ---------------------------------------------------------------- live perturbation (driven by the window's panel)
    def params(self) -> dict[str, float]:
        """Current values of the stream parameters the panel can change."""
        return {name: float(getattr(self.world.config, name)) for name, *_ in SLIDERS}

    def set_param(self, name: str, value: float) -> None:
        if not hasattr(self.world.config, name):
            raise KeyError(f"Unknown stream parameter '{name}'")
        setattr(self.world.config, name, float(value))
        if name in ('decoy_frac', 'decoy_run'):
            self.world.refresh()
        elif name == 'obs_noise':
            self.agents.obs_noise = float(value)

    def apply_event(self, event: tuple) -> None:
        kind = event[0]
        if kind == 'param':
            self.set_param(event[1], event[2])
        elif kind == 'kick':
            self.world.kick(event[1])
        elif kind == 'burst':
            self.world.burst(event[1])

    def _poll(self) -> None:
        self.window.poll_events()               # Space (pause), +/- zoom, Shift +/- fps; blocks while paused
        for event in self.window.consume_events():
            self.apply_event(event)

    def _best(self) -> int:
        sc = self.players.score_sum
        return int(np.argmax(sc if self.focus is None or len(self.focus) != len(sc) else np.where(self.focus, sc, -np.inf)))

    @property
    def rng(self) -> np.random.Generator:
        return self.world.rng

    @property
    def sequential(self) -> bool:
        return self.seq_len is not None

    def _set_spaces(self, total: int) -> None:
        shape = (self.agents.features,) if self.seq_len is None else (self.seq_len, self.agents.features)
        bound = 10.0
        self.observation_space = spaces.Box(low=-bound, high=bound, shape=(total, *shape), dtype=np.float64)
        self.action_space = spaces.Box(low=-1, high=1, shape=(total, 1), dtype=np.float64)

    def _state(self, reset: bool = False) -> Array:
        a, w = self.agents, self.world
        frame = a.get_state(w.rng)
        self._frame = frame                 # newest frame of every cursor (rendering reads this, never re-draws noise)
        if self.seq_len is None:
            return frame
        if reset or self._history is None or self._history.shape[0] != frame.shape[0]:
            self._history = np.empty((frame.shape[0], self.seq_len, frame.shape[1]))
            out = np.empty_like(self._history)
            ksensors.reset_history(self._history, out, frame)
        else:
            out = np.empty_like(self._history)
            ksensors.push_history(self._history, out, frame)
        return out

    # ---------------------------------------------------------------- gym API
    def reset(self, **params) -> tuple[Array, dict[str, Any]]:
        """``keys`` (int | list) or ``mapping`` (dict key -> index) gives the number of cursors; optional ``seed``."""
        self.terminated = False
        keys, mapping = params.get("keys"), params.get("mapping")
        if isinstance(keys, dict):
            mapping, keys = keys, None
        if keys is None and mapping is not None:
            keys = list(mapping.keys()) if isinstance(mapping, dict) else list(mapping)
        if keys is None:
            raise ValueError("Must input a list of keys or a total number of keys to reset the environment")
        total = int(keys) if isinstance(keys, (int, np.integer)) else len(keys)
        if total < 1:
            raise ValueError("Must reset with at least one cursor")
        seed = params.get("seed", self.config.seed)
        self.seed(seed)
        self.world.reset()
        self.agents.reset(total)
        self.players.reset(total)
        self._set_spaces(total)
        self.window.reset_trace()
        self._window_kinds = np.zeros(0, bool)
        if self.render_mode == "human" and not self.window.running:
            self.window.initialize()
        state = self._state(reset=True)
        if self.render_mode == "human":
            self._poll()
            self.render()                       # first frame right away: the window is never left empty
        return state, {}

    def step(self, actions: Array | Tensor) -> tuple[Array, Array, bool, bool, dict[str, Any]]:
        if self.terminated:
            raise RuntimeError("Environment is terminated!")
        if isinstance(actions, Tensor):
            actions = actions.detach().cpu().numpy()
        self.agents.update(actions)
        self.world.step()
        rewards = self.players.step(self.agents.x, self.world.target, self.config.reward.scale, self.world.kind == DECOY)
        states = self._state()
        done = self.players.frames_done >= self.horizon
        self._window_kinds = np.append(self._window_kinds, bool(self.world.kind))[-(self.seq_len or 1):]
        if self.render_mode == "human":
            self.window.record(self.world.target, self.world.cue, self.world.kind, self.agents.x, self._best(), self.players.score_sum)
            self._poll()
            if self.window.consume_stop():                  # window closed or Esc: end the episode
                done = True
        if done:
            self.terminated = True
        return states, rewards[:, None], done, done, {
            "completed": done, "disqualified": False, "time_out": done,
            "decoy": bool(self.world.kind), "target": self.world.target,
        }

    def render(self, **options):
        if self.render_mode == "ansi":
            print(render_line(self.agents.x, self.world.target, self.world.kind))
        elif self.render_mode == "human":
            best = self._best()
            fit = self.players.fitness
            if self.focus is not None and len(self.focus) == len(fit):
                fit_f = fit[self.focus]
            else:
                fit_f = fit
            hist = None if self._history is None else self._history[best]
            row = None if hist is None else hist[-1]
            if hist is None and self._frame is not None:
                row = self._frame[best]
            kinds = self._window_kinds
            if hist is not None and len(kinds) < hist.shape[0]:      # start of an episode: window = first frame repeated
                kinds = np.concatenate([np.full(hist.shape[0] - len(kinds), kinds[0] if len(kinds) else False), kinds])
            self.window.draw(self.agents.x, row, hist, {
                "step": self.players.frames_done, "horizon": self.horizon, "decoy": bool(self.world.kind),
                "target": self.world.target, "best": float(fit_f.max()), "fitness": fit,
                "mean": float(fit_f.mean()), "window_kinds": kinds,
                "mae": float(self.players.mae[best]), "mae_clean": float(self.players.mae_clean[best]),
                "mae_decoy": float(self.players.mae_decoy[best]), "frames_decoy": self.players.frames_decoy,
            })

    def close(self) -> None:
        self.window.close()