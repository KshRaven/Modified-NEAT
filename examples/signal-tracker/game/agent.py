"""The cursors (one per genome)."""
from __future__ import annotations

import numpy as np

from .config import AgentConfig
from .constants import FEATURES
from .kernels import agents as kagents, sensors as ksensors
from .world import World

__all__ = ["Agents"]


class Agents:
    features = FEATURES

    def __init__(self, world: World, config: AgentConfig, obs_noise: float = 0.0):
        self.world, self.config, self.obs_noise = world, config, obs_noise
        self.x = np.zeros(1)
        self._no_noise = np.zeros((1, FEATURES))

    def reset(self, total: int) -> None:
        self.x = np.zeros(total)

    @staticmethod
    def to_scores(actions) -> np.ndarray:
        """(N,) / (N, 1) scores in [-1, 1], (N, 2) -> argmax (down / up), integer actions 0 / 1 -> -1 / +1."""
        a = np.asarray(actions)
        if a.ndim == 2:
            a = a[:, 0] if a.shape[1] == 1 else (np.argmax(a, axis=1) * 2.0 - 1.0)
        if np.issubdtype(a.dtype, np.integer) or a.dtype == bool:
            a = a.astype(np.float64) * 2.0 - 1.0
        return np.ascontiguousarray(np.clip(a.astype(np.float64), -1.0, 1.0))

    def update(self, actions) -> None:
        kagents.move(self.x, self.to_scores(actions), float(self.config.speed))

    def noise(self, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
        """Instance noise for one frame, (N, features); a dummy + False when obs_noise is 0 (same RNG draws as before)."""
        if self.obs_noise > 0:
            return self.obs_noise * rng.normal(size=(self.x.shape[0], FEATURES)), True
        return self._no_noise, False

    def get_state(self, rng: np.random.Generator, out: np.ndarray | None = None) -> np.ndarray:
        w = self.world
        noise, use = self.noise(rng)
        if out is None:
            out = np.empty((self.x.shape[0], FEATURES))
        ksensors.fill_state(out, self.x, float(w.cue), float(w.tag), noise, use)
        return out
