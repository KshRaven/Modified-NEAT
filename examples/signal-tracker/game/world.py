"""The shared stream: hidden target, frame kinds, cue/tag readings and common-mode drift."""
from __future__ import annotations

import numpy as np

from .config import StreamConfig
from .constants import CLEAN, DECOY
from .functional import markov_rates

__all__ = ["World"]


class World:
    def __init__(self, config: StreamConfig):
        self.config = config
        self.rng = np.random.default_rng()
        self.enter, self.stay = markov_rates(config.decoy_frac, config.decoy_run)
        self.target = 0.0
        self.velocity = 0.0
        self.drift = 0.0
        self.kind = CLEAN
        self.cue = 0.0
        self.tag = 1.0
        self.frame = 0
        self.forced = 0                 # frames of forced decoy left (external perturbation from the window)

    def refresh(self) -> None:
        """Recompute the decoy Markov rates after decoy_frac / decoy_run changed live."""
        self.enter, self.stay = markov_rates(self.config.decoy_frac, self.config.decoy_run)

    def kick(self, value: float) -> None:
        """External perturbation: jump the hidden target (velocity cleared); the next frames show the new target."""
        self.target = float(np.clip(value, -1.0, 1.0))
        self.velocity = 0.0

    def burst(self, frames: int) -> None:
        """External perturbation: force the next `frames` frames to be decoys."""
        self.forced = max(self.forced, int(frames))

    def reset(self) -> None:
        c, r = self.config, self.rng
        self.target = float(r.uniform(-0.8, 0.8))
        self.velocity = 0.0
        self.drift = float(r.normal(0.0, c.drift))
        self.kind = DECOY if r.random() < c.decoy_frac else CLEAN
        self.frame = 0
        self.forced = 0
        self._read()

    def step(self) -> None:
        c, r = self.config, self.rng
        self.velocity = c.target_inertia * self.velocity + c.target_speed * float(r.normal())
        self.target += self.velocity
        if abs(self.target) > 1.0:                      # reflect at the borders
            self.target = float(np.clip(self.target, -1.0, 1.0))
            self.velocity = -self.velocity
        self.drift = c.drift_inertia * self.drift + c.drift * float(np.sqrt(1 - c.drift_inertia ** 2)) * float(r.normal())
        p = self.stay if self.kind == DECOY else self.enter
        self.kind = DECOY if r.random() < p else CLEAN
        if self.forced > 0:
            self.kind = DECOY
            self.forced -= 1
        self.frame += 1
        self._read()

    def _read(self) -> None:
        c, r = self.config, self.rng
        if self.kind == CLEAN:
            cue = self.target + c.cue_sigma * float(r.normal())
            tag = +0.5 * c.tag_sep + c.tag_sigma * float(r.normal())
        else:
            cue = float(r.uniform(-1.0, 1.0))
            tag = -0.5 * c.tag_sep + c.tag_sigma * float(r.normal())
        self.cue, self.tag = cue + self.drift, tag + self.drift