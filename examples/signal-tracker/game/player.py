"""Per-agent bookkeeping (fitness = mean reward so far, tracking error on clean / decoy frames)."""
from __future__ import annotations

import numpy as np

from .kernels import reward as kreward

__all__ = ["Players"]


class Players:
    def __init__(self):
        self.reset(1)

    def reset(self, total: int) -> None:
        self.total = total
        self.active = np.ones(total, bool)
        self.frames_done = 0
        self.frames_decoy = 0
        self.score_sum = np.zeros(total)
        self.err_sum = np.zeros(total)      # sum of |cursor - target|
        self.err_decoy = np.zeros(total)    # the part of it that fell on decoy frames
        self.last_reward = np.zeros(total)

    @property
    def active_total(self) -> int:
        return int(self.active.sum())

    @property
    def fitness(self) -> np.ndarray:
        return self.score_sum / max(self.frames_done, 1)

    @property
    def scores(self) -> np.ndarray:
        return self.score_sum

    @property
    def mae(self) -> np.ndarray:
        """Mean absolute tracking error over every frame."""
        return self.err_sum / max(self.frames_done, 1)

    @property
    def mae_decoy(self) -> np.ndarray:
        return self.err_decoy / max(self.frames_decoy, 1)

    @property
    def mae_clean(self) -> np.ndarray:
        return (self.err_sum - self.err_decoy) / max(self.frames_done - self.frames_decoy, 1)

    def step(self, x: np.ndarray, target: float, scale: float, decoy: bool) -> np.ndarray:
        """Reward of one frame (fresh array, safe to hand out) + score / error bookkeeping, in one kernel."""
        rewards = np.empty(self.total)
        kreward.step_reward(rewards, self.score_sum, self.err_sum, self.err_decoy, x, float(target), float(scale), bool(decoy))
        self.last_reward = rewards
        self.frames_done += 1
        self.frames_decoy += int(decoy)
        return rewards

    def add(self, rewards: np.ndarray) -> None:
        self.score_sum += rewards
        self.last_reward = rewards
        self.frames_done += 1
