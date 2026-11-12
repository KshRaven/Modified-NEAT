"""Reward: closeness of the cursor to the hidden target (kernel in kernels/reward.py; Players.step uses it directly)."""
from __future__ import annotations

import numpy as np

from .config import RewardConfig
from .kernels import reward as kreward


def step_reward(x: np.ndarray, target: float, config: RewardConfig) -> np.ndarray:
    n = x.shape[0]
    rewards, z = np.empty(n), np.zeros(n)
    kreward.step_reward(rewards, z, z.copy(), z.copy(), np.ascontiguousarray(x, dtype=np.float64), float(target), float(config.scale), False)
    return rewards
