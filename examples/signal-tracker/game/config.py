"""Configuration of the Signal Tracker (same dataclass pattern as the other examples)."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class StreamConfig:
    """The noisy stream. Every agent sees the SAME stream, so fitness differences come from the model only."""
    target_speed: float = 0.04     # std of the target's velocity kicks (the target drifts slowly)
    target_inertia: float = 0.90   # velocity persistence
    decoy_frac: float = 0.5        # fraction of decoy frames (cue is uniform noise)
    decoy_run: float = 4.0         # mean length of a decoy burst: no single clean frame is ever 'nearby'
    cue_sigma: float = 0.05        # reading noise of a clean frame
    tag_sep: float = 2.0           # tag(clean) - tag(decoy); smaller = harder to tell frames apart
    tag_sigma: float = 0.25        # noise of the tag
    drift: float = 0.0             # amplitude of the common-mode drift added to cue AND tag (shared by all frames)
    drift_inertia: float = 0.98    # persistence of that drift
    obs_noise: float = 0.0         # extra iid gaussian noise on every feature (instance noise, not sequence noise)


@dataclass
class AgentConfig:
    speed: float = 0.12            # cursor displacement per step at action = +-1
    action_threshold: float = 0.0


@dataclass
class RewardConfig:
    scale: float = 1.0             # reward = scale * (1 - min(|cursor - target|, 1))


@dataclass
class EnvironmentConfig:
    horizon: int = 200             # steps per episode
    seq_len: int | None = None     # None = instance states (N, F), int = sequences (N, seq_len, F)
    seed: int | None = None
    stream: StreamConfig = field(default_factory=StreamConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


def overwrite_config(section: Any, params: dict[str, Any]) -> None:
    names = {f.name for f in fields(section)}
    for k, v in params.items():
        if k in names:
            setattr(section, k, v)


def normalize_seq_len(seq_len):
    if seq_len is None:
        return None
    seq_len = int(seq_len)
    return None if seq_len < 1 else seq_len
