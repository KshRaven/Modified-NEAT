"""
Signal Tracker Environment Package (same layout as the Car-Racer, Flappy-Bird and Snake packages)

A sequentially noisy environment: a slowly drifting hidden target is only readable on clean frames, decoy frames arrive
in bursts. Used to see whether (differential) attention learns to ignore the noisy instances of a sequence.

- functional: Small helpers
- constants: dtypes and the feature layout
- config: Configuration classes (stream noise, agent, reward)
- world: The shared stream (target, frame kinds, cue / tag, common-mode drift) and the Env-private RNG
- agent: The cursors
- player: Fitness bookkeeping
- reward: Reward shaping
- renderer: Pygame drawing (scrolling time plot, live frame, model input window) and a text fallback
- window: Pygame window, keyboard controls and the step trace
- env: Gymnasium Environment wrapper
"""
from .functional import clamp, markov_rates
from .constants import FEATURES, FEAT_CURSOR, FEAT_CUE, FEAT_TAG, CLEAN, DECOY
from .config import StreamConfig, AgentConfig, RewardConfig, EnvironmentConfig
from .world import World
from .agent import Agents
from .player import Players
from .window import Window
from .env import Game
