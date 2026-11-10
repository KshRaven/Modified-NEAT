"""
Run logger used by every trainer. Writes to the ``runlytics`` dashboard (https://github.com/KshRaven/runlytics),
whose ``SummaryWriter`` is a drop-in for Tensorboard's and additionally shows an optional hint behind every card's
"i" button. Falls back to Tensorboard's writer (hints ignored) when ``runlytics`` is not installed.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from torch import Tensor

try:
    from runlytics import SummaryWriter
    BACKEND = 'runlytics'
except ImportError:  # pragma: no cover
    from torch.utils.tensorboard import SummaryWriter
    BACKEND = 'tensorboard'

__all__ = ['RunlyticsLogger', 'TensorboardLogger', 'SummaryWriter', 'BACKEND', 'DEFAULT_HINTS']

# Card hints (shown behind the "i" button) for the metrics the trainers log themselves. Keys are ``section/name``.
DEFAULT_HINTS: dict[str, str] = {
    # rollout
    'rollout/ep_len_mean': "Mean episode length (steps) over this update's episodes of the best genome's rows.",
    'rollout/ep_len_std': "Standard deviation of episode length. High = very uneven episodes.",
    'rollout/ep_rew_mean': "Mean **per-episode reward** this update. The main learning signal; higher is better.",
    'rollout/ep_rew_std': "Standard deviation of per-episode reward. Shrinking over time usually means a stabler policy.",
    'rollout/ep_cum_rew': "Cumulative reward over this update's episodes.",
    'rollout/global_rew': "Mean reward across **all** genomes, not only the best one.",
    'rollout/global_cum_rew': "Cumulative reward across all genomes.",
    'rollout/buffer_size_pri': "Records in the best genome's primary buffer (fresh experience).",
    'rollout/buffer_size_sec': "Records in the best genome's secondary (replay) buffer.",
    # time
    'time/run_time': "Seconds spent running the environment/episodes this update.",
    'time/processing_time': "Seconds spent on learning/evolution (everything except running the environment).",
    # policy
    'policy/policy_accuracy': "How closely the best genome's policy matches its training targets (1 = perfect).",
    'policy/policy_reduction': "Normalised reduction of the policy loss versus its baseline. Higher is better.",
    'policy/raw_stddev': "Std-dev of the best genome's action distribution (raw). Collapse toward 0 = no exploration.",
    'policy/policy_stddev': "Std-dev of the best genome's action distribution (normalised).",
    'policy/policy_loss': "Policy loss of the best genome. Lower is better; compare the trend, not the value.",
    'policy/policy_std': "Std-dev of the best genome's policy output. Collapse toward 0 = no exploration.",
    'policy/policy_mean': "Mean of the best genome's policy output.",
    'policy/return_score': "Score of the best genome's return estimate.",
    'policy/value_loss': "Critic (value) loss of the best genome. Lower is better.",
    'policy/value_accuracy': "How closely the critic matches the observed returns.",
    'policy/explained_variance': "Fraction of return variance the critic explains (1 = perfect, <=0 = useless).",
    'policy/raw_reduction': "Ghost reduction of the best genome (raw).",
    'policy/reduction_mean': "Mean ghost reduction over all valid records.",
    'policy/reduction_std': "Std-dev of ghost reduction over all valid records.",
    'policy/reduction_max': "Largest ghost reduction seen this update.",
    'policy/reduction_min': "Smallest ghost reduction seen this update.",
    # module
    'module/param_mean': "Mean parameter value of the best genome's network.",
    'module/param_std': "Std-dev of the best genome's parameters.",
    'module/param_min': "Smallest parameter of the best genome's network.",
    'module/param_max': "Largest parameter of the best genome's network. Very large = possible instability.",
    'module/param_zeros': "Number of exactly-zero parameters (pruned/dead weights) in the best genome.",
    'module/global_mean': "Mean parameter value over the whole population.",
    'module/global_std': "Std-dev of parameters over the whole population.",
    'module/global_max': "Largest parameter over the whole population.",
    'module/global_min': "Smallest parameter over the whole population.",
    'module/global_zeros': "Zero parameters over the whole population.",
    # population
    'population/best_genome': "Key of the best-fitness genome this update. Jumps mean a new leader.",
    'population/best_genus': "Genus (species) the best genome belongs to.",
    'population/survival_rate': "Fraction of genomes that survive this generation's culling.",
    'population/creep_score': "Mean complexity growth ('creep') of the population.",
    'population/creep_score_best': "Complexity growth ('creep') of the best genome.",
    'population/creep_score_max': "Largest complexity growth in the population.",
    'population/fitness_min': "Lowest genome fitness in the population.",
    'population/fitness_max': "Highest genome fitness in the population.",
    'population/fitness_mean': "Mean genome fitness. Should trend upward as evolution progresses.",
    'population/fitness_std': "Fitness spread. Near 0 = population has converged (little diversity).",
}


def _scalar(value):
    """Converts Tensor / numpy / python numbers to something ``add_scalar`` / ``add_histogram`` accepts."""
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.numpy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bool):
        return int(value)
    return value


class RunlyticsLogger:
    """
    Groups scalars into named sections (eg. ``rollout``, ``policy``) and writes them all to a single
    runlytics ``SummaryWriter`` in one call, so algorithms build a plain dict per section instead of
    repeating ``writer.add_scalar(...)`` for every metric.

    Values that are arrays with more than one element are written as histograms instead of scalars, and ``call`` gives
    raw access to any other ``SummaryWriter`` method (text, images, figures, hparams ...).

    Every card can carry a hint (its "i" button): the built-in metrics have one (``DEFAULT_HINTS``) and your own are
    added with ``describe``.
    """

    def __init__(self, log_path: str):
        """
        :param log_path: Directory ``SummaryWriter`` will write its event files to.
        """
        self.log_path = log_path
        self.writer: SummaryWriter | None = None
        self.hints: dict[str, str] = dict(DEFAULT_HINTS)

    @property
    def hinted(self) -> bool:
        """True when the active backend can show card hints (runlytics)."""
        return BACKEND == 'runlytics'

    def init(self):
        """Creates the underlying ``SummaryWriter`` the first time it's needed, and returns it."""
        if self.writer is None:
            if BACKEND != 'runlytics':
                warnings.warn("runlytics is not installed (pip install git+https://github.com/KshRaven/runlytics.git): falling back to Tensorboard, "
                              "card hints are not shown.", stacklevel=2)
            self.writer = SummaryWriter(self.log_path)
        return self.writer

    def describe(self, section: str | dict[str, str], **hints: str):
        """
        Sets the hint shown behind a card's "i" button (Markdown allowed). Must be called before the tag is first
        logged to take effect.

        ``tb.describe('env', goal="Current goal distance.", difficulty="Curriculum level.")`` or
        ``tb.describe({'env/goal': "Current goal distance."})``.
        """
        if isinstance(section, dict):
            self.hints.update(section)
        else:
            self.hints.update({f"{section}/{name}": text for name, text in hints.items()})

    def log(self, step: int, **sections: dict[str, Any]):
        """
        Writes every scalar of every section under ``'<section>/<name>'``.

        :param step: Shared x-axis step used for every scalar logged this call.
        :param sections: One kwarg per section (eg. ``rollout={'ep_len_mean': 12.0}``). A ``None``
            value for a scalar or an empty/``None`` section is skipped, so optional metrics (eg. only
            present for some algorithms) can be passed unconditionally.
        """
        self.init()
        for section, scalars in sections.items():
            if not scalars:
                continue
            for name, value in scalars.items():
                if value is None:
                    continue
                value, tag = _scalar(value), f"{section}/{name}"
                extra = {'description': self.hints[tag]} if self.hinted and tag in self.hints else {}
                if isinstance(value, np.ndarray):
                    self.writer.add_histogram(tag, value, step, **extra)
                else:
                    self.writer.add_scalar(tag, value, step, **extra)

    def call(self, method: str, *args, description: str | None = None, **kwargs):
        """
        Calls any ``SummaryWriter`` method, eg. ``tb.call('add_text', 'env/notes', 'hello', step)``.
        ``description`` (card hint) is forwarded only when the backend supports it.
        """
        if description and self.hinted:
            kwargs['description'] = description
        return getattr(self.init(), method)(*args, **kwargs)

    def flush(self):
        """Flushes any buffered events to disk."""
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        """Flushes and closes the writer (a later ``log`` reopens it)."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None


TensorboardLogger = RunlyticsLogger  # backwards compatible name
