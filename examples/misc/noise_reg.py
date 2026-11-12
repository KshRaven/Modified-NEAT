"""
Attention-noise regularization as an EXTENSION of the stock trainers (nothing of it lives in ModifiedNEAT.rl).

It uses only the public hooks of ``Algorithm``:
    * ``add_pass_hook``   -> wraps the algorithm's own full-sequence passes with ``Attention.set_noise_probe``
                             (each probed layer re-runs once on a perturbed input; no extra pass in PPO / NEAT+pol_reg)
    * ``add_score_term``  -> adds ``noise_reg * (1 - normalized attention noise)`` to the fitness
    * ``add_log_source``  -> logs the measured noise to the dashboard ('attn_noise/*')

Usage (examples/flappy-bird/sequential.py):
    trainer = NoiseRegPPO(population, noise_reg=0.9, noise_sigma=0.15, noise_log=True, **ppo_options)

``noise_reg`` is a live attribute, so schedulers can drive it. ``noise_reg=0`` and ``noise_log=False`` = stock trainer.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

import ModifiedNEAT as neat

__all__ = ['AttentionNoiseMixin', 'NoiseRegPPO', 'NoiseRegNEAT']

PROBE_METHODS = ('evaluate_action', 'get_policy')   # full-sequence (uncached) passes the probe can ride on

HINTS = {
    'noise_mean': "Mean attention noise over genomes: total-variation shift of the attention maps under a small "
                  "input perturbation. 0 = maps ignore the perturbation.",
    'noise_min': "Lowest attention noise in the population.",
    'noise_max': "Highest attention noise in the population.",
    'noise_best': "Attention noise of the best-fitness genome.",
    'fitness_corr_noise': "Correlation(fitness, noise). Negative = quieter genomes are fitter.",
    'noise_hist': "Histogram of attention noise over genomes.",
}


class AttentionNoiseMixin:
    """Adds attention-noise measurement / regularization to any ``neat.rl.Algorithm`` subclass."""

    def __init__(self, *args, noise_reg: float = 0.0, noise_sigma: float = 0.1, noise_log: bool = False, **kwargs):
        """
        :param noise_reg: Fitness weight of low attention noise (0 = no pressure).
        :param noise_sigma: Probe size: relative input perturbation (see ``Attention.set_noise_probe``).
        :param noise_log: Measure + log the noise even when ``noise_reg == 0`` (baseline comparisons).
        """
        super().__init__(*args, **kwargs)
        self.noise_sigma: float = float(noise_sigma)
        self.noise_log: bool = bool(noise_log)
        self.attention_noise: dict[int, float] = {}
        self._noise_acc: dict[int, list[float]] = defaultdict(list)
        self._noise_step: int = -1
        self.add_score_term('attention_noise', self._noise_source, float(noise_reg), higher_is_better=False)
        self.add_pass_hook(self._noise_hook)
        self.add_log_source('attn_noise', self._noise_log, report=True, hints=HINTS)

    # ------------------------------ live weight ------------------------------ #
    @property
    def noise_reg(self) -> float:
        return self._score_terms['attention_noise']['weight']

    @noise_reg.setter
    def noise_reg(self, value: float):
        self.set_score_weight('attention_noise', value)

    # ------------------------------ measurement ------------------------------ #
    def _noise_hook(self, trainer, method: str, keys: list[int]):
        if method in PROBE_METHODS and self.noise_sigma > 0 and (self.noise_reg > 0 or self.noise_log):
            return self._probe(keys)
        return None

    @contextmanager
    def _probe(self, keys):
        from ModifiedNEAT.nn.modules.sub import Attention   # lazy: rl does not depend on the modules
        attns = []
        for module, _ in self._module_groups(keys):
            order = sorted(module.mapping, key=module.mapping.get)

            def sink(k, v, n, _order=order):
                ks = _order if k is None else ([int(k)] if isinstance(k, (int, np.integer)) else [int(x) for x in k])
                for key, val in zip(ks, v.cpu().numpy().tolist()):
                    self._noise_acc[key].append(val)

            for attn in module.modules():
                if isinstance(attn, Attention):
                    attn.set_noise_probe(self.noise_sigma, sink)
                    attns.append(attn)
        try:
            yield
        finally:
            for attn in attns:
                attn.set_noise_probe(0.0)

    def _noise_values(self, keys) -> dict[int, float]:
        """Per-genome noise of this update (measured once; later callers in the same update reuse it)."""
        if self._noise_step != self.updates_done:
            if self.noise_sigma > 0 and not any(self._noise_acc.get(k) for k in keys):
                # the algorithm made no suitable pass this update: probe with one extra policy pass
                self.forward_pass('get_policy', ['state'], keys)
            self.attention_noise = {k: float(np.mean(v)) for k, v in self._noise_acc.items() if v}
            self._noise_acc.clear()
            self._noise_step = self.updates_done
        return self.attention_noise

    def _noise_source(self, trainer, keys) -> dict[int, float]:
        return self._noise_values(keys)

    def _noise_log(self, trainer) -> dict[str, float | np.ndarray]:
        if not (self.noise_reg > 0 or self.noise_log):
            return {}
        values = self._noise_values(list(self.primary.mapping))
        if not values:
            return {}
        keys = list(values)
        arr = np.array([values[k] for k in keys], np.float64)
        out: dict = dict(noise_mean=arr.mean(), noise_min=arr.min(), noise_max=arr.max())
        fits = [getattr(self.population.genomes.get(k), 'fitness', None) for k in keys]
        if all(f is not None for f in fits):
            fits = np.array(fits, np.float64)
            out['noise_best'] = float(arr[int(np.argmax(fits))])
            if len(arr) > 2 and arr.std() > 0 and fits.std() > 0:
                out['fitness_corr_noise'] = float(np.corrcoef(arr, fits)[0, 1])
        if len(arr) > 1:
            out['noise_hist'] = arr
        return out


class NoiseRegPPO(AttentionNoiseMixin, neat.rl.PPO):
    """``neat.rl.PPO`` + attention-noise regularization."""


class NoiseRegNEAT(AttentionNoiseMixin, neat.rl.NEAT):
    """``neat.rl.NEAT`` + attention-noise regularization."""
