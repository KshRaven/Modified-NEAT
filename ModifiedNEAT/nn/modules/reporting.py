"""
Reporting helpers for the sub-modules (dashboard log sources for ``Algorithm.add_log_source``).

Kept apart from ``sub.py`` so that file only holds the modules themselves.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Callable

import numpy as np
import torch

__all__ = ['differential_attention_logger']

SECTION = 'diff_attn'


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation, ``None`` when undefined (too few genomes or no variance)."""
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _layout(models: int, simplex: bool) -> dict:
    """
    Custom Scalars layout: a few labelled charts instead of one chart per tag.
    Open the custom scalars view, category 'Differential attention'.
    """
    def tags(m: int, *names: str) -> list[str]:
        return [f"{SECTION}/m{m}/{n}" for n in names]

    charts: dict[str, list] = {}
    for m in range(models):
        charts[f"m{m} | attention noise (probe: how far the maps move under input noise)"] = [
            'Multiline', tags(m, 'noise_best', 'noise_pop_mean', 'noise_pop_min', 'noise_pop_max')]
        charts[f"m{m} | does lower noise pay off? (correlation with fitness)"] = [
            'Multiline', tags(m, 'fitness_corr_noise')]
        charts[f"m{m} | leak of shared component (0 = rejected, 1 = plain attention)"] = [
            'Multiline', tags(m, 'leak_best', 'leak_pop_mean', 'leak_pop_min', 'leak_pop_max')]
        charts[f"m{m} | sum of subtraction coefficients"] = [
            'Multiline', tags(m, 'lambda_sum_best', 'lambda_sum_pop_mean')]
        if simplex:
            charts[f"m{m} | beta (rejection strength, 1 = full)"] = [
                'Multiline', tags(m, 'beta_best', 'beta_pop_mean', 'beta_pop_min', 'beta_pop_max')]
            charts[f"m{m} | branch split entropy (1 = even)"] = [
                'Multiline', tags(m, 'split_entropy_pop_mean')]
        charts[f"m{m} | does it pay off? (correlation with fitness)"] = [
            'Multiline', tags(m, 'fitness_corr_leak', 'fitness_corr_beta')]
        charts[f"m{m} | fitness gap: rejecting half - plain half"] = [
            'Multiline', tags(m, 'fitness_gap_rejecting')]
    return {'Differential attention': charts}


_HINT_BASE = {
    'leak': "**Leak** = |1 + sum(lambda)|: how much of the component *shared by all attention branches* (common-mode "
            "noise) still gets through. **0** = fully rejected (differential attention at full effect), **1** = plain "
            "attention (feature unused). Falling = the model is learning to cancel shared noise; flat near 1 = "
            "evolution is ignoring it.",
    'lambda_sum': "Sum of the subtraction coefficients, the raw knob behind *leak* (simplex: -beta). **-1** = "
                  "perfect rejection (leak 0), **0** = no subtraction (leak 1). Drifting past -1 over-subtracts "
                  "(leak rises again).",
    'beta': "(simplex) **Rejection strength**. **1** = shared component fully subtracted, **0** = switched off. "
            "Rising = rejection is being adopted.",
}
_HINT_SUFFIX = {
    'best': "best-fitness genome. If it sits clearly below the population mean, selection favours rejecting.",
    'pop_mean': "mean over all genomes: the population's overall trend.",
    'pop_min': "lowest genome: how far the most extreme genome has gone.",
    'pop_max': "highest genome: the laggard. A wide min-max gap = diverse population; a narrow one = converged.",
    'hist': "distribution over genomes (newer = darker). Two humps = the population split into rejecting and "
            "non-rejecting groups; one narrowing hump = convergence.",
}
_HINT_BASE['noise'] = (
    "**Attention noise** (measured, not inferred from parameters): mean total-variation distance between the attention maps "
    "on clean and slightly perturbed inputs (trainer option noise_sigma). **0** = the maps ignore input noise, higher = "
    "noise moves the attention. Works for plain attention too, so compare runs with and without differential attention. "
    "Falling while fitness holds = the model is learning robustness; flat = it is not selected for (try noise_reg > 0).")
_HINT_FIXED = {
    'fitness_corr_noise': "**Does lower noise pay off?** Correlation (-1..1) of fitness with measured attention noise across "
                          "genomes. **Negative** = quieter attention is fitter, ~0 = noise is not selected for (or fitness is "
                          "saturated), **positive** = noisier genomes are fitter.",
    'split_entropy_pop_mean': "(simplex) How evenly the subtraction is **split across the extra branches**. **1** = "
                              "all branches share it equally, **0** = one branch does everything and the rest are "
                              "redundant. Falling over time = branches specialising or collapsing.",
    'fitness_corr_leak': "**Does rejecting pay off?** Correlation (-1..1) of fitness with leak across genomes. "
                         "**Negative** = genomes that reject more are fitter (good), ~0 = no relation (feature not "
                         "selected for, or noise), **positive** = rejecting hurts. Correlation, not proof; noisy "
                         "with small populations.",
    'fitness_corr_beta': "(simplex) Correlation (-1..1) of fitness with beta. **Positive** = genomes that reject "
                         "more are fitter (good), ~0 = no relation, **negative** = rejecting hurts.",
    'fitness_gap_rejecting': "Mean fitness of the more-rejecting half minus the less-rejecting half, in fitness "
                             "units. **> 0** = rejecting pays off; compare its size with the fitness spread "
                             "(population/fitness_std) to judge if it matters.",
}
_HINT_LAYER = {
    'leak': "Leak of this one attention layer (see diff_attn leak). Compare layers to see **where** rejection is "
            "adopted: lower = this layer cancels shared noise more.",
    'beta': "(simplex) Rejection strength of this one attention layer. Compare layers to see where it is used.",
}
_LAYOUT_HINT = ("How the genomes use **Differential Attention**. Per model: leak (is shared noise rejected, 0 = yes), "
                "lambda sum (the knob behind it), beta and branch-split entropy (simplex), then whether it pays off "
                "(fitness correlation and the rejecting-vs-plain fitness gap). Trust the 'pays off' charts before "
                "the rest: rejection that does not raise fitness is just drift.")


def _hints(out: dict[str, Any], layers: dict[str, Any] | None = None) -> dict[str, str]:
    """Card hints (the dashboard's \"i\" button) for every tag in ``out`` (and per-layer ``layers``), keyed by full tag."""
    hints = {}
    for key in out:
        name = key.split('/', 1)[1]
        text = _HINT_FIXED.get(name)
        if text is None:
            for suffix, shows in _HINT_SUFFIX.items():
                base = name[:-len(suffix) - 1]
                if name.endswith('_' + suffix) and base in _HINT_BASE:
                    text = f"{_HINT_BASE[base]} **Showing:** {shows}"
                    break
        if text:
            hints[f"{SECTION}/{key}"] = text
    for key in layers or ():
        text = _HINT_LAYER.get(key.rsplit('/', 1)[-1])
        if text:
            hints[f"diff_attn_layers/{key}"] = text
    return hints


def differential_attention_logger(population: bool = True, detail: bool = False) -> Callable[[Any], dict]:
    """
    Log source for ``Algorithm.add_log_source`` that shows how the genomes use Differential Attention.

    ``trainer.add_log_source('diff_attn', differential_attention_logger())``

    Everything is reduced over attention layers and heads, so a model gives a FIXED, small set of tags
    ``diff_attn/m<i>/<name>`` (m<i> = i-th model of the trainer), also grouped into labelled charts under
    the dashboard's *Custom Scalars* view ('Differential attention'); every card carries a hint behind its "i" button. ``_best`` = best-fitness genome,
    ``_pop_*`` = statistics over all genomes of the model.

    ``leak_*``                 |1 + sum(lambda)|: how much of a component shared by all branches survives.
                               0 = exactly rejected (differential attention fully used), 1 = plain attention.
    ``lambda_sum_*``           sum of the subtraction coefficients (simplex: -beta).
    ``beta_*``                 (simplex) rejection strength; -> 1 used, -> 0 switched off.
    ``split_entropy_pop_mean`` (simplex) how the subtraction is split over the extra branches (1 = even, 0 = one).
    ``fitness_corr_leak``      correlation(fitness, leak): negative = genomes that reject more are fitter.
    ``fitness_corr_beta``      (simplex) correlation(fitness, beta): positive = genomes that reject more are fitter.
    ``fitness_gap_rejecting``  mean fitness of the more rejecting half minus the less rejecting half (>0: it pays off).
    ``leak_hist`` / ``beta_hist``  histograms over genomes.

    :param population: Also log the population statistics / benefit tags (False: only the best genome).
    :param detail: Additionally log ``diff_attn_layers/m<i>_<layer>/{leak,beta}`` per attention layer.
    """
    state = {'layout_done': False}

    def source(trainer) -> dict[str, dict[str, Any]]:
        from .sub import (
            AttentionLambda,  # lazy: keeps this module importable without the nn package
        )

        out: dict[str, Any] = {}
        layers_out: dict[str, Any] = {}
        genomes = trainer.population.genomes
        simplex_any = False
        with torch.no_grad():
            for mi, model in enumerate(trainer.models):
                keys = [k for k in genomes.keys() if k in model.mapping]
                if not keys:
                    continue
                fitness = np.array(
                    [-math.inf if genomes[k].fitness is None else float(genomes[k].fitness) for k in keys])
                bi = int(np.argmax(fitness))

                leak, lsum, beta, ent = [], [], [], []
                for name, owner in model.named_modules():
                    lam = getattr(owner, 'diff_lambda', None)
                    if not isinstance(lam, AttentionLambda):
                        continue
                    d = lam.diagnostics(keys)
                    n = len(keys)
                    leak.append(d['cm_gain'].float().reshape(n, -1).mean(-1).cpu().numpy())
                    lsum.append(d['lambda_sum'].float().reshape(n, -1).mean(-1).cpu().numpy())
                    if 'beta' in d:
                        beta.append(d['beta'].float().reshape(n, -1).mean(-1).cpu().numpy())
                        w = d['weights'].float()
                        if w.shape[-1] > 1:
                            e = -(w * torch.log(w.clamp_min(1e-12))).sum(-1) / math.log(w.shape[-1])
                            ent.append(e.reshape(n, -1).mean(-1).cpu().numpy())
                    if detail:
                        tag = f"m{mi}_{name.replace('.', '_')}" if name else f"m{mi}"
                        layers_out[f"{tag}/leak"] = float(leak[-1].mean())
                        if beta:
                            layers_out[f"{tag}/beta"] = float(beta[-1].mean())
                noise_map = getattr(trainer, 'attention_noise', None) or {}
                nz = np.array([noise_map.get(k, np.nan) for k in keys], dtype=float)
                ok = np.isfinite(nz)
                if ok.any():
                    out[f"m{mi}/noise_best"] = float(nz[bi]) if ok[bi] else float(np.nanmean(nz))
                    if population:
                        out[f"m{mi}/noise_pop_mean"] = float(np.nanmean(nz))
                        out[f"m{mi}/noise_pop_min"] = float(np.nanmin(nz))
                        out[f"m{mi}/noise_pop_max"] = float(np.nanmax(nz))
                        both = ok & np.isfinite(fitness)
                        corr = _pearson(nz[both], fitness[both]) if both.sum() >= 3 else None
                        if corr is not None:
                            out[f"m{mi}/fitness_corr_noise"] = corr
                if not leak:
                    continue

                def put(prefix: str, values: list[np.ndarray]) -> np.ndarray:
                    v = np.mean(values, axis=0)  # over layers -> (genomes,)
                    out[f"m{mi}/{prefix}_best"] = float(v[bi])
                    if population:
                        out[f"m{mi}/{prefix}_pop_mean"] = float(v.mean())
                        if prefix in ('leak', 'beta'):
                            out[f"m{mi}/{prefix}_pop_min"] = float(v.min())
                            out[f"m{mi}/{prefix}_pop_max"] = float(v.max())
                            out[f"m{mi}/{prefix}_hist"] = v
                    return v

                leak_v = put('leak', leak)
                put('lambda_sum', lsum)
                beta_v = put('beta', beta) if beta else None
                simplex_any |= bool(beta)
                if ent and population:
                    out[f"m{mi}/split_entropy_pop_mean"] = float(np.mean(ent))

                finite = np.isfinite(fitness)  # genomes without a fitness yet are left out
                if population and finite.sum() >= 3:
                    lk, ft = leak_v[finite], fitness[finite]
                    corr = _pearson(lk, ft)
                    if corr is not None:
                        out[f"m{mi}/fitness_corr_leak"] = corr
                    if beta_v is not None:
                        corr = _pearson(beta_v[finite], ft)
                        if corr is not None:
                            out[f"m{mi}/fitness_corr_beta"] = corr
                    if ft.size >= 4:
                        order, half = np.argsort(lk), ft.size // 2
                        out[f"m{mi}/fitness_gap_rejecting"] = float(ft[order[:half]].mean() - ft[order[-half:]].mean())

        if out:  # every call: tags such as the fitness correlations only appear once enough genomes have a fitness
            trainer.tb.describe(_hints(out, layers_out))
        if not state['layout_done'] and out:
            try:
                trainer.tb.call('add_custom_scalars', _layout(len(trainer.models), simplex_any), description=_LAYOUT_HINT)
                state['layout_done'] = True
            except Exception as e:  # layout is cosmetic: never break training for it
                warnings.warn(f"differential_attention_logger: could not write the chart layout: {e!r}")
        result = {SECTION: out}
        if layers_out:
            result['diff_attn_layers'] = layers_out
        return result

    return source