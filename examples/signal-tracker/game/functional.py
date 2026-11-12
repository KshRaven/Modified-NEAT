"""Small helpers."""
import numpy as np


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def markov_rates(frac: float, run: float) -> tuple[float, float]:
    """(P(clean->decoy), P(decoy->decoy)) so that a fraction `frac` of frames are decoys in runs of ~`run` frames."""
    run = max(float(run), 1.0)
    frac = min(max(float(frac), 0.0), 0.95)
    stay = 1.0 - 1.0 / run
    enter = min(1.0, frac / max(1.0 - frac, 1e-9) * (1.0 - stay))
    return enter, stay
