"""GPU (numba.cuda / cupy) functional helpers. Device independent ones are re-exported from ``ModifiedNEAT.base.functional``."""

from ModifiedNEAT.base.functional import calc, calc_grid, clamp, set_seed

from .generation import gauss, get_rng_states, normal, prob, uniform
from .launch import get_value, set_value

__all__ = [
	'calc', 'calc_grid', 'clamp', 'gauss', 'get_rng_states', 'get_value',
	'normal', 'prob', 'set_seed', 'set_value', 'uniform',
]
