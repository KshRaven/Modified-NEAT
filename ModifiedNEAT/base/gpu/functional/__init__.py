"""GPU (numba.cuda / cupy) functional helpers. Device independent ones are re-exported from ``ModifiedNEAT.base.functional``."""

from ModifiedNEAT.base.functional import calc_grid, calc, clamp, SEED, set_seed
from .launch import get_value, set_value
from .generation import get_rng_states, gauss, normal, uniform, prob
