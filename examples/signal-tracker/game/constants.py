"""Dtypes, feature layout and defaults of the Signal Tracker."""
import numpy as np

DTYPE = np.float64

# Feature layout of one frame (state = last SEQ_LEN frames, oldest first)
FEAT_CURSOR = 0     # own cursor position in [-1, 1]
FEAT_CUE    = 1     # noisy reading of the hidden target (+ common-mode drift)
FEAT_TAG    = 2     # reliability tag: high on clean frames, low on decoys (+ common-mode drift)
FEATURES    = 3

CLEAN, DECOY = 0, 1  # frame kinds
