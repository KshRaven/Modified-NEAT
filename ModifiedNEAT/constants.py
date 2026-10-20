"""
Numeric type constants shared across ModifiedNEAT.

Numba (``types.*``) and NumPy (``np.*``) flavours of every dtype are defined once here, so a file that needs "the
genome float" or "the distance matrix float" imports it instead of redefining its own (which is how ``float64``
genomes and ``float32`` species ended up mixed before). Change a dtype here and every user follows.
"""

from numba import types

import numpy as np

# ---------- General (genome / parameters / typed containers) ---------- #
FLOAT, NP_FLOAT = types.float64, np.float64
INT, NP_INT     = types.int64, np.int64
BOOL, NP_BOOL   = types.boolean, np.bool_
STR             = types.unicode_type # NOTE: Old Numba versions fail to get this and use in custom class so
#                                            switch back to typeof('str') or raise Numba required version

# ---------- Species statistics (fitness, adjusted fitness, fitness history) ---------- #
SPECIES_FLOAT, NP_SPECIES_FLOAT = types.float32, np.float32

# ---------- Genetic distance matrix (genomes x genomes, the largest array of speciation) ---------- #
DISTANCE_FLOAT, NP_DISTANCE_FLOAT = types.float32, np.float32