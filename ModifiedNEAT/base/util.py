"""
Device independent utilities shared by the CPU and GPU implementations.

Nothing in here imports cupy / numba.cuda, so it is safe on machines without a GPU. Array helpers receive the array
module (`xp`, i.e. numpy or cupy) from the active backend.
"""

from ModifiedNEAT.config import Config

from torch import Tensor
from typing import Union

import numpy as np
import torch


def init_type_code(config: Config) -> int:
    """Numeric code of the configured genome init type (used inside kernels)."""
    if config.genome.init_type == 'normal':
        return 0
    elif config.genome.init_type == 'uniform':
        return 1
    raise NotImplementedError(f"Unsupported NEAT Genome init type '{config.genome.init_type}'")


# TODO: Use a global mapping of PyTorch dtypes instead of just specifically for
#       the case bfloat16
def parameter_dtype(dtype: torch.dtype) -> torch.dtype:
    """dtype used for computation: bfloat16 has no numpy/cupy equivalent so float32 is used instead."""
    return dtype if dtype != torch.bfloat16 else torch.float32

# TODO: Maybe add extra functionality for case of balancing dimension sizes for the
#       the sake of optimal GPU kernel dimension declaration. Assuming this even
#       an effect so test out different shapes of similar sized kernels to prove
#       the necessity, in <...>/tests
def reshape_to_ndim(tensor: Union[Tensor, np.ndarray], required_ndim: int, xp=np):
    """
    Flatten the extra trailing dims (or append singleton dims) so that the tensor has exactly `required_ndim` dims.
    Works for torch tensors and for numpy / cupy arrays (pass the matching array module as `xp`).
    :return: (reshaped tensor, original shape)
    """
    original_shape: tuple[int, ...] = tensor.shape
    if tensor.ndim > required_ndim:
        tensor = tensor.reshape(*tensor.shape[:required_ndim - 1], -1)
    elif tensor.ndim < required_ndim:
        for _ in range(required_ndim - tensor.ndim):
            if isinstance(tensor, Tensor):
                tensor = tensor.unsqueeze(-1)
            else:
                tensor = xp.expand_dims(tensor, -1)
    return tensor, original_shape


def check_param_compatibility(parameters: list) -> bool:
    """Whether the parameters of a group (one per genus) have identical ndim and shape."""
    if len(parameters) == 0:
        return False

    reference_ndim = parameters[0].ndim
    for p in parameters:
        if p.ndim != reference_ndim:
            return False

    reference_shape = parameters[0].original_shape
    for p in parameters:
        for p_dim, ref_dim in zip(p.original_shape, reference_shape):
            if p_dim != ref_dim:
                return False

    return True


def effective_epsilon(config: Config, dtype) -> float:
    """
    Smallest magnitude a parameter may have. When weight deletion is disabled an exact 0.0 is illegal, so even with
    param_epsilon = 0.0 (or an epsilon that underflows in the parameter dtype, e.g. 1e-12 in float16) the floor becomes
    the smallest positive normal number of that dtype. With deletion enabled, zeros are legitimate and epsilon is kept.
    """
    epsilon = float(config.genome.param_epsilon)
    if config.genome.weight_del_prob == 0.0:
        try:
            epsilon = max(epsilon, float(torch.finfo(dtype).tiny))
        except TypeError:
            pass
    return epsilon


def clamp_to_epsilon(xp, array, epsilon: float):
    """In place: push every value with |v| < epsilon to +-epsilon (sign of the value, 0.0 -> +epsilon)."""
    small = xp.abs(array) < epsilon
    xp.copyto(array, epsilon, where=small & (array >= 0))
    xp.copyto(array, -epsilon, where=small & (array < 0))
    return small


def repair_zero_sources(sources, epsilon: float) -> int:
    """
    Parents can carry exact 0.0 values (e.g. from the very first initialization, or a float cast underflow).
    Crossover copies them verbatim, which the illegal-zero check then flags. The post-mutation epsilon clamp would
    turn them into +epsilon anyway (mutation also treats |v| <= epsilon as "zero"), so repair them at the source.
    """
    repaired = 0
    for source in sources:
        mask = (source == 0)
        total = int(mask.sum())
        if total:
            source[mask] = epsilon
            repaired += total
    return repaired


def check_distinct_genomes(total_distance: np.ndarray, mapping):
    """Two different genomes can never be at distance exactly 0 (crossover and mutation always change values)."""
    zero = (total_distance == 0.0)
    np.fill_diagonal(zero, False)
    if zero.any():
        i, j = [int(v) for v in np.argwhere(zero)[0]]
        keys = {index: key for key, index in mapping.items()}
        print(f"Was running genomes '{keys.get(i)}', and '{keys.get(j)}', "
              f"distances = {total_distance[i, j]}, {total_distance[j, i]}")
        raise ValueError(
            f"The genetic distance between two different genomes cannot be 0, "
            f"\nsince crossover and mutation exists. Even clones are mutated after creation"
            f"\nWithin the config file:"
            f"\n1. Check the weight and disjoint coefficient values."
            f"\n2. Check mutation coefficients' values  "
        )
