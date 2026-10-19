"""
Parameter initialization, written once for every device. Only the init kernel (``cpu/initialization.py`` and
``gpu/initialization.py``) and the array/RNG handling of the :class:`~ModifiedNEAT.base.backend.Backend` differ.
"""

from ModifiedNEAT.nn.base import NeatModule
from ModifiedNEAT.config import Config
from ModifiedNEAT.base.backend import Backend
from ModifiedNEAT.base.functional import calc_grid
from ModifiedNEAT.base.util import init_type_code, parameter_dtype, reshape_to_ndim, effective_epsilon, clamp_to_epsilon
from ModifiedNEAT.util.fancy_text import CM, Fore

import torch
import time as clock


def initialize(backend: Backend, config: Config, module: NeatModule, tpb=1, verbose: int = None):
    ts = clock.perf_counter()
    init_type = init_type_code(config)

    for param in module.neat_parameters():
        dtype = parameter_dtype(param.data.dtype)
        # Clone the existing parameter (the init kernel writes into it in place)
        array = param.data.clone().to(dtype)
        original_shape = param.data.shape
        # In case dims are greater than 3, flatten the extra dimensions
        array = reshape_to_ndim(array, 3)[0]
        # Convert param to a device array
        array = backend.asarray(array)
        # Get kernel dims
        kernel_shape = calc_grid(*array.shape, tpb=tpb)
        # Get rng states for each element in the param
        rng_states, threads_total = backend.normal_states(kernel_shape, config.general.seed)
        # Run kernel
        backend.initialize(
            array, init_type, config.genome.weight_init_mean, config.genome.weight_init_std,
            config.genome.weight_min_value, config.genome.weight_max_value, rng_states, kernel_shape,
            info=(param.dtype, param.device, kernel_shape, threads_total, array.shape, original_shape)
        )

        # Never hand out exact zeros (reproduction's post-mutation clamp does the same for children)
        clamp_to_epsilon(backend.xp, array, effective_epsilon(config, dtype))

        # Copy data back to parameter
        param.data = torch.from_dlpack(array.reshape(original_shape))

        # Remove data from the device
        del rng_states, array

    # Remove data
    backend.release()

    # Limit init values according the module specifications
    module.update_limit()
    for m in module.neat_modules():
        m.update_limit()

    if verbose and verbose >= 2:
        print(f"{CM('Initialized genomes in ', Fore.CYAN)} in {round(clock.perf_counter() - ts, 2)}s")
