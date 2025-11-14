
from .functional import clamp, prob, normal, get_enumeration
from .initialization import initialize_genome

from numba import njit, prange
from numpy import ndarray as CPUArray

# NOTE:
# 1. Activations are immutable
# 2. The size of a genome's network is immutable
# 3. Nodes(biases) and Connections are indistinguishable at this level and all are just Parameters/ Weights
# 4. Deletion of nodes and connections can be simulated by setting weights and biases of a module to 0
# 5. Addition of nodes and connections can be simulated by re-initializing values that were set to 0 (deleted)


@njit
def mutate_genome(parameter: CPUArray, g: int, x: int, y: int, mutate_rate: float, mutate_power: float,
                  replace_rate: float, init_type: int, mean: float, std: float, minimum: float, maximum: float,
                  ssm: bool, add_param: float, delete_param: float, epsilon: float,
                  probabilities: tuple[CPUArray, ...], normals: CPUArray, rng_index: int):
    value = parameter[g, x, y]

    zero_param = abs(value) <= epsilon
    if ssm:
        div = max(1.0, add_param + delete_param)
        r = prob(probabilities[0], rng_index)
        if r < (add_param / div):
            if zero_param:
                initialize_genome(
                    parameter, g, x, y,
                    init_type, mean, std, minimum, maximum,
                    normals, rng_index
                )
            # else:
            #     clamp(
            #         value + normal(normals, rng_index, 0., mutate_power/3),
            #         minimum, maximum
            #     )
        elif r < ((delete_param + add_param) / div):
            parameter[g, x, y] = epsilon
    else:
        if prob(probabilities[0], rng_index) < add_param:
            if zero_param:
                initialize_genome(
                    parameter, g, x, y,
                    init_type, mean, std, minimum, maximum,
                    normals, rng_index
                )
            # else:
            #     clamp(
            #         value + normal(normals, rng_index, 0., mutate_power/3),
            #         minimum, maximum
            #     )
        elif prob(probabilities[1], rng_index) < delete_param:
            parameter[g, x, y] = epsilon

    r = prob(probabilities[-1], rng_index)
    if r < mutate_rate:
        parameter[g, x, y] = clamp(
            value + normal(normals, rng_index, 0., mutate_power),
            minimum, maximum
        )
    elif r < replace_rate + mutate_rate:
        initialize_genome(
            parameter, g, x, y,
            init_type, mean, std, minimum, maximum,
            normals, rng_index
        )


@njit(nogil=True)
def mutate(
        updates: CPUArray, children: CPUArray,
        mutate_rate: float, mutate_power: float,
        replace_rate: float, init_type: int, mean: float, std: float, minimum: float, maximum: float,
        ssm: bool, add_param: float, delete_param: float, epsilon: float,
        probabilities: tuple[CPUArray, ...], normals: CPUArray,
        # debugging: CPUArray
):
    if updates.ndim != 3:
        raise ValueError(f"Expected a 3D array, got {updates.ndim}")

    s_g, s_x, s_y = updates.shape
    # Parameter shape (genomes, *spatial_dims)
    g_lim = updates.shape[0]
    x_lim = 1 if updates.ndim <= 1 else updates.shape[1]
    y_lim = 1 if updates.ndim <= 2 else updates.shape[2]

    indices, total = get_enumeration(updates)
    for i in prange(total):
        genome_idx, x, y = indices[i]

        # Linearized thread index
        rng_index = (y * s_x * s_g) + (x * s_g) + genome_idx

        if genome_idx < g_lim and x < x_lim and y < y_lim:
            if children[genome_idx] is True:
                mutate_genome(
                    updates, genome_idx, x, y,
                    mutate_rate, mutate_power,
                    replace_rate, init_type, mean, std, minimum, maximum,
                    ssm, add_param, delete_param, epsilon,
                    probabilities, normals, rng_index
                )
        else:
            raise ValueError("Out of bounds")
