"""
Device independent parts of mutation. The kernels themselves (``mutate_genome`` / ``mutate``) are device specific
and live in ``cpu/mutation.py`` and ``gpu/mutation.py``.

NOTE:
1. Activations are immutable
2. The size of a genome's network (ie. total parameter count) is immutable
3. Nodes(biases) and Connections are indistinguishable at this level and all are just Parameters/ Weights
4. Deletion of nodes and connections can be simulated by setting weights and biases of a module to 0
5. Addition of nodes and connections can be simulated by re-initializing values that were set to 0 (deleted)
"""

from ModifiedNEAT.config import Config


def mutation_arguments(config: Config, init_type: int, epsilon: float) -> tuple:
    """Scalar arguments of the mutate kernels (in kernel signature order, after `updates` and `children`)."""
    return (
        config.genome.weight_mutate_rate, config.genome.weight_mutate_power,
        config.genome.weight_replace_rate, init_type,
        config.genome.weight_init_mean, config.genome.weight_init_std,
        config.genome.weight_min_value, config.genome.weight_max_value,
        config.genome.single_structural_mutation,
        config.genome.weight_add_prob, config.genome.weight_del_prob, epsilon,
    )
