"""Regression coverage for CPU reproduction parameter transfer.

The convolution notebook uses float64 parameters, multi-dimensional tensors,
and a nonzero ``param_epsilon``.  This test keeps those characteristics while
using a small module so reproduction can be exercised without the notebook.
"""

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ModifiedNEAT.config import Config
from ModifiedNEAT.nn.base import NeatModule, NeatParameter
from ModifiedNEAT.population import Population

class ReproductionModule(NeatModule):
    """Small convolution-shaped parameter surface for reproduction tests."""

    def __init__(self):
        super().__init__()
        self.projection = NeatParameter((8, 1, 1), dtype=torch.float64)
        self.bias = NeatParameter((8,), dtype=torch.float64)
        self.kernel = NeatParameter((8, 8, 3), dtype=torch.float64)
        self.output = NeatParameter((8, 5), dtype=torch.float64)


def reproduction_config() -> Config:
    config = Config("cpu_reproduction_test", "tests")
    config.general.seed                     = 1
    config.genome.init_type                 = "normal"
    config.genome.weight_init_mean          = 0.0
    config.genome.weight_init_std           = 0.1
    config.genome.weight_min_value          = -np.inf
    config.genome.weight_max_value          = np.inf
    config.genome.weight_mutate_power       = 0.05
    config.genome.weight_mutate_rate        = 0.65
    config.genome.weight_replace_rate       = 0.0
    config.genome.weight_add_prob           = 0.99
    config.genome.weight_del_prob           = 0.0
    config.genome.param_epsilon             = 1e-12
    config.genome.single_structural_mutation = False
    config.reproduction.min_species_size    = 20
    config.reproduction.purge               = 1
    config.reproduction.elitism             = 0.33
    config.reproduction.clone_threshold     = 0.0
    config.reproduction.survival_threshold  = 0.10
    config.reproduction.cross_threshold     = 0.0
    config.species.compatibility_threshold  = np.inf
    config.stagnation.max_stagnation        = 1
    config.stagnation.species_elitism       = 1
    config.reproduction.darwin_multiplier   = 0.50
    config.reproduction.cross_multiplier    = 0.75
    return config


def test_sample_parameters_are_initialized_with_non_epsilon_values():
    module      = ReproductionModule()
    population  = Population(20, module, reproduction_config(), verbose=False)
    epsilon     = population.config.genome.param_epsilon

    for parameter in module.neat_parameters():
        values = parameter.detach().cpu()
        assert torch.count_nonzero(values).item() == values.numel()
        assert torch.count_nonzero(values == epsilon).item() < values.numel()


def test_cpu_reproduction_preserves_parameter_variation():
    module = ReproductionModule()
    population = Population(20, module, reproduction_config(), verbose=False)

    def evaluate(current_population, **_options):
        for index, genome in enumerate(current_population.genomes.values()):
            genome.fitness = float(index)

    reproduction_error = None
    try:
        population.run(evaluate, generations=1, verbose=False)
    except ValueError as error:
        reproduction_error = error
    epsilon = population.config.genome.param_epsilon
    diagnostics = []

    for parameter in module.neat_parameters():
        values = parameter.detach().cpu().reshape(parameter.shape[0], -1)
        epsilon_count = torch.count_nonzero(values == epsilon).item()
        zero_count = torch.count_nonzero(values == 0).item()
        unique_genomes = torch.unique(values, dim=0).shape[0]
        diagnostics.append(
            f"p{parameter.param_index}: epsilon={epsilon_count}, "
            f"zero={zero_count}, unique_genomes={unique_genomes}/{values.shape[0]}"
        )

    if reproduction_error is not None:
        raise AssertionError(
            "CPU reproduction raised before preserving parameter variation: "
            f"{reproduction_error}\n" + "\n".join(diagnostics)
        ) from reproduction_error

    for parameter in module.neat_parameters():
        values = parameter.detach().cpu().reshape(parameter.shape[0], -1)
        assert torch.count_nonzero(values == epsilon).item() < values.numel(), (
            f"parameter {parameter.param_index} was collapsed to param_epsilon"
        )
        assert torch.unique(values, dim=0).shape[0] > 1, (
            f"parameter {parameter.param_index} lost all genome variation"
        )