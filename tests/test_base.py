from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

import ModifiedNEAT.nn as mn

from ModifiedNEAT.config import Config
from ModifiedNEAT.nn.base import NeatModule, NeatParameter
from ModifiedNEAT.population import Population

class TestModule(NeatModule):
    """Small convolution-shaped parameter surface for reproduction tests."""

    def __init__(self, modules: int = 0, dim_size: int = 64, dtype=torch.float64):
        super().__init__()
        self.projection = NeatParameter((dim_size, 1, 1), dtype=dtype)
        self.bias = NeatParameter((dim_size,), dtype=dtype)
        self.kernel = NeatParameter((dim_size, dim_size, 3), dtype=dtype)
        self.output = NeatParameter((dim_size, 5), dtype=dtype)
        self.extra = nn.ModuleList([
            mn.Linear(dim_size, dim_size, True, dtype=dtype) for _ in range(modules)
        ])


def test_config() -> Config:
    config = Config("cpu_test_base", "tests")
    config.general.seed                     = 1
    config.genome.init_type                 = "normal"
    config.genome.weight_init_mean          = 0.0
    config.genome.weight_init_std           = 0.5
    config.genome.weight_min_value          = -np.inf
    config.genome.weight_max_value          = np.inf
    config.genome.weight_mutate_power       = 0.25
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


def evaluate(current_population, **_options):
    for index, genome in enumerate(current_population.genomes.values()):
        genome.fitness = float(index)


if __name__ == "__main__":
    module      = TestModule(modules=20, dim_size=128)
    genomes     = 20
    population  = Population(genomes, module, test_config(), verbose=2)

    population.run(evaluate, generations=3, verbose=2)
