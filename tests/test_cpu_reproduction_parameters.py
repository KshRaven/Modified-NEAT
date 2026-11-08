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

    def __init__(self, output_size=5):
        super().__init__()
        self.projection = NeatParameter((8, 1, 1), dtype=torch.float64)
        self.bias = NeatParameter((8,), dtype=torch.float64)
        self.kernel = NeatParameter((8, 8, 3), dtype=torch.float64)
        self.output = NeatParameter((8, output_size), dtype=torch.float64)


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


def test_population_can_reproduce_multiple_genera_on_one_module():
    module = ReproductionModule()
    config = reproduction_config()
    config.reproduction.min_species_size = 2
    config.reproduction.purge = 0
    config.reproduction.cross_threshold = 0.5
    population = Population(20, module, config, genera=2, verbose=False)

    assert len(population.genera) == 2
    assert len(population.genomes) == 20
    assert all(parameter.shape[0] == 20 for parameter in module.neat_parameters())
    assert all(len(mapping) == 10 for mapping in population.get_mapping())

    def evaluate(current_population, **_options):
        for index, genome in enumerate(current_population.genomes.values()):
            genome.fitness = float(index)

    population.run(evaluate, generations=1, verbose=False)

    assert set(genome.genus for genome in population.genomes.values()) == set(population.genera)
    assert all(parameter.shape[0] == len(population.genomes) for parameter in module.neat_parameters())

    restored = Population(20, ReproductionModule(), config, genera=2, verbose=False)
    restored.load(state_dict=population.state_dict(), strict=False, debug=False)
    assert restored.genera == population.genera
    assert restored.genus == population.genus
    assert all(len(mapping) == 10 for mapping in restored.get_mapping())

    retained_genus = population.genera[-1]
    retained_keys = [key for key, genome in population.genomes.items() if genome.genus == retained_genus]
    population.crop(retained_keys, device="cpu")
    assert population.genera == [retained_genus]
    assert population.genus == retained_genus
    assert len(population.genomes) == 10
    assert all(parameter.shape[0] == 10 for parameter in module.neat_parameters())


def test_population_consolidates_compatible_absorbed_modules():
    config = reproduction_config()
    first_module = ReproductionModule()
    first = Population(4, first_module, config, init_reporter=False, verbose=False)
    second_module = ReproductionModule()
    second = Population(6, second_module, config, init_reporter=False, verbose=False)

    expected_rows = {}
    for population, module in ((first, first_module), (second, second_module)):
        for genome in population.genomes.values():
            genome.fitness = float(genome.key)
            expected_rows[genome.key] = [
                parameter.data[parameter.mapping[genome.key]].detach().clone()
                for parameter in module.neat_parameters()
            ]

    first.absorb_population(second)
    expected_species = {
        key: (specie.genus, set(specie.members))
        for key, specie in first.species_set.species.items()
    }
    consolidated = first.consolidate()

    assert consolidated is not first
    assert consolidated.genera == first.genera
    assert set(consolidated.genomes) == set(first.genomes)
    assert len({id(module) for module in consolidated.modules.values()}) == 1
    assert len(consolidated.get_mapping(consolidated=True)) == 10
    assert len(consolidated.species_set.genome_to_species) == 10
    assert {
        key: (specie.genus, set(specie.members))
        for key, specie in consolidated.species_set.species.items()
    } == expected_species

    merged_module = next(iter(consolidated.modules.values()))
    for parameter_index, parameter in enumerate(merged_module.neat_parameters()):
        assert parameter.shape[0] == 10
        for key, expected_parameters in expected_rows.items():
            row = parameter.data[parameter.mapping[key]]
            assert torch.equal(row, expected_parameters[parameter_index])
    for key, genome in consolidated.genomes.items():
        assert genome.fitness == float(key)

    assert len({id(module) for module in first.modules.values()}) == 2
    assert all(parameter.shape[0] == 4 for parameter in first_module.neat_parameters())
    assert all(parameter.shape[0] == 6 for parameter in second_module.neat_parameters())

    def evaluate(current_population, **_options):
        for genome in current_population.genomes.values():
            genome.fitness = float(genome.key)

    consolidated.config.reproduction.min_species_size = 2
    consolidated.config.reproduction.purge = 0
    consolidated.run(evaluate, generations=1, verbose=False)
    assert set(genome.genus for genome in consolidated.genomes.values()) == set(consolidated.genera)


def test_population_consolidate_rejects_incompatible_parameter_shapes():
    config = reproduction_config()
    population = Population(4, ReproductionModule(), config, init_reporter=False, verbose=False)
    incompatible = Population(
        4, ReproductionModule(output_size=6), config, init_reporter=False, verbose=False
    )
    population.absorb_population(incompatible)

    try:
        population.consolidate()
    except ValueError as error:
        assert "different NEAT parameter counts or shapes" in str(error)
    else:
        raise AssertionError("Expected incompatible parameter shapes to be rejected")