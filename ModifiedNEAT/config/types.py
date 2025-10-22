
from ModifiedNEAT.config.base import Configuration
from ModifiedNEAT.util.storage import STORAGE_DIR
from ModifiedNEAT.cuda.functional import SEED

import os
import math


class GeneralConfig(Configuration):
    def __init__(self):
        super(GeneralConfig, self).__init__('general config')

        self.fitness_criterion   = 'max'
        self.fitness_threshold   = math.inf
        self.pop_size            = 100
        self.reset_on_extinction = True
        self.seed                = SEED


class GenomeConfig(Configuration):
    def __init__(self):
        super(GenomeConfig, self).__init__('genome config')

        self.weight_init_mean       = 0.0
        self.weight_init_std        = 1.0
        self.weight_max_value       = +math.inf
        self.weight_min_value       = -math.inf
        self.weight_mutate_power    = 0.5
        self.weight_mutate_rate     = 0.50
        self.weight_replace_rate    = 0.01
        self.weight_add_prob        = 0.01
        self.weight_del_prob        = 0.01
        self.param_epsilon          = 1e-6

        self.compatibility_disjoint_coefficient = 1.0
        self.compatibility_weight_coefficient   = 0.1

        self.init_type = 'normal'
        self.single_structural_mutation = False


class SpeciesConfig(Configuration):
    def __init__(self):
        super(SpeciesConfig, self).__init__('species config')

        self.compatibility_threshold = 5.0 # 3.0


class StagnationConfig(Configuration):
    def __init__(self):
        super(StagnationConfig, self).__init__('stagnation config')

        self.species_fitness_func = 'max'
        self.max_stagnation       = 5
        self.species_elitism      = 2


class ReproductionConfig(Configuration):
    def __init__(self):
        super(ReproductionConfig, self).__init__('reproduction config')

        self.elitism            = 10
        self.clone_threshold    = 0.00
        self.survival_threshold = 0.10
        self.cross_threshold    = 0.05
        self.cross_multiplier   = 0.75
        self.darwin_multiplier  = 0.10
        self.min_species_size   = 100
        self.purge              = 0
        self.preserve_elite     = False


class Config:
    def __init__(self, file_name: str = None, directory: str = None):
        if file_name is None:
            file_name = "default"
        if directory is None:
            directory = f"{STORAGE_DIR}configs"
        self.dir = directory
        self.path = f"{directory}\\{file_name}-neat_config.txt"

        self.general      = GeneralConfig()
        self.genome       = GenomeConfig()
        self.species      = SpeciesConfig()
        self.stagnation   = StagnationConfig()
        self.reproduction = ReproductionConfig()

    def save(self, debug=True):
        create = not os.path.exists(self.dir)
        if create:
            os.makedirs(self.dir, exist_ok=True)
        self.general.create(self.path, debug=debug)
        self.genome.create(self.path, debug=debug)
        self.species.create(self.path, debug=debug)
        self.stagnation.create(self.path, debug=debug)
        self.reproduction.create(self.path, debug=debug)

    def load(self, verbose: int = None):
        self.general.load(self.path, verbose=verbose)
        self.genome.load(self.path, verbose=verbose)
        self.species.load(self.path, verbose=verbose)
        self.stagnation.load(self.path, verbose=verbose)
        self.reproduction.load(self.path, verbose=verbose)

    def update(self, verbose: int = None):
        self.general.update(self.path, verbose=verbose)
        self.genome.update(self.path, verbose=verbose)
        self.species.update(self.path, verbose=verbose)
        self.stagnation.update(self.path, verbose=verbose)
        self.reproduction.update(self.path, verbose=verbose)
