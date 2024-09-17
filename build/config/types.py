
from build.config.base import Configuration
from build.util.storage import STORAGE_DIR


class GeneralConfig(Configuration):
    def __init__(self):
        super(GeneralConfig, self).__init__('general config')

        self.fitness_criterion   = 'max'
        self.fitness_threshold   = 100
        self.pop_size            = 10000
        self.reset_on_extinction = True


RANGE = 3.14


class GenomeConfig(Configuration):
    def __init__(self):
        super(GenomeConfig, self).__init__('genome config')

        self.weight_init_mean       = 0.0
        self.weight_init_std        = 1.0
        self.weight_max_value       = RANGE
        self.weight_min_value       = -RANGE
        self.weight_mutate_power    = 0.05
        self.weight_mutate_rate     = 0.50
        self.weight_replace_rate    = 0.15

        self.bias_init_mean         = 0.0
        self.bias_init_std          = 1.0
        self.bias_max_value         = RANGE
        self.bias_min_value         = -RANGE
        self.bias_mutate_power      = 0.05
        self.bias_mutate_rate       = 0.40
        self.bias_replace_rate      = 0.15

        self.compatibility_disjoint_coefficient = 1.0
        self.compatibility_weight_coefficient   = 0.5

        self.conn_add_prob   = 0.5
        self.conn_del_prob   = 0.5
        self.node_add_prob   = 0.2
        self.node_del_prob   = 0.2

        self.initial_connection = 'full'
        self.init_type = 'normal'

        self.single_structural_mutation = False


class SpeciesConfig(Configuration):
    def __init__(self):
        super(SpeciesConfig, self).__init__('species config')

        self.compatibility_threshold = 7.0 # 3.0


class StagnationConfig(Configuration):
    def __init__(self):
        super(StagnationConfig, self).__init__('stagnation config')

        self.species_fitness_func = 'max'
        self.max_stagnation       = 5
        self.species_elitism      = 1


class ReproductionConfig(Configuration):
    def __init__(self):
        super(ReproductionConfig, self).__init__('reproduction config')

        self.elitism            = 10
        self.clone_threshold    = 0.00
        self.survival_threshold = 0.20
        self.darwin_multiplier  = 1
        self.min_species_size   = 20


class Config:
    def __init__(self, file_name: str = None, directory: str = None):
        if file_name is None:
            file_name = "default"
        if directory is None:
            directory = f"{STORAGE_DIR}configs"
        self.path = f"{directory}\\{file_name}-neat_config.txt"

        self.general      = GeneralConfig()
        self.genome       = GenomeConfig()
        self.species      = SpeciesConfig()
        self.stagnation   = StagnationConfig()
        self.reproduction = ReproductionConfig()

    def save(self, debug=True):
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

    def update(self, debug=True):
        self.general.update(self.path, debug=debug)
        self.genome.update(self.path, debug=debug)
        self.species.update(self.path, debug=debug)
        self.stagnation.update(self.path, debug=debug)
        self.reproduction.update(self.path, debug=debug)
