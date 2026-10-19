from __future__ import annotations

from ModifiedNEAT.config.base import Configuration
from ModifiedNEAT.util.storage import STORAGE_DIR

import os
import math


class GeneralConfig(Configuration):
    def __init__(self):
        super(GeneralConfig, self).__init__('general config')

        self.fitness_criterion   = 'max'
        self.fitness_threshold   = math.inf
        self.pop_size            = 100
        self.reset_on_extinction = True
        # None -> no reproducibility (fresh entropy); int -> reproducible. Only affects the ModifiedNEAT backend.
        self.seed                = None


class GenomeConfig(Configuration):
    def __init__(self):
        super(GenomeConfig, self).__init__('genome config')

        self.weight_init_mean       = 0.0
        self.weight_init_std        = 1.0
        self.weight_max_value       = +math.inf
        self.weight_min_value       = -math.inf
        self.weight_mutate_power    = 0.50
        self.weight_mutate_rate     = 0.60
        self.weight_replace_rate    = 0.01
        self.weight_add_prob        = 0.25
        self.weight_del_prob        = 0.01
        self.param_epsilon          = 1e-12

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

        self.elitism            = 0.33
        self.clone_threshold    = 0.50
        self.survival_threshold = 0.20
        self.cross_threshold    = 0.00
        self.cross_multiplier   = 0.80
        self.darwin_multiplier  = 0.20
        self.min_species_size   = 100
        self.purge              = 0
        self.preserve_elite     = False


class RLConfig(Configuration):
    """
    Open-ended, initially empty section for RL / scheduler-driven hyper-parameters (e.g. ``rew_reg``, ``ent_reg``).

    Behaves like the other sections (saved / loaded / updated through ``Config``), but new variables can be
    added at any time with ``add`` -- or automatically by a ``Scheduler`` -- and are persisted as well, since
    ``Configuration.create`` / ``load`` operate on whatever public attributes the object currently has.
    """
    def __init__(self):
        super(RLConfig, self).__init__('rl config')
        # Intentionally empty: each algorithm registers only the variables it uses.

    def has(self, name: str) -> bool:
        return name in vars(self) and name[0] != '_'

    def add(self, name: str, value=0.0, overwrite: bool = False):
        """Register a new variable (no-op if it already exists, unless ``overwrite`` is True)."""
        if not isinstance(name, str) or not name or name[0] == '_':
            raise ValueError(f"Invalid RL config variable name: {name!r}")
        if overwrite or not self.has(name):
            setattr(self, name, value)

    def pull(self, source, names):
        """Register/overwrite variables from ``source`` attributes (e.g. an algorithm's own starting values)."""
        for name in names:
            self.add(name, getattr(source, name), overwrite=True)

    def apply(self, target, names=None):
        """Push current values onto ``target`` (e.g. a PPO instance) for every attribute it already has."""
        for label, value in vars(self).items():
            if label[0] == '_' or (names is not None and label not in names):
                continue
            if hasattr(target, label):
                setattr(target, label, value)


class Config:
    def __init__(self, file_name: str = None, directory: str = None):
        if file_name is None:
            file_name = "default"
        if directory is None:
            directory = f"{STORAGE_DIR}configs"
        self.dir = directory
        self.path = f"{directory}/{file_name}.mnc.json"

        self.general      = GeneralConfig()
        self.genome       = GenomeConfig()
        self.species      = SpeciesConfig()
        self.stagnation   = StagnationConfig()
        self.reproduction = ReproductionConfig()
        self.rl           = RLConfig()

    def save(self, debug=True):
        create = not os.path.exists(self.dir)
        if create:
            os.makedirs(self.dir, exist_ok=True)
        self.general.create(self.path, debug=debug)
        self.genome.create(self.path, debug=debug)
        self.species.create(self.path, debug=debug)
        self.stagnation.create(self.path, debug=debug)
        self.reproduction.create(self.path, debug=debug)
        self.rl.create(self.path, debug=debug)

    def load(self, verbose: int = None):
        self.general.load(self.path, verbose=verbose)
        self.genome.load(self.path, verbose=verbose)
        self.species.load(self.path, verbose=verbose)
        self.stagnation.load(self.path, verbose=verbose)
        self.reproduction.load(self.path, verbose=verbose)
        self.rl.load(self.path, verbose=verbose)

    def update(self, verbose: int = None):
        self.general.update(self.path, verbose=verbose)
        self.genome.update(self.path, verbose=verbose)
        self.species.update(self.path, verbose=verbose)
        self.stagnation.update(self.path, verbose=verbose)
        self.reproduction.update(self.path, verbose=verbose)
        self.rl.update(self.path, verbose=verbose)