from ModifiedNEAT.config import Config
from ModifiedNEAT.constants import INT, SPECIES_FLOAT, NP_SPECIES_FLOAT, NP_DISTANCE_FLOAT, DISTANCE_FLOAT
from ModifiedNEAT.nn.genome import Genome
from ModifiedNEAT.reporter.base import ReporterSet
from ModifiedNEAT.util.qol import Indexer

from numba import types, optional
from numba.experimental import jitclass
from numba.typed import List, Dict
from numbers import Real
from typing import Union
from numpy import ndarray as CPUArray

import numpy as np

# Names other modules import from here (``from ModifiedNEAT.species import FLOAT``): species statistics use float32
FLOAT    = SPECIES_FLOAT
NP_FLOAT = NP_SPECIES_FLOAT
GENOME   = Genome.class_type.instance_type


@jitclass([
    ('key', INT),
    ('created', INT),
    ('last_improved', INT),
    ('representative', optional(GENOME)),
    ('members', types.DictType(INT, GENOME)),
    ('fitness', optional(FLOAT)),
    ('adjusted_fitness', optional(FLOAT)),
    ('fitness_history', types.ListType(FLOAT)),
    ('genus', INT),
])
class Species(object):
    def __init__(self, key: int, generation: int, genus: int):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative: Genome = None
        self.members: dict[int, Genome] = Dict.empty(INT, GENOME)
        self.fitness: float = None
        self.adjusted_fitness: float = None
        self.fitness_history: list[float] = List.empty_list(FLOAT)
        self.genus = genus

    def update(self, representative: Genome, members: dict[int, Genome]):
        self.representative = representative
        self.members = members

    def get_fitnesses(self):
        return [genome.fitness for genome in self.members.values()]


@jitclass([
    ('mapping', types.DictType(INT, INT)),                 # genome key -> row/column in total_distance
    ('total_distance', types.Array(DISTANCE_FLOAT, 2, 'C')),
    ('hits', INT),
    ('misses', INT),
])
class GenomeDistanceCache(object):
    def __init__(self):
        self.mapping: dict[int, int] = Dict.empty(INT, INT)
        self.total_distance: CPUArray = np.zeros((1, 1), dtype=NP_DISTANCE_FLOAT)
        self.hits = 0
        self.misses = 0

    def get(self, genome1: Genome, genome2: Genome):
        # Read straight from the distance matrix instead of an n^2 tuple-keyed dict
        if genome1.key not in self.mapping or genome2.key not in self.mapping:
            raise ValueError(f"distances between Genomes {genome1.key} and {genome2.key} have not been computed")
        self.hits += 1
        return self.total_distance[self.mapping[genome1.key], self.mapping[genome2.key]]

    def list(self):
        return self.total_distance.ravel()


SPECIES = Species.class_type.instance_type


class SpeciesSet:
    """
    Container of the species of every genus. The speciation itself (distances, representatives, grouping, species
    cap) lives in ``ModifiedNEAT.base.speciation`` and runs on the active device backend.
    """
    species_indexer = Indexer(0)

    def __init__(self, configuration: Config, reporters: ReporterSet):
        self.config = configuration
        self.reporters = reporters
        self.species: dict[int, Species] = Dict.empty(INT, SPECIES)
        self.genome_to_species: dict[int, int] = Dict.empty(INT, INT)
        self.distances_cache = GenomeDistanceCache()
        # (mean, std) of the genetic distances of the last speciation
        self.last_ct: tuple[float, float] | None = None
        # Adaptive compatibility threshold per genus (only used when `species.compatibility_threshold` is None)
        self.thresholds: dict[int, float] = {}

    def reset_genome_mapping(self):
        self.genome_to_species = Dict.empty(INT, INT)

    def get_species_id(self, gid: int):
        return self.genome_to_species.get(gid)

    def get_species(self, gid: int):
        sid = self.genome_to_species[gid]
        return self.species[sid]


def load_species(genomes: dict[int, Genome], struct: Dict):
    key                     = struct['key']
    genus                   = struct['genus']
    specie                  = Species(key, 0, genus)
    specie.created          = struct['created']
    specie.last_improved    = struct['last_improved']
    specie.representative   = genomes[struct['representative']]
    specie.members          = Dict([(k, genomes[k]) for k in struct['members']])
    specie.fitness          = struct['fitness']
    specie.adjusted_fitness = struct['adjusted_fitness']
    specie.fitness_history  = List.empty_list(FLOAT)
    for i in struct['fitness_history']:
        specie.fitness_history.append(i)
    return specie


def get_ct(ct: Union[int, float, str, None], last_ct_info=None, population=None):
    """
    Resolve ``config.species.compatibility_threshold``.

    * ``None`` (or the legacy string ``'auto'``) -> ``None``: the threshold is adaptive and managed per genus by
      ``base.speciation`` so the number of species stays within :func:`get_max_species`.
    * a positive number -> that fixed threshold (the species cap is still enforced by merging).
    ``last_ct_info`` / ``population`` are accepted for backwards compatibility and ignored.
    """
    if ct is None or (isinstance(ct, str) and ct.lower() == 'auto'):
        return None
    if isinstance(ct, bool) or not isinstance(ct, Real):
        raise ValueError(f"Unsupported compatibility threshold '{ct}' ({type(ct)}), use a positive number or None")
    if ct <= 0:
        raise ValueError(f"compatibility_threshold must be positive, got {ct}")
    return float(ct)


def get_max_species(max_species: Union[int, None], pop_size: int, min_species_size: int) -> int:
    """
    Maximum number of species a genus of `pop_size` genomes may have. Every species is guaranteed up to
    `min_species_size` offspring, so by default the population can afford `pop_size // min_species_size` of them;
    ``config.species.max_species`` overrides that. Always within [1, pop_size].
    """
    if max_species is None:
        max_species = pop_size // max(1, int(min_species_size))
    return int(max(1, min(int(max_species), max(1, pop_size))))