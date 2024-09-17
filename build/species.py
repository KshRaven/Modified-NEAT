
from build.config import Config
from build.nn.genome import Genome, FLOAT, INT
from build.reporter.base import ReporterSet
from build.util.fancy_text import CM, Fore

from numba import types, njit, optional, prange
from numba.experimental import jitclass
from numba.typed import List, Dict

import numpy as np
import time as clock

GENOME  = Genome.class_type.instance_type


@jitclass([
    ('key', INT),
    ('created', INT),
    ('last_improved', INT),
    ('representative', optional(GENOME)),
    ('members', types.DictType(INT, GENOME)),
    ('fitness', optional(FLOAT)),
    ('adjusted_fitness', optional(FLOAT)),
    ('fitness_history', types.ListType(FLOAT)),
])
class Species(object):
    def __init__(self, key: int, generation: int):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative: Genome = None
        self.members: dict[int, Genome] = Dict.empty(INT, GENOME)
        self.fitness: float = None
        self.adjusted_fitness: float = None
        self.fitness_history: list[float] = List.empty_list(FLOAT)

    def update(self, representative: Genome, members: dict[int, Genome]):
        self.representative = representative
        self.members = members

    def get_fitnesses(self):
        return [genome.fitness for genome in self.members.values()]


DISTANCE_TUPLE = types.Tuple([INT, INT])


@jitclass([
    ('distances', types.DictType(DISTANCE_TUPLE, FLOAT)),
    ('hits', INT),
    ('misses', INT),
])
class GenomeDistanceCache(object):
    def __init__(self):
        self.distances: dict[tuple[int, int], float] = Dict.empty(DISTANCE_TUPLE, FLOAT)
        self.hits = 0
        self.misses = 0

    def get(self, genome0: Genome, genome1: Genome, cwc: float, cdc: float) -> float:
        # Search for distance
        distance = self.distances.get((genome0.key, genome1.key))
        if distance is None:
            # Distance is not already computed.
            distance = genome0.distance(genome1, cwc, cdc)
            self.distances[(genome0.key, genome1.key)] = distance
            self.distances[(genome1.key, genome0.key)] = distance
            self.misses += 1
        else:
            self.hits += 1

        return distance

    def list(self):
        return list(self.distances.values())


SPECIES = Species.class_type.instance_type


class SpeciesSet:
    def __init__(self, configuration: Config, reporters: ReporterSet):
        self.config = configuration
        self.reporters = reporters
        self.species: dict[int, Species] = Dict.empty(INT, SPECIES)
        self.genome_to_species: dict[int, int] = Dict.empty(INT, INT)
        self.species_indexer = 1

    @staticmethod
    @njit(nogil=True)
    def _get_representatives(species_dict: dict[int, Species], population: dict[int, Genome], unspeciated: list[int],
                             representatives: dict[int, int], members: dict[int, list[int]],
                             distance_cache: GenomeDistanceCache, cwc: float, cdc: float):
        def gamma(candidates_: list[tuple[int, Genome]]):
            if len(candidates_) > 0:
                candidate = candidates_[0] # (distance, genome)
                for x in range(1, len(candidates_)):
                    comp = candidates_[x]
                    if comp[0] < candidate[0]:
                        candidate = comp
                return candidate
            else:
                raise ValueError(f"empty list")

        mapping = List(species_dict.keys())
        for i in prange(len(species_dict)):
            sid = mapping[i]
            species = species_dict[sid]
            candidates: list[tuple[float, Genome]] = []
            for j in prange(len(unspeciated)):
                gid = unspeciated[j]
                genome = population[gid]
                distance = distance_cache.get(species.representative, genome, cwc, cdc)
                candidates.append((distance, genome))

            # The new representative is the genome closest to the current representative.
            ignored_rdist, new_rep = gamma(candidates)
            new_rid = new_rep.key
            representatives[sid] = new_rid
            members[sid] = List([new_rid])
            unspeciated.remove(new_rid)

    @staticmethod
    @njit(nogil=True)
    def _get_species(available_key: int, population: dict[int, Genome], unspeciated: list[int],
                     representatives: dict[int, int], members: dict[int, list[int]],
                     distance_cache: GenomeDistanceCache, cwc: float, cdc: float, ct: float):
        def gamma(candidates_: list[tuple[int, int]]):
            if len(candidates_) > 0:
                candidate = candidates_[0] # (distance, species_key)
                for x in range(1, len(candidates_)):
                    comp = candidates_[x]
                    if comp[0] < candidate[0]:
                        candidate = comp
                return candidate
            else:
                raise ValueError(f"empty list")

        key = available_key
        done = List.empty_list(INT)
        for idx in prange(len(unspeciated)):
            gid = unspeciated[idx]
            genome = population[gid]

            # Find the species with the most similar representative.
            candidates: list[tuple[int, int]] = List()
            mapping = List(representatives.keys())
            for i in prange(len(representatives)):
                sid = mapping[i]
                rid = representatives[sid]
                rep = population[rid]
                distance = distance_cache.get(rep, genome, cwc, cdc)
                if distance < ct:
                    candidates.append((distance, sid))

            if candidates:
                ignored_sdist, sid = gamma(candidates)
                members[sid].append(gid)
            else:
                # No species is similar enough, create a new species, using this genome as its representative.
                sid = key
                key += 1
                representatives[sid] = gid
                members[sid] = List([gid])
            done.append(gid)
        if len(done) != len(unspeciated):
            raise ValueError(f"Filling species wasn't completed properly")

        return key

    @staticmethod
    @njit(nogil=True)
    def _update_collection(species_dict: dict[int, Species], population: dict[int, Genome],
                           representatives: dict[int, int], members: dict[int, list[int]], generation: int):
        genome_to_species: dict[int, int] = Dict.empty(INT, INT)
        mapping = List(representatives.keys())
        for i in prange(len(representatives)):
            sid = mapping[i]
            rid = representatives[sid]
            species = species_dict.get(sid)
            if species is None:
                species = Species(sid, generation)
                species_dict[sid] = species

            members_ = members[sid]
            for j in prange(len(members_)):
                gid = members_[j]
                genome_to_species[gid] = sid

            member_dict = {gid: population[gid] for gid in members_}
            species.update(population[rid], member_dict)
        return genome_to_species

    def speciate(self, population: dict[int, Genome], generation: int, verbose: int = None):
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        assert isinstance(population, (Dict, dict))

        compatibility_threshold = self.config.species.compatibility_threshold
        cwc = self.config.genome.compatibility_weight_coefficient
        cdc = self.config.genome.compatibility_disjoint_coefficient

        unspeciated: list[int]              = List(set(population.keys()))
        distances_cache                     = GenomeDistanceCache()
        new_representatives: dict[int, int] = Dict.empty(INT, INT)
        new_members: dict[int, list[int]]   = Dict.empty(INT, types.ListType(INT))

        # Find the best representatives for each existing species.
        gts = clock.perf_counter()
        self._get_representatives(
            self.species, population, unspeciated, new_representatives, new_members,  distances_cache, cwc, cdc
        )
        if verbose:
            print(f"\n{CM('Collected species representatives', Fore.CYAN)} in {round(clock.perf_counter() - gts, 2)} s")

        # Partition population into species based on genetic similarity.
        ts = clock.perf_counter()
        self.species_indexer = self._get_species(
            0, population, unspeciated, new_representatives, new_members,
            distances_cache, cwc, cdc, compatibility_threshold
        )
        if verbose:
            print(f"{CM('Filled species', Fore.CYAN)} in {round(clock.perf_counter() - ts, 2)} s")

        # Update species collection based on new speciation.
        ts = clock.perf_counter()
        self.genome_to_species = self._update_collection(
            self.species, population, new_representatives, new_members, generation
        )
        if verbose:
            print(f"{CM('Updated species mapping', Fore.CYAN)} in {round(clock.perf_counter() - ts, 2)} s")
            print(f"{CM('Completed speciating', Fore.CYAN)} in {round(clock.perf_counter() - gts, 2)} s")

        distances = distances_cache.list()
        gdmean = np.mean(distances)
        gdstdev = np.std(distances)
        self.reporters.info(f"Mean genetic distance {CM(f'{gdmean:.3f}', Fore.LIGHTYELLOW_EX)}, "
                            f"standard deviation {CM(f'{gdstdev:.3f}', Fore.LIGHTYELLOW_EX)}")

    def get_species_id(self, gid: int):
        return self.genome_to_species.get(gid)

    def get_species(self, gid: int):
        sid = self.genome_to_species[gid]
        return self.species[sid]


def load_species(genomes: dict[int, Genome], struct: Dict):
    key                     = struct['key']
    specie                  = Species(key, 0)
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
