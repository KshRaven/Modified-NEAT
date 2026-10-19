"""
Speciation, written once for every device. Only the distance computation (``cpu/speciation.py`` and
``gpu/speciation.py``, exposed through ``Backend.compute_distances``) differs; grouping genomes into species works
on genome objects and is identical on both devices.
"""

from ModifiedNEAT.config import Config
from ModifiedNEAT.nn.base import NeatModule
from ModifiedNEAT.nn.genome import Genome, INT
from ModifiedNEAT.species import Species, SpeciesSet, GenomeDistanceCache, get_ct
from ModifiedNEAT.base.backend import Backend
from ModifiedNEAT.base.util import check_distinct_genomes
from ModifiedNEAT.util.fancy_text import CM, Fore

from numba import types, njit
from numba.typed import List, Dict

import numpy as np
import time as clock


def update_distances_cache(backend: Backend, config: Config, module: NeatModule, genome_cache: GenomeDistanceCache,
                           tpb=10, verbose: int = None):
    ts = clock.perf_counter()
    total_distance = backend.compute_distances(config, module, genome_cache.total_distance.dtype, tpb, verbose)
    if verbose and verbose >= 2:
        print(f"{CM('Ran distance kernel', Fore.CYAN)} in {round(clock.perf_counter() - ts, 2)} s")

    check_distinct_genomes(total_distance, module.mapping)
    genome_cache.mapping = Dict(module.mapping.items())
    genome_cache.total_distance = total_distance

    # Remove temporary data
    backend.release()


@njit(nogil=True)
# NOTE: _get_representatives/_get_species/_update_collection below are left as
#       nogil rather than parallel=True: unlike crossover()/mutate()/_execute()/
#       get_distance(), these mutate shared dicts (representatives, members,
#       species) and, in _get_species, hand out new species keys from a single
#       running counter and call unspeciated.remove(...) - all inherently
#       sequential/order-dependent, not an embarrassingly-parallel per-element
#       grid like the array kernels above.
def _get_representatives(species_dict: dict[int, Species], population: dict[int, Genome], unspeciated: list[int],
                         representatives: dict[int, int], members: dict[int, list[int]],
                         distance_cache: GenomeDistanceCache):
    def gamma(candidates_: list[tuple[float, Genome]]):
        if len(candidates_) > 0:
            candidate = candidates_[0] # (distance, genome)
            for x in range(1, len(candidates_)):
                comp = candidates_[x]
                if comp[0] < candidate[0]:
                    candidate = comp
            return candidate
        else:
            raise ValueError(f"Empty list")

    # Loop through each existing species
    mapping = list(species_dict.keys())
    for i in range(len(species_dict)):
        sid = mapping[i]
        species = species_dict[sid]
        # Loop through unspeciated genomes
        candidates: list[tuple[float, Genome]] = []
        for j in range(len(unspeciated)):
            gid = unspeciated[j]
            genome = population[gid]
            # Replaced calculating distance with GPU function for updating distance_cache
            distance = distance_cache.get(species.representative, genome)
            # Add candidate to list
            candidates.append((distance, genome))

        # The new representative is the genome closest to the current representative.
        ignored_rdist, new_rep = gamma(candidates)
        new_rid = new_rep.key
        representatives[sid] = new_rid
        members[sid] = List([new_rid])
        unspeciated.remove(new_rid)


@njit(nogil=True)
def _get_species(available_key: int, population: dict[int, Genome], unspeciated: list[int],
                 representatives: dict[int, int], members: dict[int, list[int]],
                 distance_cache: GenomeDistanceCache, ct: float):
    def gamma(candidates_: list[tuple[int, int]]):
        if len(candidates_) > 0:
            candidate = candidates_[0]  # (distance, species_key)
            for x in range(1, len(candidates_)):
                comp = candidates_[x]
                if comp[0] < candidate[0]:
                    candidate = comp
            return candidate
        else:
            raise ValueError(f"empty list")

    key = available_key
    done = List.empty_list(INT)
    # Representative keys list is built once and appended to when a new species is created
    mapping = List(representatives.keys())
    # Loop through unspeciated genomes
    for idx in range(len(unspeciated)):
        gid = unspeciated[idx]
        genome = population[gid]

        # Find the species with the most similar representative.
        candidates: list[tuple[int, int]] = List()
        for i in range(len(mapping)):
            sid = mapping[i]
            rid = representatives[sid]
            rep = population[rid]
            # NOTE: Calculating distance is what takes most time in this function
            distance = distance_cache.get(rep, genome)
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
            mapping.append(sid)
            members[sid] = List([gid])
        done.append(gid)
    if len(done) != len(unspeciated):
        raise ValueError(f"Filling species wasn't completed properly")

    return key


@njit(nogil=True)
def _update_collection(genus: int, population: dict[int, Genome], species: dict[int, Species],
                       genome_to_species: dict[int, int], representatives: dict[int, int],
                       members: dict[int, list[int]], generation: int):
    mapping = list(representatives.keys())
    for i in range(len(representatives)):
        sid = mapping[i]
        rid = representatives[sid]
        specie = species.get(sid)
        if specie is None:
            specie = Species(sid, generation, genus)
            species[sid] = specie

        members_ = members[sid]
        for j in range(len(members_)):
            gid = members_[j]
            genome_to_species[gid] = sid

        member_dict = {gid: population[gid] for gid in members_}
        specie.update(population[rid], member_dict)


def speciate(backend: Backend, config: Config, genera: list[int], modules: dict[int, NeatModule],
             species_set: SpeciesSet, population: dict[int, Genome],
             generation: int, tpb=10, verbose: int = None):
    """
    Place genomes into species by genetic similarity.

    Note that this method assumes the current representatives of the species are from the old
    generation, and that after speciation has been performed, the old representatives should be
    dropped and replaced with representatives from the new generation.  If you violate this
    assumption, you should make sure other necessary parts of the code are updated to reflect
    the new behavior.
    """
    assert isinstance(population, (Dict, dict))
    assert all([module.genus in genera for module in modules.values()])

    for sid in list(species_set.species.keys()):
        if species_set.species[sid].representative.key not in population:
            del species_set.species[sid]

    compatibility_threshold = get_ct(config.species.compatibility_threshold, species_set.last_ct, population)

    species_set.reset_genome_mapping()
    population_genera: list[dict[int, Genome]] = [
        Dict([(g.key, g) for g in population.values() if g.genus == genus]) for genus in genera
    ]
    species_genera: list[dict[int, Genome]] = [
        Dict([(s.key, s) for s in species_set.species.values() if s.genus == genus]) for genus in genera
    ]
    modules_genera: list[NeatModule] = [modules[genus] for genus in genera]
    distances = []
    for species, genomes, module, genus in zip(species_genera, population_genera, modules_genera, genera):
        assert module.genus == genus
        assert isinstance(genomes, Dict)
        unspeciated: list[int]              = List(set(genomes.keys()))
        distances_cache                     = GenomeDistanceCache() # species_set.distances_cache
        new_representatives: dict[int, int] = Dict.empty(INT, INT)
        new_members: dict[int, list[int]]   = Dict.empty(INT, types.ListType(INT))

        # Update distances cache
        gts = clock.perf_counter()
        update_distances_cache(backend, config, module, distances_cache, tpb=tpb, verbose=verbose)
        if verbose and verbose >= 2:
            print(f"{CM('Created distances cache', Fore.CYAN)} in {round(clock.perf_counter() - gts, 2)} s")

        # Find the best representatives for each existing species.
        ts = clock.perf_counter()
        species = species_set.species.copy()
        for key, specie in list(species.items()):
            if specie.genus != genus:
                del species[key]
        try:
            _get_representatives(
                species, genomes, unspeciated, new_representatives, new_members, distances_cache
            )
        except Exception as e:
            print(distances_cache.total_distance)
            raise e
        if verbose and verbose >= 2:
            print(f"{CM('Collected species representatives', Fore.CYAN)} in {round(clock.perf_counter() - ts, 2)} s")

        # Partition genomes into species based on genetic similarity.
        ts = clock.perf_counter()
        species_set.species_indexer.set(_get_species(
            species_set.species_indexer.get(), genomes, unspeciated, new_representatives, new_members,
            distances_cache, compatibility_threshold
        ))
        if verbose and verbose >= 2:
            print(f"{CM('Filled species', Fore.CYAN)} in {round(clock.perf_counter() - ts, 2)} s")

        # Update species collection based on new speciation.
        ts = clock.perf_counter()
        _update_collection(
            genus, genomes, species_set.species, species_set.genome_to_species,
            new_representatives, new_members, generation
        )
        if verbose and verbose >= 2:
            print(f"{CM('Updated species mapping', Fore.CYAN)} in {round(clock.perf_counter() - ts, 2)} s")
            print(f"{CM('Completed speciating', Fore.CYAN)} in {round(clock.perf_counter() - gts, 2)} s")

        distances.append(np.asarray(distances_cache.total_distance).ravel())

    distances = np.concatenate(distances) if len(distances) > 0 else np.zeros(0, np.float32)
    distances = distances[np.isfinite(distances)]
    gd_mean = np.mean(distances).item()
    gd_std = np.std(distances).item()
    species_set.last_ct = (gd_mean, gd_std)
    if verbose:
        species_set.reporters.info(
            f"Mean genetic distance {CM(f'{gd_mean:.3f}', Fore.LIGHTYELLOW_EX)}, "
            f"standard deviation {CM(f'{gd_std:.3f}', Fore.LIGHTYELLOW_EX)}, "
            f"with {CM(len(species_set.species), Fore.LIGHTMAGENTA_EX)} species"
        )
