
from build.config import Config
from build.nn.base import NeatModule
from build.nn.genome import Genome, FLOAT, INT
from build.species import Species, SpeciesSet, GenomeDistanceCache, get_ct
from build.functional import get_value, calc_grid
from build.util.fancy_text import CM, Fore

from numba import types, njit, optional, prange, cuda
from numba.experimental import jitclass
from numba.cuda.cudadrv.devicearray import DeviceNDArray as GPUArray
from numba.typed import List, Dict
from numpy import ndarray as CPUArray

import numpy as np
import time as clock


DISTANCE_TUPLE = types.Tuple([INT, INT])


@cuda.jit(device=True)
def calc_distance(parameter: GPUArray, total_distance: GPUArray, genome1: int, genome2: int, x,
                  compatibility_weight_coefficient: float, compatibility_disjoint_coefficient: float):
    """
    Returns the genetic distance between this genome and the other. This distance value
    is used to compute genome compatibility for speciation.
    """

    v1 = get_value(parameter, genome1, x, 0)
    v2 = get_value(parameter, genome2, x, 0)
    distance = 0.0
    disjoint_values = 0

    if v2 == 0:
        disjoint_values += 1
    if v1 == 0:
        disjoint_values += 1
    else:
        distance = distance + abs(v1 - v2) * compatibility_weight_coefficient
    distance = distance + disjoint_values * compatibility_disjoint_coefficient
    size = parameter.shape[-1]
    if size != 0:
        distance = distance / size

    total_distance[genome1, genome2] = distance


@cuda.jit
def get_distance(parameter: GPUArray, total_distance: GPUArray,
                 compatibility_weight_coefficient: float, compatibility_disjoint_coefficient: float):
    # select one genome
    genome1_idx, genome2_idx, x = cuda.grid(3)
    g_lim = total_distance.shape[0]
    x_lim = parameter.shape[1]
    if genome1_idx < g_lim and genome2_idx < g_lim and x < x_lim:
        # Compare it with all other genomes
        calc_distance(parameter, total_distance, genome1_idx, genome2_idx, x,
                      compatibility_weight_coefficient, compatibility_disjoint_coefficient)
        pass


@njit(nogil=True)
def update_dict(distances: dict[tuple[int, int], float], total_distance: CPUArray, mapping: dict[int, int]):
    hits, misses = 0, 0
    keys = List(mapping.keys())
    for i in prange(len(keys)):
        gi = keys[i]
        ii = mapping[gi]
        for j in prange(len(keys)):
            gj = keys[j]
            ij = mapping[gj]

            # if ki != kj:
            key = (gi, gj)
            if key not in distances:
                distances[key] = total_distance[ii, ij]
                misses += 1
            else:
                hits += 1
    return hits, misses


def update_distances_cache(config: Config, module: NeatModule, genome_cache: GenomeDistanceCache,
                           tpb=10, verbose: int = None) -> float:
    total_distance = cuda.to_device(np.zeros((module.genome_num, module.genome_num), np.float64))

    ts = clock.perf_counter()
    for param in module.neat_parameters():
        with cuda.defer_cleanup():
            array = param.data.cpu().numpy()
            genome_num = len(module.mapping)
            array = array.reshape((genome_num, -1, 1))
            elements = array.shape[1]
            array = cuda.to_device(array)
            kernel_shape = calc_grid(genome_num, genome_num, elements, tpb=tpb)
            # if verbose:
            #     print(param.dtype, param.device, kernel_shape, array.shape, param.data.shape)

            get_distance[*kernel_shape](
                array, total_distance,
                config.genome.compatibility_weight_coefficient,
                config.genome.compatibility_disjoint_coefficient
            )

        # remove data from GPU
        _ = array.copy_to_host()

    total_distance = total_distance.copy_to_host()
    if verbose and verbose >= 2:
        print(f"\n{CM('Ran distance kernel', Fore.CYAN)} in {round(clock.perf_counter() - ts, 2)} s")

    h, m = update_dict(genome_cache.distances, total_distance, Dict(module.mapping.items()))
    genome_cache.total_distance = total_distance
    genome_cache.hits += h
    genome_cache.misses += m


@njit(nogil=True)
def _get_representatives(species_dict: dict[int, Species], population: dict[int, Genome], unspeciated: list[int],
                         representatives: dict[int, int], members: dict[int, list[int]],
                         distance_cache: GenomeDistanceCache):
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

    # Loop through each existing species
    mapping = List(species_dict.keys())
    for i in prange(len(species_dict)):
        sid = mapping[i]
        species = species_dict[sid]
        # Loop through unspeciated genomes
        candidates: list[tuple[float, Genome]] = []
        for j in prange(len(unspeciated)):
            gid = unspeciated[j]
            genome = population[gid]
            # NOTE: Calculating distance is what takes most time in this function. Replace with GPU functionality
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
    # Loop through unspeciated genomes
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
            members[sid] = List([gid])
        done.append(gid)
    if len(done) != len(unspeciated):
        raise ValueError(f"Filling species wasn't completed properly")

    return key


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


def speciate(config: Config, module: NeatModule, species_set: SpeciesSet, population: dict[int, Genome],
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

    compatibility_threshold = get_ct(config.species.compatibility_threshold, species_set.last_ct, population)

    unspeciated: list[int]              = List(set(population.keys()))
    distances_cache                     = GenomeDistanceCache() # species_set.distances_cache
    new_representatives: dict[int, int] = Dict.empty(INT, INT)
    new_members: dict[int, list[int]]   = Dict.empty(INT, types.ListType(INT))

    # Update distances cache
    gts = clock.perf_counter()
    update_distances_cache(config, module, distances_cache, tpb=tpb, verbose=verbose)
    if verbose:
        print(f"\n{CM('Created distances cache', Fore.CYAN)} in {round(clock.perf_counter() - gts, 2)} s")

    # Find the best representatives for each existing species.
    ts = clock.perf_counter()
    for sid in list(species_set.species.keys()):
        if species_set.species[sid].representative.key not in population:
            del species_set.species[sid]
    _get_representatives(
        species_set.species, population, unspeciated, new_representatives, new_members, distances_cache
    )
    if verbose:
        print(f"{CM('Collected species representatives', Fore.CYAN)} in {round(clock.perf_counter() - ts, 2)} s")

    # Partition population into species based on genetic similarity.
    ts = clock.perf_counter()
    species_set.species_indexer = _get_species(
        species_set.species_indexer, population, unspeciated, new_representatives, new_members,
        distances_cache, compatibility_threshold
    )
    if verbose:
        print(f"{CM('Filled species', Fore.CYAN)} in {round(clock.perf_counter() - ts, 2)} s")

    # Update species collection based on new speciation.
    ts = clock.perf_counter()
    species_set.genome_to_species = _update_collection(
        species_set.species, population, new_representatives, new_members, generation
    )
    if verbose:
        print(f"{CM('Updated species mapping', Fore.CYAN)} in {round(clock.perf_counter() - ts, 2)} s")
        print(f"{CM('Completed speciating', Fore.CYAN)} in {round(clock.perf_counter() - gts, 2)} s")

    distances = distances_cache.list()
    gd_mean = np.mean(distances)
    gd_std = np.std(distances)
    species_set.last_ct = (gd_mean, gd_std)
    species_set.reporters.info(
        f"Mean genetic distance {CM(f'{gd_mean:.3f}', Fore.LIGHTYELLOW_EX)}, "
        f"standard deviation {CM(f'{gd_std:.3f}', Fore.LIGHTYELLOW_EX)}, "
        f"with {CM(len(species_set.species), Fore.LIGHTMAGENTA_EX)} species"
    )
