from ModifiedNEAT.config import Config
from ModifiedNEAT.nn.base import NeatModule
from ModifiedNEAT.nn.genome import Genome, INT
from ModifiedNEAT.species import Species, SpeciesSet, GenomeDistanceCache, get_ct
# from .functional import calc_grid
from ModifiedNEAT.util.fancy_text import CM, Fore

from numba import types, njit, prange
from numba.typed import List, Dict
from numpy import ndarray as CPUArray
from typing import Union
from torch import Tensor

import torch
import numpy as np
import time as clock
import gc


DISTANCE_TUPLE = types.Tuple([INT, INT])


_Index  = Union[int, tuple[int]]
_Number = Union[float, int]


@njit
def calc_distance(parameter: CPUArray, total_distance: CPUArray,
                  genome0: int, genome1: int, x: int,
                  compatibility_weight_coefficient: float, compatibility_disjoint_coefficient: float):
    """
    Returns the genetic distance between this genome and the other. This distance value
    is used to compute genome compatibility for speciation.
    """

    value0 = parameter[genome0, x]
    value1 = parameter[genome1, x]

    current_distance = 0 # total_distance[genome0, genome1]
    disjoint_value   = 0
    if value0 == 0:
        disjoint_value += 1
    if value1 == 0:
        disjoint_value = disjoint_value + 1
    if value0 != 0 and value1 != 0:
        current_distance = current_distance + (abs(value0 - value1) * compatibility_weight_coefficient)
    current_distance = current_distance + disjoint_value * compatibility_disjoint_coefficient

    # Parameter shape = (genomes, x, y | 1)
    x_s = parameter.shape[1] # [:]
    size = x_s # * y_s
    # if size != 0:
    current_distance = current_distance / size

    return current_distance


@njit(parallel=True) # , cache=True)
# NOTE: Same fix as crossover(): a single return point and ret_code instead of
#       raising inside a prange body. Only the outer `genome0` loop is a prange -
#       each iteration exclusively owns row `genome0` of total_distance (lower
#       triangle writes total_distance[genome0, genome1>=genome0], upper triangle
#       writes total_distance[genome0, genome1<genome0] after reading the already
#       completed lower triangle) so there is no cross-thread write conflict.
#       The inner loops stay plain `range` - nesting prange inside prange gives
#       numba nothing extra here and only muddies which loop actually gets parallelized.
def get_distance(parameter: CPUArray, total_distance: CPUArray,
                 compatibility_weight_coefficient: float, compatibility_disjoint_coefficient: float):
    ret_code: int = 0
    if parameter.ndim != 2: ret_code = 1

    if not ret_code:
        g_lim = total_distance.shape[0]
        x_lim = parameter.shape[1]

        # TODO: The nested pranges may raise performance warning so find a way to combine
        #       and iterate over a single integer like other parralel functions
        # Fill the lower triangle
        # Rows i and (g_lim - 1 - i) are handled by the same iteration to balance the triangular workload.
        # Each row is owned by exactly one iteration, the diagonal is skipped (it only ever added 0.0), and
        # the per-element sum is accumulated in a local (same order as before, so results are identical).
        half = (g_lim + 1) // 2
        for r in prange(half):
            row = np.int64(r)  # prange index is unsigned; keep everything int64 (uint64 with int64 -> float64)
            for side in range(2):
                genome0 = row if side == 0 else g_lim - 1 - row
                if side == 1 and genome0 == row:
                    continue
                for genome1 in range(genome0 + 1, g_lim):
                    acc = total_distance[genome0, genome1]
                    for x in range(x_lim):
                        acc += calc_distance(
                            parameter, total_distance, genome0, genome1, x,
                            compatibility_weight_coefficient, compatibility_disjoint_coefficient
                        )
                    total_distance[genome0, genome1] = acc
        # Fill the upper triangle
        for genome0 in prange(g_lim):
            for genome1 in prange(genome0):
                # Assert limits
                if genome0 < g_lim and genome1 < g_lim:
                    distance = total_distance[genome1, genome0]
                    total_distance[genome0, genome1] = distance
                else: ret_code = 2

    return ret_code


def _check_distinct_genomes(total_distance, mapping):
    """Two different genomes can never be at distance exactly 0 (crossover and mutation always change values)."""
    zero = (total_distance == 0.0)
    np.fill_diagonal(zero, False)
    if zero.any():
        i, j = [int(v) for v in np.argwhere(zero)[0]]
        keys = {index: key for key, index in mapping.items()}
        print(f"Was running genomes '{keys.get(i)}', and '{keys.get(j)}', "
              f"distances = {total_distance[i, j]}, {total_distance[j, i]}")
        raise ValueError(
            f"The genetic distance between two different genomes cannot be 0, "
            f"\nsince crossover and mutation exists. Even clones are mutated after creation"
            f"\nWithin the config file:"
            f"\n1. Check the weight and disjoint coefficient values."
            f"\n2. Check mutation coefficients' values  "
        )


def update_distances_cache(config: Config, module: NeatModule, genome_cache: GenomeDistanceCache,
                           tpb=10, verbose: int = None):
    total_distance = np.zeros((module.genome_num, module.genome_num), genome_cache.total_distance.dtype)

    def reshape(tensor: Union[Tensor, CPUArray], required_ndim: int):
        original_shape: tuple[int, ...] = tensor.shape
        if tensor.ndim > required_ndim:
            tensor = tensor.reshape(*tensor.shape[:required_ndim-1], -1)
        elif tensor.ndim < required_ndim:
            for _ in range(required_ndim-tensor.ndim):
                if isinstance(tensor, Tensor):
                    tensor = tensor.unsqueeze(-1)
                elif isinstance(tensor, CPUArray):
                    tensor = np.expand_dims(tensor, -1)
                else:
                    raise ValueError(f"Unsupported dtype = {type(tensor)}")
        return tensor, original_shape

    ts = clock.perf_counter()
    for param in module.neat_parameters():
        dtype = param.data.dtype if param.data.dtype != torch.bfloat16 else torch.float32
        array = np.asarray(reshape(param.data.to(dtype), 2)[0])

        try:
            ret_code = get_distance(
                array, total_distance,
                config.genome.compatibility_weight_coefficient,
                config.genome.compatibility_disjoint_coefficient
            )
            if ret_code == 1: raise ValueError(f"Expected a 2D array, got {array.ndim}")
            if ret_code == 2: raise ValueError("Out of bounds")
        except Exception as e:
            print(param.dtype, param.device, array.shape, param.data.shape)
            raise e

        # Release memory
        del array

    if verbose and verbose >= 2:
        print(f"{CM('Ran distance kernel', Fore.CYAN)} in {round(clock.perf_counter() - ts, 2)} s")

    _check_distinct_genomes(total_distance, module.mapping)
    genome_cache.mapping = Dict(module.mapping.items())
    genome_cache.total_distance = total_distance

    # Remove GPU data
    gc.collect()


@njit(nogil=True)
# NOTE: _get_representatives/_get_species/_update_collection below are left as
#       nogil rather than parallel=True: unlike crossover()/mutate()/_execute()/
#       get_distance(), these mutate shared dicts (representatives, members,
#       species) and, in _get_species, hand out new species keys from a single
#       running counter and call unspeciated.remove(...) - all inherently
#       sequential/order-dependent, not an embarrassingly-parallel per-element
#       grid like the array kernels above.
# TODO: Can be parallelized like update_dict above. Though caching may not be possible due to custom classes
def _get_representatives(species_dict: dict[int, Species], population: dict[int, Genome], unspeciated: list[int],
                         representatives: dict[int, int], members: dict[int, list[int]],
                         distance_cache: GenomeDistanceCache):
    def gamma(candidates_: list[tuple[float, Genome]]):
        if len(candidates_) > 0:
            candidate = candidates_[0] # (distance, genome)
            for x in prange(1, len(candidates_)):
                comp = candidates_[x]
                if comp[0] < candidate[0]:
                    candidate = comp
            return candidate
        else:
            raise ValueError(f"Empty list")

    # Loop through each existing species
    mapping = list(species_dict.keys())
    for i in prange(len(species_dict)):
        sid = mapping[i]
        species = species_dict[sid]
        # Loop through unspeciated genomes
        candidates: list[tuple[float, Genome]] = []
        for j in prange(len(unspeciated)):
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
            for x in prange(1, len(candidates_)):
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
    for idx in prange(len(unspeciated)):
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
    for i in prange(len(representatives)):
        sid = mapping[i]
        rid = representatives[sid]
        specie = species.get(sid)
        if specie is None:
            specie = Species(sid, generation, genus)
            species[sid] = specie

        members_ = members[sid]
        for j in prange(len(members_)):
            gid = members_[j]
            genome_to_species[gid] = sid

        member_dict = {gid: population[gid] for gid in members_}
        specie.update(population[rid], member_dict)


def speciate(config: Config, genera: list[int], modules: dict[int, NeatModule],
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
        update_distances_cache(config, module, distances_cache, tpb=tpb, verbose=verbose)
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