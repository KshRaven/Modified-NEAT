"""
Reproduction, written once for every device.

* ``compute_spawn`` / ``select_parents`` / ``create_children`` are numba CPU functions that only touch genome objects,
  so they are identical for both devices (the GPU only ever accelerates the parameter arrays).
* ``update_children`` builds the parameter arrays of the new population and runs crossover and mutation through the
  device :class:`~ModifiedNEAT.base.backend.Backend`.
* ``reproduce`` drives a whole generation.
"""

from ModifiedNEAT.nn.base import NeatModule, check_for_illegal_zeros
from ModifiedNEAT.nn.genome import Genome
from ModifiedNEAT.config import Config
from ModifiedNEAT.species import Species, SpeciesSet, FLOAT, INT, SPECIES
from ModifiedNEAT.stagnation import Stagnation
from ModifiedNEAT.reporter.base import ReporterSet
from ModifiedNEAT.base.backend import Backend
from ModifiedNEAT.base.functional import calc_grid, threads_total
from ModifiedNEAT.base.mutation import mutation_arguments
from ModifiedNEAT.base.util import (
    init_type_code, parameter_dtype, reshape_to_ndim, check_param_compatibility,
    effective_epsilon, clamp_to_epsilon, repair_zero_sources
)
from ModifiedNEAT.util.fancy_text import CM, Fore

from numba import njit, types, prange
from numba.typed import List, Dict

import torch
import numpy as np
import time as clock
import math

NP_FLOAT = float # np.float64
GENOME   = Genome.class_type.instance_type
GENOME_LIST = types.ListType(GENOME)


@njit(parallel=True) # , cache=True)
def compute_spawn(adjusted_fitness: list[float], previous_sizes: list[int], pop_size: int, min_species_size: int,
                  purge: int, generation: int):
    """Compute the proper number of offspring per species (proportional to fitness)."""
    if len(adjusted_fitness) != len(previous_sizes):
        raise ValueError(f"Mismatch in reproduction data")

    af_sum = sum(adjusted_fitness)
    # NOTE: pre-sized and written by index (instead of .append()) so each prange
    #       iteration only ever touches its own slot - .append() from a real
    #       parallel loop would race and scramble the species <-> amount alignment.
    spawn_amounts = [0 for _ in range(len(adjusted_fitness))]

    for idx in prange(len(adjusted_fitness)):
        af = adjusted_fitness[idx]
        ps = previous_sizes[idx]
        if af_sum > 0:
            s = max(float(min_species_size), af / af_sum * pop_size)
        else:
            s = min_species_size

        d = (s - ps) * 0.5
        c = int(round(d))
        spawn_amount = ps
        if abs(c) > 0:
            spawn_amount += c
        elif d > 0:
            spawn_amount += 1
        elif d < 0:
            spawn_amount -= 1

        spawn_amounts[idx] = spawn_amount

    # Normalize the spawn amounts so that the next generation is roughly the population size requested by the user.
    total_spawn = sum(spawn_amounts)
    norm = pop_size / total_spawn
    for idx in prange(len(spawn_amounts)):
        spawn_amounts[idx] = max(min_species_size, int(round(spawn_amounts[idx] * norm)))

    # Limit to min_species_size when enabled
    if purge > 0:
        if generation % purge == 0:
            for idx in prange(len(spawn_amounts)):
                spawn_amounts[idx] = min_species_size

    return spawn_amounts


@njit(parallel=True)
def select_parents(pool_fitness, pool_foreign, pool_keys, spawn, clone_cutoff, coeff_cross, coeff_fitness,
                   first_parents, second_parents):
    """
    Array based parent selection (same scoring as the former per-child `choice`): the fitness/genus factors are
    computed once per pool and every child only draws its random numbers, so children are independent and can be
    chosen in parallel. Genome objects are only touched afterwards, serially.
    """
    size = len(pool_fitness)
    coeff_cross = max(-1.0, min(1.0, coeff_cross))
    coeff_fitness = max(-1.0, min(1.0, coeff_fitness))
    global_max = np.max(pool_fitness)
    global_min = np.min(pool_fitness)
    diff = global_max - global_min

    base = np.zeros(size, np.float64)
    for i in range(size):
        if coeff_fitness != 0 and global_max != global_min:
            factor = (pool_fitness[i] - global_min) / diff * coeff_fitness
            if coeff_cross > 0 and pool_foreign[i]:
                factor *= coeff_cross
            if coeff_cross < 0 and not pool_foreign[i]:
                factor *= 1 - coeff_cross
        else:
            factor = 0.0
        base[i] = factor

    for s in prange(spawn):
        probabilities = np.random.rand(size)
        best = np.argmax(probabilities + base)
        first_parents[s] = best
        if s < clone_cutoff:
            second_parents[s] = best
        else:
            probabilities = np.random.rand(size)
            full = np.empty(size, np.float64)
            for i in range(size):
                p = probabilities[i]
                f = base[i]
                if pool_keys[i] == pool_keys[best]:
                    p *= 0.1
                    f *= 0.0
                full[i] = p + f
            second_parents[s] = np.argmax(full)


@njit(nogil=True)
def create_children(genus: int, genus_population: dict[int, Genome], population: dict[int, Genome], species: dict[int, Species],
                    available_gid: int, spawn_amounts: list[int], remaining_species: list[Species], to_delete: list[int],
                    elitism: float, survival_threshold: float, clone_threshold: float, cross_threshold: float,
                    cross_multiplier: float, darwin_multiplier: float,
                    criteria: str, ancestors: dict[int, tuple[Genome, Genome]], equal_params: bool, preserve: bool):
    if len(spawn_amounts) != len(remaining_species):
        raise ValueError(f"Mismatch in reproduction data")

    # TODO: Fix and re-enable
    def fix_infinities(specie_population: list[Genome], full_population: list[Genome], ufp: bool = False):
        if np.any(np.array([g.fitness is None for g in specie_population])):
            raise ValueError(f"A genome's fitness has not been set")
        fitnesses = np.zeros(len(specie_population if not ufp else full_population), NP_FLOAT)
        fitnesses[:] = np.array([g.fitness for g in (specie_population if not ufp else full_population)])
        inf_fit_vals = np.isinf(fitnesses)
        all_infinite = np.all(inf_fit_vals).item()
        if all_infinite:
            if not ufp:
                ufp = True
                specie_keys = [g.key for g in specie_population]
                for g in full_population:
                    if g.key not in specie_keys:
                        ufp = False
                        break
            if ufp:
                raise ValueError(f"Entire population has infinite fitness values")
            else:
                return False
        _min, _max = np.min(fitnesses[~inf_fit_vals]).item(), np.max(fitnesses[~inf_fit_vals]).item()
        # _range = _max - _min
        # if _range == 0.0:
        #     _range = abs(_max)
        # _half_range = _range / 2
        for i, g in enumerate(specie_population):
            # g.fitness = _min - _half_range if fitnesses[i] < 0.0 else _max + _half_range
            g.fitness = _min if fitnesses[i] < 0.0 else _max
        return True

    def get_limits(genomes: list[Genome]) -> tuple[int, int]:
        if np.any(np.array([g.fitness is None for g in genomes])):
            raise ValueError(f"A genome's fitness has not been set")
        fitnesses = np.zeros(len(genomes), NP_FLOAT)
        fitnesses[:] = np.array([g.fitness for g in genomes])
        inf_fit_vals = np.isinf(fitnesses)
        has_nan, all_inf = np.any(np.isnan(fitnesses)), np.all(inf_fit_vals)
        if has_nan or all_inf:
            verdict = "Has NaN fitness value" if has_nan else "All fitnesses are Inf"
            raise ValueError(f"Invalid fitness value within specie's members. {verdict}")
        _min, _max = np.min(fitnesses[~inf_fit_vals]).item(), np.max(fitnesses[~inf_fit_vals]).item()
        _range = _max - _min
        if _range == 0.0:
            _range = abs(_max)
        _red_range = _range * 0.10
        fitnesses[inf_fit_vals & (fitnesses < 0.0)] = _min - _red_range
        fitnesses[inf_fit_vals & (fitnesses > 0.0)] = _max + _red_range
        _minimum, _maximum = np.min(fitnesses).item(), np.max(fitnesses).item()
        return _minimum, _maximum

    def sort(members: dict[int, Genome], crit: str) -> list[Genome]:
        genomes = List(members.values())
        genomes_total = len(genomes)

        # Stable O(n log n) argsort (same ordering as the previous stable bubble sort)
        keys = np.zeros(genomes_total, np.float64)
        if crit == 'min':
            for i in range(genomes_total):
                keys[i] = genomes[i].fitness
        elif crit == 'max':
            for i in range(genomes_total):
                keys[i] = -genomes[i].fitness
        elif crit == 'mean':
            # Sort by distance to mean fitness (mean is constant for the whole sort)
            mean_fitness = 0.0
            for i in range(genomes_total):
                mean_fitness += genomes[i].fitness
            mean_fitness = mean_fitness / max(genomes_total, 1)
            for i in range(genomes_total):
                keys[i] = abs(genomes[i].fitness - mean_fitness)
        order = np.argsort(keys, kind='mergesort')
        result = List.empty_list(GENOME)
        for i in range(genomes_total):
            result.append(genomes[order[i]])
        return result

    # Populate global genera members
    genera: dict[int, list[Genome]] = Dict.empty(INT, GENOME_LIST)
    for genome in population.values():
        if genome.genus not in genera:
            genera[genome.genus] = List.empty_list(GENOME)
        genera[genome.genus].append(genome)

    # Foreign-genus members are identical for every species, so they are sorted once outside the species loop.
    extra_members: list[Genome] = List.empty_list(GENOME)
    extra_min, extra_max = 0.0, 0.0
    if cross_threshold > 0:
        for gn, genus_members in genera.items():
            if gn != genus and len(genus_members) > 0:
                # Sort genus members in descending order and keep the best n members
                genus_members = sort({g.key: g for g in genus_members}, criteria)
                cross_cutoff = math.ceil(cross_threshold * len(genus_members))
                extra_members.extend(genus_members[:cross_cutoff])
    if len(extra_members) > 1:
        extra_members = sort({g.key: g for g in extra_members}, criteria)
        if len(remaining_species) > 0:
            extra_min, extra_max = get_limits(extra_members)

    for idx in range(len(remaining_species)):
        spawn  = spawn_amounts[idx]
        specie = remaining_species[idx]
        # If elitism is enabled, each species always at least gets to retain its elites.
        elite_total = max(1, math.ceil(elitism * len(specie.members)))
        spawn = max(spawn, elite_total + 1)
        assert spawn >= 2

        # Get fitness limits
        sg, ag = List(specie.members.values()), List(population.values())
        # if not fix_infinities(sg, ag):
        #     fix_infinities(sg, ag, True)
        minimum, maximum = get_limits(sg)

        # Delete unwanted members
        executions = [gid for gid in specie.members.keys() if gid in to_delete]
        execution_count = 0
        # The species has at least one member for the next generation, so retain it.
        for member in sort(specie.members, criteria)[::-1]:
            purge_limit_reached = not preserve or (len(specie.members) - execution_count <= elite_total and preserve)
            if not purge_limit_reached and len(specie.members) > 1:
                if member.key in executions:
                    del specie.members[member.key]
                    execution_count += 1
            else:
                break
        # Sort members in order of descending fitness.
        old_members = sort(specie.members, criteria)
        # Clear specie's members
        specie.members = Dict.empty(INT, GENOME)
        species[specie.key] = specie

        # Transfer elites to new generation.
        if elite_total > 0:
            for m in old_members[:elite_total]:
                if m.key not in executions:
                    genus_population[m.key] = m
                    spawn -= 1

        if spawn <= 0:
            continue

        # Only use the survival threshold fraction to use as parents for the next generation.
        member_count = len(old_members)
        repro_cutoff = max(2, math.ceil(survival_threshold * member_count))
        clone_cutoff = max(0, math.ceil(min(clone_threshold, 0.75) * spawn))
        # Use at least two parents no matter what the threshold fraction result is.
        old_members = old_members[:repro_cutoff]

        if len(extra_members) > 1:
            minimum, maximum = min(minimum, extra_min), max(maximum, extra_max)
            
        # NOTE:
        #   survival_threshold: Fraction of old population used to produce offspring for the next generation.
        #   cross_threshold: Fraction of each other genus' population used to produce offspring for the next generation.
        #   elitism: Fraction of old population that is retained in the next generation.
        #   clone_threshold: Fraction of remaining spawn produced through direct cloning instead of crossover.

        # Randomly choose parents and produce the number of offspring allotted to the species.
        pool = List(list(old_members) + list(extra_members)) # TODO: Numba currently cannot add typed lists
        pool_size = len(pool)
        pool_fitness = np.empty(pool_size, np.float64)
        pool_foreign = np.zeros(pool_size, np.bool_)
        pool_keys    = np.empty(pool_size, np.int64)
        for pi in range(pool_size):
            pg = pool[pi]
            pool_fitness[pi] = pg.fitness if pg.fitness is not None else -np.inf
            pool_foreign[pi] = pg.genus != genus
            pool_keys[pi]    = pg.key
        first_parents  = np.zeros(spawn, np.int64)
        second_parents = np.zeros(spawn, np.int64)
        # NOTE: Previously, at least one genome has to be from the same genus in order to preserve genus integrity.
        # Now you can fill current genus with foreign ones with hopes that they get filtered out by fitness if they do not perform
        select_parents(
            pool_fitness, pool_foreign, pool_keys, spawn, clone_cutoff, cross_multiplier, darwin_multiplier,
            first_parents, second_parents
        )
        for spawn_index in range(spawn):
            # Cloning when parent2 == parent1, interbreeding or crossbreeding otherwise
            parent1: Genome = pool[first_parents[spawn_index]]
            parent2: Genome = pool[second_parents[spawn_index]]

            # Note that if the parents are not distinct, crossover will produce a genetically identical clone of the parent (but with a different ID).
            gid = available_gid
            available_gid += 1
            child = Genome(gid, genus)
            # child.update_from_build()
            genus_population[gid] = child
            ancestors[gid] = (parent1, parent2)
    return available_gid


def update_children(
        backend: Backend, config: Config, genus: int, modules: dict[int, NeatModule],
        old_population: dict[int, Genome], new_population: dict[int, Genome],
        ancestors: dict[int, tuple[Genome, Genome]], tpb=1, seed: int = None, verbose: int = None
):
    xp = backend.xp
    for m in modules[genus].neat_modules():
        m.updated = False

    init_type = init_type_code(config)

    genus_mapping = {g: index for index, g in enumerate(modules.keys())}
    old_mapping = modules[genus].mapping
    sources = {}
    genera  = {}
    emergency_keys = [key for key, genome in new_population.items() if key in old_population and genome.genus == genus]
    if len(emergency_keys) == 0:
        emergency_keys = [key for key, genome in old_population.items() if genome.genus == genus]
    emergency_prob = torch.softmax(torch.tensor([old_population[key].fitness for key in emergency_keys]), -1).numpy()
    emergency_choices = np.random.choice(emergency_keys, size=len(new_population), p=emergency_prob)
    try:
        for index, (key, genome) in enumerate(new_population.items()):
            emeg_key = emergency_choices[index]
            if genome.key in old_population:
                sources[index] = (old_mapping[key], old_mapping[key], old_mapping[key])
                # NOTE: genera must store RAW genus IDs, not genus_mapping indices.
                # crossover() only ever compares these against the raw genus ids it
                # receives (see the `genus_` loop below), never uses them to index
                # into an array. Mapping them here made the equality checks inside
                # crossover() fail whenever a genus's id didn't coincide with its
                # position in modules.keys() (e.g. after an earlier genus went
                # extinct and the enumeration shifted) - silently leaving
                # array_update at its 0.0 fill value for that genus.
                genera[index]  = (genome.genus, genome.genus, genus)
            else:
                parent1, parent2 = ancestors[key]
                if parent1.fitness < parent2.fitness:
                    parent1, parent2 = parent2, parent1
                mapping1, mapping2 = modules[parent1.genus].mapping, modules[parent2.genus].mapping
                sources[index] = (mapping1[parent1.key], mapping2[parent2.key], old_mapping[emeg_key])
                genera[index]  = (parent1.genus, parent2.genus, genus)
    except Exception as e:
        for m in modules.values():
            print(m.mapping)
        print(list(new_population.keys()))
        print(genus_mapping)
        print(f"Current genus = {genus}")
        raise e
    sources = backend.asarray(np.array([list(g) for g in sources.values()]))
    genera  = backend.asarray(np.array([list(m) for m in genera.values()]))
    child_filter = backend.asarray(np.array([
        True if gid not in old_population else False for gid in new_population.keys()
    ]))

    genera_parameters = [m.neat_parameters() for m in modules.values()]
    pgs = list(zip(*genera_parameters))
    equal_param_num = all([len(c) == len(genera_parameters[0]) for c in genera_parameters])

    valid_perc = np.count_nonzero([check_param_compatibility(pg) for pg in pgs])
    if verbose and verbose >= 2:
        print(f"{CM('Valid Count', Fore.CYAN)} = {valid_perc} / {len(pgs)}")
    if not equal_param_num:
        # TODO: Work on this section
        raise NotImplementedError()

    mutation_planes = 2 if config.genome.single_structural_mutation else 3
    fallback_runs = 0
    # NOTE: Parameters are handled one at a time on purpose, so that a single parameter can be as large as the
    #       device's dedicated memory allows.
    for genus_param, param_group in zip(modules[genus].neat_parameters(), pgs):
        dtype = parameter_dtype(genus_param.data.dtype)
        eps_eff = effective_epsilon(config, dtype)
        # ------------------------------ Sending data to the device ------------------------------ #
        array_update, original_shape = reshape_to_ndim(
            backend.zeros((len(new_population), *genus_param.original_shape)), 3, xp
        )
        array_sources = tuple([backend.asarray(reshape_to_ndim(p.data.to(dtype), 3)[0]) for p in param_group])
        # ------------------------------ Define crossover kernel and randomizer values ------------------------------ #
        genome_num = len(new_population)
        kernel_shape = calc_grid(genome_num, *array_update.shape[1:3], tpb=tpb)
        # One RNG call for the crossover plane and all mutation planes (row 0 = crossover, rows 1.. = mutation)
        rng_planes = backend.uniform((1 + mutation_planes, threads_total(kernel_shape)))
        probabilities = rng_planes[0]
        # ------------------------------ Run crossover using sources ------------------------------ #
        equal_params = check_param_compatibility(param_group)

        # Attempt 1 uses the device kernel. Exact-zero parents are repaired and the run repeated; if nothing is
        # repairable and the backend has a fallback implementation, it is used once as a cross-check before giving up.
        attempts = 2
        use_fallback = False
        fallback_cache = {}
        while attempts > 0:
            attempts -= 1
            array_update[...] = 0  # clean slate for every attempt
            filled = xp.zeros(array_update.shape + (sources.shape[-1],), bool)  # freed per parameter (low peak memory)
            updates_total = 0
            for genus_, source_ in zip(genus_mapping.keys(), array_sources):
                if config.reproduction.cross_threshold == 0.0 and genus_ != genus:
                    continue
                updates_total += 1
                if not use_fallback:
                    backend.crossover(
                        genus_, source_, array_update, sources, genera, filled, probabilities, equal_params,
                        kernel_shape
                    )
                else:
                    fallback_runs += 1
                    backend.crossover_fallback(
                        genus_, source_, array_update, sources, genera, filled, probabilities, equal_params,
                        fallback_cache
                    )

            # There shouldn't be any zero values when epsilon or weight deletion is enabled for parameters
            try:
                check_for_illegal_zeros(config, array_update, genus_param, modules)
            except Exception as e:
                repaired = repair_zero_sources(array_sources, eps_eff)
                if repaired:
                    if verbose and verbose >= 1:
                        print(f"Repaired {repaired} exact-zero parent value(s) for param_index = {genus_param.param_index}")
                    attempts = max(attempts, 1)  # always allow one clean re-run after a repair
                    continue
                if attempts > 0 and backend.has_fallback and not use_fallback:
                    use_fallback = True
                    continue
                print(f"Attempts left = {attempts}, fallback runs = {fallback_runs}")
                print(f"updates total = {updates_total} for param_index = {genus_param.param_index}")
                print(f"indices = {backend.to_host(xp.where(xp.any(array_update == 0, axis=(1, 2)))[0]).tolist()[:20]}")
                print(f"zeros total = {int(xp.sum(xp.abs(array_update) == 0))}")
                print(f"sources = {[s.shape for s in array_sources]}")
                print(f"update = {array_update.shape}, probabilities = {probabilities.shape}")
                print(f"genus = {genus}, kernel shape = {kernel_shape}")
                raise e
            break

        if verbose and verbose >= 4:
            genus_param.cd = backend.to_host(array_update.copy())
        # ------------------------------ Remove source data from the device ------------------------------ #
        del array_sources
        # ------------------------------ Define mutation kernel and randomizer values ------------------------------ #
        probabilities = rng_planes[1:]
        normals = backend.normal_states(kernel_shape, seed)[0]
        # ------------------------------ Run mutation using configuration ------------------------------ #
        backend.mutate(
            array_update, child_filter, *mutation_arguments(config, init_type, eps_eff), probabilities, normals,
            kernel_shape=kernel_shape
        )
        zero_values = clamp_to_epsilon(xp, array_update, eps_eff)
        if verbose and verbose >= 4:
            genus_param.md = backend.to_host(array_update.copy()) - genus_param.cd
        # ------------------------------ Update parameters ------------------------------ #
        update = array_update.reshape(*original_shape)
        genus_param.update(new_population, update, True)
        # ------------------------------ Remove data from the device ------------------------------ #
        del probabilities, rng_planes, normals, array_update, update, filled, zero_values

    # Free memory
    del sources, child_filter
    backend.release()


def reproduce(
        backend: Backend,
        config: Config, genera: list[int], modules: dict[int, NeatModule], population: dict[int, Genome],
        ancestors: dict[int, tuple[Genome, Genome]], generation: int, to_delete: list[int], genome_indexer: int,
        stagnation: Stagnation, species_set: SpeciesSet, reporters: ReporterSet,
        tpb=1, verbose: int = None):
    assert all([module.genus in genera for module in modules.values()])
    # Filter out stagnated species, collect the set of non-stagnated
    # species members, and compute their average adjusted fitness.

    population_genera: list[dict[int, Genome]] = [
        {g.key: g for g in population.values() if g.genus == genus} for genus in genera
    ]
    modules_genera: list[NeatModule] = [modules[genus] for genus in genera]
    new_population: dict[int, Genome] = Dict.empty(INT, GENOME)
    new_genomes: dict[int, dict[int, Genome]] = {}
    for genomes, module, genus in zip(population_genera, modules_genera, genera):
        assert module.genus == genus

        remaining_species: list[Species] = List.empty_list(SPECIES)
        all_fitnesses: list[float]       = List.empty_list(FLOAT)
        for sid, specie, stagnant in stagnation.update(genus, species_set, generation):
            if stagnant and verbose:
                reporters.species_stagnant(sid, specie)
            else:
                all_fitnesses.extend([FLOAT(m.fitness) for m in specie.members.values()])
                remaining_species.append(specie)

        # No species left.
        if not remaining_species:
            for key in list(species_set.species.keys()):
                specie = species_set.species.get(key)
                if specie.genus == genus:
                    del species_set.species[key]

        # Find minimum/maximum fitness across the entire population, for use in species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for specie in remaining_species:
            # Compute adjusted fitness.
            mean_fitness = np.mean([m.fitness for m in specie.members.values()]).item()
            adj_fitness = (mean_fitness - min_fitness) / fitness_range
            specie.adjusted_fitness = adj_fitness

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = np.mean(adjusted_fitnesses)
        if verbose:
            reporters.info(f"Average adjusted fitness: {avg_adjusted_fitness:.3f}")

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = config.reproduction.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size, self.reproduction_config.elitism)?
        # That would probably produce more accurate tracking of population sizes and relative fitnesses... doing.
        # TODO: document.
        min_species_size = max(min_species_size, max([np.ceil(config.reproduction.elitism * len(s.members)) for s in remaining_species]))
        # TODO: add pop size to arguments or just get from current population?
        pop_size = math.ceil(len(population) / len(genera))

        spawn_amounts = compute_spawn(adjusted_fitnesses, previous_sizes, pop_size, min_species_size,
                                      config.reproduction.purge, generation)

        genus_population: dict[int, Genome] = Dict.empty(INT, GENOME)

        for key, specie in list(species_set.species.items()):
            if specie.genus == genus:
                del species_set.species[key]
        ts = clock.perf_counter()
        # print(f"Current genome index = {genome_indexer}")
        genome_indexer = create_children(
            genus, genus_population, population, species_set.species, genome_indexer, spawn_amounts, remaining_species,
            to_delete, config.reproduction.elitism,
            config.reproduction.survival_threshold, config.reproduction.clone_threshold, config.reproduction.cross_threshold,
            config.reproduction.cross_multiplier, config.reproduction.darwin_multiplier,
            config.general.fitness_criterion, ancestors, False, config.reproduction.preserve_elite
        )
        update_children(
            backend, config, genus, modules, population, genus_population, ancestors, tpb, config.general.seed, verbose
        )
        if verbose and verbose >= 2:
            print(f"{CM('Spawned new genomes in ', Fore.CYAN)} in {round(clock.perf_counter() - ts, 2)}s")

        # Update genus
        for genome in genus_population.values():
            new_population[genome.key] = genome
        new_genomes[genus] = genus_population

        pass

    # Update modules
    for genus, module in modules.items():
        module.update(new_genomes[genus], None, True)
        module.update_limit()
        for m in module.neat_modules():
            m.update_limit()

    # Update population
    for key in list(population.keys()):
        del population[key]
    for genome in new_population.values():
        population[genome.key] = genome

    return genome_indexer
