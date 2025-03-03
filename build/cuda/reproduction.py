
from build.nn.base import NeatModule
from build.nn.genome import Genome
from build.config import Config
from build.species import Species, SpeciesSet, FLOAT, INT, SPECIES
from build.stagnation import Stagnation
from build.reporter.base import ReporterSet
from build.cuda.functional import calc_grid, prob, normal, clamp, get_rng_states, get_value, set_value
from build.cuda.initialization import initialize_genome
from build.util.fancy_text import CM, Fore

from numba import njit, types, prange, cuda
from numba.typed import List, Dict
from numba.cuda.cudadrv.devicearray import DeviceNDArray as GPUArray
from torch import Tensor

import torch
import numpy as np
import time as clock

NP_FLOAT = types.float64
GENOME   = Genome.class_type.instance_type


@njit(nogil=True)
def compute_spawn(adjusted_fitness: list[float], previous_sizes: list[int], pop_size: int, min_species_size: int,
                  purge: int, generation: int):
    """Compute the proper number of offspring per species (proportional to fitness)."""
    if len(adjusted_fitness) != len(previous_sizes):
        raise ValueError(f"Mismatch in reproduction data")

    af_sum = sum(adjusted_fitness)
    spawn_amounts = []

    for idx in prange(len(adjusted_fitness)):
        af = adjusted_fitness[idx]
        ps = previous_sizes[idx]
        if af_sum > 0:
            s = max(min_species_size, af / af_sum * pop_size)
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

        spawn_amounts.append(spawn_amount)

    # Normalize the spawn amounts so that the next generation is roughly
    # the population size requested by the user.
    total_spawn = sum(spawn_amounts)
    norm = pop_size / total_spawn
    spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

    # Limit to min_species_size when enabled
    if purge > 0:
        if generation % purge == 0:
            spawn_amounts = [min_species_size for _ in spawn_amounts]

    return spawn_amounts


@njit(nogil=True)
def create_children(new_population: dict[int, Genome], species: dict[int, Species], available_gid: int,
                    spawn_amounts: list[int], remaining_species: list[Species], to_delete: list[int],
                    elitism: int, survival_threshold: float, darwin_multiplier: float, criteria: str,
                    ancestors: dict[int, tuple[Genome, Genome]]):
    if len(spawn_amounts) != len(remaining_species):
        raise ValueError(f"Mismatch in reproduction data")

    def sort(members: dict[int, Genome], crit: str) -> list[Genome]:
        genomes = List(members.values())
        genomes_total = len(genomes)

        # Bubble sort
        for i in range(genomes_total):
            for j in range(0, genomes_total - i - 1):
                if crit == 'min':
                    if genomes[j].fitness > genomes[j + 1].fitness:
                        genomes[j], genomes[j + 1] = genomes[j + 1], genomes[j]
                elif crit == 'max':
                    if genomes[j].fitness < genomes[j + 1].fitness:
                        genomes[j], genomes[j + 1] = genomes[j + 1], genomes[j]
                elif crit == 'mean':
                    # Sort by distance to mean fitness
                    mean_fitness = sum([g.fitness for g in genomes]) / len(genomes)
                    dist_j = abs(genomes[j].fitness - mean_fitness)
                    dist_j1 = abs(genomes[j + 1].fitness - mean_fitness)
                    if dist_j > dist_j1:
                        genomes[j], genomes[j + 1] = genomes[j + 1], genomes[j]
        return genomes

    def choice(genomes: list[Genome], multiplier: float):
        if multiplier is None:
            multiplier = 1
        factors = np.array([g.fitness for g in genomes])
        maximum = np.max(factors)
        minimum = np.min(factors) + 1e-8
        probabilities = np.full(len(genomes), 0.0)
        for i, p in enumerate(factors):
            probabilities[i] = (p - minimum) / (maximum - minimum) * np.random.rand() * multiplier
        return genomes[np.argmax(probabilities)]

    for idx in range(len(remaining_species)):
        spawn  = spawn_amounts[idx]
        specie = remaining_species[idx]
        # If elitism is enabled, each species always at least gets to retain its elites.
        spawn = max(spawn, elitism)
        assert spawn > 0

        # Delete unwanted members
        executions = [gid for gid in specie.members.keys() if gid in to_delete]
        if len(specie.members.keys()) - len(executions) > 1:
            for gid in executions:
                if gid in to_delete and len(specie.members) > 1:
                    del specie.members[gid]
        # The species has at least one member for the next generation, so retain it.
        # old_members: list[Genome] = List(specie.members.values())
        # Sort members in order of descending fitness.
        old_members = sort(specie.members, criteria)
        # Clear specie's members
        specie.members = Dict.empty(INT, GENOME)
        species[specie.key] = specie

        # Transfer elites to new generation.
        if elitism > 0:
            for m in old_members[:elitism]:
                new_population[m.key] = m
                spawn -= 1

        if spawn <= 0:
            continue

        # Only use the survival threshold fraction to use as parents for the next generation.
        repro_cutoff = max(2, int(np.ceil(survival_threshold * len(old_members))))
        # Use at least two parents no matter what the threshold fraction result is.
        old_members = old_members[:repro_cutoff]
        # TODO: Enable probabilities when numba supports prob in numpy.random.choice()

        # Randomly choose parents and produce the number of offspring allotted to the species.
        for _ in prange(spawn):
            parent1: Genome = choice(old_members, darwin_multiplier)
            parent2: Genome = choice(old_members, darwin_multiplier)

            # Note that if the parents are not distinct, crossover will produce a
            # genetically identical clone of the parent (but with a different ID).
            gid = available_gid
            available_gid += 1
            # print(gid)
            child = Genome(gid)
            # child.update_from_build()
            new_population[gid] = child
            ancestors[gid] = (parent1, parent2)
            spawn -= 1
    return available_gid


@cuda.jit(device=True)
def _crossover(value1: float, value2: float, states: GPUArray, index: int):
    if prob(states, index) > 0.5:
        return value1
    else:
        return value2


@cuda.jit
def crossover(source: GPUArray, updates: GPUArray, build: GPUArray, rng_states: GPUArray):
    genome_idx, x, y = cuda.grid(3)
    # Parameter shape (genomes, *spatial_dims)
    g_lim = updates.shape[0]
    x_lim = 1 if updates.ndim <= 1 else updates.shape[1]
    y_lim = 1 if updates.ndim <= 2 else updates.shape[2]
    s_g, s_x, s_y = cuda.gridsize(3)

    # Linearized thread index
    rng_index = (y * s_x * s_g) + (x * s_g) + genome_idx

    if genome_idx < g_lim and x < x_lim and y < y_lim:
        parent1, parent2 = build[genome_idx]
        if parent1 == parent2:
            value = get_value(source, parent1, x, y)
        else:
            value1 = get_value(source, parent1, x, y)
            value2 = get_value(source, parent2, x, y)
            value = _crossover(value1, value2, rng_states, rng_index)

        set_value(updates, genome_idx, x, y, value)


@cuda.jit(device=True)
def mutate_genome(parameter: GPUArray, g: int, x: int, y: int, mutate_rate: float, mutate_power: float,
                  replace_rate: float, init_type: str, mean: float, std: float, minimum: float, maximum: float,
                  rng_states: GPUArray, rng_index: int):
    r = prob(rng_states, rng_index)
    if r < mutate_rate:
        value = clamp(get_value(parameter, g, x, y) + normal(rng_states, rng_index, 0., mutate_power), minimum, maximum)
        set_value(parameter, g, x, y, value)
    elif r < replace_rate + mutate_rate:
        initialize_genome(parameter, g, x, y, init_type, mean, std, minimum, maximum, rng_states, rng_index)


@cuda.jit
def mutate(
        updates: GPUArray, children: GPUArray, mutate_rate: float, mutate_power: float, replace_rate: float,
        init_type: str, mean: float, std: float, minimum: float, maximum: float, rng_states: GPUArray,
        # debugging: GPUArray
):
    genome_idx, x, y = cuda.grid(3)
    # Parameter shape (genomes, *spatial_dims)
    g_lim = updates.shape[0]
    x_lim = 1 if updates.ndim <= 1 else updates.shape[1]
    y_lim = 1 if updates.ndim <= 2 else updates.shape[2]
    s_g, s_x, s_y = cuda.gridsize(3)

    # Linearized thread index
    rng_index = (y * s_x * s_g) + (x * s_g) + genome_idx

    if genome_idx < g_lim and x < x_lim and y < y_lim:
        if children[genome_idx] is True:
            mutate_genome(updates, genome_idx, x, y, mutate_rate, mutate_power, replace_rate, init_type,
                          mean, std, minimum, maximum, rng_states, rng_index)


def update_children(
        config: Config, module: NeatModule, old_population: dict[int, Genome], new_population: dict[int, Genome],
        ancestors: dict[int, tuple[Genome, Genome]], tpb=10, seed: int = None, verbose: int = None
):
    for m in module.neat_modules():
        m.updated = False

    if config.genome.init_type == 'normal':
        init_type = 0
    elif config.genome.init_type == 'uniform':
        init_type = 1
    else:
        raise NotImplementedError(f"Unsupported NEAT Genome init type '{config.genome.init_type}'")

    old_mapping = module.mapping
    sources = {}
    for index, (key, genome) in enumerate(new_population.items()):
        if genome.key in old_population:
            sources[index] = (old_mapping[key], old_mapping[key])
        else:
            parent1, parent2 = ancestors[key]
            if parent1.fitness < parent2.fitness:
                parent1, parent2 = parent2, parent1
            sources[index] = (old_mapping[parent1.key], old_mapping[parent2.key])
    updates: dict[int, Tensor] = {}
    sources = cuda.to_device(np.array([list(g) for g in sources.values()]))
    child_filter = cuda.to_device(np.array([
        True if gid not in old_population else False for gid in new_population.keys()
    ]))
    max_pop_size = max(len(old_population), len(new_population))

    for param in module.neat_parameters():
        # with cuda.defer_cleanup():
        array_source = param.data.cpu().numpy()
        array_update = np.zeros((len(new_population), *param.original_shape))
        as_shape, au_shape = array_source.shape, array_update.shape
        if array_source.ndim > 3:
            array_source = array_source.reshape((*array_source.shape[:2], -1))
            array_update = array_update.reshape((*array_update.shape[:2], -1))
        elif array_source.ndim < 3:
            for _ in range(3-array_source.ndim):
                array_source = np.expand_dims(array_source, -1)
                array_update = np.expand_dims(array_update, -1)
        # mutate_debug = cuda.to_device(np.zeros_like(array_update))
        array_source, array_update = cuda.to_device(array_source), cuda.to_device(array_update)
        kernel_shape = calc_grid(max_pop_size, *array_update.shape[1:3], tpb=tpb)
        rng_states, threads_total = get_rng_states(kernel_shape, seed)
        # if verbose and verbose >= 3:
        #     print(param.dtype, param.device, kernel_shape, as_shape, au_shape, array_source.shape, array_update.shape)

        crossover[*kernel_shape](
            array_source, array_update, sources, rng_states
        )
        if verbose and verbose >= 4:
            param.cd = array_update.copy_to_host()

        rng_states = rng_states.copy_to_host()
        rng_states = get_rng_states(kernel_shape, seed)[0]
        mutate[*kernel_shape](
            array_update, child_filter, config.genome.weight_mutate_rate, config.genome.weight_mutate_power,
            config.genome.weight_replace_rate, init_type, config.genome.weight_init_mean, config.genome.weight_init_std,
            config.genome.weight_min_value, config.genome.weight_max_value, rng_states, # mutate_debug
        )
        if verbose and verbose >= 4:
            param.md = array_update.copy_to_host() - param.cd

        # remove data from GPU
        rng_states = rng_states.copy_to_host()
        array_source = array_source.copy_to_host()
        # Add update to update list
        # array_update = array_update.copy_to_host()
        update = torch.tensor(array_update, device=param.device, dtype=param.dtype).view(au_shape)

        updates[param.param_index] = update

    # remove GPU data
    sources = sources.copy_to_host()
    child_filter = child_filter.copy_to_host()

    # Limit init values according the module specifications
    module.update(new_population, updates, True)
    module.update_limit()
    for m in module.neat_modules():
        m.update_limit()


def reproduce(
        config: Config, module: NeatModule, population: dict[int, Genome], ancestors: dict[int, tuple[Genome, Genome]],
        generation: int, to_delete: list[int], genome_indexer: int,
        stagnation: Stagnation, species_set: SpeciesSet, reporters: ReporterSet,
        tpb=10, verbose: int = None):
    # Filter out stagnated species, collect the set of non-stagnated
    # species members, and compute their average adjusted fitness.
    all_fitnesses: list[float]       = List.empty_list(FLOAT)
    remaining_species: list[Species] = List.empty_list(SPECIES)
    for sid, specie, stagnant in stagnation.update(species_set, generation):
        if stagnant:
            reporters.species_stagnant(sid, specie)
        else:
            all_fitnesses.extend([m.fitness for m in specie.members.values()])
            remaining_species.append(specie)

    # No species left.
    if not remaining_species:
        for key in list(species_set.species.keys()):
            del species_set.species[key]
        return

    # Find minimum/maximum fitness across the entire population, for use in species adjusted fitness computation.
    min_fitness = min(all_fitnesses)
    max_fitness = max(all_fitnesses)
    # Do not allow the fitness range to be zero, as we divide by it below.
    # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
    fitness_range = max(1.0, max_fitness - min_fitness)
    for specie in remaining_species:
        # Compute adjusted fitness.
        mean_fitness = np.mean([m.fitness for m in specie.members.values()])
        adj_fitness = (mean_fitness - min_fitness) / fitness_range
        specie.adjusted_fitness = adj_fitness

    adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
    avg_adjusted_fitness = np.mean(adjusted_fitnesses)
    reporters.info(f"Average adjusted fitness: {avg_adjusted_fitness:.3f}")

    # Compute the number of new members for each species in the new generation.
    previous_sizes = [len(s.members) for s in remaining_species]
    min_species_size = config.reproduction.min_species_size
    # Isn't the effective min_species_size going to be max(min_species_size, self.reproduction_config.elitism)?
    # That would probably produce more accurate tracking of population sizes and relative fitnesses... doing.
    # TODO: document.
    min_species_size = max(min_species_size, config.reproduction.elitism)
    # TODO: add pop size to arguments or just get from current population?
    pop_size = len(population)

    spawn_amounts = compute_spawn(adjusted_fitnesses, previous_sizes, pop_size, min_species_size,
                                  config.reproduction.purge, generation)

    new_population: dict[int, Genome] = Dict.empty(INT, GENOME)
    species_set.species = Dict.empty(INT, SPECIES)
    ts = clock.perf_counter()
    genome_indexer = create_children(
        new_population, species_set.species, genome_indexer, spawn_amounts, remaining_species, to_delete,
        int(config.reproduction.elitism), config.reproduction.survival_threshold, config.reproduction.darwin_multiplier,
        config.general.fitness_criterion, ancestors
    )
    update_children(
        config, module, population, new_population, ancestors, tpb, config.general.seed, verbose
    )
    if verbose:
        print(f"\n{CM('Spawned new genomes in ', Fore.CYAN)} in {round(clock.perf_counter() - ts, 2)}s")

    # update population
    for key in list(population.keys()):
        del population[key]
    for genome in new_population.values():
        population[genome.key] = genome

    return genome_indexer
