
from ModifiedNEAT.nn.base import NeatModule, NeatParameter
from ModifiedNEAT.nn.genome import Genome
from ModifiedNEAT.config import Config
from ModifiedNEAT.species import Species, SpeciesSet, FLOAT, INT, SPECIES
from ModifiedNEAT.stagnation import Stagnation
from ModifiedNEAT.reporter.base import ReporterSet
from ModifiedNEAT.cuda.functional import calc_grid, prob, normal, clamp, get_rng_states, get_value, set_value
from ModifiedNEAT.cuda.initialization import initialize_genome
from ModifiedNEAT.util.fancy_text import CM, Fore

from numba import njit, types, prange, cuda
from numba.typed import List, Dict
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from torch import Tensor
from typing import Union

import torch
import numpy as np
import cupy as cp
import time as clock
import gc
import math

GPUArray = Union[DeviceNDArray, cp.ndarray]

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

    # Normalize the spawn amounts so that the next generation is roughly the population size requested by the user.
    total_spawn = sum(spawn_amounts)
    norm = pop_size / total_spawn
    spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

    # Limit to min_species_size when enabled
    if purge > 0:
        if generation % purge == 0:
            spawn_amounts = [min_species_size for _ in spawn_amounts]

    return spawn_amounts


@njit(nogil=True)
def create_children(genus: int, genus_population: dict[int, Genome], population: dict[int, Genome], species: dict[int, Species],
                    available_gid: int, spawn_amounts: list[int], remaining_species: list[Species], to_delete: list[int],
                    elitism: int, survival_threshold: float, cross_threshold: float, cross_balance: bool,
                    darwin_multiplier: float, criteria: str, ancestors: dict[int, tuple[Genome, Genome]]):
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
                genus_population[m.key] = m
                spawn -= 1

        if spawn <= 0:
            continue

        # Only use the survival threshold fraction to use as parents for the next generation.
        repro_cutoff = max(2, math.ceil(survival_threshold * len(old_members)))
        # Use at least two parents no matter what the threshold fraction result is.
        old_members = old_members[:repro_cutoff]

        if cross_threshold > 0:
            extra_members = []
            for genome in population.values():
                if genome.genus != genus:
                    extra_members.append(genome)
            if len(extra_members) > 0:
                extra_members = sort({g.key: g for g in extra_members}, criteria)
                cross_cutoff = math.ceil(cross_threshold * len(extra_members))
                old_members.extend(extra_members[:cross_cutoff])
                old_members = sort({g.key: g for g in old_members}, criteria)

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
            child = Genome(gid, genus)
            # child.update_from_ModifiedNEAT()
            genus_population[gid] = child
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
def crossover(genus: int, source: GPUArray, updates: GPUArray, parents: GPUArray, genera: GPUArray, filled: GPUArray,
              probabilities: GPUArray, equal_sources: bool):
    genome_idx, x, y = cuda.grid(3)
    # Parameter shape (genomes, *spatial_dims)
    g_lim = updates.shape[0]
    x_lim = 1 if updates.ndim <= 1 else updates.shape[1]
    y_lim = 1 if updates.ndim <= 2 else updates.shape[2]
    s_g, s_x, s_y = cuda.gridsize(3)

    # Linearized thread index
    rng_index = (y * s_x * s_g) + (x * s_g) + genome_idx

    if genome_idx < g_lim and x < x_lim and y < y_lim:
        parent0, parent1, parent_   = parents[genome_idx]
        genus0, genus1, genus_      = genera[genome_idx]
        filled0, filled1, filled_   = filled[genome_idx]
        if equal_sources or genus0 == genus1 == genus:
            # For elite genomes
            if parent0 == parent1 and genus == genus0 == genus1:
                value = get_value(source, parent0, x, y)
                filled[genome_idx, :2] = True
            # For child crossover
            else:
                value = get_value(updates, genome_idx, x, y)
                if value == 0 and not (filled0 or filled1):
                    if genus == genus0 and not filled0:
                        value = get_value(source, parent0, x, y)
                        filled[genome_idx, 0] = True
                    elif genus == genus1 and not filled1:
                        value = get_value(source, parent1, x, y)
                        filled[genome_idx, 1] = True
                if value != 0 and not (filled0 and filled1):
                    if genus == genus0 and not filled0:
                        value_c = get_value(source, parent0, x, y)
                        value = _crossover(value, value_c, probabilities, rng_index)
                        filled[genome_idx, 0] = True
                    elif genus == genus1 and not filled1:
                        value_c = get_value(source, parent1, x, y)
                        value = _crossover(value, value_c, probabilities, rng_index)
                        filled[genome_idx, 1] = True
        else:
            # Emergency fill on unmatched param groups
            if not filled_ and genus == genus_:
                # TODO: Should random emergency parents be used instead of random cloning?
                value = get_value(source, parent_, x, y)
                filled[genome_idx, 2] = True
            else:
                value = get_value(updates, genome_idx, x, y)

        set_value(updates, genome_idx, x, y, value)


@cuda.jit(device=True)
def mutate_genome(parameter: GPUArray, g: int, x: int, y: int, mutate_rate: float, mutate_power: float,
                  replace_rate: float, init_type: str, mean: float, std: float, minimum: float, maximum: float,
                  probabilities: GPUArray, normals: GPUArray, rng_index: int):
    r = prob(probabilities, rng_index)
    if r < mutate_rate:
        value = clamp(get_value(parameter, g, x, y) + normal(normals, rng_index, 0., mutate_power), minimum, maximum)
        set_value(parameter, g, x, y, value)
    elif r < replace_rate + mutate_rate:
        initialize_genome(parameter, g, x, y, init_type, mean, std, minimum, maximum, normals, rng_index)


@cuda.jit
def mutate(
        updates: GPUArray, children: GPUArray, mutate_rate: float, mutate_power: float, replace_rate: float,
        init_type: str, mean: float, std: float, minimum: float, maximum: float,
        probabilities: GPUArray, normals: GPUArray,
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
                          mean, std, minimum, maximum, probabilities, normals, rng_index)


def update_children(
        config: Config, genus: int, modules: dict[int, NeatModule],
        old_population: dict[int, Genome], new_population: dict[int, Genome],
        ancestors: dict[int, tuple[Genome, Genome]], tpb=1, seed: int = None, verbose: int = None
):
    for m in modules[genus].neat_modules():
        m.updated = False

    if config.genome.init_type == 'normal':
        init_type = 0
    elif config.genome.init_type == 'uniform':
        init_type = 1
    else:
        raise NotImplementedError(f"Unsupported NEAT Genome init type '{config.genome.init_type}'")

    genus_mapping = {g: index for index, g in enumerate(modules.keys())}
    old_mapping = modules[genus].mapping
    sources = {}
    genera  = {}
    emergency_keys = [key for key, genome in new_population.items() if key in old_population and genome.genus == genus]
    if len(emergency_keys) == 0:
        emergency_keys = [key for key, genome in old_population.items() if genome.genus == genus]
    emergency_prob = torch.softmax(torch.tensor([old_population[key].fitness for key in emergency_keys]), -1).numpy()
    try:
        for index, (key, genome) in enumerate(new_population.items()):
            emeg_key = np.random.choice(emergency_keys, p=emergency_prob)
            if genome.key in old_population:
                sources[index] = (old_mapping[key], old_mapping[key], old_mapping[emeg_key])
                genera[index]  = (genus_mapping[genome.genus], genus_mapping[genome.genus], genus)
            else:
                parent1, parent2 = ancestors[key]
                if parent1.fitness < parent2.fitness:
                    parent1, parent2 = parent2, parent1
                mapping1, mapping2 = modules[parent1.genus].mapping, modules[parent2.genus].mapping
                sources[index] = (mapping1[parent1.key], mapping2[parent2.key], old_mapping[emeg_key])
                genera[index]  = (genus_mapping[parent1.genus], genus_mapping[parent2.genus], genus)
    except Exception as e:
        for m in modules.values():
            print(m.mapping)
        print(list(new_population.keys()))
        print(genus_mapping)
        print(f"Current genus = {genus}")
        raise e
    sources = cp.array([list(g) for g in sources.values()])
    genera  = cp.array([list(m) for m in genera.values()])
    child_filter = cp.array([
        True if gid not in old_population else False for gid in new_population.keys()
    ])
    max_pop_size = max(len(old_population), len(new_population))

    def check_param_compatibility(parameters: list[NeatParameter]):
        if len(parameters) == 0:
            return False

        reference_ndim = parameters[0].ndim
        for p in parameters:
            if p.ndim != reference_ndim:
                return False

        reference_shape = parameters[0].original_shape
        for p in parameters:
            for p_dim, ref_dim in zip(p.original_shape, reference_shape):
                if p_dim != ref_dim:
                    return False

        return True

    def reshape(tensor: Union[Tensor, cp.ndarray]):
        original_shape: tuple[int, ...] = tensor.shape
        if tensor.ndim > 3:
            tensor = tensor.reshape(*tensor.shape[:2], -1)
        elif tensor.ndim < 3:
            for _ in range(3-tensor.ndim):
                if isinstance(tensor, Tensor):
                    tensor = tensor.unsqueeze(-1)
                elif isinstance(tensor, cp.ndarray):
                    tensor = cp.expand_dims(tensor, -1)
                else:
                    raise ValueError(f"Unsupported dtype = {type(tensor)}")
        return tensor, original_shape

    updates: dict[int, Tensor] = {}
    genera_parameters = [m.neat_parameters() for m in modules.values()]
    pgs: list[tuple[NeatParameter, ...]] = list(zip(*genera_parameters))
    equal_param_num = all([len(c) == len(genera_parameters[0]) for c in genera_parameters])

    valid_perc = np.count_nonzero([check_param_compatibility(pg) for pg in pgs])
    if verbose and verbose >= 2:
        print(f"Valid Count = {valid_perc} / {len(pgs)}")
    if equal_param_num:
        step = 0
        for genus_param, param_group in zip(modules[genus].neat_parameters(), pgs):
            step += 1
            # with cuda.defer_cleanup():
            # ------------------------------ Sending data to GPU ------------------------------ #
            # ctype = cp.float32 if genus_param.dtype == torch.float32 else cp.float64
            array_update, original_shape = reshape(cp.zeros((len(new_population), *genus_param.original_shape),)) # ctype))
            array_sources = tuple([cp.asarray(reshape(p.data)[0]) for p in param_group])
            # ------------------------------ Define crossover kernel and randomizer values ------------------------------ #
            kernel_shape = calc_grid(max_pop_size, *array_update.shape[1:3], tpb=tpb)
            # if verbose and verbose >= 3:
            #     print(param.dtype, param.device, kernel_shape, as_shape, au_shape, array_source.shape, array_update.shape)
            probabilities, threads_total = get_rng_states(kernel_shape, seed, get_normal=False, use_cuda=True)
            # ------------------------------ Run crossover using sources ------------------------------ #
            equal_params = check_param_compatibility(param_group)
            filled = cp.zeros(sources.shape, bool)
            for genus_, source_ in zip(genus_mapping.keys(), array_sources):
                crossover[*kernel_shape](
                    genus_, source_, array_update, sources, genera, filled, probabilities, equal_params
                )
            if verbose and verbose >= 4:
                genus_param.cd = array_update.copy().get()
            # ------------------------------ Remove data from GPU ------------------------------ #
            del array_sources
            # ------------------------------ Define mutation kernel and randomizer values ------------------------------ #
            probabilities = get_rng_states(kernel_shape, seed, get_normal=False, use_cuda=True)[0]
            normals = get_rng_states(kernel_shape, seed, get_normal=True, use_cuda=True)[0]
            # ------------------------------ Run mutation using configuration ------------------------------ #
            mutate[*kernel_shape](
                array_update, child_filter, config.genome.weight_mutate_rate, config.genome.weight_mutate_power,
                config.genome.weight_replace_rate, init_type, config.genome.weight_init_mean, config.genome.weight_init_std,
                config.genome.weight_min_value, config.genome.weight_max_value, probabilities, normals,
            )
            if verbose and verbose >= 4:
                genus_param.md = array_update.copy().get() - genus_param.cd
            # ------------------------------ Add update to update list ------------------------------ #
            update = array_update.reshape(*original_shape)
            updates[genus_param.param_index] = update
            # ------------------------------ Remove data from GPU ------------------------------ #
            del probabilities, normals, array_update

        # Remove GPU data
        del sources, child_filter
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    else:
        for genus_param in modules[genus].neat_parameters():
            # with cuda.defer_cleanup():
            # ------------------------------ Sending data to GPU ------------------------------ #
            ctype = cp.float32 if genus_param.dtype == torch.float32 else cp.float64
            array_update, original_shape = reshape(cp.zeros((len(new_population), *genus_param.original_shape), ctype))
            array_sources = (array_update,)
            # ------------------------------ Define crossover kernel and randomizer values ------------------------------ #
            kernel_shape = calc_grid(max_pop_size, *array_update.shape[1:3], tpb=tpb)
            # if verbose and verbose >= 3:
            #     print(param.dtype, param.device, kernel_shape, as_shape, au_shape, array_source.shape, array_update.shape)
            probabilities, threads_total = get_rng_states(kernel_shape, seed, get_normal=False, use_cuda=True)
            # ------------------------------ Run crossover using sources ------------------------------ #
            crossover[*kernel_shape](
                array_sources, array_update, sources, genera, probabilities, False
            )
            if verbose and verbose >= 4:
                genus_param.cd = array_update.copy().get()
            # ------------------------------ Remove data from GPU ------------------------------ #
            del array_sources
            # ------------------------------ Define mutation kernel and randomizer values ------------------------------ #
            probabilities = get_rng_states(kernel_shape, seed, get_normal=False, use_cuda=True)[0]
            normals = get_rng_states(kernel_shape, seed, get_normal=True, use_cuda=True)[0]
            # ------------------------------ Run mutation using configuration ------------------------------ #
            mutate[*kernel_shape](
                array_update, child_filter, config.genome.weight_mutate_rate, config.genome.weight_mutate_power,
                config.genome.weight_replace_rate, init_type, config.genome.weight_init_mean, config.genome.weight_init_std,
                config.genome.weight_min_value, config.genome.weight_max_value, probabilities, normals,
            )
            if verbose and verbose >= 4:
                genus_param.md = array_update.copy().get() - genus_param.cd
            # ------------------------------ Add update to update list ------------------------------ #
            update = array_update.reshape(*original_shape)
            updates[genus_param.param_index] = update
            # ------------------------------ Remove data from GPU ------------------------------ #
            del probabilities, normals, array_update

        # Remove GPU data
        del sources, child_filter
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    return updates


def reproduce(
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
    updates = {}
    new_genomes = {}
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
            mean_fitness = np.mean([m.fitness for m in specie.members.values()])
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
        min_species_size = max(min_species_size, config.reproduction.elitism)
        # TODO: add pop size to arguments or just get from current population?
        pop_size = math.ceil(len(population) / len(genera))

        spawn_amounts = compute_spawn(adjusted_fitnesses, previous_sizes, pop_size, min_species_size,
                                      config.reproduction.purge, generation)

        genus_population: dict[int, Genome] = Dict.empty(INT, GENOME)
        new_genomes[genus] = genus_population

        for key, specie in list(species_set.species.items()):
            if specie.genus == genus:
                del species_set.species[key]
        ts = clock.perf_counter()
        print(f"Current genome index = {genome_indexer}")
        genome_indexer = create_children(
            genus, genus_population, population, species_set.species, genome_indexer, spawn_amounts, remaining_species, to_delete,
            int(config.reproduction.elitism), config.reproduction.survival_threshold, config.reproduction.cross_threshold,
            config.reproduction.cross_balance, config.reproduction.darwin_multiplier,
            config.general.fitness_criterion, ancestors,
        )
        updates[genus] = update_children(
            config, genus, modules, population, genus_population, ancestors, tpb, config.general.seed, verbose
        )
        if verbose and verbose >= 2:
            print(f"{CM('Spawned new genomes in ', Fore.CYAN)} in {round(clock.perf_counter() - ts, 2)}s")

        # Update genus
        for genome in genus_population.values():
            new_population[genome.key] = genome

        pass

    # Update modules
    for genus, update in updates.items():
        module = modules[genus]
        module.update(new_genomes[genus], update, True)
        module.update_limit()
        for m in module.neat_modules():
            m.update_limit()

    # Update population
    for key in list(population.keys()):
        del population[key]
    for genome in new_population.values():
        population[genome.key] = genome

    return genome_indexer
