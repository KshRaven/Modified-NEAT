
from ModifiedNEAT.nn.base import NeatModule, NeatParameter, check_for_illegal_zeros
from ModifiedNEAT.nn.genome import Genome
from ModifiedNEAT.config import Config
from ModifiedNEAT.species import Species, SpeciesSet, FLOAT, INT, SPECIES
from ModifiedNEAT.stagnation import Stagnation
from ModifiedNEAT.reporter.base import ReporterSet
from ModifiedNEAT.base.gpu.functional import calc_grid, prob, get_rng_states, get_value, set_value
from ModifiedNEAT.base.gpu.mutation import mutate
from ModifiedNEAT.base.cpu.reproduction import crossover as  crossover_cpu
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
GENOME_LIST = types.ListType(GENOME)


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
                    elitism: int, survival_threshold: float, clone_threshold: float, cross_threshold: float,
                    cross_multiplier: float, darwin_multiplier: float,
                    criteria: str, ancestors: dict[int, tuple[Genome, Genome]], equal_params: bool, preserve: bool):
    if len(spawn_amounts) != len(remaining_species):
        raise ValueError(f"Mismatch in reproduction data")

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

    def choice(main_genus: int, genomes: list[Genome], multiplier_cross: float, multiplier_fitness: float,
               # Using global limits to ensure when genome count low, this func doesn't focus on a given genome only
               global_min: float, global_max: float, last_choice: Genome = None):
        if multiplier_cross is None:
            multiplier_cross = 0
        multiplier_cross = max(-1, min(+1, multiplier_cross))
        if multiplier_fitness is None:
            multiplier_fitness = 0
        multiplier_fitness = max(-1, min(+1, multiplier_fitness))
        if global_max == global_min:
            global_min += 1e-12
        probabilities = np.random.rand(len(genomes))
        if multiplier_fitness != 0:
            for i, g in enumerate(genomes):
                factor = (g.fitness - global_min) / (global_max - global_min) * multiplier_fitness
                if multiplier_cross > 0 and g.genus != main_genus:
                    factor *= multiplier_cross
                if multiplier_cross < 0 and g.genus == main_genus:
                    factor *= 1 - multiplier_cross
                if last_choice is not None and g.key == last_choice.key:
                    factor *= 0.01
                probabilities[i] += factor
        return genomes[np.argmax(probabilities)]

    # Populate global genera members
    genera: dict[int, list[Genome]] = Dict.empty(INT, GENOME_LIST)
    for genome in population.values():
        if genome.genus not in genera:
            genera[genome.genus] = List.empty_list(GENOME)
        genera[genome.genus].append(genome)

    for idx in range(len(remaining_species)):
        spawn  = spawn_amounts[idx]
        specie = remaining_species[idx]
        # If elitism is enabled, each species always at least gets to retain its elites.
        spawn = max(spawn, elitism)
        assert spawn > 0

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
            purge_limit_reached = not preserve or (len(specie.members) - execution_count <= elitism and preserve)
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
        if elitism > 0:
            for m in old_members[:elitism]:
                if m.key not in executions:
                    genus_population[m.key] = m
                    spawn -= 1

        if spawn <= 0:
            continue

        # Only use the survival threshold fraction to use as parents for the next generation.
        member_count = len(old_members)
        repro_cutoff = max(2, math.ceil(survival_threshold * member_count))
        clone_cutoff = max(0, math.ceil(min(0.75, clone_threshold) * member_count))
        # Use at least two parents no matter what the threshold fraction result is.
        old_members = old_members[:repro_cutoff]

        extra_members: list[Genome] = List.empty_list(GENOME)
        if cross_threshold > 0:
            for gn, genus_members in genera.items():
                if gn != genus and len(genus_members) > 0:
                    cross_cutoff = math.ceil(cross_threshold * len(genus_members))
                    extra_members.extend(genus_members[:cross_cutoff])
        if len(extra_members) > 1:
            extra_members = sort({g.key: g for g in extra_members}, criteria)
            extra_min, extra_max = get_limits(extra_members)
            minimum, maximum = min(minimum, extra_min), max(maximum, extra_max)

        # Randomly choose parents and produce the number of offspring allotted to the species.
        for spawn_index in prange(spawn):
            # At least one genome has to be from the same genus in order to preserve genus integrity
            parent1: Genome = choice(genus, old_members, cross_multiplier, darwin_multiplier, minimum, maximum)
            if spawn_index < clone_cutoff:
                parent2 = parent1
            else:
                parent2: Genome = choice(genus, List(list(old_members)+list(extra_members)), cross_multiplier, darwin_multiplier, minimum, maximum, parent1)

            # Note that if the parents are not distinct, crossover will produce a genetically identical clone of the parent (but with a different ID).
            gid = available_gid
            available_gid += 1
            child = Genome(gid, genus)
            # child.update_from_build()
            genus_population[gid] = child
            ancestors[gid] = (parent1, parent2)
            spawn -= 1
    return available_gid


@cuda.jit(device=True)
def _crossover(value1: float, value2: float, states: GPUArray, index: int):
    if prob(states, index) >= 0.5:
        return value1
    else:
        return value2


@cuda.jit() # debug=True, opt=False)
def crossover(genus: int, source: GPUArray, updates: GPUArray, parents: GPUArray, genera: GPUArray, filled: GPUArray,
              probabilities: GPUArray, equal_sources: bool):
    genome_idx, x, y = cuda.grid(3)
    # Parameter shape (genomes, *spatial_dims)
    g_lim = updates.shape[0]
    x_lim = 1 if updates.ndim <= 1 else updates.shape[1]
    y_lim = 1 if updates.ndim <= 2 else updates.shape[2]
    s_g, s_x, s_y = updates.shape # cuda.gridsize(3)

    # Linearized thread index
    rng_index = (y * s_x * s_g) + (x * s_g) + genome_idx

    if genome_idx < g_lim and x < x_lim and y < y_lim:
        parent0, parent1, parent_   = parents[genome_idx]
        genus0, genus1, genus_      = genera[genome_idx]
        filled0, filled1, filled_   = filled[genome_idx, x, y]
        if equal_sources or (genus0 == genus1 == genus):
            # For elite genomes or clones
            if (parent0 == parent1) and (genus == genus0 == genus1):
                value = get_value(source, parent0, x, y)
                filled[genome_idx, x, y, :2] = True
            # For child crossover
            else:
                already_filled = filled0 or filled1

                # If not filled, fill from current available source
                value = get_value(updates, genome_idx, x, y) # Get value in case param has been filled from another source
                if not already_filled:
                    if genus == genus0:
                        value = get_value(source, parent0, x, y)
                        filled[genome_idx, x, y, 0] = filled0 = True
                    elif genus == genus1:
                        value = get_value(source, parent1, x, y)
                        filled[genome_idx, x, y, 1] = filled1 = True

                # Crossover when both parents' values are available
                if genus == genus0 and not filled0 and filled1:
                    value_c = get_value(source, parent0, x, y)
                    value = _crossover(value, value_c, probabilities, rng_index)
                    filled[genome_idx, x, y, 0] = True
                if genus == genus1 and not filled1 and filled0:
                    value_c = get_value(source, parent1, x, y)
                    value = _crossover(value, value_c, probabilities, rng_index)
                    filled[genome_idx, x, y, 1] = True
        else:
            # Emergency fill on unmatched param groups
            if not filled_ and genus == genus_:
                # TODO: Should random emergency parents be used instead of random cloning?
                value = get_value(source, parent_, x, y)
                filled[genome_idx, x, y, 2] = True
            else:
                value = get_value(updates, genome_idx, x, y)

        set_value(updates, genome_idx, x, y, value)


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
                sources[index] = (old_mapping[key], old_mapping[key], old_mapping[key])
                genera[index]  = (genus_mapping[genome.genus], genus_mapping[genome.genus], genus_mapping[genus])
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

    genera_parameters = [m.neat_parameters() for m in modules.values()]
    pgs: list[tuple[NeatParameter, ...]] = list(zip(*genera_parameters))
    equal_param_num = all([len(c) == len(genera_parameters[0]) for c in genera_parameters])

    valid_perc = np.count_nonzero([check_param_compatibility(pg) for pg in pgs])
    gpu_fails = 0
    if verbose and verbose >= 2:
        print(f"{CM('Valid Count', Fore.CYAN)} = {valid_perc} / {len(pgs)}")
    if equal_param_num:
        step = 0
        for genus_param, param_group in zip(modules[genus].neat_parameters(), pgs):
            dtype = genus_param.data.dtype if genus_param.data.dtype != torch.bfloat16 else torch.float32
            step += 1
            # with cuda.defer_cleanup():
            # ------------------------------ Sending data to GPU ------------------------------ #
            # TODO: Re-enable this if necessary after testing if it raises error in crossover
            # ctype = cp.float32 if genus_param.dtype == torch.float32 else cp.float64
            array_update, original_shape = reshape(cp.zeros((len(new_population), *genus_param.original_shape),)) # ctype))
            array_sources = tuple([cp.asarray(reshape(p.data.clone().to(dtype))[0]) for p in param_group])
            # ------------------------------ Define crossover kernel and randomizer values ------------------------------ #
            genome_num = len(new_population)
            kernel_shape = calc_grid(genome_num + 5, *array_update.shape[1:3], tpb=tpb)
            # if verbose and verbose >= 3:
            #     print(param.dtype, param.device, kernel_shape, as_shape, au_shape, array_source.shape, array_update.shape)
            probabilities, threads_total = get_rng_states(kernel_shape, seed, get_normal=False, use_cuda=True)
            # ------------------------------ Run crossover using sources ------------------------------ #
            equal_params = check_param_compatibility(param_group)

            attempts = 5
            while attempts > 0:
                filled = cp.zeros(array_update.shape + (sources.shape[-1],), bool)
                updates_total = 0
                for genus_, source_ in zip(genus_mapping.keys(), array_sources):
                    if config.reproduction.cross_threshold == 0.0 and genus_ != genus:
                        continue
                    updates_total += 1
                    if attempts > 2:
                        crossover[*kernel_shape](
                            genus_, source_, array_update, sources, genera, filled, probabilities, equal_params
                        )
                    else:
                        if attempts <= 2:
                            gpu_fails += 1
                        array_update, filled = array_update.get(), filled.get()
                        filled[:] = False
                        crossover_cpu(
                            genus_, source_.get(), array_update, sources.get(), genera.get(), filled,
                            probabilities.get(), equal_params
                        )
                        array_update, filled = cp.asarray(array_update), cp.asarray(filled)
                attempts -= 1

                # There shouldn't be any zero values when epsilon or weight deletion is enabled for parameters
                try:
                    check_for_illegal_zeros(config, array_update, genus_param, modules)
                except Exception as e:
                    if attempts > 0:
                        continue
                    else:
                        print(f"Attempts left = {attempts}, GPU fails = {gpu_fails}")
                        print(f"updates total = {updates_total} for param_index = {genus_param.param_index}")
                        print(f"indices = {cp.where(cp.any(array_update == 0, axis=(1, 2)))[0].get().tolist()[:20]}")
                        print(f"zeros total = {cp.sum(cp.abs(array_update) == 0).get().item()}")
                        print(f"sources = {[s.shape for s in array_sources]}")
                        print(f"update = {array_update.shape}, probabilities = {probabilities.shape}")
                        print(f"genus = {genus}, kernel shape = {kernel_shape}")
                        raise e
                break

            if verbose and verbose >= 4:
                genus_param.cd = array_update.copy().get()
            # ------------------------------ Remove data from GPU ------------------------------ #
            del array_sources
            # ------------------------------ Define mutation kernel and randomizer values ------------------------------ #
            probabilities = cp.stack([
                get_rng_states(kernel_shape, seed, get_normal=False, use_cuda=True)[0]
                for _ in range(2 if config.genome.single_structural_mutation else 3)
            ], axis=0)
            normals = get_rng_states(kernel_shape, seed, get_normal=True, use_cuda=True)[0]
            # ------------------------------ Run mutation using configuration ------------------------------ #
            mutate[*kernel_shape](
                array_update, child_filter, config.genome.weight_mutate_rate, config.genome.weight_mutate_power,
                config.genome.weight_replace_rate, init_type,
                config.genome.weight_init_mean, config.genome.weight_init_std,
                config.genome.weight_min_value, config.genome.weight_max_value,
                config.genome.single_structural_mutation,
                config.genome.weight_add_prob, config.genome.weight_del_prob, config.genome.param_epsilon,
                probabilities, normals,
            )
            epsilon = config.genome.param_epsilon
            zero_values = cp.abs(array_update) < epsilon
            array_update[zero_values & (array_update >= 0)] = epsilon
            array_update[zero_values & (array_update < 0)] = -epsilon
            if verbose and verbose >= 4:
                genus_param.md = array_update.copy().get() - genus_param.cd
            # ------------------------------ Update parameters ------------------------------ #
            update = array_update.reshape(*original_shape)
            genus_param.update(new_population, update, True)
            # ------------------------------ Remove data from GPU ------------------------------ #
            del probabilities, normals, array_update, update, filled, zero_values

        # Remove GPU data
        del sources, child_filter
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    else:
        # TODO: Work on this section
        raise NotImplementedError()


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

        for key, specie in list(species_set.species.items()):
            if specie.genus == genus:
                del species_set.species[key]
        ts = clock.perf_counter()
        # print(f"Current genome index = {genome_indexer}")
        genome_indexer = create_children(
            genus, genus_population, population, species_set.species, genome_indexer, spawn_amounts, remaining_species,
            to_delete, int(config.reproduction.elitism),
            config.reproduction.survival_threshold, config.reproduction.clone_threshold, config.reproduction.cross_threshold,
            config.reproduction.cross_multiplier, config.reproduction.darwin_multiplier,
            config.general.fitness_criterion, ancestors, False, config.reproduction.preserve_elite
        )
        update_children(
            config, genus, modules, population, genus_population, ancestors, tpb, config.general.seed, verbose
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
