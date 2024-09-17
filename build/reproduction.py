
from build.repoduction.types import ReproductionMethods
from build.config import Config
from build.nn.base import NeatModule, bind_modules
from build.nn.genome import Genome, initialize_genome, FLOAT, INT
from build.reporter.base import ReporterSet
from build.species import SpeciesSet, Species, SPECIES
from build.stagnation import Stagnation
from build.util.datetime import eta, clock
from build.util.conversion import float_to_str

from numba import types, typeof, njit, optional, prange
from numba.experimental import jitclass
from numba.typed import List, Dict

from itertools import count

import numpy as np

GENOME  = Genome.class_type.instance_type
GENOME_TUPLE = types.Tuple([GENOME, GENOME])
ADPR_ZIP = types.Tuple([FLOAT, INT])
SPRE_ZIP = types.Tuple([INT, SPECIES])


@njit()
def _zip_adpr(adjusted_fitness: list[float], previous_sizes: list[int]):
    filling: list = List.empty_list(ADPR_ZIP)
    for idx in range(len(adjusted_fitness)):
        filling.append((adjusted_fitness[idx], previous_sizes[idx]))
    return filling


@njit()
def _zip_spre(spawn_amounts: list[int], remaining_species: list[Species]):
    filling: list = List.empty_list(SPRE_ZIP)
    for idx in range(len(spawn_amounts)):
        filling.append((spawn_amounts[idx], remaining_species[idx]))
    return filling


class Reproduction:
    def __init__(self, reporters: ReporterSet, configuration: Config):
        self.genome_indexer = 0
        self._reporters = reporters
        self._stagnation = Stagnation(configuration, reporters)
        self._config = configuration
        self.ancestors: dict[int, tuple[Genome, Genome]] = Dict.empty(INT, GENOME_TUPLE)
        self.types = ReproductionMethods()

    def create_new(self, pop_size: int, build: list[NeatModule]) -> dict[int, Genome]:
        # Create keys
        genome_ids = {self.genome_indexer + idx: idx for idx in range(pop_size)}
        self.genome_indexer += pop_size
        # Set genomes
        genomes = Dict.empty(INT, GENOME)
        for gid, idx in genome_ids.items():
            genome = Genome(gid)
            genomes[gid] = genome
        # Bind networks to genomes
        bind_modules(build, List(genomes.values()))
        # Initialize each network
        for genome in genomes.values():
            initialize_genome(genome, self._config)
        return genomes

    @staticmethod
    @njit(nogil=True)
    def compute_spawn(adjusted_fitness: list[float], previous_sizes: list[int], pop_size: int, min_species_size: int):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        af_sum = sum(adjusted_fitness)

        spawn_amounts = []
        for af, ps in _zip_adpr(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            d = (s - ps) * 0.5
            c = int(round(d))
            spawn = ps
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1

            spawn_amounts.append(spawn)

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

        return spawn_amounts

    @staticmethod
    @njit(nogil=True)
    def spawn(new_population: dict[int, Genome], species: dict[int, Species], available_gid: int,
              spawn_amounts: list[int], remaining_species: list[Species], to_delete: list[int],
              elitism: int, survival_threshold: float, darwin_multiplier: float, criteria: str,
              structure_params: tuple, weight_params: tuple, bias_params: tuple,
              ancestors: dict[int, tuple[Genome, Genome]] = None):
        def sort(members: dict[int, Genome], criteria: str) -> list[Genome]:
            genomes = List(members.values())
            genomes_total = len(genomes)

            # Bubble sort
            for i in range(genomes_total):
                for j in range(0, genomes_total - i - 1):
                    if criteria == 'min':
                        if genomes[j].fitness > genomes[j + 1].fitness:
                            genomes[j], genomes[j + 1] = genomes[j + 1], genomes[j]
                    elif criteria == 'max':
                        if genomes[j].fitness < genomes[j + 1].fitness:
                            genomes[j], genomes[j + 1] = genomes[j + 1], genomes[j]
                    elif criteria == 'mean':
                        # Sort by distance to mean fitness
                        mean_fitness = sum([g.fitness for g in genomes]) / len(genomes)
                        dist_j = abs(genomes[j].fitness - mean_fitness)
                        dist_j1 = abs(genomes[j + 1].fitness - mean_fitness)
                        if dist_j > dist_j1:
                            genomes[j], genomes[j + 1] = genomes[j + 1], genomes[j]
            return genomes

        temp = _zip_spre(spawn_amounts, remaining_species)
        for idx in range(len(temp)):
            spawn, specie = temp[idx]
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, elitism)
            assert spawn > 0

            # Delete unwanted members
            for gid in list(specie.members.keys()):
                if gid in to_delete and len(specie.members) > 2:
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
            old_members_indexes = np.array([i for i in range(len(old_members))])
            # TODO: Enable probabilities when numba supports prob in numpy.random.choice()
            # probs = np.array([g.fitness for g in old_members]) * darwin_multiplier
            # probs += (np.abs(probs.min()) + 1e-10)
            # probs = probs / probs.sum()

            # Randomly choose parents and produce the number of offspring allotted to the species.
            for _ in prange(spawn):
                parent1: Genome = old_members[np.random.choice(old_members_indexes)] # p=probs)
                parent2: Genome = old_members[np.random.choice(old_members_indexes)]

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = available_gid
                available_gid += 1
                # print(gid)
                child = Genome(gid)
                # Gain new networks from crossover
                child.update_from_crossover(parent1, parent2)
                # Mutate structure and values of networks
                child.mutate(*structure_params, weight_params, bias_params)
                # TODO: Enable this when mutation of structure is available
                # Update layers in case of modification
                # child.update_from_build()
                new_population[gid] = child
                if ancestors is not None:
                    ancestors[gid] = (parent1, parent2)
                spawn -= 1
        return available_gid

    def reproduce(self, species_set: SpeciesSet, population: dict[int, Genome], build: list[NeatModule], generation: int,
                  to_delete: list[int], repro_function: callable = None, verbose: int = None):
        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        all_fitnesses: list[float] = List.empty_list(FLOAT)
        remaining_species: list[Species] = List.empty_list(SPECIES)
        for sid, specie, stagnant in self._stagnation.update(species_set, generation):
            if stagnant:
                self._reporters.species_stagnant(sid, specie)
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
        self._reporters.info(f"Average adjusted fitness: {avg_adjusted_fitness:.3f}")

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self._config.reproduction.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size, self.reproduction_config.elitism)?
        # That would probably produce more accurate tracking of population sizes and relative fitnesses... doing.
        # TODO: document.
        min_species_size = max(min_species_size, self._config.reproduction.elitism)
        # TODO: add pop size to arguments or just get from current population?
        pop_size = len(population)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes, pop_size, min_species_size)

        structure_ = (
            self._config.genome.single_structural_mutation,
            self._config.genome.node_add_prob, self._config.genome.node_del_prob,
            self._config.genome.conn_add_prob, self._config.genome.conn_del_prob,
        )
        weight_ = (
            self._config.genome.weight_mutate_rate, self._config.genome.weight_mutate_power,
            self._config.genome.weight_min_value, self._config.genome.weight_max_value,
            self._config.genome.weight_replace_rate, self._config.genome.init_type,
            self._config.genome.weight_init_mean, self._config.genome.weight_init_std,
        )
        bias_ = (
            self._config.genome.bias_mutate_rate, self._config.genome.bias_mutate_power,
            self._config.genome.bias_min_value, self._config.genome.bias_max_value,
            self._config.genome.bias_replace_rate, self._config.genome.init_type,
            self._config.genome.bias_init_mean, self._config.genome.bias_init_std,
        )
        new_population = Dict.empty(INT, GENOME)
        species_set.species = Dict.empty(INT, SPECIES)
        ts = clock.perf_counter()
        self.genome_indexer = self.spawn(
            new_population, species_set.species, self.genome_indexer, spawn_amounts, remaining_species, to_delete,
            self._config.reproduction.elitism, self._config.reproduction.survival_threshold,
            self._config.reproduction.darwin_multiplier, self._config.general.fitness_criterion,
            structure_, weight_, bias_, None
        )
        if verbose:
            print(f"spawned genomes in {round(clock.perf_counter() - ts, 2)}s")

        # Clear old population
        for gid in list(population.keys()):
            del population[gid]
        # Enter new population
        for gid, genome in new_population.items():
            population[gid] = genome
        # Copy weights to modules
        bind_modules(build, population)

        if verbose and verbose >= 2:
            print(f"Population sample: {[g.key for g in population.values()][:20]}")
