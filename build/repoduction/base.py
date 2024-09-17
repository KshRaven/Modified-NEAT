
from build.repoduction.types import ReproductionMethods
from build.config import Config
from build.nn.base import NeatModule, bind_modules
from build.nn.genome import Genome, initialize_genome
from build.reporter.base import ReporterSet
from build.species import SpeciesSet, Species, SPECIES
from build.stagnation import Stagnation

from numba import types, typeof, njit, optional, prange
from numba.experimental import jitclass
from numba.typed import List, Dict

from itertools import count

import numpy as np

FLOAT   = types.float64
INT     = types.int64
# STR     = typeof('str')
# BOOL    = types.boolean
GENOME  = Genome.class_type.instance_type


class Reproduction:
    def __init__(self, reporters: ReporterSet, configuration: Config):
        self._genome_indexer = count(1)
        self._reporters = reporters
        self._stagnation = Stagnation(configuration, reporters)
        self._config = configuration
        self.ancestors: dict[int, tuple[Genome, Genome]] = {}
        self.types = ReproductionMethods()

    def create_new(self, pop_size: int, build: list[NeatModule]) -> dict[int, Genome]:
        # Create keys
        genome_ids = {next(self._genome_indexer): idx for idx in range(pop_size)}
        # Set genomes
        genomes = Dict.empty(INT, GENOME)
        for gid, idx in genome_ids.items():
            genome = Genome(gid)
            genomes[gid] = genome
        # Bind networks to genomes
        bind_modules(build, list(genomes.values()))
        # Initialize each network
        for genome in genomes.values():
            initialize_genome(genome, self._config)
        return genomes

    @staticmethod
    def compute_spawn(adjusted_fitness: list[float], previous_sizes: list[int], pop_size: int, min_species_size: int):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        af_sum = sum(adjusted_fitness)

        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
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

    def reproduce(self, species_set: SpeciesSet, population: dict[int, Genome], build: list[NeatModule], generation: int,
                  repro_function: callable = None):
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

        new_population = Dict.empty(INT, GENOME)
        species_set.species = Dict.empty(INT, SPECIES)
        for spawn, specie in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self._config.reproduction.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members: list[Genome] = list(specie.members.values())
            specie.members = Dict.empty(INT, GENOME)
            species_set.species[specie.key] = specie

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x.fitness)

            # Transfer elites to new generation.
            if self._config.reproduction.elitism > 0:
                for m in old_members[:self._config.reproduction.elitism]:
                    new_population[m.key] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = max(2, int(np.ceil(self._config.reproduction.survival_threshold * len(old_members))))
            # Use at least two parents no matter what the threshold fraction result is.
            old_members = old_members[:repro_cutoff]
            probs = np.array([g.fitness for g in old_members]) * self._config.reproduction.darwin_multiplier
            probs += (np.abs(probs.min()) + 1e-10)
            probs = probs / probs.sum()

            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                parent1: Genome = np.random.choice(old_members, p=probs)
                parent2: Genome = np.random.choice(old_members, p=probs)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = next(self._genome_indexer)
                # print(gid)
                child = Genome(gid)
                child.update_from_build()
                child.update_from_crossover(parent1, parent2)
                # TODO: Enable custom reproduction functions
                # repro_function(child, parent1, parent2)
                # TODO: Enable mutation of offspring
                # child.mutate(self._config.genome)
                new_population[gid] = child
                self.ancestors[gid] = (parent1, parent2)
                spawn -= 1

        # Clear old population
        for gid in list(population.keys()):
            del population[gid]
        # Enter new population
        for gid, genome in new_population.items():
            population[gid] = genome
        # Copy weights to modules
        bind_modules(build, list(population.values()), None)

        print(f"Population sample: {[g.key for g in population.values()][:20]}")
