"""Implements the core evolution algorithm."""
import sys

from build.nn.base import get_modules, NeatModule, bind_modules
from build.nn.genome import Genome, load_genome, INT
from build.config import Config
from build.species import SpeciesSet, load_species, GENOME, SPECIES
from build.reporter.base import ReporterSet
from build.reporter.reporters import StdOutReporter
from build.reproduction import Reproduction
from build.util.qol import manage_params
from build.util.storage import save, load
from build.util.fancy_text import CM, Fore
from build.util.datetime import eta, clock
from build.sequential import RollbackBuffer

from typing import Union
from numba import njit
from numba.typed import List, Dict

import numpy as np
import torch.nn as nn


class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, genomes: int, model: Union[nn.Module, list[nn.Module]], config: Config = None,
                 save_dict: dict = None, **options):
        init_rep = manage_params(options, 'init_reporter', False)
        self._init_pop_size = genomes
        self.modules: list[NeatModule] = get_modules(model)
        if len(self.modules) is None:
            raise ValueError(f"Model has no NeatModules()")
        self.config       = config
        self.reporters    = ReporterSet()
        if init_rep:
            self.add_reporter(StdOutReporter(True))
        self.reproduction = Reproduction(self.reporters, self.config)
        self.to_delete: list[int] = List.empty_list(INT)
        self.buffers      = RollbackBuffer()
        if config.general.fitness_criterion == 'max':
            self.fitness_criterion = np.max
        elif config.general.fitness_criterion == 'min':
            self.fitness_criterion = np.min
        elif config.general.fitness_criterion == 'mean':
            self.fitness_criterion = np.mean
        else:
            raise ValueError(f"Unexpected fitness_criterion: {config.general.fitness_criterion}")

        if save_dict is None:
            # Create a population from scratch, then partition into species.
            self.genomes = self.reproduction.create_new(self.pop_size, self.modules)
            self.generation = 0
            self.species = SpeciesSet(self.config, self.reporters)
            self.species.speciate(self.genomes, self.generation, True)
        self.best_genome: Genome = None
        self.loop_idx: int = 0
        self._skipped = False
        self.ranking: dict[int, Genome] = {}

    @property
    def pop_size(self):
        return self._init_pop_size if not hasattr(self, 'genomes') else len(self.genomes)

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def add_buffers(self, buffers: list[str]):
        self.buffers.add_buffers(buffers)

    def reset_buffers(self):
        self.buffers.reset()

    def rollout_buffers(self, sequence_length: int = None, buffers: [str, list[str]] = None):
        return self.buffers.get(sequence_length, buffers)

    def _init_population_update(self):
        # Gather and report statistics.
        @njit
        def get_best_genomes(genomes: list[Genome], criteria: str, best_genome: Union[Genome, None]) -> Genome:
            if not genomes:
                return None  # Return None if the list is empty
            if best_genome is None:
                best_genome = genomes[0]

            if criteria == 'min':
                # Find the genome with the minimum fitness
                for genome in genomes:
                    if genome.fitness < best_genome.fitness:
                        best_genome = genome

            elif criteria == 'max':
                # Find the genome with the maximum fitness
                for genome in genomes:
                    if genome.fitness > best_genome.fitness:
                        best_genome = genome

            elif criteria == 'mean':
                # Find the genome whose fitness is closest to the mean fitness
                mean_fitness = sum([g.fitness for g in genomes]) / len(genomes)
                closest_distance = abs(best_genome.fitness - mean_fitness)

                for genome in genomes:
                    distance = abs(genome.fitness - mean_fitness)
                    if distance < closest_distance:
                        best_genome = genome
                        closest_distance = distance

            return best_genome
        # Track the best genome ever seen.
        self.best_genome = get_best_genomes(List(self.genomes.values()), self.config.general.fitness_criterion, self.best_genome)
        self.reporters.post_evaluate(self.config, self.genomes, self.species, self.best_genome)

        # End if the fitness threshold is reached.
        fitness_aggr = self.fitness_criterion([genome.fitness for genome in self.genomes.values()])
        if fitness_aggr >= self.config.general.fitness_threshold:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)
            return False

        return True

    def _adv_population_update(self) -> None:
        # Check for complete extinction.
        if not self.species.species:
            self.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            if self.config.general.reset_on_extinction:
                self.genomes = self.reproduction.create_new(self.pop_size, self.modules)
            # otherwise raise an exception.
            else:
                raise CompleteExtinctionException(f"Complete extinction of Population")

        # Divide the new population into species.
        self.species.speciate(self.genomes, self.generation, True)

        self.reporters.end_generation(self.config, self.genomes, self.species)

        self.generation += 1

    def run(self, fitness_function, generations: int = None,
            reproduction_function=None, mutation_function=None, skip=False, verbose: int = None, **options):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """

        if skip:
            generations = 1

        gen = 0
        try:
            while generations is None or gen < generations:
                if not self._skipped:
                    # Update reporters
                    self.reporters.start_generation(self.generation)

                    # Evaluate all genomes using the user-provided function.
                    print(f"executing fitness function {fitness_function} on population, skip-enabled={skip}")
                    fitness_function(self, **options)

                    self.ranking = {}
                    for genome in self.genomes.values():
                        self.ranking[genome.key] = genome

                if skip:
                    if not self._skipped:
                        # print(CM(f"SKIPPED!", Fore.LIGHTGREEN_EX))
                        self._skipped = True
                        break
                    else:
                        self._skipped = False

                if not self._skipped:
                    print(f"updating population") # , skip-enabled={skip}")
                    # Update Population
                    if not self._init_population_update():
                        break
                    # Create the next generation from the current generation.
                    if reproduction_function is None:
                        reproduction_function = self.reproduction.types.value_crossover
                    self.reproduction.reproduce(self.species, self.genomes, self.modules, self.generation,
                                                self.to_delete, reproduction_function, verbose)
                    self.to_delete = List.empty_list(INT)
                    # Mutate all genomes using the user-provided function.
                    if mutation_function is not None:
                        print(f"executing fitness function {fitness_function} on population")
                        mutation_function(self)
                    else:
                        # TODO: Implement what to do when no mutation function is not set
                        pass

                    self._adv_population_update()

                    if verbose and verbose >= 2:
                        for p in self.modules:
                            print(p)

                    if skip:
                        _, ranking = self.run(fitness_function, generations, reproduction_function, mutation_function,
                                              skip, verbose, **options)
                        break
                gen += 1

        except KeyboardInterrupt:
            pass

        return self.best_genome, self.ranking

    def save_dict(self, name: str = None, directory: str = None, file_no: int = None, replace=False):
        # Genomes
        genomes = []
        for genome in self.genomes.values():
            networks = []
            for network in genome.networks.values():
                network_dict = {
                    'key': network.key,
                    'inp_num': network.inp_num,
                    'out_num': network.out_num,
                    'hidden_layers_num': list(network.hidden_layers_num),
                    'input_keys': list(network.input_keys),
                    'output_keys': list(network.output_keys),
                    'nodes': [(n.key, n.bias) for n in network.nodes.values()],
                    'connections': [(c.key, c.weight) for c in network.connections.values()],
                    'layers': [list(i) for i in network.layers],
                    'build': list(network.build),
                }
                networks.append(network_dict)
            genome_dict = {
                'key': genome.key,
                'fitness': genome.fitness,
                'networks': networks,
            }
            genomes.append(genome_dict)
        # Species
        species = []
        for specie in self.species.species.values():
            specie_dict = {
                'key': specie.key,
                'created': specie.created,
                'last_improved': specie.last_improved,
                'representative': specie.representative.key,
                'members': list(specie.members.keys()),
                'fitness': specie.fitness,
                'adjusted_fitness': specie.adjusted_fitness,
                'fitness_history': list(specie.fitness_history),
            }
            species.append(specie_dict)
        state = {
            'generation': self.generation,
            'genomes': genomes,
            'genome_indexer': self.reproduction.genome_indexer,
            'species': species,
            'species_indexer': self.species.species_indexer,
        }

        if name is not None:
            if directory is None:
                directory = 'neat_save'
            save(state, name, directory, file_no, replace, items_name='NEAT Population')

        return state

    def load_dict(self, save_state: dict = None, name: str = None, directory: str = None, file_no: int = None):
        if save_state is None and name is not None:
            if directory is None:
                directory = 'neat_save'
            file = load(name, directory, file_no, items_name='NEAT Population')
            if file is not None:
                save_state = file
            else:
                return
        else:
            raise ValueError(f"Cannot load save state with no save_dict nor filename")
        self.generation = save_state['generation']
        self.genomes = Dict.empty(INT, GENOME)
        for gs in save_state['genomes']:
            # gs['networks'] = [dict(ns) for ns in gs['networks']]
            # gs = Dict(gs)
            genome = load_genome(gs)
            self.genomes[genome.key] = genome
        self.reproduction.genome_indexer = save_state['genome_indexer']
        self.species.species = Dict.empty(INT, SPECIES)
        for ss in save_state['species']:
            # ss = Dict(ss)
            specie = load_species(self.genomes, ss)
            self.species.species[specie.key] = specie
        self.species.species_indexer = save_state['species_indexer']

        bind_modules(self.modules, list(self.genomes.values()))
        self.species.speciate(self.genomes, self.generation, True)
