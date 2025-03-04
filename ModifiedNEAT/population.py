"""Implements the core evolution algorithm."""

from ModifiedNEAT.nn.base import NeatModule
from ModifiedNEAT.nn.genome import Genome, load_genome, INT
from ModifiedNEAT.config import Config
from ModifiedNEAT.species import SpeciesSet, load_species, GENOME, SPECIES
from ModifiedNEAT.reporter.base import ReporterSet
from ModifiedNEAT.reporter.reporters import StdOutReporter
from ModifiedNEAT.reproduction import Reproduction
from ModifiedNEAT.cuda.reproduction import reproduce
from ModifiedNEAT.cuda.speciation import speciate
from ModifiedNEAT.util.qol import manage_params
from ModifiedNEAT.util.storage import save, load
from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.util.datetime import eta, clock
from ModifiedNEAT.util.replay import ReplayBuffer

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

    def __init__(self, genomes: int, module: NeatModule, config: Config = None,
                 save_dict: dict = None, **options):
        init_rep = manage_params(options, 'init_reporter', True)
        self._init_pop_size = genomes
        self.module: NeatModule = module
        if len(module.neat_parameters()) is None:
            raise ValueError(f"Model has no NeatParameters()")
        self.config       = config
        self.reporters    = ReporterSet()
        if init_rep:
            self.add_reporter(StdOutReporter(True))
        self.reproduction = Reproduction(self.reporters, self.config)
        self.to_delete: list[int] = List.empty_list(INT)
        self.survival_rate: float = None
        self.buffers      = ReplayBuffer()
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
            self.genomes = self.reproduction.create_new(self.pop_size, self.module, tpb=4, verbose=2)
            self.generation = 0
            self.species = SpeciesSet(self.config, self.reporters)
            # self.species.speciate(self.genomes, self.generation, True)
            speciate(self.config, self.module, self.species, self.genomes, self.generation, tpb=4, verbose=2)
        self.best_genome: Genome = None
        self.ranking: dict[int, Genome] = {}
        self.avatars: list[Genome] = [] # List.empty_list(GENOME)
        self.loop_idx: int = 0
        self._skipped = False

    @property
    def pop_size(self):
        return self._init_pop_size if not hasattr(self, 'genomes') else len(self.genomes)

    def get_mapping(self):
        return self.module.mapping

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def add_buffers(self, buffers: list[str]):
        self.buffers.add_buffers(buffers)

    def reset_buffers(self):
        self.buffers.reset()

    def rollout_buffers(self, sequence_length: int = None, buffers: [str, list[str]] = None, keys=None):
        return self.buffers.rollout(buffers, sequence_length, keys, stack=True)

    def _init_population_update(self, verbose: int = None):
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
        self.best_genome = get_best_genomes(List(self.genomes.values()), self.config.general.fitness_criterion, None)

        def update_avatars(avatars: list[Genome], best_genome: Genome, genomes: dict[int, Genome]):
            # Remove avatars that are no longer within population
            index = 0
            while index < len(avatars):
                genome = avatars[index]
                if genome.key not in genomes:
                    avatars.remove(genome)
                index += 1
            # Append best genome to avatars
            if best_genome not in avatars:
                avatars.append(best_genome)
            else:
                avatars.remove(best_genome)
                avatars.append(best_genome)
        update_avatars(list(set(self.avatars)), self.best_genome, self.genomes)

        if verbose:
            self.reporters.post_evaluate(self.config, self.genomes, self.species, self.best_genome)

        # End if the fitness threshold is reached.
        fitness_aggr = self.fitness_criterion([genome.fitness for genome in self.genomes.values()])
        if fitness_aggr >= self.config.general.fitness_threshold:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)
            return False

        return True

    def _adv_population_update(self, verbose: int = None) -> None:
        # Check for complete extinction.
        if not self.species.species:
            self.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            if self.config.general.reset_on_extinction:
                self.genomes = self.reproduction.create_new(self.pop_size, self.module, tpb=4, verbose=verbose)
            # otherwise raise an exception.
            else:
                raise CompleteExtinctionException(f"Complete extinction of Population")

        # Divide the new population into species.
        speciate(self.config, self.module, self.species, self.genomes, self.generation, tpb=4, verbose=verbose)

        if verbose:
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

        terminate = manage_params(options, 'terminate_skip', False)
        if skip:
            generations = 1

        gen = 0
        try:
            while generations is None or gen < generations:
                if not self._skipped:
                    # Update reporters
                    if verbose:
                        self.reporters.start_generation(self.generation)

                    # Evaluate all genomes using the user-provided function.
                    if verbose and verbose >= 2:
                        print(f"executing fitness function {fitness_function} on population, skip-enabled={skip}")
                    fitness_function(self, **options)

                    self.ranking = {}
                    for genome in self.genomes.values():
                        self.ranking[genome.key] = genome
                    self.ranking = dict(sorted(self.ranking.items(), key=lambda item: item[1].fitness, reverse=True))

                if skip:
                    if not self._skipped:
                        # print(CM(f"SKIPPED!", Fore.LIGHTGREEN_EX))
                        self._skipped = True
                        break
                    else:
                        self._skipped = False

                if not self._skipped:
                    if verbose:
                        print(f"------ Updating population ------") # , skip-enabled={skip}")
                    # Update Population
                    if not self._init_population_update(verbose):
                        break
                    # Create the next generation from the current generation.
                    temp = np.unique(list(self.to_delete))
                    self.to_delete: list[int] = List.empty_list(INT)
                    for i in temp:
                        self.to_delete.append(i)
                    self.survival_rate = 1 - (len(self.to_delete) / len(self.genomes))
                    # self.reproduction.reproduce(self.species, self.genomes, self.modules, self.generation,
                    #                             self.to_delete, reproduction_function, verbose)

                    self.reproduction.genome_indexer = reproduce(
                        self.config, self.module, self.genomes, self.reproduction.ancestors, self.generation,
                        self.to_delete, self.reproduction.genome_indexer, self.reproduction._stagnation, self.species,
                        self.reporters, tpb=4, verbose=verbose
                    )

                    for key in List(self.ranking.keys()):
                        if key not in self.genomes:
                            del self.ranking[key]

                    self.to_delete = List.empty_list(INT)
                    # Mutate all genomes using the user-provided function.
                    if mutation_function is not None:
                        print(f"executing fitness function {fitness_function} on population")
                        mutation_function(self)
                    else:
                        # TODO: Implement what to do when no mutation function is not set
                        pass

                    self._adv_population_update(verbose)

                    # if verbose and verbose >= 2:
                    #     for p in self.modules:
                    #         print(p)

                    if skip and not terminate:
                        _, ranking = self.run(fitness_function, generations, reproduction_function, mutation_function,
                                              skip, verbose, **options)
                        break
                gen += 1

        except KeyboardInterrupt:
            pass

        return self.best_genome, self.ranking

    def save_dict(self, name: str = None, directory: str = None, file_no: int = None, replace=False):
        # Model
        module_state = self.module.state_dict()
        # Genomes
        genomes = []
        for genome in self.genomes.values():
            # networks = []
            # for network in genome.networks.values():
            #     network_dict = {
            #         'key': network.key,
            #         'inp_num': network.inp_num,
            #         'out_num': network.out_num,
            #         'hidden_layers_num': list(network.hidden_layers_num),
            #         'input_keys': list(network.input_keys),
            #         'output_keys': list(network.output_keys),
            #         'nodes': [(n.key, n.bias) for n in network.nodes.values()],
            #         'connections': [(c.key, c.weight) for c in network.connections.values()],
            #         'layers': [list(i) for i in network.layers],
            #         'ModifiedNEAT': list(network.ModifiedNEAT),
            #     }
            #     networks.append(network_dict)
            genome_dict = {
                'key': genome.key,
                'fitness': genome.fitness,
                # 'networks': networks,
            }
            genomes.append(genome_dict)
        best_key = self.best_genome.key if self.best_genome else None
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
            'module_state': module_state,
            'generation': self.generation,
            'genomes': genomes,
            'genome_indexer': self.reproduction.genome_indexer,
            'best_genome': best_key if best_key is not None and best_key in self.genomes else None,
            'avatars': [genome.key for genome in self.avatars if genome.key in self.genomes],
            'species': species,
            'species_indexer': self.species.species_indexer,
        }

        if name is not None:
            if directory is None:
                directory = 'neat_save'
            _, file_no = save(state, name, directory, file_no, replace, items_name='NEAT Population')

        return state, file_no

    def load_dict(self, save_state: dict = None, name: str = None, directory: str = None, file_no: int = None, verbose: int = None):
        if save_state is None and name is not None:
            if directory is None:
                directory = 'neat_save'
            file = load(name, directory, file_no, items_name='NEAT Population')
            if file is not None:
                save_state: dict = file
            else:
                return
        else:
            raise ValueError(f"Cannot load save state with no save_dict nor filename")
        gts = clock.perf_counter()
        self.generation = save_state['generation']
        self.genomes = Dict.empty(INT, GENOME)
        ts, ud, ut = clock.perf_counter(), 0, len(save_state['genomes'])
        genomes_data = save_state['genomes']
        for gs in genomes_data:
            # gs['networks'] = [dict(ns) for ns in gs['networks']]
            # gs = Dict(gs)
            genome = load_genome(gs)
            self.genomes[genome.key] = genome
            ud += 1
            eta(ts, ud, ut, f'loading {len(genomes_data)}')
        for m in self.module.neat_modules():
            m.updated = False
        for p in self.module.neat_parameters():
            p.reset()
        self.module.update(self.genomes)
        self.module.load_state_dict(save_state['module_state'])
        print(f"\rloaded genomes in {round(clock.perf_counter() - ts, 2)}s")
        self.reproduction.genome_indexer = save_state['genome_indexer']
        best_genome_key = save_state.get('best_genome')
        if best_genome_key is not None:
            self.best_genome = self.genomes[best_genome_key]
        avatars = save_state.get('avatars')
        if avatars is not None:
            for key in save_state['avatars']:
                self.avatars.append(self.genomes[key])
        self.species.species = Dict.empty(INT, SPECIES)
        ts, ud, ut = clock.perf_counter(), 0, len(save_state['species'])
        for ss in save_state['species']:
            # ss = Dict(ss)
            specie = load_species(self.genomes, ss)
            self.species.species[specie.key] = specie
            ud += 1
            eta(ts, ud, ut, 'loading species')
        print(f"\rloaded species in {round(clock.perf_counter() - ts, 2)}s")
        self.species.species_indexer = save_state['species_indexer']
        print(f"Loaded NEAT Population species in {round(clock.perf_counter() - gts, 2)}s")

        speciate(self.config, self.module, self.species, self.genomes, self.generation, tpb=4, verbose=verbose)
