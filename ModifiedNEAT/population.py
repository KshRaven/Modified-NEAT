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
from ModifiedNEAT.util.qol import manage_params, Indexer
from ModifiedNEAT.util.storage import save, load
# from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.util.datetime import eta, clock
from ModifiedNEAT.util.replay import ReplayBuffer

from typing import Union, Iterable
from numba import njit
from numba.typed import List, Dict
from itertools import count

import numpy as np
import torch


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

    genus_indexer = Indexer(0)
    threads_per_block = 8
    group_indexer = Indexer(0)

    def __init__(self, genomes: int, module: NeatModule, config: Config = None, save_state: dict = None, **options):
        # ------------------------------ Globals ------------------------------ #
        self.genus: int = next(self.genus_indexer)
        self.genera = [self.genus]
        self._init_pop_size = genomes
        params = module.neat_parameters()
        if not params or len(params) == 0:
            raise ValueError(f"Module has no NeatParameters()")
        module.genus = self.genus
        for m in module.neat_modules():
            m.genus = self.genus
        self.modules: dict[int, NeatModule] = {self.genus: module}
        self.config = config
        self.generation = 0
        self.group_up: bool = manage_params(options, 'group_up', True)

        # ------------------------------ Debugging ------------------------------ #
        self.reporters = ReporterSet()
        if manage_params(options, ['init_reporter', 'init_rep'], True):
            self.add_reporter(StdOutReporter(True))

        # ------------------------------ Evaluation ------------------------------ #
        self.genomes: dict[int, Genome] = Dict.empty(INT, GENOME)
        self.species_set = SpeciesSet(self.config, self.reporters)
        self.reproduction = Reproduction(self.reporters, self.config)
        self.to_delete: list[int] = List.empty_list(INT)

        # ------------------------------ Data Loading ------------------------------ #
        self._initialized = False
        verbose = manage_params(options, 'verbose', 2)
        # TODO: Fix this variable name
        if save_state is None:
            # TODO: Implement initialization for CPU functions
            # Create a population from scratch, then partition into species.
            self.genomes = self.reproduction.create_new(
                self.genus, self.size, self.modules[self.genus],
                tpb=self.threads_per_block, verbose=verbose
            )
            # TODO: Implement initial speciation for CPU functions
            # self.species.speciate(self.genomes, self.generation, True)
            speciate(
                self.config, self.genera, self.modules, self.species_set, self.genomes, self.generation,
                tpb=self.threads_per_block, verbose=verbose
            )
        else:
            self.load_dict(save_state, verbose=verbose)
        self._initialized = True

        # ------------------------------ Post Evaluation ------------------------------ #
        if config.general.fitness_criterion == 'max':
            self.fitness_criterion = np.max
        elif config.general.fitness_criterion == 'min':
            self.fitness_criterion = np.min
        elif config.general.fitness_criterion == 'mean':
            self.fitness_criterion = np.mean
        else:
            raise ValueError(f"Unexpected fitness_criterion: {config.general.fitness_criterion}")
        self.best_genomes: dict[int, Union[Genome, None]] = {self.genus: None}
        self.rankings: dict[int, dict[int, Genome]] = {self.genus: {}}
        self.legends: dict[int, list[Genome]] = {self.genus: []} # List.empty_list(GENOME)
        self.survival_rate: float = None
        self.loop_idx: int = 0
        self._skipped = False

        # ------------------------------ Miscellaneous ------------------------------ #
        self.buffers = ReplayBuffer()

    def absorb_population(self, population: 'Population'):
        for genome in population.genomes.values():
            try:
                assert genome.key not in self.genomes
            except Exception as e:
                print(self, population)
                print(list(self.genomes.keys()))
                print(list(population.genomes.keys()))
                raise e
            self.genomes[genome.key] = genome
        for specie in population.species_set.species.values():
            assert specie.key not in self.species_set.species
            self.species_set.species[specie.key] = specie
        for gid, sid in population.species_set.genome_to_species.items():
            self.species_set.genome_to_species[gid] = sid
        for genus, module in population.modules.items():
            assert genus not in self.genera
            self.modules[genus] = module
        for genus, best_genome in population.best_genomes.items():
            assert genus not in self.genera
            self.best_genomes[genus] = best_genome
        for genus, ranking in population.rankings.items():
            assert genus not in self.genera
            self.rankings[genus] = ranking
        for genus, avatars in population.legends.items():
            assert genus not in self.genera
            self.legends[genus] = avatars
        self.genera.extend(population.genera)

        pass

    @property
    def size(self):
        return self._init_pop_size if not self._initialized else len(self.genomes)

    @property
    def best_genome(self):
        res = tuple(self.best_genomes.values())
        if len(res) == 1:
            res = res[0]
        return res

    @property
    def ranking(self):
        res = tuple(self.rankings.values())
        if len(res) == 1:
            res = res[0]
        return res

    @property
    def avatars(self):
        res = tuple(self.legends.values())
        if len(res) == 1:
            res = res[0]
        return res

    def get_mapping(self, consolidated=False, grouped=False):
        mapping = tuple([m.mapping for m in self.modules.values()])
        if len(mapping) == 1:
            mapping = mapping[0]
        else:
            index = 0
            if consolidated:
                consolidated_mapping = {}
                for genus_mapping in mapping:
                    for key in genus_mapping.keys():
                        consolidated_mapping[key] = index
                        index += 1
                mapping = consolidated_mapping
            elif grouped:
                grouped_mapping = {}
                for group in zip(*[list(genus_mapping.keys()) for genus_mapping in mapping]):
                    grouped_mapping[group] = index
                    index += 1
                mapping = grouped_mapping
        return mapping

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def add_buffers(self, buffers: list[str]):
        self.buffers.add_buffers(buffers)

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
        for genus in self.genera:
            self.best_genomes[genus] = get_best_genomes(
                List([g for g in self.genomes.values() if g.genus == genus]),
                self.config.general.fitness_criterion, None
            )

        def update_avatars(avatars: list[Genome], best_genome: Genome, genomes: dict[int, Genome]):
            # Remove avatars that are no longer within population
            index = 0
            while index < len(avatars):
                genome = avatars[index]
                if genome.key not in genomes:
                    avatars.remove(genome)
                else:
                    index += 1
            # Append best genome to avatars
            if best_genome not in avatars:
                avatars.append(best_genome)
            else:
                avatars.remove(best_genome)
                avatars.append(best_genome)

            return avatars
        for genus in self.genera:
            self.legends[genus] = update_avatars(
                self.legends[genus], self.best_genomes[genus],
                {g.key: g for g in self.genomes.values() if g.genus == genus}
            )

        if verbose:
            self.reporters.post_evaluate(self.config, self.genomes, self.species_set, self.best_genome)

        # End if the fitness threshold is reached.
        fitness_aggr: list[Union[float, int, complex]] = [
            self.fitness_criterion([
                genome.fitness for genome in self.genomes.values() if genome.genus == genus
            ])
            for genus in self.genera
        ]
        if all([aggr >= self.config.general.fitness_threshold for aggr in fitness_aggr]):
            self.reporters.found_solution(self.config, self.generation, self.best_genome)
            return False

    def _adv_population_update(self, verbose: int = None) -> None:
        # Check for complete extinction.
        if not self.species_set.species:
            self.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            if self.config.general.reset_on_extinction:
                self.genomes = Dict.empty(INT, GENOME)
                for genus, module in self.modules.items():
                    assert genus in self.genera
                    for genome in self.reproduction.create_new(
                            genus, self.size, module,
                        tpb=self.threads_per_block, verbose=verbose
                    ).values():
                        self.genomes[genome.key] = genome
            # otherwise raise an exception.
            else:
                raise CompleteExtinctionException(f"Complete extinction of Population")

        # Divide the new population into species.
        speciate(
            self.config, self.genera, self.modules, self.species_set, self.genomes, self.generation,
            tpb=self.threads_per_block, verbose=verbose
        )

        if verbose:
            self.reporters.end_generation(self.config, self.genomes, self.species_set)

        self.generation += 1

    def run(self, fitness_function, generations: int = None, skip=False, verbose: int = None, **options):
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
                    if verbose and verbose >= 2:
                        print(f"------ Executing fitness function {fitness_function} - Skip={skip} ------")

                    # Update reporters
                    if verbose:
                        self.reporters.start_generation(self.generation)

                    # Evaluate all genomes using the user-provided function.
                    fitness_function(self, **options)

                    # Create rankings
                    for genus in self.genera:
                        ranking = {}
                        for genome in self.genomes.values():
                            if genome.genus == genus:
                                ranking[genome.key] = genome
                        self.rankings[genus] = dict(sorted(ranking.items(), key=lambda item: item[1].fitness, reverse=True))
                    # self.rankings = dict(sorted(self.rankings.items(), key=lambda item: item[0]))

                if skip:
                    if not self._skipped:
                        # print(CM(f"SKIPPED!", Fore.LIGHTGREEN_EX))
                        self._skipped = True
                        break
                    else:
                        self._skipped = False

                if not self._skipped:
                    if verbose and verbose >= 2:
                        print(f"------ Updating population ------") # , skip-enabled={skip}")
                    # Update Population
                    self._init_population_update(verbose)

                    # Create the next generation from the current generation.
                    to_delete_hold = np.unique(list(self.to_delete))
                    self.to_delete: list[int] = List.empty_list(INT)
                    for i in to_delete_hold:
                        self.to_delete.append(i)
                    self.survival_rate = 1 - (len(self.to_delete) / len(self.genomes))
                    # self.reproduction.reproduce(self.species, self.genomes, self.modules, self.generation,
                    #                             self.to_delete, reproduction_function, verbose)

                    self.reproduction.genome_indexer.set(reproduce(
                        self.config, self.genera, self.modules, self.genomes, self.reproduction.ancestors,
                        self.generation, self.to_delete, self.reproduction.genome_indexer.get(), self.reproduction.stagnation,
                        self.species_set, self.reporters,
                        tpb=self.threads_per_block, verbose=verbose
                    ))

                    for ranking in self.rankings.values():
                        for key in list(ranking.keys()):
                            if key not in self.genomes:
                                del ranking[key]

                    self.to_delete = List.empty_list(INT)

                    self._adv_population_update(verbose)

                    # if verbose and verbose >= 2:
                    #     for p in self.modules:
                    #         print(p)

                    if skip and not terminate:
                        _, ranking = self.run(fitness_function, generations, skip, verbose, **options)
                        break
                gen += 1

        except KeyboardInterrupt:
            pass

        return self.best_genome, self.ranking

    def save_dict(self, name: str = None, directory: str = None, file_no: int = None, replace=False):
        # ------------------------------ Taxonomy Info ------------------------------ #
        genera = self.genera
        # ------------------------------ Modules' Params ------------------------------ #
        module_state = {genus: module.state_dict() for genus, module in self.modules.items()}
        # ------------------------------ Genomes and Evaluations ------------------------------ #
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
                'genus': genome.genus,
            }
            genomes.append(genome_dict)
        best_keys = {
            genus: g.key if g and g.key in self.genomes else None for genus, g in self.best_genomes.items()
        } if self.best_genomes else None
        avatar_keys = {
            genus: [g.key for g in listing if g.key in self.genomes] for genus, listing in self.legends.items()
        }
        # ------------------------------ Species ------------------------------ #
        species = []
        for specie in self.species_set.species.values():
            specie_dict = {
                'key': specie.key,
                'created': specie.created,
                'last_improved': specie.last_improved,
                'representative': specie.representative.key,
                'members': list(specie.members.keys()),
                'fitness': specie.fitness,
                'adjusted_fitness': specie.adjusted_fitness,
                'fitness_history': list(specie.fitness_history),
                'genus': specie.genus,
            }
            species.append(specie_dict)
        # ------------------------------ State ------------------------------ #
        state = {
            'genera': genera,
            'generation': self.generation,
            'module_state': module_state,
            'genomes': genomes,
            'genome_indexer': self.reproduction.genome_indexer.get(),
            'best_genomes': best_keys,
            'avatars': avatar_keys,
            'species': species,
            'species_indexer': self.species_set.species_indexer.get(),
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
        # ------------------------------ General ------------------------------ #
        self.genera = save_state['genera']
        self.genus_indexer.set(max(self.genera)+1)
        self.generation = save_state['generation']
        # ------------------------------ Genomes ------------------------------ #
        genomes_data = save_state['genomes']
        ts, ud, ut = clock.perf_counter(), 0, len(genomes_data)
        self.genomes = Dict.empty(INT, GENOME)
        for gs in genomes_data:
            # gs['networks'] = [dict(ns) for ns in gs['networks']]
            # gs = Dict(gs)
            genome = load_genome(gs)
            self.genomes[genome.key] = genome
            ud += 1
            eta(ts, ud, ut, f'loading {len(genomes_data)}')
        self.reproduction.genome_indexer.set(save_state['genome_indexer'])
        print(f"\rloaded genomes in {round(clock.perf_counter() - ts, 2)}s")
        # ------------------------------ Modules' Params ------------------------------ #
        modules_params = save_state['module_state']
        for (genus_depr, module), (genus, params) in zip(list(self.modules.items()), list(modules_params.items())):
            del self.modules[genus_depr]
            module.genus = genus
            for m in module.neat_modules():
                m.genus = genus
                m.updated = False
            for p in module.neat_parameters():
                p.reset()
            module.update({g.key: g for g in self.genomes.values() if g.genus == genus})
            module.load_state_dict(params)
            del modules_params[genus]
            self.modules[genus] = module
        # ------------------------------ Evaluations ------------------------------ #
        best_genome_keys = save_state.get('best_genomes')
        if best_genome_keys is not None:
            self.best_genomes = {genus: self.genomes.get(key) for genus, key in best_genome_keys.items()}
        avatars = save_state.get('avatars')
        if avatars is not None:
            self.legends = {genus: [self.genomes[key] for key in keys if key in self.genomes] for genus, keys in avatars.items()}
        # ------------------------------ Species ------------------------------ #
        species_data = save_state['species']
        ts, ud, ut = clock.perf_counter(), 0, len(species_data)
        self.species_set.species = Dict.empty(INT, SPECIES)
        for ss in species_data:
            # ss = Dict(ss)
            specie = load_species(self.genomes, ss)
            self.species_set.species[specie.key] = specie
            ud += 1
            eta(ts, ud, ut, 'loading species')
        self.species_set.species_indexer.set(save_state['species_indexer'])
        print(f"\rloaded species in {round(clock.perf_counter() - ts, 2)}s")

        print(f"Loaded NEAT Population species in {round(clock.perf_counter() - gts, 2)}s")

        speciate(
            self.config, self.genera, self.modules, self.species_set, self.genomes, self.generation,
            tpb=self.threads_per_block, verbose=verbose
        )

    def crop(self, keys: Union[int, Iterable[int]], device: torch.device = None):
        if not isinstance(keys, Iterable):
            keys = [keys]
        if device is None:
            device = list(self.modules.values())[0].dev

        def remove(dictionary: dict, key):
            if key in dictionary:
                del dictionary[key]

        gm = self.get_mapping(consolidated=False, grouped=False)
        if not isinstance(gm, tuple):
            gm = (gm,)
        for genus, genus_mapping in zip(self.genera, gm):
            new_genomes = {key: self.genomes[key] for key in keys if key in genus_mapping}
            new_updates = {}
            module = self.modules[genus]

            for parameter in module.neat_parameters():
                indices_to_keep = torch.unique(
                    torch.tensor([parameter.mapping[key] for key in new_genomes.keys() if key in parameter.mapping],
                                 device=parameter.device if not device else device, dtype=torch.int64)
                )
                update = torch.index_select(parameter.data.to(parameter.device if not device else device),
                                            dim=0, index=indices_to_keep)
                new_updates[parameter.param_index] = update

            module.update(new_genomes, new_updates, True)

        if device:
            for module in self.modules.values():
                module.to(device)
                for label, attr in vars(module).items():
                    if isinstance(attr, torch.Tensor):
                        setattr(module, label, attr.to(device))
                if hasattr(module, 'device'):
                    module.device = device
                for sub_module in module.neat_modules():
                    for label, attr in vars(sub_module).items():
                        if isinstance(attr, torch.Tensor):
                            setattr(sub_module, label, attr.to(device))
                    if hasattr(sub_module, 'device'):
                        sub_module.device = device

        genera_to_remove = [genus for genus in self.genera if not np.any([self.genomes[key].genus == genus for key in keys])]
        for genus in genera_to_remove:
            remove(self.modules, genus)
            remove(self.best_genomes, genus)
            remove(self.rankings, genus)
            remove(self.legends, genus)

        genomes_to_remove = [key for key in self.genomes.keys() if key not in keys]
        for key in genomes_to_remove:
            remove(self.genomes, key)

        torch.cuda.empty_cache()
