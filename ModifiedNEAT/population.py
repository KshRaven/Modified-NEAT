"""Implements the core evolution algorithm."""

from ModifiedNEAT.nn.base import NeatModule, NeatParameter
from ModifiedNEAT.nn.genome import Genome, load_genome, INT
from ModifiedNEAT.config import Config
from ModifiedNEAT.species import SpeciesSet, load_species, GENOME, SPECIES
from ModifiedNEAT.reporter.base import ReporterSet
from ModifiedNEAT.reporter.reporters import StdOutReporter
from ModifiedNEAT.reproduction import Reproduction
from ModifiedNEAT.base import reproduce, speciate, get_tpb
from ModifiedNEAT.util.qol import manage_params, Indexer
from ModifiedNEAT.util import storage as storage_util
from ModifiedNEAT.util.storage import save, load
# from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.util.datetime import eta, clock
from ModifiedNEAT.util.replay import ReplayBuffer

from typing import Union, Iterable, Any
from collections.abc import Mapping
from pathlib import Path
from os import PathLike
import copy
import shutil
from numba import njit
from numba.typed import List, Dict

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
    group_indexer = Indexer(0)

    def __init__(self, genomes: int, module: NeatModule, config: Config = None, state_dict: dict[str, Any] = None, **options):
        # self.genus_indexer = Indexer(0)
        if config is None: config = Config()
        
        # ------------------------------ Globals ------------------------------ #
        requested_genera = manage_params(options, 'genera', 1)
        if isinstance(requested_genera, int):
            if requested_genera < 1:
                raise ValueError("genera must be a positive count or a non-empty iterable of IDs")
            self.genera = [next(self.genus_indexer) for _ in range(requested_genera)]
        else:
            self.genera = list(dict.fromkeys(requested_genera))
            if not self.genera or not all(isinstance(genus, int) for genus in self.genera):
                raise ValueError("genera must be a positive count or a non-empty iterable of integer IDs")
            self.genus_indexer.set(max(self.genus_indexer.get(), max(self.genera) + 1))
        self.genus: int = self.genera[0]
        self._init_pop_size = genomes
        params = module.neat_parameters()
        if not params or len(params) == 0:
            raise ValueError(f"Module has no NeatParameters()")
        module.genus = self.genus
        for m in module.neat_modules():
            m.genus = self.genus
        self.modules: dict[int, NeatModule] = {genus: module for genus in self.genera}
        self.config = config
        self.generation = 0
        self.group_up: bool = manage_params(options, 'group_up', True)
        self.threads_per_block = get_tpb()

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
        verbose: bool | int = manage_params(options, 'verbose', False)
        # Only restore the config saved inside ``state_dict`` when explicitly requested; otherwise keep ``config``
        load_config: bool = manage_params(options, 'load_config', False)
        if state_dict is None:
            # Create a population from scratch, then partition into species.
            genus_sizes = manage_params(options, 'genus_sizes', None)
            if genus_sizes is not None and sum(genus_sizes.values()) != genomes:
                raise ValueError("genus_sizes must sum to the population size")
            self.genomes = self.reproduction.create_new(
                self.genera, genus_sizes if genus_sizes is not None else self.size, self.modules[self.genus],
                tpb=self.threads_per_block, verbose=verbose
            )
            speciate(
                self.config, self.genera, self.modules, self.species_set, self.genomes, self.generation,
                tpb=self.threads_per_block, verbose=verbose
            )
        else:
            self.load_dict(state_dict, verbose=verbose, load_config=load_config)
        self._initialized = True

        # ------------------------------ Post Evaluation ------------------------------ #
        if self.config.general.fitness_criterion == 'max':
            self.fitness_criterion = np.max
        elif self.config.general.fitness_criterion == 'min':
            self.fitness_criterion = np.min
        elif self.config.general.fitness_criterion == 'mean':
            self.fitness_criterion = np.mean
        else:
            raise ValueError(f"Unexpected fitness_criterion: {self.config.general.fitness_criterion}")
        if not hasattr(self, 'best_genomes'):
            self.best_genomes: dict[int, Union[Genome, None]] = {genus: None for genus in self.genera}
        if not hasattr(self, 'rankings'):
            self.rankings: dict[int, dict[int, Genome]] = {genus: {} for genus in self.genera}
        if not hasattr(self, 'legends'):
            self.legends: dict[int, list[Genome]] = {genus: [] for genus in self.genera}
            if not hasattr(self, 'survival_rate'):
                self.survival_rate: float | None = None
            if not hasattr(self, 'loop_idx'):
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

    def consolidate(self, module: NeatModule = None, verbose: int | bool = False) -> 'Population':
        """Return an independent population with compatible genus modules packed into one module."""
        if not self.genomes:
            raise ValueError("Cannot consolidate an empty population")

        source_modules = {}
        for genus in self.genera:
            source_module = self.modules.get(genus)
            if source_module is None:
                raise ValueError(f"No module is registered for genus {genus}")
            source_modules.setdefault(id(source_module), source_module)

        template_source = module or next(iter(source_modules.values()))
        template_parameters = template_source.neat_parameters()
        if not template_parameters:
            raise ValueError("Consolidation requires modules with NEAT parameters")
        expected_shapes = [parameter.original_shape for parameter in template_parameters]

        parameters_by_module = {}
        for module_id, source_module in source_modules.items():
            parameters = source_module.neat_parameters()
            shapes = [parameter.original_shape for parameter in parameters]
            if shapes != expected_shapes:
                raise ValueError(
                    "Cannot consolidate modules with different NEAT parameter counts or shapes"
                )
            parameters_by_module[module_id] = parameters

        consolidated_module = copy.deepcopy(template_source)
        consolidated_parameters = consolidated_module.neat_parameters()
        if [parameter.original_shape for parameter in consolidated_parameters] != expected_shapes:
            raise ValueError("The destination module's NEAT parameter shapes do not match the population")

        combined_genomes = Dict.empty(INT, GENOME)
        for genome in self.genomes.values():
            if genome.genus not in self.modules:
                raise ValueError(f"Genome {genome.key} belongs to unregistered genus {genome.genus}")
            combined_genomes[genome.key] = genome

        combined_parameters = {}
        for parameter_index, target_parameter in enumerate(consolidated_parameters):
            rows = []
            for genome in combined_genomes.values():
                source_module = self.modules[genome.genus]
                source_parameter = parameters_by_module[id(source_module)][parameter_index]
                if genome.key not in source_parameter.mapping:
                    raise ValueError(
                        f"Module for genus {genome.genus} has no parameter row for genome {genome.key}"
                    )
                row = source_parameter.data[source_parameter.mapping[genome.key]]
                rows.append(row.detach().to(device=target_parameter.device, dtype=target_parameter.dtype))
            combined_parameters[target_parameter.param_index] = torch.stack(rows)

        for sub_module in consolidated_module.neat_modules():
            sub_module.mapping = None
            sub_module.genome_num = None
            sub_module.updated = False
        for parameter in consolidated_parameters:
            parameter.data = parameter.data[:1].clone()
            parameter.mapping = None
            parameter.genome_num = None

        consolidated_module.genus = self.genera[0]
        for sub_module in consolidated_module.neat_modules():
            sub_module.genus = self.genera[0]
        consolidated_module.update(combined_genomes, combined_parameters)

        state = self.state_dict()
        serialized_module = consolidated_module.neat_dict()
        state['module_state_dicts'] = {
            genus: serialized_module for genus in self.genera
        }
        state['module_groups'] = {
            genus: self.genera[0] for genus in self.genera
        }

        return Population(
            len(combined_genomes), consolidated_module, self.config, state_dict=state,
            genera=self.genera, verbose=verbose, init_reporter=False
        )

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

    def get_mapping(
            self, consolidated=False, grouped=False
        ) -> dict[int, int] | tuple[dict[int, int], ...] | dict[tuple[int, ...], int]:
        """
        Mapping of keys / group of keys to an index, for each genus or entire population
        """
        mapping = tuple(
            {key: index for key, index in self.modules[genus].mapping.items()
             if key in self.genomes and self.genomes[key].genus == genus}
            for genus in self.genera
        )
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
                assert all([len(gm) == len(mapping[0]) for gm in mapping])
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
                module_groups = {}
                for genus, module in self.modules.items():
                    module_groups.setdefault(id(module), (module, []))[1].append(genus)
                for module, genera_group in module_groups.values():
                    if len(genera_group) == 1:
                        genomes = self.reproduction.create_new(
                            genera_group[0], self._init_pop_size, module,
                            tpb=self.threads_per_block, verbose=verbose
                        )
                    else:
                        genomes = self.reproduction.create_new(
                            genera_group, self._init_pop_size, module,
                            tpb=self.threads_per_block, verbose=verbose
                        )
                    for genome in genomes.values():
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
    
    def reset_keys(self):
        # TODO: Implement
        pass

    def state_dict(self) -> dict[str, Any]:
        # ------------------------------ Taxonomy Info ------------------------------ #
        genera = list(self.genera)
        # ------------------------------ Modules' State and Architecture ------------------------------ #
        module_state_dicts = {}
        module_groups = {}
        module_group_ids = {}
        for genus, module in self.modules.items():
            module_groups[genus] = module_group_ids.setdefault(id(module), genus)
            # Use new state_dict_with_metadata() to save both weights and architecture
            serialized = module.neat_dict()
            module_state_dicts[genus] = serialized
        # ------------------------------ Genomes and Evaluations ------------------------------ #
        genomes = []
        for genome in self.genomes.values():
            genome_dict = {
                'key': genome.key,
                'fitness': genome.fitness,
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
        return {
            'version': 1,
            'genera': genera,
            'generation': self.generation,
            'module_state_dicts': module_state_dicts,
            'module_groups': module_groups,
            'genomes': genomes,
            'genome_indexer': self.reproduction.genome_indexer.get(),
            'config': self.config,
            'best_genomes': best_keys,
            'avatars': avatar_keys,
            'rankings': {genus: list(ranking.keys()) for genus, ranking in self.rankings.items()},
            'species': species,
            'species_indexer': self.species_set.species_indexer.get(),
            'survival_rate': self.survival_rate,
            'loop_idx': self.loop_idx,
        }

    def _save_state_dict(
        self,
        filename: str = None,
        directory: str = None,
        file_no: int = None,
        replace: bool = False,
        filepath: str | None = None,
        extra_files: Mapping[str, str | bytes | bytearray | PathLike[str]] = None,
        debug: bool = True,
    ) -> tuple[bool, int]:
        if filepath is not None:
            target = Path(filepath)
            bundle_name = target.name.removesuffix('.neat.pkl') if target.name.endswith('.neat.pkl') else target.stem
            extra_directory = target.parent / bundle_name
        elif extra_files:
            if filename is None:
                filename = 'population'
            if directory is None:
                directory = 'neat_save'
            storage_directory = Path(directory) if Path(directory).is_absolute() else Path(storage_util.STORAGE_DIR) / directory
            storage_directory.mkdir(parents=True, exist_ok=True)
            latest = 0 if Path(directory).is_absolute() else storage_util.get_last_index(filename, directory, '.neat.pkl')
            if Path(directory).is_absolute():
                for candidate in storage_directory.glob(f'{filename}-*.neat.pkl'):
                    try:
                        latest = max(latest, int(candidate.name[len(filename) + 1:-len('.neat.pkl')]))
                    except ValueError:
                        pass
            for candidate in storage_directory.glob(f'{filename}-*'):
                if candidate.is_dir():
                    try:
                        latest = max(latest, int(candidate.name[len(filename) + 1:]))
                    except ValueError:
                        pass
            if file_no is None:
                file_no = latest if replace else latest + 1
            suffix = '' if file_no == 0 else f'-{file_no:0{storage_util.FILE_INDEX_PADDING}d}'
            extra_directory = storage_directory / f'{filename}{suffix}'
            target = extra_directory / f'{filename}.neat.pkl'
        else:
            if directory is None:
                directory = 'neat_save'
            if Path(directory).is_absolute():
                storage_directory = Path(directory)
                storage_directory.mkdir(parents=True, exist_ok=True)
                latest = 0
                for candidate in storage_directory.glob(f'{filename or "population"}-*.neat.pkl'):
                    try:
                        latest = max(latest, int(candidate.name[len(filename or "population") + 1:-len('.neat.pkl')]))
                    except ValueError:
                        pass
                if file_no is None:
                    file_no = latest if replace else latest + 1
                suffix = '' if file_no == 0 else f'-{file_no:0{storage_util.FILE_INDEX_PADDING}d}'
                target = storage_directory / f'{filename or "population"}{suffix}.neat.pkl'
                return save(
                    self.state_dict(), extension='.neat.pkl', items_name='NEAT Population',
                    debug=debug, filepath=str(target)
                )
            return save(
                self.state_dict(), filename or 'population', directory, file_no, replace,
                extension='.neat.pkl', items_name='NEAT Population', debug=debug, filepath=filepath
            )

        success, file_no = save(
            self.state_dict(), extension='.neat.pkl', items_name='NEAT Population', debug=debug,
            filepath=str(target)
        )
        if not success or not extra_files:
            return success, file_no

        try:
            extra_directory.mkdir(parents=True, exist_ok=True)
            for relative_path, content in extra_files.items():
                relative_path = Path(relative_path)
                if relative_path.is_absolute() or '..' in relative_path.parts:
                    raise ValueError(f"Extra file path must stay within the population bundle: {relative_path}")
                extra_path = extra_directory / relative_path
                extra_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, (bytes, bytearray)):
                    extra_path.write_bytes(content)
                elif isinstance(content, PathLike):
                    shutil.copyfile(content, extra_path)
                elif isinstance(content, str):
                    extra_path.write_text(content, encoding='utf-8')
                else:
                    raise TypeError(f"Unsupported extra file content for {relative_path}: {type(content).__name__}")
        except Exception as error:
            if debug:
                print(f"Failed to write population extra files: {error}")
            return False, file_no

        return success, file_no

    def save_dict(self, name: str = None, directory: str = None, file_no: int = None, replace=False):
        state = self.state_dict()
        if name is not None:
            if directory is None:
                directory = 'neat_save'
            _, file_no = self._save_state_dict(name, directory, file_no, replace)

        return state, file_no

    def _load_state_dict(
        self,
        state_dict: dict[str, Any] = None,
        filename: str = None,
        directory: str = None,
        file_no: int = None,
        filepath: str = None,
        strict: bool = True,
        verbose: int = None,
        load_config: bool = False,
    ):
        """
        :param load_config: When True, the config stored in the saved state replaces this population's current config.
            By default (False) the current config is kept, so a new/edited config is not overwritten by an old one.
        """
        if state_dict is None:
            if filepath is None and filename is None:
                raise ValueError("Cannot load population without a state_dict, filename, or filepath")
            if directory is None:
                directory = 'neat_save'
            if filepath is None:
                storage_directory = Path(directory) if Path(directory).is_absolute() else Path(storage_util.STORAGE_DIR) / directory
                if Path(directory).is_absolute():
                    if file_no is None:
                        latest = 0
                        for candidate in storage_directory.glob(f'{filename}-*.neat.pkl'):
                            try:
                                latest = max(latest, int(candidate.name[len(filename) + 1:-len('.neat.pkl')]))
                            except ValueError:
                                pass
                        for candidate in storage_directory.glob(f'{filename}-*'):
                            if candidate.is_dir():
                                try:
                                    latest = max(latest, int(candidate.name[len(filename) + 1:]))
                                except ValueError:
                                    pass
                        file_no = latest if latest > 0 else 0
                    suffix = '' if file_no == 0 else f'-{file_no:0{storage_util.FILE_INDEX_PADDING}d}'
                    bundled_file = storage_directory / f'{filename}{suffix}' / f'{filename}.neat.pkl'
                    flat_file = storage_directory / f'{filename}{suffix}.neat.pkl'
                    if bundled_file.is_file():
                        filepath = str(bundled_file)
                    elif flat_file.is_file():
                        filepath = str(flat_file)
                else:
                    bundle_directory = storage_directory
                    suffix = '' if file_no in (None, 0) else f'-{file_no:0{storage_util.FILE_INDEX_PADDING}d}'
                    if file_no is None:
                        latest = storage_util.get_last_index(filename, directory, '.neat.pkl')
                        for candidate in bundle_directory.glob(f'{filename}-*'):
                            if candidate.is_dir():
                                try:
                                    latest = max(latest, int(candidate.name[len(filename) + 1:]))
                                except ValueError:
                                    pass
                        if latest > 0:
                            suffix = f'-{latest:0{storage_util.FILE_INDEX_PADDING}d}'
                    bundled_file = bundle_directory / f'{filename}{suffix}' / f'{filename}.neat.pkl'
                    if bundled_file.is_file():
                        filepath = str(bundled_file)
            if filepath is not None:
                state_dict = load(
                    extension='.neat.pkl', items_name='NEAT Population', debug=bool(verbose), filepath=filepath
                )
            else:
                state_dict = load(
                    filename, directory, file_no=file_no, extension='.neat.pkl',
                    items_name='NEAT Population', debug=bool(verbose)
                )
            if state_dict is None:
                return None

        gts = clock.perf_counter()
        # ------------------------------ General ------------------------------ #
        self.genera = state_dict['genera']
        self.genus = self.genera[0]
        self.genus_indexer.set(max(self.genera)+1)
        self.generation = state_dict['generation']
        # The saved config is only restored when explicitly requested (``load_config=True``)
        if load_config and 'config' in state_dict:
            self.config = state_dict['config']
            self.species_set.config = self.config
            self.reproduction._config = self.config
            self.reproduction.stagnation._config = self.config
            self.reproduction.stagnation.fitness_criterion = self.config.stagnation.species_fitness_func
            if hasattr(self, 'fitness_criterion'):
                self.fitness_criterion = getattr(np, self.config.general.fitness_criterion)
        # ------------------------------ Genomes ------------------------------ #
        genomes_data = state_dict['genomes']
        ts, ud, ut = clock.perf_counter(), 0, len(genomes_data)
        self.genomes = Dict.empty(INT, GENOME)
        for gs in genomes_data:
            genome = load_genome(gs)
            self.genomes[genome.key] = genome
            ud += 1
            eta(ts, ud, ut, f'loading {len(genomes_data)}')
        self.reproduction.genome_indexer.set(state_dict['genome_indexer'])
        if verbose:
            print(f"\rloaded genomes in {round(clock.perf_counter() - ts, 2)}s")
        # ------------------------------ Modules' Params and Architecture ------------------------------ #
        module_state_dicts = state_dict['module_state_dicts']
        saved_module_groups = state_dict.get(
            'module_groups', {genus: genus for genus in module_state_dicts}
        )
        saved_group_ids = list(dict.fromkeys(saved_module_groups[genus] for genus in self.genera))
        provided_modules = {}
        for module in self.modules.values():
            provided_modules.setdefault(id(module), module)
        if len(provided_modules) != len(saved_group_ids):
            raise ValueError(
                f"Population state has {len(saved_group_ids)} distinct modules, "
                f"but {len(provided_modules)} were provided"
            )
        modules_by_group = dict(zip(saved_group_ids, provided_modules.values()))
        self.modules = {
            genus: modules_by_group[saved_module_groups[genus]] for genus in self.genera
        }
        for group_id, module in modules_by_group.items():
            neat_dict = module_state_dicts[group_id]
            module.genus = group_id
            for m in module.neat_modules():
                m.genus = group_id
                m.updated = False
            for p in module.neat_parameters():
                p.reset()
            
            module.load_neat_dict(
                neat_dict,
                strict=strict,
                verbose=verbose and verbose >= 2
            )
            
            module_genomes = {
                genome.key: genome for genome in self.genomes.values()
                if saved_module_groups[genome.genus] == group_id
            }
            module.update(module_genomes)
        # ------------------------------ Evaluations ------------------------------ #
        best_genome_keys = state_dict.get('best_genomes')
        if best_genome_keys is not None:
            self.best_genomes = {
                genus: self.genomes.get(key) if key is not None else None
                for genus, key in best_genome_keys.items()
            }
        avatars = state_dict.get('avatars')
        if avatars is not None:
            self.legends = {genus: [self.genomes[key] for key in keys if key in self.genomes] for genus, keys in avatars.items()}
        saved_rankings = state_dict.get('rankings')
        if saved_rankings is not None:
            self.rankings = {
                genus: {key: self.genomes[key] for key in keys if key in self.genomes}
                for genus, keys in saved_rankings.items()
            }
        self.survival_rate = state_dict.get('survival_rate', getattr(self, 'survival_rate', None))
        self.loop_idx = state_dict.get('loop_idx', getattr(self, 'loop_idx', 0))
        # ------------------------------ Species ------------------------------ #
        species_data = state_dict['species']
        ts, ud, ut = clock.perf_counter(), 0, len(species_data)
        self.species_set.species = Dict.empty(INT, SPECIES)
        for ss in species_data:
            specie = load_species(self.genomes, ss)
            self.species_set.species[specie.key] = specie
            ud += 1
            eta(ts, ud, ut, 'loading species')
        self.species_set.species_indexer.set(state_dict['species_indexer'])
        if verbose:
            print(f"\rloaded species in {round(clock.perf_counter() - ts, 2)}s")
            print(f"Loaded NEAT Population species in {round(clock.perf_counter() - gts, 2)}s")

        speciate(
            self.config, self.genera, self.modules, self.species_set, self.genomes, self.generation,
            tpb=self.threads_per_block, verbose=verbose if verbose and verbose >= 2 else False
        )

    def load_dict(
        self, save_state: dict = None, name: str = None, directory: str = None, file_no: int = None,
        strict: bool = True, verbose: int = None, load_config: bool = False
    ):
        return self._load_state_dict(
            state_dict=save_state, filename=name, directory=directory, file_no=file_no,
            strict=strict, verbose=verbose, load_config=load_config
        )

    def crop(self, keys: Union[int, Iterable[int]], device: torch.device | str = None):
        if not isinstance(keys, Iterable):
            keys = [keys]
        if device is None:
            device = list(list(self.modules.values())[0].parameters())[0].device

        def remove(dictionary: dict, key):
            if key in dictionary:
                del dictionary[key]

        genomes_to_keep = {key: self.genomes[key] for key in keys if key in self.genomes}
        module_groups = {}
        for genus, module in self.modules.items():
            module_groups.setdefault(id(module), (module, []))[1].append(genus)
        for module, genera_group in module_groups.values():
            new_genomes = {
                key: genome for key, genome in genomes_to_keep.items()
                if genome.genus in genera_group
            }
            if not new_genomes:
                continue
            new_updates = {}
            for parameter in module.neat_parameters():
                indices_to_keep = torch.tensor(
                    [parameter.mapping[key] for key in new_genomes],
                    device=parameter.device if not device else device, dtype=torch.int64
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

        genera_to_remove = [genus for genus in self.genera if not any(
            genome.genus == genus for genome in genomes_to_keep.values()
        )]
        for genus in genera_to_remove:
            remove(self.modules, genus)
            remove(self.best_genomes, genus)
            remove(self.rankings, genus)
            remove(self.legends, genus)
            self.genera.remove(genus)

        if self.genera and self.genus not in self.genera:
            self.genus = self.genera[0]
        for module, genera_group in module_groups.values():
            remaining_genera = [genus for genus in genera_group if genus in self.genera]
            if remaining_genera and module.genus not in remaining_genera:
                module.genus = remaining_genera[0]
                for sub_module in module.neat_modules():
                    sub_module.genus = module.genus

        genomes_to_remove = [key for key in self.genomes.keys() if key not in genomes_to_keep]
        for key in genomes_to_remove:
            remove(self.genomes, key)

        torch.cuda.empty_cache()

    def save(
        self,
        filename: str = None,
        directory: str = None,
        file_no: int = None,
        replace: bool = False,
        debug: bool = True,
        filepath: str | None = None,
        extra_files: Mapping[str, str | bytes | bytearray | PathLike[str]] = None,
    ) -> tuple[bool, int]:
        """Save the population state, optionally alongside related files.

        Extra file values may be text, bytes, or source ``Path`` objects. Relative keys
        determine their paths inside the bundle. ``filepath`` is used exactly as given;
        with extra files, those files are placed in a sibling folder named after it.
        """
        return self._save_state_dict(
            filename=filename, directory=directory, file_no=file_no, replace=replace,
            filepath=filepath, extra_files=extra_files, debug=debug
        )

    def load(
        self,
        filename: str | dict[str, Any] = None,
        directory: str = None,
        file_no: int = None,
        strict: bool = True,
        debug: bool = True,
        filepath: str | None = None,
        state_dict: dict[str, Any] = None,
        load_config: bool = False,
    ) -> 'Population':
        """Restore this population from a state dictionary or saved file.

        The saved config is ignored unless ``load_config=True``; the current config is kept otherwise.
        """
        if isinstance(filename, dict) and state_dict is None:
            state_dict = filename
            filename = None
        self._load_state_dict(
            state_dict=state_dict, filename=filename, directory=directory, file_no=file_no,
            filepath=filepath, strict=strict, verbose=2 if debug else 0, load_config=load_config
        )
        if debug:
            print(f"Successfully loaded Population with {len(self.genomes)} genomes and "
                  f"{len(self.species_set.species)} species")
        return self
