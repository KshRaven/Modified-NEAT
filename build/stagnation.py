
from build.config import Config
from build.nn.genome import Genome, FLOAT, INT, BOOL
from build.species import SpeciesSet, Species, SPECIES
from build.reporter.base import ReporterSet
from build.util.fancy_text import CM, Fore

from numba import types, njit, optional, prange
from numba.typed import List, Dict

import numpy as np

GENOME  = Genome.class_type.instance_type


class Stagnation(object):
    def __init__(self, config: Config, reporters: ReporterSet):
        self._config = config
        self.reporters = reporters
        self.fitness_criterion = config.stagnation.species_fitness_func

    @staticmethod
    @njit(nogil=True)
    def get_species_data(species_data: list[tuple[int, Species]], species_dict: dict[int, Species],
                         generation: int, criterion: str):
        mapping = List(species_dict.keys())
        for idx in prange(len(species_dict)):
            sid = mapping[idx]
            specie = species_dict[sid]
            if specie.fitness_history:
                prev_fitness = max(specie.fitness_history)
            else:
                prev_fitness = -np.inf

            array = specie.get_fitnesses()
            if criterion == 'max':
                specie.fitness = max(array)
            elif criterion == 'min':
                specie.fitness = min(array)
            elif criterion == 'mean':
                specie.fitness = sum(array) / len(array)
            else:
                raise ValueError(f"Unexpected fitness_criterion: {criterion}")
            specie.fitness_history.append(specie.fitness)
            specie.adjusted_fitness = None
            if prev_fitness is None or specie.fitness > prev_fitness:
                specie.last_improved = generation

            species_data.append((sid, specie))

    @staticmethod
    @njit(nogil=True)
    def get_result(result: list[tuple[int, Species, bool]], species_data: list[tuple[int, Species]],
                   generation: int, species_elitism: int, max_stagnation: int):
        species_fitnesses: list[float] = List.empty_list(FLOAT)
        num_non_stagnant = len(species_data)

        for idx in prange(len(species_data)):
            # Override stagnant state if marking this species as stagnant would
            # result in the total number of species dropping below the limit.
            # Because species are in ascending fitness order, less fit species
            # will be marked as stagnant first.
            sid, specie = species_data[idx]
            stagnant_time = generation - specie.last_improved
            is_stagnant = False
            if num_non_stagnant > species_elitism:
                is_stagnant = stagnant_time >= max_stagnation

            if (len(species_data) - idx) <= species_elitism:
                is_stagnant = False

            if is_stagnant:
                num_non_stagnant -= 1

            result.append((sid, specie, is_stagnant))
            species_fitnesses.append(specie.fitness)

    def update(self, species_set: SpeciesSet, generation: int):
        """
        Required interface method. Updates species fitness history information,
        checking for ones that have not improved in max_stagnation generations,
        and - unless it would result in the number of species dropping below the configured
        species_elitism parameter if they were removed,
        in which case the highest-fitness species are spared -
        returns a list with stagnant species marked for removal.
        """

        species_data: list[tuple[int, Species]] = List.empty_list(types.Tuple([INT, SPECIES]))
        self.get_species_data(species_data, species_set.species, generation, self.fitness_criterion)

        # Sort in ascending fitness order.
        species_data = sorted(species_data, key=lambda s: s[1].fitness)

        result: list[tuple[int, Species, bool]] = List.empty_list(types.Tuple([INT, SPECIES, BOOL]))
        self.get_result(result, species_data, generation,
                        self._config.stagnation.species_elitism, self._config.stagnation.max_stagnation)

        return result
