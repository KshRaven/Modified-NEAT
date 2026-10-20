from ModifiedNEAT.config import Config
from ModifiedNEAT.constants import INT, FLOAT, BOOL
from ModifiedNEAT.species import SpeciesSet, Species, SPECIES
from ModifiedNEAT.reporter.base import ReporterSet

from numba import types, njit
from numba.typed import List, Dict

import numpy as np


class Stagnation(object):
    def __init__(self, config: Config, reporters: ReporterSet):
        self._config = config
        self.reporters = reporters
        self.fitness_criterion = config.stagnation.species_fitness_func

    @staticmethod
    @njit(nogil=True)
    def get_species_data(species_data: list[tuple[int, Species]], species_dict: dict[int, Species],
                         generation: int, criterion: str):
        # NOTE: sequential on purpose. Every iteration appends to the shared `species_data` list, which would race
        #       (and scramble the order) in a parallel loop, and there are only a handful of species anyway.
        mapping = List(species_dict.keys())
        for idx in range(len(species_dict)):
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
            if specie.fitness > prev_fitness:
                specie.last_improved = generation

            species_data.append((sid, specie))

    @staticmethod
    @njit(nogil=True)
    def get_result(result: list[tuple[int, Species, bool]], species_data: list[tuple[int, Species]],
                   generation: int, species_elitism: int, max_stagnation: int):
        species_fitnesses: list[float] = List.empty_list(FLOAT)
        num_non_stagnant = len(species_data)

        # NOTE: sequential on purpose. `num_non_stagnant` is decremented as species are marked stagnant and later
        #       iterations depend on it (weakest species go first, `species_elitism` protects the best ones), and
        #       `result` has to keep the ascending-fitness order.
        for idx in range(len(species_data)):
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

        # Fitness of every species in the same (ascending) order as `result`, handy for reporting
        return species_fitnesses

    def update(self, genus: int, species_set: SpeciesSet, generation: int):
        """
        Required interface method. Updates species fitness history information,
        checking for ones that have not improved in max_stagnation generations,
        and - unless it would result in the number of species dropping below the configured
        species_elitism parameter if they were removed,
        in which case the highest-fitness species are spared -
        returns a list with stagnant species marked for removal.
        """

        species_data: list[tuple[int, Species]] = List.empty_list(types.Tuple([INT, SPECIES]))
        self.get_species_data(species_data, Dict([(s.key, s) for s in species_set.species.values() if s.genus == genus]),
                              generation, self.fitness_criterion)

        # Sort in ascending fitness order.
        ordered = sorted(species_data, key=lambda s: s[1].fitness)
        # Back to a typed list: numba cannot take a reflected Python list of jitclass instances
        species_data = List.empty_list(types.Tuple([INT, SPECIES]))
        for item in ordered:
            species_data.append(item)

        result: list[tuple[int, Species, bool]] = List.empty_list(types.Tuple([INT, SPECIES, BOOL]))
        self.get_result(result, species_data, generation,
                        self._config.stagnation.species_elitism, self._config.stagnation.max_stagnation)

        return result