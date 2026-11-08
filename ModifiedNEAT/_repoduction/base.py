
from itertools import count

from ModifiedNEAT._repoduction.types import ReproductionMethods
from ModifiedNEAT.config import Config
from ModifiedNEAT.nn.base import NeatModule
from ModifiedNEAT.nn.genome import Genome
from ModifiedNEAT.reporter.base import ReporterSet
from ModifiedNEAT.species import SpeciesSet
from ModifiedNEAT.stagnation import Stagnation

# TODO: Re-support this module for NEAT CPU capability

class Reproduction:
    def __init__(self, reporters: ReporterSet, configuration: Config):
        self._genome_indexer = count(1)
        self._reporters = reporters
        self._stagnation = Stagnation(configuration, reporters)
        self._config = configuration
        self.ancestors: dict[int, tuple[Genome, Genome]] = {}
        self.types = ReproductionMethods()

    def create_new(self, pop_size: int, ModifiedNEAT: list[NeatModule]) -> dict[int, Genome]:
        raise NotImplementedError(
            "The legacy reproduction implementation is unsupported; "
            "use ModifiedNEAT.reproduction.Reproduction."
        )

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

    def reproduce(self, species_set: SpeciesSet, population: dict[int, Genome], ModifiedNEAT: list[NeatModule], generation: int,
                  repro_function: callable = None):
        raise NotImplementedError(
            "The legacy reproduction implementation is unsupported; "
            "use ModifiedNEAT.reproduction.Reproduction."
        )
