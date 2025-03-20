
from ModifiedNEAT.config import Config
from ModifiedNEAT.nn.genome import Genome
from ModifiedNEAT.reporter.base import BaseReporter
from ModifiedNEAT.util.fancy_text import CM, Fore

import time as clock
import numpy as np


class StdOutReporter(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""

    def __init__(self, show_species_detail: bool):
        self.show_species_detail = show_species_detail
        self.generation: int = None
        self.generation_start_time: int = None
        self.generation_times: list[int] = []
        self.num_extinctions = 0

    def post_reproduction(self, config: Config, population: dict[int, Genome], species):
        pass

    def start_generation(self, generation: int):
        self.generation = generation
        print(f"\n ****** {CM(f'Running generation {generation+1}', Fore.LIGHTYELLOW_EX)} ****** \n")
        self.generation_start_time = clock.time()

    def end_generation(self, config: Config, population: dict[int, Genome], species_set):
        genome_num = len(population)
        species_num = len(species_set.species)
        if self.show_species_detail:
            print('Population of {0:d} members in {1:d} species:'.format(genome_num, species_num))
            sids = list(species_set.species.keys())
            sids.sort()
            print("   ID   age  size  fitness  adj fit  stag")
            print("  ====  ===  ====  =======  =======  ====")
            for sid, species in species_set.species.items():
                age = self.generation - species.created
                member_num = len(species.members)
                fitness = "--" if species.fitness is None else f"{species.fitness:.1f}"
                fitness_adj = "--" if species.adjusted_fitness is None else f"{species.adjusted_fitness:.3f}"
                gen_start = self.generation - species.last_improved
                print(f"{sid: >4}  {age: >3}  {member_num: >4}  {fitness: >7}  {fitness_adj: >7}  {gen_start: >4}")
        else:
            print('Population of {0:d} members in {1:d} species'.format(genome_num, species_num))

        elapsed = clock.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        print(f'Total extinctions: {self.num_extinctions:d}')
        if len(self.generation_times) > 1:
            print(f"Generation time: {elapsed:.3f} sec ({average:.3f} average)")
        else:
            print(f"Generation time: {elapsed:.3f} sec")

    def post_evaluate(self, config: Config, population: dict[int, Genome], species, best_genome: Genome):
        fitness  = [genome.fitness for genome in population.values()]
        fit_mean = np.mean(fitness)
        fit_std  = np.std(fitness)
        best_genomes = best_genome
        if isinstance(best_genomes, Genome):
            best_genome = best_genomes
            best_key = best_genome.key
            best_fitness = best_genome.fitness
            best_species_id = species.get_species_id(best_key)
        else:
            best_key = tuple([genome.key for genome in best_genomes])
            best_fitness = tuple([round(genome.fitness, 4) for genome in best_genomes])
            best_species_id = tuple([species.get_species_id(key) for key in best_key])
        print(f"\nPopulation's average fitness: {fit_mean:3.5f} std_dev: {fit_std:3.5f}")
        print(f"Best fitness: {CM(f'{best_fitness}', Fore.LIGHTGREEN_EX)}"
              # TODO: Enable Genome complexity calculation, or scrap it
              # f" - size: { best_genome.size()!r}"
              f" - species {best_species_id} - id {best_key}")

    def complete_extinction(self):
        self.num_extinctions += 1
        print('All species extinct.')

    def found_solution(self, config: Config, generation: int, best: Genome):
        print(f"\nBest individual in generation {CM(self.generation, Fore.LIGHTGREEN_EX)} "
              f"meets fitness threshold"
              # TODO: Enable Genome complexity calculation, or scrap it
              # f" - complexity: {best.size()!r}"
              )

    def species_stagnant(self, sid: int, species):
        if self.show_species_detail:
            print(f"\nSpecies {sid} with {len(species.members)} members is stagnated: removing it")

    def info(self, msg: str):
        print(msg)
