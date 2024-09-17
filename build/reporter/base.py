
from build.config import Config
from build.nn.genome import Genome


class BaseReporter(object):
    """Definition of the reporter interface expected by ReporterSet."""
    def start_generation(self, generation: int):
        raise NotImplementedError()

    def end_generation(self, config: Config, population: dict[int, Genome], species_set):
        raise NotImplementedError()

    def post_evaluate(self, config: Config, population: dict[int, Genome], species, best_genome: Genome):
        raise NotImplementedError()

    def post_reproduction(self, config: Config, population: dict[int, Genome], species):
        raise NotImplementedError()

    def complete_extinction(self):
        raise NotImplementedError()

    def found_solution(self, config: Config, generation: int, best: Genome):
        raise NotImplementedError()

    def species_stagnant(self, sid: int, species):
        raise NotImplementedError()

    def info(self, msg: str):
        raise NotImplementedError()


class ReporterSet(object):
    """
    Keeps track of the set of reporters
    and gives methods to dispatch them at appropriate points.
    """
    def __init__(self):
        self.reporters: list[BaseReporter] = []

    def add(self, reporter: BaseReporter):
        self.reporters.append(reporter)

    def remove(self, reporter: BaseReporter):
        self.reporters.remove(reporter)

    def start_generation(self, gen: int):
        for r in self.reporters:
            r.start_generation(gen)

    def end_generation(self, config: Config, population: dict[int, Genome], species_set):
        for r in self.reporters:
            r.end_generation(config, population, species_set)

    def post_evaluate(self, config: Config, population: dict[int, Genome], species, best_genome: Genome):
        for r in self.reporters:
            r.post_evaluate(config, population, species, best_genome)

    def post_reproduction(self, config: Config, population: dict[int, Genome], species):
        for r in self.reporters:
            r.post_reproduction(config, population, species)

    def complete_extinction(self):
        for r in self.reporters:
            r.complete_extinction()

    def found_solution(self, config: Config, generation: int, best: Genome):
        for r in self.reporters:
            r.found_solution(config, generation, best)

    def species_stagnant(self, sid: int, species):
        for r in self.reporters:
            r.species_stagnant(sid, species)

    def info(self, msg: str):
        for r in self.reporters:
            r.info(msg)
