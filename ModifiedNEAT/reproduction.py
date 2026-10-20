from ModifiedNEAT.config import Config
from ModifiedNEAT.nn.base import NeatModule
from ModifiedNEAT.nn.genome import Genome, INT
from ModifiedNEAT.reporter.base import ReporterSet
from ModifiedNEAT.stagnation import Stagnation
from ModifiedNEAT.util.qol import Indexer
from ModifiedNEAT.base import initialize, get_tpb

from numba import types
from numba.typed import Dict

GENOME  = Genome.class_type.instance_type
GENOME_TUPLE = types.Tuple([GENOME, GENOME])


class Reproduction:
    """
    Holds the reproduction state of a population (genome id counter, ancestry, stagnation tracker) and creates the
    initial genomes. Spawn computation, parent selection, crossover and mutation live in ``ModifiedNEAT.base``
    (``base.reproduce``), which runs on the CPU or GPU backend.
    """
    genome_indexer = Indexer(0)

    def __init__(self, reporters: ReporterSet, configuration: Config):
        self._reporters = reporters
        self.stagnation = Stagnation(configuration, reporters)
        self._config = configuration
        self.ancestors: dict[int, tuple[Genome, Genome]] = Dict.empty(INT, GENOME_TUPLE)

    def create_new(self, genus: int, pop_size: int, module: NeatModule, tpb: int = None, verbose: int = None) -> dict[int, Genome]:
        assert module.genus == genus
        if tpb is None:
            tpb = get_tpb()
        # Set genomes
        genomes = Dict.empty(INT, GENOME)
        for _ in range(pop_size):
            gid = next(self.genome_indexer)
            genome = Genome(gid, genus)
            genomes[gid] = genome
        # Initialize each parameter
        for m in module.neat_modules():
            m.updated = False
        module.update(genomes, verify=False)
        initialize(self._config, module, tpb, verbose)
        return genomes