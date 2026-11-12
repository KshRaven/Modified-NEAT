from numba import types
from numba.typed import Dict

from ModifiedNEAT.base import get_tpb, initialize
from ModifiedNEAT.config import Config
from ModifiedNEAT.nn.base import NeatModule
from ModifiedNEAT.nn.genome import INT, Genome
from ModifiedNEAT.reporter.base import ReporterSet
from ModifiedNEAT.stagnation import Stagnation
from ModifiedNEAT.util.qol import Indexer

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

    def create_new(
        self, genus: int | list[int], pop_size: int | dict[int, int], module: NeatModule,
        tpb: int = None, verbose: bool | int = False
    ) -> dict[int, Genome]:
        if isinstance(genus, int):
            genus_sizes = {genus: pop_size}
        else:
            genera = list(genus)
            if isinstance(pop_size, dict):
                genus_sizes = pop_size
                if set(genus_sizes) != set(genera):
                    raise ValueError("Genus sizes must match the requested genera")
            else:
                base_size, remainder = divmod(pop_size, len(genera))
                genus_sizes = {
                    genus: base_size + (index < remainder)
                    for index, genus in enumerate(genera)
                }
        if module.genus not in genus_sizes:
            raise ValueError(f"Module genus {module.genus} is not in the requested genera")
        if any(size < 1 for size in genus_sizes.values()):
            raise ValueError("Each genus must have at least one genome")
        if tpb is None:
            tpb = get_tpb()
        genomes = Dict.empty(INT, GENOME)
        for genus, size in genus_sizes.items():
            for _ in range(size):
                gid = next(self.genome_indexer)
                genomes[gid] = Genome(gid, genus)
        for m in module.neat_modules():
            m.updated = False
        module.update(genomes, verify=False)
        initialize(self._config, module, tpb, verbose)
        return genomes