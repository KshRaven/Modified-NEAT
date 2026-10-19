# ---------- CPU (numba, prange) implementation of the genome methods ---------- #
# Device independent logic lives in ModifiedNEAT.base.{initialization, reproduction, speciation, util, functional};
# this package only holds the numba kernels and the CPU backend that plugs them into it.

from ModifiedNEAT.base import initialization as _initialization
from ModifiedNEAT.base import reproduction as _reproduction
from ModifiedNEAT.base import speciation as _speciation
from ModifiedNEAT.config import Config
from ModifiedNEAT.nn import NeatModule, Genome
from ModifiedNEAT.species import SpeciesSet
from ModifiedNEAT.stagnation import Stagnation
from ModifiedNEAT.reporter import ReporterSet

from . import initialization, mutation, reproduction, speciation
from .backend import BACKEND, CpuBackend
from .mutation import mutate


def initialize(config: Config, module: NeatModule, tpb=1, verbose: int = None):
    return _initialization.initialize(BACKEND, config, module, tpb, verbose)


def reproduce(
        config: Config, genera: list[int], modules: dict[int, NeatModule], population: dict[int, Genome],
        ancestors: dict[int, tuple[Genome, Genome]], generation: int, to_delete: list[int], genome_indexer: int,
        stagnation: Stagnation, species_set: SpeciesSet, reporters: ReporterSet,
        tpb=1, verbose: int = None):
    return _reproduction.reproduce(
        BACKEND, config, genera, modules, population, ancestors, generation, to_delete, genome_indexer,
        stagnation, species_set, reporters, tpb, verbose
    )


def speciate(config: Config, genera: list[int], modules: dict[int, NeatModule],
             species_set: SpeciesSet, population: dict[int, Genome],
             generation: int, tpb=10, verbose: int = None):
    return _speciation.speciate(BACKEND, config, genera, modules, species_set, population, generation, tpb, verbose)


# Keep the old module level entry points (e.g. `base.cpu.reproduction.reproduce`) available
initialization.initialize = initialize
reproduction.reproduce = reproduce
speciation.speciate = speciate
