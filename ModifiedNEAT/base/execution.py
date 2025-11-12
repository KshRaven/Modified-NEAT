
from ModifiedNEAT.config import Config
from ModifiedNEAT.nn import NeatModule, Genome
from ModifiedNEAT.species import SpeciesSet
from ModifiedNEAT.stagnation import Stagnation
from ModifiedNEAT.reporter import ReporterSet
from ModifiedNEAT.base import cpu

import torch
import warnings

DEVICE = cpu
_device = 'cpu'

def device():
    return _device

def cuda_is_available():
    try:
        from numba import cuda
        return cuda.is_available()
    except ModuleNotFoundError as e:
        warnings.warn(message=str(e), category=ImportWarning, stacklevel=2)
        return False

def set_device(name):
    global DEVICE, _device
    if isinstance(name, torch.device):
        name = name.type
    if 'cpu' in name:
        DEVICE = cpu
        _device = 'cpu'
    elif 'cuda' in name:
        if cuda_is_available():
            from ModifiedNEAT.base import gpu
            DEVICE = gpu
            _device = 'cuda'
        else:
            warnings.warn("No CUDA device is available!", category=UserWarning, stacklevel=2)
            set_device('cpu')
    else:
        raise ValueError(f"Unsupported device: '{name}'")

def initialize(config: Config, module: NeatModule, tpb: int = 1, verbose: int | bool | None = None):
    return DEVICE.initialize(
        config, module, tpb, verbose
    )

def speciate(config: Config, genera: list[int], modules: dict[int, NeatModule],
             species_set: SpeciesSet, population: dict[int, Genome],
             generation: int, tpb=10, verbose: int = None):
   return DEVICE.speciate(
       config, genera, modules, species_set, population, generation, tpb, verbose
   )

def reproduce(
        config: Config, genera: list[int], modules: dict[int, NeatModule], population: dict[int, Genome],
        ancestors: dict[int, tuple[Genome, Genome]], generation: int, to_delete: list[int], genome_indexer: int,
        stagnation: Stagnation, species_set: SpeciesSet, reporters: ReporterSet,
        tpb=1, verbose: int | bool | None = None):
    return DEVICE.reproduce(
        config, genera, modules, population, ancestors, generation, to_delete, genome_indexer,
        stagnation, species_set, reporters, tpb, verbose
    )

if __name__ == '__main__':
    print(f"Device Type: {DEVICE}")
    print(f"CUDA Available: {cuda_is_available()}")
    print(f"Device Type: {DEVICE}")
    print(f"Initialization -> {initialize}")
    print(f"Speciation -> {speciate}")
    print(f"Reproduction -> {reproduce}")