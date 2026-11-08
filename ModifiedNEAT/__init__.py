
# Core imports - these are imported immediately
# Initialize device early
from . import base, config, util
from .base import (
    cuda_is_available,
    get_device,
    get_tpb,
    initialize,
    reproduce,
    set_device,
    set_tpb,
    speciate,
)

# Import key utilities first (no dependencies on other NEAT modules)
from .config import Config

# Standard imports for commonly used components
from .nn import Genome, Model, NeatModule, NeatParameter, activations, modules
from .population import Population
from .reporter import BaseReporter, ReporterSet, StdOutReporter
from .reproduction import Reproduction
from .rl import NEAT, PPO
from .species import Species, SpeciesSet
from .stagnation import Stagnation
from .util.fancy_text import CM, Fore

# Lazy imports for less commonly used modules
_lazy_modules = {
    'nn': 'ModifiedNEAT.nn',
    'optim': 'ModifiedNEAT.optim',
    'reproduction': 'ModifiedNEAT.reproduction',
    'reporter': 'ModifiedNEAT.reporter',
    'rl': 'ModifiedNEAT.rl',
    'population': 'ModifiedNEAT.population',
    'species': 'ModifiedNEAT.species',
    'stagnation': 'ModifiedNEAT.stagnation',
    'multproc': 'ModifiedNEAT.multproc',
}

__all__ = [
    'BaseReporter', 'CM', 'Config', 'Fore', 'Genome', 'Model', 'NEAT', 'NeatModule',
    'NeatParameter', 'PPO', 'Population', 'ReporterSet', 'Reproduction', 'Species',
    'SpeciesSet', 'Stagnation', 'StdOutReporter', 'activations', 'base',
    'config', 'cuda_is_available', 'get_device', 'get_tpb', 'initialize', 'modules',
    'multproc', 'nn', 'optim', 'population', 'reproduce', 'reporter', 'reproduction',
    'rl', 'set_device', 'set_tpb', 'species', 'speciate', 'stagnation', 'util',
]

def __getattr__(name):
    """Lazy import for less commonly used submodules."""
    if name in _lazy_modules:
        import importlib
        module = importlib.import_module(_lazy_modules[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
