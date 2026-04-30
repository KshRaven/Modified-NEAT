
# Core imports - these are imported immediately
from . import config
from . import util

# Import key utilities first (no dependencies on other NEAT modules)
from .config import Config
from .util.fancy_text import CM, Fore

# Initialize device early
from . import base
from .base import (
    cuda_is_available, set_device, get_device, 
    set_tpb, get_tpb,
    initialize, reproduce, speciate
)

# Standard imports for commonly used components
from .nn import modules, activations, NeatModule, NeatParameter, Model, Genome
from .reporter import BaseReporter, ReporterSet, StdOutReporter
from .population import Population
from .reproduction import Reproduction
from .species import Species, SpeciesSet
from .stagnation import Stagnation
from .rl import NEAT

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

def __getattr__(name):
    """Lazy import for less commonly used submodules."""
    if name in _lazy_modules:
        import importlib
        module = importlib.import_module(_lazy_modules[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
