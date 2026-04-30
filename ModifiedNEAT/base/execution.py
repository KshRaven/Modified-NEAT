
from ModifiedNEAT.config import Config
from ModifiedNEAT.nn import NeatModule, Genome
from ModifiedNEAT.species import SpeciesSet
from ModifiedNEAT.stagnation import Stagnation
from ModifiedNEAT.reporter import ReporterSet
from ModifiedNEAT.base import cpu

import os
import torch
import warnings


# Environment variables for configuration
DEVICE_ENV_NAME = 'MODIFIEDNEAT_DEVICE'
TPB_ENV_NAME = 'MODIFIEDNEAT_TPB'


# Initialize device from environment variable or default to CPU
try:
    _default_device = os.environ.get(DEVICE_ENV_NAME, 'cpu').lower()
    if _default_device not in ('cpu', 'cuda'):
        warnings.warn(f"Invalid device '{_default_device}' in {DEVICE_ENV_NAME}, defaulting to 'cpu'", UserWarning)
        _default_device = 'cpu'
except ValueError:
    _default_device = 'cpu'

# Initialize threads per block from environment variable or default to 10
# Value of 10 ensures 10^3 = 1000 total threads (under 1024 limit for most CUDA kernels)
try:
    _default_tpb = int(os.environ.get(TPB_ENV_NAME, '10'))
    # Validate range
    if _default_tpb < 1 or _default_tpb > 32:
        warnings.warn(f"Invalid TPB value {_default_tpb}, using default 10", UserWarning)
        _default_tpb = 10
except ValueError:
    _default_tpb = 10

DEVICE = cpu
_device = _default_device
_tpb = _default_tpb

def get_device():
    """Get the current device type."""
    return _device

def cuda_is_available():
    """Check if CUDA is available."""
    try:
        from numba import cuda
        return cuda.is_available()
    except ModuleNotFoundError as e:
        warnings.warn(message=str(e), category=ImportWarning, stacklevel=2)
        return False

def set_device(name, persist_env=False):
    f"""
    Set the device for ModifiedNEAT execution.
    
    Parameters
    ----------
    name : str or torch.device
        Device name ('cpu' or 'cuda').
    persist_env : bool, optional
        If False (default), also updates the system environment variable
        {DEVICE_ENV_NAME} so the setting persists for the current process
        and any subprocesses spawned from it.
    
    Raises
    ------
    ValueError
        If device type is not supported.
    UserWarning
        If CUDA is requested but not available.
    """
    global DEVICE, _device
    
    if isinstance(name, torch.device):
        name = name.type
    
    name = name.lower()
    
    if 'cpu' in name:
        DEVICE = cpu
        _device = 'cpu'
        if persist_env:
            os.environ[DEVICE_ENV_NAME] = 'cpu'
    elif 'cuda' in name:
        if cuda_is_available():
            from ModifiedNEAT.base import gpu
            DEVICE = gpu
            _device = 'cuda'
            if persist_env:
                os.environ[DEVICE_ENV_NAME] = 'cuda'
        else:
            warnings.warn("No CUDA device is available!", category=UserWarning, stacklevel=2)
            set_device('cpu', persist_env=persist_env)
    else:
        raise ValueError(f"Unsupported device: '{name}'")

def get_device_env():
    """Get the current device setting from environment variable."""
    return os.environ.get(DEVICE_ENV_NAME, 'cpu')

def set_device_env(name: str):
    """
    Permanently set the device environment variable for this process and subprocesses.
    
    Parameters
    ----------
    name : str
        Device name ('cpu' or 'cuda').
    
    Notes
    -----
    This updates os.environ and can be used to configure the default device
    before importing modules that depend on it. To change the active device
    in the current process, use set_device() instead.
    """
    name = name.lower()
    if name not in ('cpu', 'cuda'):
        raise ValueError(f"Unsupported device: '{name}'")
    os.environ[DEVICE_ENV_NAME] = name

def get_tpb():
    """Get the current threads per block (TPB) setting."""
    return _tpb

def set_tpb(value: int, persist_env=True):
    """
    Set the threads per block (TPB) for GPU kernel execution.
    
    Parameters
    ----------
    value : int
        Threads per block (typically 4-32). Default is 10, which ensures 10^3 = 1000 total
        threads (under the 1024 limit for most CUDA kernels).
    persist_env : bool, optional
        If True (default), also updates the system environment variable MODIFIEDNEAT_TPB
        so the setting persists for the current process and any subprocesses spawned from it.
    
    Raises
    ------
    ValueError
        If value is not in valid range [1, 32].
    """
    global _tpb
    
    if not isinstance(value, int) or value < 1 or value > 32:
        raise ValueError(f"TPB must be an integer in range [1, 32], got {value}")
    
    _tpb = value
    if persist_env:
        os.environ[TPB_ENV_NAME] = str(value)

def get_tpb_env():
    """Get the threads per block setting from environment variable."""
    try:
        return int(os.environ.get(TPB_ENV_NAME, '10'))
    except ValueError:
        return 10

def set_tpb_env(value: int):
    """
    Permanently set the threads per block environment variable for this process and subprocesses.
    
    Parameters
    ----------
    value : int
        Threads per block (typically 4-32).
    
    Raises
    ------
    ValueError
        If value is not in valid range [1, 32].
    
    Notes
    -----
    This updates os.environ only. Use set_tpb() to also update the active Python session.
    """
    if not isinstance(value, int) or value < 1 or value > 32:
        raise ValueError(f"TPB must be an integer in range [1, 32], got {value}")
    os.environ[TPB_ENV_NAME] = str(value)

def initialize(
        config: Config, module: NeatModule, tpb: int = 10, verbose: int | bool | None = None
    ):
    return DEVICE.initialize(
        config, module, tpb, verbose
    )

def speciate(
        config: Config, genera: list[int], modules: dict[int, NeatModule],
        species_set: SpeciesSet, population: dict[int, Genome],
        generation: int, tpb=10, verbose: int = None
    ):
   return DEVICE.speciate(
       config, genera, modules, species_set, population, generation, tpb, verbose
   )

def reproduce(
        config: Config, genera: list[int], modules: dict[int, NeatModule], population: dict[int, Genome],
        ancestors: dict[int, tuple[Genome, Genome]], generation: int, to_delete: list[int], genome_indexer: int,
        stagnation: Stagnation, species_set: SpeciesSet, reporters: ReporterSet,
        tpb=10, verbose: int | bool | None = None
    ):
    return DEVICE.reproduce(
        config, genera, modules, population, ancestors, generation, to_delete, genome_indexer,
        stagnation, species_set, reporters, tpb, verbose
    )

if __name__ == '__main__':
    print(f"Device Type: {DEVICE}")
    print(f"CUDA Available: {cuda_is_available()}")
    print(f"Current Device: {_device}")
    print(f"Environment Device: {get_device_env()}")
    print(f"Initialization -> {initialize}")
    print(f"Speciation -> {speciate}")
    print(f"Reproduction -> {reproduce}")