"""
Seeding of the ModifiedNEAT backend (``config.general.seed``).

The seed only drives *private* generators owned by this package, it never calls ``np.random.seed``,
``torch.manual_seed``, ``cp.random.seed`` or ``torch.cuda.manual_seed``. Anything else running in the same process
(an Env, examples, user code) keeps its own random state and can use a different seed (or none).

* ``seed = None``  -> every generator is initialised from OS entropy: no reproducibility.
* ``seed = <int>`` -> all generators are derived from that int (independent streams), reproducible.

CPU mode uses NumPy + PyTorch(cpu) generators. CuPy and PyTorch(cuda) generators are created lazily, only when a
GPU backend asks for them, so CPU-only machines never import cupy or touch torch.cuda.
"""

import numpy as np
import torch

_UNSET = object()
_applied = _UNSET          # seed that is currently applied (None = entropy)
_entropy = None            # SeedSequence entropy of the current run
_streams: dict = {}        # lazily created generators


# Stream ids, one independent stream per library
_NUMPY, _TORCH_CPU, _CUPY, _TORCH_CUDA = 0, 1, 2, 3


def _check(seed):
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
        raise TypeError(f"seed must be an int or None, got {type(seed).__name__}")
    if seed is not None and seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")


def set_seed(seed: int | None):
    """(Re)seed all backend generators. ``None`` -> non-reproducible (fresh OS entropy)."""
    global _applied, _entropy
    _check(seed)
    _streams.clear()
    _applied = None if seed is None else int(seed)
    _entropy = np.random.SeedSequence(_applied).entropy


def configure(seed: int | None):
    """
    Apply ``config.general.seed`` if it is not the one already applied. Called at the start of initialize /
    speciate / reproduce, so the streams keep advancing across generations instead of restarting every call.
    Call :func:`set_seed` explicitly to restart the same sequence (e.g. a second run in the same process).
    """
    if _applied is _UNSET or seed != _applied:
        set_seed(seed)


def get_seed() -> int | None:
    return None if _applied is _UNSET else _applied


def _stream_seed(stream: int) -> int:
    if _applied is _UNSET:
        set_seed(None)
    ss = np.random.SeedSequence(_entropy, spawn_key=(stream,))
    return int(ss.generate_state(1, np.uint32)[0])


def numpy_rng() -> np.random.Generator:
    if _NUMPY not in _streams:
        _streams[_NUMPY] = np.random.default_rng(_stream_seed(_NUMPY))
    return _streams[_NUMPY]


def torch_generator(device: str = 'cpu') -> torch.Generator:
    """PyTorch generator for 'cpu', or for 'cuda' (only created when requested, i.e. in GPU mode)."""
    is_cuda = 'cuda' in str(device)
    stream = _TORCH_CUDA if is_cuda else _TORCH_CPU
    if stream not in _streams:
        generator = torch.Generator(device='cuda' if is_cuda else 'cpu')
        generator.manual_seed(_stream_seed(stream))
        _streams[stream] = generator
    return _streams[stream]


def cupy_rng():
    """CuPy generator. cupy is imported here, so it is only ever loaded by the GPU backend."""
    if _CUPY not in _streams:
        import cupy as cp
        _streams[_CUPY] = cp.random.default_rng(_stream_seed(_CUPY))
    return _streams[_CUPY]


def numba_seed() -> int:
    """Seed to hand to a numba kernel (numba has its own RNG state); drawn from the numpy stream."""
    return int(numpy_rng().integers(0, 2 ** 32, dtype=np.uint64))
