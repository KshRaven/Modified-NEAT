# TODO

## Population and Reproduction

- Support reproduction when genera have unequal numbers of `NeatModule` instances or incompatible `NeatParameters` shapes (`ModifiedNEAT/base/reproduction.py`).
- Implement `Population.reset_keys()` and define how genome, species, and genus identifiers are remapped (`ModifiedNEAT/population.py`).
- Make cross-genus parent-selection weighting configurable instead of relying on the current fixed multiplier (`ModifiedNEAT/base/reproduction.py`).

## Reinforcement Learning

- Add save/load support for current RL algorithms, including the state needed to resume training (`ModifiedNEAT/rl/base.py`, `ModifiedNEAT/rl/neat.py`, `ModifiedNEAT/rl/ppo.py`).
- Implement the CUDA RL backend or document and test the supported CPU fallback behavior (`ModifiedNEAT/rl/functional.py`).
- Review whether RL score normalization should separate genomes by species rather than genus (`ModifiedNEAT/rl/base.py`).

## Neural Modules

- Bring `ConvSelfAttention` and `ConvCrossAttention` up to feature parity for differential/branch modes and attention-probe hooks (`ModifiedNEAT/nn/modules/sub.py`).
- Complete `ConvCrossAttention` and add tests for its supported input layouts (`ModifiedNEAT/nn/modules/sub.py`).
- Support dimensional stride and dilation in convolution and pooling modules (`ModifiedNEAT/nn/modules/base/convolution.py`, `ModifiedNEAT/nn/modules/base/pooling.py`).
- Add running-statistics variants of batch normalization and adaptive average pooling (`ModifiedNEAT/nn/modules/base/normalization.py`, `ModifiedNEAT/nn/modules/base/pooling.py`).

## Installation and Tests

- Extend `install_cuda.py` with platform- and device-aware GPU setup/preflight checks for Windows and Linux, including NVIDIA/CUDA compatibility and graceful handling of AMD, Intel, and systems without a supported GPU.
- Organize and mark the test suite by package area and execution requirements so CPU, GPU, integration, and optional-dependency tests can be run selectively.
- Synchronize the published repository copy with the current package API and dependencies.
