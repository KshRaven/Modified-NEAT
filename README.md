# ModifiedNEAT

ModifiedNEAT is a PyTorch-based implementation of NEAT for evolving neural networks expressed as PyTorch modules. It combines population-based evolution, configurable mutation and speciation rules, and optional reinforcement-learning style training loops.

This repository focuses on the public package under [ModifiedNEAT](ModifiedNEAT) and its main entry points:

- [ModifiedNEAT/config/README.md](ModifiedNEAT/config/README.md) for configuration behavior and experiment setup
- [ModifiedNEAT/dashboard/README.md](ModifiedNEAT/dashboard/README.md) for the built-in dashboard and inspection workflow
- [ModifiedNEAT/rl/README.md](ModifiedNEAT/rl/README.md) for rollout-based training and scoring

## What is included

- Neural-network modules and genome wrappers under [ModifiedNEAT/nn](ModifiedNEAT/nn)
- Population, species, reproduction, and stagnation logic under [ModifiedNEAT](ModifiedNEAT)
- Persistent JSON-backed configuration under [ModifiedNEAT/config](ModifiedNEAT/config)
- Optional dashboard tooling under [ModifiedNEAT/dashboard](ModifiedNEAT/dashboard)
- Reinforcement-learning integrations under [ModifiedNEAT/rl](ModifiedNEAT/rl)

## Installation

Install the package from the repository root:

```bash
pip install -e .
```

For GPU-enabled installs, use the optional dependency set defined in [setup.py](setup.py):

```bash
pip install -e .[gpu]
```

## Quick start

```python
import ModifiedNEAT as neat

config = neat.Config(file_name="demo", directory="demo")
population = neat.Population(100, neat.nn.Model, config)
print("Population created")
```

The public API is exposed from [ModifiedNEAT/__init__.py](ModifiedNEAT/__init__.py), so most experiments can be built around the top-level imports from that module.

## Documentation map

- [ModifiedNEAT/config/README.md](ModifiedNEAT/config/README.md) explains the configuration sections in [ModifiedNEAT/config/types.py](ModifiedNEAT/config/types.py) and how they affect evolution.
- [ModifiedNEAT/dashboard/README.md](ModifiedNEAT/dashboard/README.md) covers the NeatBoard dashboard workflow for inspecting serialized modules.
- [ModifiedNEAT/rl/README.md](ModifiedNEAT/rl/README.md) summarizes the rollout and scoring loop used by the RL trainer.

## Notes

The repository includes several legacy or internal directories that are not part of the public package surface. The documentation above focuses on the maintained package under [ModifiedNEAT](ModifiedNEAT) and the public modules that are imported from it.
