# ModifiedNEAT

ModifiedNEAT is a PyTorch-based implementation of NEAT (Neuro‑Evolution of Augmenting Topologies).  
It extends NEAT concepts to work with `torch.nn.Module` graphs and provides utilities for:

- **NEAT-aware PyTorch modules** (`ModifiedNEAT.nn`).
- **Population-based evolution** (`ModifiedNEAT.population.Population`).
- **Configurable evolution hyperparameters** (`ModifiedNEAT.config.Config`).
- **Reinforcement-learning style training loops** (`ModifiedNEAT.rl.NEAT`).

---

## 1. Installation

### 1.1. Requirements

- Python **3.10+** (see `setup.py`).
- A working **PyTorch** installation (CPU or CUDA).
- Packages in the root `requirements.txt`, including:
  - `torch`, `numpy`, `numba`, `numba-cuda[cu13]`, `cuda-python`, `cuda-toolkit`, `cupy-cuda13x`,
  - `matplotlib`, `colorama`, `tensorboard`,
  - `gymnasium`, `pygame` (needed for the game / RL examples).

> If you are on CPU-only or a different CUDA version, install an appropriate PyTorch build first and then adapt/remove the GPU-specific lines in `requirements.txt`.

### 1.2. Install from source

Clone the repository and install in editable (development) mode:

```bash
git clone https://github.com/KshRaven/Modified-NEAT.git
cd Modified-NEAT

python -m pip install -r requirements.txt
python -m pip install -e .
```

Verify the installation:

```bash
python - << "PY"
import ModifiedNEAT as neat
print("ModifiedNEAT imported from:", neat.__file__)
PY
```

---

## 2. Core API overview

The top-level package exposes the main building blocks (see `ModifiedNEAT/__init__.py`):

```python
import ModifiedNEAT as neat

# Configuration
config = neat.Config("my_experiment")

# NEAT modules and models
from ModifiedNEAT import nn as mn

# Population and RL trainer
from ModifiedNEAT import Population, NEAT
```

Key concepts:

- **Config**: groups all evolution hyperparameters.
- **NEAT-aware model**: a subclass of `mn.Model` or `ModifiedNEAT.nn.base.Model` that uses NEAT parameters.
- **Population**: manages genomes, speciation, reproduction and the evolutionary loop.
- **RL trainer (optional)**: `ModifiedNEAT.rl.NEAT` provides a rollout + scoring loop for RL problems.

---

## 3. Storage and configuration files

### 3.1. Storage directory

`ModifiedNEAT.util.storage` defines where configuration and population snapshots are stored.

```python
import ModifiedNEAT as neat

# Default storage directory (platform-dependent)
from ModifiedNEAT.util.storage import STORAGE_DIR
print("Default storage dir:", STORAGE_DIR)

# Change storage location (recommended in your scripts)
neat.util.storage.set_storage_location("./storage/")
```

Most examples use relative storage subdirectories such as `"pong"`, `"snake"`, etc.  These sit under the global `STORAGE_DIR`.

### 3.2. Config object and config file

The main configuration wrapper lives in `ModifiedNEAT/config/types.py`:

```python
from ModifiedNEAT.config import Config

# file_name becomes the base name of the config file
# directory is the subfolder under STORAGE_DIR
config = Config(file_name="original", directory="pong")
```

Internally this creates a text config file at:

```text
{STORAGE_DIR}/pong/original-neat_config.txt
```

and populates it with multiple sections via:

- `GeneralConfig` (general settings)
- `GenomeConfig` (mutation / weight parameters)
- `SpeciesConfig` (speciation threshold)
- `StagnationConfig` (species stagnation handling)
- `ReproductionConfig` (elitism, survival ratios, etc.)

You typically:

1. Create the config object.
2. Adjust parameters in Python.
3. Call `save()` to write the text config file.
4. Call `load()` to reload values (optionally with verbosity).

Example (adapted from `examples/pong/main.py`):

```python
import math
import ModifiedNEAT as neat

CONFIG = neat.Config("original", "pong")

# Adjust genome section
CONFIG.genome.init_type              = "normal"
CONFIG.genome.weight_init_mean       = 0.0
CONFIG.genome.weight_init_std        = 2.0
CONFIG.genome.weight_min_value       = -math.inf
CONFIG.genome.weight_max_value       = +math.inf
CONFIG.genome.weight_mutate_power    = 2e-1
CONFIG.genome.weight_mutate_rate     = 0.50
CONFIG.genome.weight_replace_rate    = 0.00
CONFIG.genome.weight_add_prob        = 0.00
CONFIG.genome.weight_del_prob        = 0.00
CONFIG.genome.single_structural_mutation = False
CONFIG.genome.param_epsilon          = 1e-6

# Adjust reproduction / species / stagnation
CONFIG.reproduction.min_species_size   = 100
CONFIG.reproduction.purge              = 1
CONFIG.reproduction.clone_threshold    = 0.05
CONFIG.reproduction.survival_threshold = 0.20
CONFIG.reproduction.cross_threshold    = 0.00
CONFIG.reproduction.elitism            = 30
CONFIG.reproduction.darwin_multiplier  = 0.50
CONFIG.reproduction.cross_multiplier   = 0.50
CONFIG.reproduction.preserve_elite     = False
CONFIG.species.compatibility_threshold = math.inf
CONFIG.stagnation.max_stagnation       = 1
CONFIG.stagnation.species_elitism      = 2

# Write configuration to a NEAT config text file
CONFIG.save()

# Optionally reload from file (with printing)
CONFIG.load(verbose=2)
```

The same pattern is used in `examples/snake/main.py` with slightly different hyperparameters.

### 3.3. Configuration reference

Below is a reference for each configuration section and field. Unless otherwise stated, values are floats.

#### GeneralConfig (`config.general`)

- **`fitness_criterion`**
  - What: Aggregation used when computing the global fitness for stopping and reporting.
  - Allowed: `'max'`, `'min'`, `'mean'`.
  - Notes: Used in `Population` and `Reproduction` to sort/select genomes; wrong values will raise `ValueError`.
- **`fitness_threshold`**
  - What: Target aggregated fitness at which evolution terminates early.
  - Allowed: Any real number. Default: `math.inf` (effectively disables threshold termination).
  - Notes: If `fitness_criterion(…) >= fitness_threshold` for all genera, `Population.run` stops.
- **`pop_size`**
  - What: Intended nominal population size.
  - Allowed: Positive integer (e.g. `10`–`1000`+). Default: `100`.
  - Notes: Currently used mainly in legacy CPU/GPU reproduction helpers; effective size is driven by how you instantiate `Population` (`Population(genomes=GENOMES, ...)`). Keep `pop_size` close to that value for consistency.
- **`reset_on_extinction`**
  - What: Whether to automatically create a fresh population if all species go extinct.
  - Allowed: `True` or `False`.
  - Notes: If `False`, complete extinction raises `CompleteExtinctionException`. If `True`, the population is re‑initialized with the current config.
- **`seed`**
  - What: Random seed used in CUDA/CPU initialization kernels.
  - Allowed: Integer in `[0, 2**31 - 1]` (practically any non‑negative `int`).
  - Notes: Defaults to a random value; set manually for reproducible runs.

#### GenomeConfig (`config.genome`)

Weights (and, by analogy in some examples, biases) share a common pattern:

- **`weight_init_mean`**
  - What: Mean of the initial weight distribution.
  - Typical: `0.0`.
  - Notes: Used along with `weight_init_std` when initializing NEAT parameters.
- **`weight_init_std`**
  - What: Standard deviation of the initial weight distribution.
  - Typical: `0.1`–`2.0` depending on task.
  - Constraint: Non‑negative.
- **`weight_max_value` / `weight_min_value`**
  - What: Hard clamp bounds on weights after mutation.
  - Typical: `-math.inf`, `+math.inf` (no clamp), or finite symmetric bounds like `[-30, 30]`.
  - Notes: Choose finite bounds if you want to keep parameters numerically stable.
- **`weight_mutate_power`**
  - What: Magnitude of weight perturbations during mutation.
  - Typical: `1e-2`–`1.0`.
  - Constraint: Non‑negative; larger values produce more aggressive changes.
- **`weight_mutate_rate`**
  - What: Probability that a given weight is perturbed on mutation.
  - Range: `[0.0, 1.0]`.
  - Typical: Around `0.5`–`0.8` in examples.
- **`weight_replace_rate`**
  - What: Probability of completely re‑initializing a weight instead of perturbing it.
  - Range: `[0.0, 1.0]`.
  - Typical: `0.0`–`0.15`.
- **`weight_add_prob` / `weight_del_prob`**
  - What: Structural mutation rates (adding/removing connections), when structural mutations are enabled.
  - Range: `[0.0, 1.0]`.
  - Notes: Many examples keep these at `0.0` (structure fixed) and only evolve values.
- **`param_epsilon`**
  - What: Small epsilon used in some numerical operations to avoid division by zero.
  - Typical: `1e-6`–`1e-9`.

Genome compatibility (for speciation):

- **`compatibility_disjoint_coefficient`**
  - What: Weight given to disjoint / excess genes when computing distance between genomes.
  - Typical: `1.0`.
  - Constraint: Non‑negative; larger values penalize topological differences more strongly.
- **`compatibility_weight_coefficient`**
  - What: Weight given to parameter differences (same connection/node, different value).
  - Typical: `0.1`–`0.5`.
  - Notes: Larger values emphasize weight mismatch over structure.

Other genome settings:

- **`init_type`**
  - What: Name of the initialization regime used by the kernels.
  - Allowed: Currently `'normal'` (aligned with existing initializers).
- **`single_structural_mutation`**
  - What: Whether to limit each mutation step to at most one structural change.
  - Allowed: `True` or `False`.
  - Notes: When `True`, helps keep topology changes gradual.

> **Bias parameters**: Some CPU/GPU kernels and examples reference `bias_init_mean`, `bias_init_std`, `bias_min_value`, `bias_max_value`, `bias_mutate_power`, `bias_mutate_rate`, `bias_replace_rate`. These follow the same meaning and typical ranges as the `weight_*` counterparts, but are applied to bias tensors.

#### SpeciesConfig (`config.species`)

- **`compatibility_threshold`**
  - What: Maximum allowed genetic distance for a genome to join an existing species.
  - Allowed:
    - Numeric (int/float): fixed threshold.
    - String `'auto'`: automatically adapted based on distance statistics (see `ModifiedNEAT.species.get_ct`).
  - Typical: `3.0` in neat‑style configs, `math.inf` when you want a single giant species.
  - Notes: Lower values → more, smaller species; higher values → fewer, broader species.

#### StagnationConfig (`config.stagnation`)

- **`species_fitness_func`**
  - What: Aggregation for per‑species fitness when checking stagnation.
  - Allowed: `'max'`, `'min'`, `'mean'`.
  - Notes: Invalid values raise `ValueError`.
- **`max_stagnation`**
  - What: Number of generations a species is allowed to go without improvement before being marked stagnant.
  - Allowed: Non‑negative integer.
  - Typical: Small integer (e.g. `1`–`20`), depending on how quickly you want to prune species.
- **`species_elitism`**
  - What: Minimum number of top species that are never marked stagnant, even if they do not improve.
  - Allowed: Non‑negative integer.
  - Notes: Prevents complete collapse to a single species due to temporary plateaus.

#### ReproductionConfig (`config.reproduction`)

- **`elitism`**
  - What: Number of top genomes per species copied unchanged into the next generation.
  - Allowed: Integer `>= 0`.
  - Typical: `2`–`30` depending on population size.
  - Notes: Effective minimum species size is `max(min_species_size, elitism)`.
- **`clone_threshold`**
  - What: Fraction of top genomes in a species that can be copied directly (cloned) without crossover.
  - Range: `[0.0, 1.0]` (internally clipped to at most `0.75`).
  - Notes: Higher values → more pure cloning; lower values → more crossover.
- **`survival_threshold`**
  - What: Fraction of genomes per species eligible to become parents.
  - Range: `(0.0, 1.0]`.
  - Typical: `0.2`.
  - Notes: At least 2 parents are always used, even if this fraction would give fewer.
- **`cross_threshold`**
  - What: Lower bound on the fraction of offsprings produced via crossover vs cloning.
  - Range: `[0.0, 1.0]`.
  - Notes: Small positive values encourage some crossover; `0.0` means crossover is optional.
- **`cross_multiplier`**
  - What: Strength of bias towards fitter parents during crossover selection.
  - Typical: `0.5`–`1.0`.
  - Notes: Passed as a multiplier into parent selection; larger values emphasize high‑fitness parents.
- **`darwin_multiplier`**
  - What: Multiplier used when sampling parents based on fitness in some reproduction kernels.
  - Typical: `0.1`–`1.0`.
  - Notes: Larger values sharpen the probability differences between high‑ and low‑fitness genomes.
- **`min_species_size`**
  - What: Minimum spawn size per species when allocating the next generation.
  - Allowed: Integer `>= 1`.
  - Typical: Set close to `GENOMES` (total population size) for experiments that maintain one large species per genus, or much smaller if you expect many species.
  - Notes: Effective value is `max(min_species_size, elitism)`.
- **`purge`**
  - What: When `> 0`, every `purge` generations all species are temporarily forced to exactly `min_species_size` individuals.
  - Allowed: Integer `>= 0` (`0` disables purging).
  - Notes: Can be used as a periodic reset to avoid species blow‑up.
- **`preserve_elite`**
  - What: Whether elites are guaranteed to be preserved during certain GPU reproduction flows.
  - Allowed: `True` or `False`.

> **Tip:** Start from the Pong or Snake example config values and change **one group** of parameters at a time (e.g., only `weight_mutate_*`, or only `compatibility_*`) so you can see how each group influences training behavior.

---

## 4. Basic usage: evolving a population

A typical workflow:

1. Build a NEAT-aware PyTorch model (subclass of `ModifiedNEAT.nn.base.Model`).
2. Construct a `Config` and adjust its sections.
3. Create a `Population` with a number of genomes and the model.
4. Define an evaluation / fitness function that assigns `fitness` to each genome.
5. Run evolution with `Population.run`.

A minimal skeleton (omitting details of the model and fitness function):

```python
import ModifiedNEAT as neat
import ModifiedNEAT.nn as mn
import torch
from torch import nn, Tensor
from typing import Union

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32

class SimpleModel(mn.Model):
    def __init__(self, inputs: int, outputs: int, dim: int, device, dtype):
        super().__init__()
        # Use NEAT-aware modules instead of raw torch.nn layers
        self.proj = mn.Linear(inputs, dim, True, device, dtype)
        self.head = mn.Linear(dim, outputs, True, device, dtype)

    def get_policy(self, state: Tensor, keys: Union[int, list[int]] = None, **kwargs) -> Tensor:
        x = self.proj(state, keys=keys)
        x = torch.tanh(x)
        x = self.head(x, keys=keys)
        return x

    def forward(self, state: Tensor, keys: Union[int, list[int]] = None, **kwargs) -> Tensor:
        return self.get_policy(state, keys=keys, **kwargs)

# 1. Config
CONFIG = neat.Config("demo", "demo")

# 2. Model and population
INPUTS   = 3
OUTPUTS  = 3
DIM_SIZE = 32
MODEL    = SimpleModel(INPUTS, OUTPUTS, DIM_SIZE, DEVICE, DTYPE)
GENOMES  = 40

POP = neat.Population(GENOMES, MODEL, CONFIG, init_rep=True)

# 3. Fitness function

def eval_population(population: neat.Population, **options):
    # Map genome keys to indices (if needed by your environment/model)
    mapping = population.get_mapping(consolidated=True)
    # Compute a score per genome and assign genome.fitness
    for key, genome in population.genomes.items():
        # Dummy fitness: use the genome key (replace with real evaluation)
        genome.fitness = float(key)

# 4. Run evolution
best_genome, ranking = POP.run(eval_population, generations=10, verbose=1)
```

For full, working examples with environments and rendering, see:

- `examples/pong/main.py`
- `examples/snake/main.py`
- `examples/flappy-bird/flappy_bird.py`

These demonstrate how to:

- Connect a `Game` environment (e.g. Pong/Snake/FlappyBird) to a NEAT population.
- Use `population.get_mapping` to map genome keys to model indices.
- Save and reload populations with `Population.save_dict` / `Population.load_dict`.

---

## 5. Reinforcement-learning style NEAT (ModifiedNEAT.rl.NEAT)

For RL-style training with rollouts and episodic returns, see `ModifiedNEAT/rl/neat.py` and the **Snake** example.

High-level flow (see `examples/snake/main.py`):

1. Create `POPULATION = neat.Population(...)`.
2. Create schedulers (optional) with `neat.optim.scheduler`.
3. Construct `TRAINER = neat.NEAT(POPULATION, schedulers=[...], device=..., dtype=..., ...)`.
4. Implement `eval_genomes(population: neat.Population, trainer: neat.NEAT, ...)` that:
   - runs the environment,
   - feeds transitions into the trainer via `trainer.update(...)`,
   - sets `population.to_delete` for bad genomes when desired.
5. Call `TRAINER.learn(eval_genomes, steps, epochs, ...)`.

Because this interface is more complex, it is best to start from `examples/snake/main.py` and adapt it to your own environment.

---

## 6. Saving and loading populations

`ModifiedNEAT.population.Population` provides helpers for checkpointing:

```python
# Save population state
state, file_no = POP.save_dict(name="original", directory="pong")

# Later, load it back (into an existing Population object with matching model/config)
POP.load_dict(name="original", directory="pong", file_no=file_no, verbose=1)
```

This uses the same storage utilities as the configuration system (`ModifiedNEAT.util.storage`).  Population snapshots include:

- genome keys and fitness values,
- module state_dicts,
- species metadata and indices,
- best genomes and avatar lists.

---

## 7. Examples

The `examples/` directory contains ready-made experiments that you can run and modify:

- `examples/pong/` – NEAT-controlled paddles for Pong.
- `examples/snake/` – NEAT + RL trainer on a Snake environment (supports convolutional observations).
- `examples/flappy-bird/` – NEAT on Flappy Bird with various architectures.

Use these as templates for integrating ModifiedNEAT into your own projects:

1. Design a PyTorch model using `ModifiedNEAT.nn` modules.
2. Build a `Config` and save/load it.
3. Create a `Population` and an evaluation function.
4. Optionally integrate with `ModifiedNEAT.rl.NEAT` for episodic RL.

---

## 8. Further reading

- Original NEAT documentation (conceptual background):  
  https://neat-python.readthedocs.io/en/latest/
- Repository and issues:  
  https://github.com/KshRaven/Modified-NEAT
