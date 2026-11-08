# ModifiedNEAT

ModifiedNEAT is a PyTorch-based neuroevolution library. It evolves parameters in `NeatModule` models through initialization, mutation, speciation, and reproduction. RL trainers and example environments are included, but environment design and fitness evaluation remain the caller's responsibility.

## Requirements and installation

Python 3.11 or newer is required. From the repository root, install the package in editable mode:

```bash
pip install -e .
```

The optional GPU dependencies can be installed with:

```bash
pip install -e .[gpu]
```

The GPU extra installs CUDA-related packages; PyTorch model passes can use a CUDA device, while the current RL rollout bookkeeping runs on the CPU. During installation, the setup script may also build the dashboard frontend. If its build artifacts are missing, Node.js with npm or Bun is needed; set `MODIFIEDNEAT_SKIP_DASHBOARD_BUILD=1` to skip that build.

## Getting started

Create a `Config`, implement a concrete `NeatModule` containing NEAT parameters, then pass an instance of that module to `Population`:

```python
import ModifiedNEAT as neat

config = neat.Config(file_name="demo", directory="demo")
model = MyPolicy()
population = neat.Population(100, model, config)
```

The base `neat.nn.Model` is not a complete policy by itself. For working environment loops and policy examples, see the [maze example](examples/maze/main.py) and other directories under [examples](examples).

## Current capabilities

| Area | Current state |
|---|---|
| Evolution | `Population` manages genomes, species, fitness, mutation, reproduction, and checkpoints. CPU is the default backend; an optional CUDA backend accelerates supported operations. |
| Genera | A population can assign genomes to multiple genera on one shared model with `genera=N`, or use separate models and combine populations with `absorb_population()`. Mapping, speciation, and reproduction retain genus identity. |
| Models | `NeatModule` and `NeatParameter` define evolvable parameters; compose them into a `Model` for policy inference. Different model parameter layouts are not generally compatible for crossover. |
| Reinforcement learning | `NEAT`, `Ghost`, and `PPO` collect rollouts and assign fitness through a user-provided evaluation loop. Model inference can use the configured Torch device; RL rollout bookkeeping currently runs on CPU. |
| Inspection | NeatBoard inspects saved module and population artifacts. Treat pickle files as trusted input and keep the unauthenticated dashboard on trusted networks. |

For example, `Population(200, model, config, genera=2)` creates 200 genomes split across two genera while keeping the parameter tensors in one model. Pass `genus_sizes={id: count, ...}` to choose the distribution explicitly. After combining compatible populations with `absorb_population()`, `consolidate()` returns a new population that packs their parameter rows into one model; parameter counts and shapes must match. See [TODO.md](TODO.md) for known unfinished work, including crossover between incompatible model layouts and saving/loading RL algorithms.

## Example use cases

| Example | What it demonstrates |
|---|---|
| [Maze](examples/maze/README.md) | Gymnasium-style grid environment, observation/action design, and RL training. |
| [Flappy Bird](examples/flappy-bird/README.md) | Policy training; `main.py` uses separate model populations, while [`_main.py`](examples/flappy-bird/_main.py) uses one model with multiple genera. |
| [Car Racer](examples/car-racer/README.md) | Vectorized driving environment, NEAT policies, race modes, and demos. |
| [Snake](examples/snake/README.md) | Vectorized game, manual and rule-based play, NEAT training, and reward-shaping experiments. |
| [Pong](examples/pong/README.md) | A small game example for NEAT-driven paddle control. |
| [MNIST](examples/mnist/main.ipynb) | Notebook-based image-classification experiment and dataset loading. |

## Saving and loading

`Population.state_dict()` returns a serializable snapshot of the population, including module architecture and parameters, genomes, species, configuration, and evaluation state. Use it for in-memory transfer, or use `save()` and `load()` for checkpoints:

```python
state = population.state_dict()
population.load(state_dict=state)

population.save(filename="experiment", directory="neat_save")
population.load(filename="experiment", directory="neat_save")
```

Pass `filepath` to read or write a checkpoint at an exact path instead of using ModifiedNEAT's storage directory and numbered filenames:

```python
population.save(filepath="checkpoints/current.neat.pkl")
population.load(filepath="checkpoints/current.neat.pkl")
```

Related text, bytes, or files can be stored alongside a checkpoint with `extra_files`. Keys are relative paths inside the checkpoint's bundle folder:

```python
population.save(
	filename="experiment",
	directory="neat_save",
	extra_files={
		"config.json": '{"seed": 42}',
		"logs/training.log": "Training completed",
	},
)
```

Without extra files, the checkpoint is a single `.neat.pkl` file. With extra files, the checkpoint and related files are placed in a folder named for that checkpoint. The legacy `save_dict()` and `load_dict()` methods remain available for dictionary-based workflows.

## Documentation

- [Configuration](ModifiedNEAT/config/README.md): experiment settings and schedulers.
- [Reinforcement learning](ModifiedNEAT/rl/README.md): trainers, rollout behavior, and metrics.
- [Dashboard](ModifiedNEAT/dashboard/README.md): inspecting saved modules and artifacts.

The public imports are listed in [ModifiedNEAT/__init__.py](ModifiedNEAT/__init__.py). Implementations under `ModifiedNEAT/rl/legacy/` and `examples/` are retained for older workflows and experiments; prefer the current package interfaces documented above for new code.
