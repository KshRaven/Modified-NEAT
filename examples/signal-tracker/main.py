import os, sys
import torch
import torch.nn as nn
import time as clock
import numpy as np
import warnings
import multiprocessing as mp
import argparse

from torch import Tensor, device as TDEVICE, dtype as TDTYPE
from torch.nn import Module
from typing import Union
from numba.core.errors import NumbaPerformanceWarning

EXMP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(os.path.dirname(EXMP_DIR))
print(f"Example path = {EXMP_DIR}")
if EXMP_DIR not in sys.path: sys.path.insert(0, EXMP_DIR)
if PROJ_DIR not in sys.path: sys.path.insert(0, PROJ_DIR)
EXMP_ROOT = os.path.dirname(EXMP_DIR)   # examples/ (holds the shared 'misc' extensions)
if EXMP_ROOT not in sys.path: sys.path.insert(0, EXMP_ROOT)

import ModifiedNEAT as neat
import ModifiedNEAT.nn as mn

from ModifiedNEAT.optim import scheduler
from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.util.datetime import unix_to_datetime_file
from ModifiedNEAT.util.qol import manage_params

from game import Game
from misc import BinaryStepper

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
torch.set_printoptions(threshold=10)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float32


def _T(x):
    """Env arrays (NumPy) -> tensors on the training device."""
    return torch.as_tensor(x, device=DEVICE, dtype=DTYPE)

neat.set_device(DEVICE)
neat.util.storage.set_storage_location(f"{PROJ_DIR}/storage/")
print(f"Using PyTorch device: '{DEVICE}'. NEAT device: '{neat.get_device()}'")
print(f"Storage path: {neat.util.storage.STORAGE_DIR}")


class Block(mn.NeatModule):
    def __init__(
        self, inputs: int, outputs: int, dim_size: int, coefficients: int,
        normalize: bool = False, residual: bool = True,
        activation: nn.Module | None = None, bias: bool = True,
        device: torch.device = "cpu", dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            inp=inputs, out=outputs, dim=dim_size, coeffs=coefficients, norm=normalize,
            res=residual, actv=activation.__class__.__name__ if not None else str(None), bias=bias
        )
        self.inputs     = inputs
        self.outputs    = outputs
        self.dim_size   = dim_size
        self.normalize  = normalize
        self.residual   = residual
        self.activation = activation if activation is not None else nn.SiLU()
        self.bias       = bias
        self.device     = device
        self.dtype      = dtype

        self.pre_norm = mn.LayerNorm(inputs, bias=False, device=device, dtype=dtype) if normalize else None
        self.inp_proj = mn.Polynomial(inputs, dim_size, coefficients, bias, device, dtype)
        self.out_proj = mn.Polynomial(dim_size, outputs, coefficients, bias, device, dtype)
        self.res_proj = (
            mn.Identity()
            if inputs == outputs else
            mn.Polynomial(inputs, outputs, 1, False, device, dtype)
        ) if residual else None

    def forward(self, tensor: Tensor, keys: int | list[int] | None = None, **kwargs):
        residue = tensor
        if self.normalize: tensor = self.pre_norm(tensor, keys=keys)
        tensor = self.out_proj(self.activation(self.inp_proj(tensor, keys=keys)), keys=keys)
        if self.residual: tensor = tensor + self.res_proj(residue, keys=keys)
        return tensor


class BaseModel(mn.Model):
    """Feed-forward policy network."""

    def __init__(
        self, inputs: int, outputs: int, dim_size: int, layers: int,
        activation: nn.Module = nn.SiLU(), probabilistic: bool = False, bias: bool = True,
        device: torch.device = "cpu", dtype: torch.dtype = torch.float32, **options,
    ):
        super().__init__()
        self.inputs             = inputs
        self.outputs            = outputs
        self.coefficients: int  = options.get('coefficients', 1)
        self.dim_size           = dim_size
        self.layers             = layers
        self.distribution       = options.get('distribution', 'normal')
        self.probabilistic      = probabilistic
        self.normalize          = options.get('normalize', True)
        self.clip_min: int | None = options.get('clip_min', -24) # -12 apprx -28
        self.clip_max: int | None = options.get('clip_max', +0)
        self.clip_range         = self.clip_max - self.clip_min
        self.lower              = 10 ** self.clip_min
        self.upper              = 10 ** self.clip_max
        self.fwd_exp: int       = options.get('fwd_exp', 1)
        self.epsilon            = options.get('epsilon', 1e-9)
        self.steps: int         = options.get('steps', 1)         # resolution levels per continuous output (see stepper.py)
        self.bipolar: bool      = options.get('bipolar', True)    # env actions live in [-1, 1]
        self.stepper: BinaryStepper | None = None
        if self.distribution != 'discrete':
            self.stepper = BinaryStepper(outputs, self.steps, self.bipolar)
        self.act_mult           = 1 if self.stepper is None else self.stepper.bits   # raw outputs per env output = log2 levels
        self.raw_outputs        = outputs * self.act_mult

        if activation is None: activation = nn.SiLU()
        if dtype is None: dtype = torch.float32

        self.lat_proj = mn.Sequential(
            # mn.Linear(inputs, dim_size, True, device, dtype),
            mn.Polynomial(inputs, dim_size, self.coefficients, True, device, dtype),
            # activation,
            *sum([
                [
                    # # mn.Linear(dim_size, dim_size, bias, device, dtype),
                    # mn.Polynomial(dim_size, dim_size, self.coefficients, bias, device, dtype),
                    # activation,
                    Block(dim_size, dim_size, dim_size, self.coefficients, self.normalize,
                          True, activation, bias, device, dtype)
                ]
                    for _ in range(layers)
            ], []),
        )
        self.pol_proj = mn.Sequential(
            (
                mn.LayerNorm(dim_size, bias=False, device=device, dtype=dtype) 
                if self.normalize else 
                mn.Identity()
            ),
            # mn.Linear(dim_size, dim_size, bias, device, dtype),
            mn.Polynomial(dim_size, dim_size, self.coefficients, bias, device, dtype),
            activation,
            # mn.Linear(dim_size, 2 * outputs, True, device, dtype),
            mn.Polynomial(dim_size, 2 * self.raw_outputs, self.coefficients, True, device, dtype),
        )

    def extra_repr(self) -> str:
        return f"probabilistic={self.probabilistic}, distro='{self.distribution}'"

    def forward(self, state: Tensor, keys: int | list[int] | None = None, **kwargs):
        return self.get_policy(state, keys=keys, **kwargs)

    def _get_mean_std(self, latent: Tensor, keys: int | list[int] | None = None):
        mean_std      = self.pol_proj(latent, keys=keys)
        mean, log_std = torch.chunk(mean_std, 2, -1)
        std           = torch.exp(log_std.clamp(self.clip_min, self.clip_max))
        return mean, std

    def _reduce(self, tensor: Tensor) -> Tensor:
        """(..., outputs * bits) -> (..., outputs): joint over the bits of one output."""
        if self.stepper is None:
            return tensor
        return tensor.unflatten(-1, (self.outputs, self.act_mult)).sum(-1)

    def to_env(self, actions: Tensor) -> Tensor:
        """Raw (pre-tanh) actions, as stored in the rollout buffer -> value for ENV.step. 1 level == plain tanh."""
        return actions if self.stepper is None else self.stepper(actions)

    def get_action(self, state: Tensor, keys: int | list[int] | None = None, **options) -> tuple[Tensor, Tensor]:
        latent    = self.lat_proj(state, keys=keys)   # state: (genomes, batch, features)
        mean, std = self._get_mean_std(latent, keys=keys)
        dist: torch.distributions.Distribution = self.dist(mean, std)
        if self.probabilistic:
            action = dist.sample()
        else:
            action = mean
            if self.distribution == "discrete": action = action.argmax(-1)
        log_prob = self._reduce(dist.log_prob(action))
        return action, log_prob

    def evaluate_action(self, state: Tensor, action: Tensor, keys: int | list[int] | None = None) -> tuple[Tensor, Tensor]:
        latent    = self.lat_proj(state, keys=keys)
        mean, std = self._get_mean_std(latent, keys=keys)
        dist: torch.distributions.Distribution = self.dist(mean, std)
        return self._reduce(dist.log_prob(action)), self._reduce(dist.entropy())

    def get_policy(self, state: Tensor, keys: int | list[int] | None = None, **options) -> Tensor:
        """state: (genomes, batch, features) -> action: (genomes, batch, outputs)."""
        latent    = self.lat_proj(state, keys=keys)
        mean, std = self._get_mean_std(latent, keys=keys)
        action = mean
        if options.get('normal', self.probabilistic):
            action = action + (std * torch.randn_like(std))
        if self.distribution == 'discrete':
            action = torch.argmax(action, dim=-1)
        # NOTE: stays PRE-tanh on purpose. get_action() stores the pre-tanh mean in the rollout buffer, and the trainer
        # compares get_policy() against that buffer (policy accuracy), so both must live in the same space.
        # The caller converts once with MODELS[0].to_env(actions) (= tanh for LEVELS=1) before ENV.step.
        return action

    # def get_value(self, state: Tensor, keys: Union[int, list[int]] = None) -> Tensor:
    #     latent      = self.projection(state, keys=keys)
    #     value       = self.val_proj(latent, keys=keys)
    #     return value


def fix(value: float, default: float = 1):
    if np.isinf(value) or np.isnan(value):
        return default
    else:
        return value


# ENVIRONMENT SETTINGS (Signal Tracker: a hidden target readable only on clean frames; see game/env.py)
HORIZON             = 200    # steps per episode
EPISODES            = 4      # episodes per training update (STEPS)
DECOY_FRAC          = 0.5    # fraction of decoy frames (cue = uniform noise): sequence noise between instances
DECOY_RUN           = 4.0    # mean decoy burst length (longer = needs a longer memory)
CUE_SIGMA           = 0.05   # reading noise of clean frames
TAG_SEP             = 2.0    # how well the tag separates clean frames from decoys (smaller = harder)
TAG_SIGMA           = 0.25
DRIFT               = 0.0    # common-mode drift shared by every frame (what differential attention subtracts)
OBS_NOISE           = 0.0    # plain per-feature noise (instance noise)
TARGET_SPEED        = 0.04
AGENT_SPEED         = 0.12
FULL_STATES         = True
DELAY               = 0
DISCRETE            = False

# Model properties
GENOMES             = 100
INPUTS              = None   # set in run() from ENV.observation_space
OUTPUTS             = None   # set in run() from ENV.action_space
EMBED_SIZE          = 32
COEFFICIENTS        = 2
LAYERS              = 2
FWD_EXP             = 1
LEVELS              = 10      # resolution levels per continuous output: raw outputs per action = bit_length(LEVELS) (1 = plain tanh)
BIAS                = True
PROBABILISTIC       = False
NORMALIZE           = True
TEST_ACTIVATION     = nn.SiLU()
CLIP_MIN            = -13
CLIP_MAX            = +0
DISTRIBUTION        = 'normal' if not DISCRETE else 'discrete'
MODEL_NUM           = 1
GENERA              = 2   # >1: ONE model shared by this many genera (Population(genera=N), GENOMES per genus); 1: MODEL_NUM separate models
if GENERA > 1: MODEL_NUM = GENERA   # the same model object is listed once per genus

# Trainer properties
MEMORY_SIZE         = 5
GAMMA               = fix(np.exp(np.log(0.10) / 256), 0.0)
KAPPA               = 0.0 # fix(np.exp(np.log(0.01) / 128), 0.0)
ALPHA               = fix(np.exp(np.log(1.00) / (MEMORY_SIZE - 1)), 1.0)
ALPHA_ORDER         = 0
REW_NORM            = 3
REW_REG             = 1.00
LOSS_REG            = 0.00
VAL_REG             = 0.00
POL_REG             = 0.95
ENT_REG             = 0.00
DIV_REG             = 0.00
CPY_REG             = 0.00

# Post-training test
TEST_RUNS    = 5
TEST_GOAL    = 400
TEST_ELITES  = 10      # best genomes per genus kept for the test

# Training 
GOAL        = HORIZON
STEPS       = HORIZON * EPISODES
EPOCHS      = 100
FILTER_DEAD = False   # cursors never die in this env

MODELS: list = []


def build_models() -> list:
    """Models are sized from the Env's observation / action spaces, so they are created after the Env."""
    return [
        BaseModel(
            INPUTS, OUTPUTS, EMBED_SIZE, LAYERS,
            activation, PROBABILISTIC, BIAS, DEVICE, DTYPE,
            clip_min=CLIP_MIN, clip_max=CLIP_MAX, distribution=DISTRIBUTION,
            fwd_exp=FWD_EXP, normalize=NORMALIZE, coefficients=COEFFICIENTS, steps=LEVELS, bipolar=True
        )
        for activation in [TEST_ACTIVATION for _ in range(1 if GENERA > 1 else MODEL_NUM)]
    ]

GAME_TYPE = int(round(DECOY_FRAC * 10))   # decoy fraction in tenths, for the file / log names
BASE_NAME: str = (
    f"GT{GAME_TYPE}_DC{int(DISCRETE)}_T{MODEL_NUM}{f'G{GENERA}' if GENERA > 1 else ''}_"
    f"E{EMBED_SIZE}_L{LAYERS}_C{COEFFICIENTS}_"
    f"A-{TEST_ACTIVATION.__class__.__name__}_B{int(BIAS)}" # fwd_exp
    f"{f'_R-{CLIP_MIN}~{CLIP_MAX}' if PROBABILISTIC else ''}"
    f"{f'_Lv{LEVELS}' if LEVELS != 1 else ''}"
)
FILE_NAME: str = f"SignalTrackerModel-{BASE_NAME}"
FILE_DIR: str | None = "signal-tracker/main"
FILE_NO: int | None = None
INIT_GEN: int | None = None

print(f"\ncreating config")
CONFIG = neat.Config('signal-tracker-main', '.config')
MULTIPLIER = 1.0 # 0.33
CONFIG.genome.init_type = "normal"
CONFIG.genome.weight_init_mean          = 0.0
CONFIG.genome.weight_init_std           = 1.00 * MULTIPLIER
CONFIG.genome.weight_min_value          = -np.inf
CONFIG.genome.weight_max_value          = +np.inf
CONFIG.genome.weight_mutate_power       = 0.25 * MULTIPLIER
CONFIG.genome.weight_mutate_rate        = 0.65
CONFIG.genome.weight_replace_rate       = 0.00
CONFIG.genome.weight_add_prob           = 0.00
CONFIG.genome.weight_del_prob           = 0.00
CONFIG.genome.single_structural_mutation = True
CONFIG.genome.param_epsilon             = 0e-9
CONFIG.reproduction.min_species_size    = GENOMES
CONFIG.reproduction.purge               = 1
CONFIG.reproduction.clone_threshold     = 0.50
CONFIG.reproduction.survival_threshold  = 0.10
CONFIG.reproduction.cross_threshold     = 0.00
CONFIG.reproduction.elitism             = 0.33
CONFIG.reproduction.darwin_multiplier   = 0.20
CONFIG.reproduction.cross_multiplier    = 0.80
CONFIG.reproduction.preserve_elite      = False
CONFIG.species.compatibility_threshold  = np.inf
CONFIG.stagnation.max_stagnation        = 1
CONFIG.stagnation.species_elitism       = 2
CONFIG.save()
CONFIG.load(verbose=2)

ENV: Game | None = None
POPULATION: neat.Population | None = None


def evaluate(population: neat.Population, **options):
    trainer: neat.rl.PPO = options['trainer']
    split_mappings = population.get_mapping()
    if len(population.genera) == 1: split_mappings = (split_mappings,)
    split_sizes = [len(m) for m in split_mappings]
    cons_mapping = population.get_mapping(consolidated=True)
    trainer.update_mapping(cons_mapping)
    global INIT_GEN, FILE_NO
    if INIT_GEN is None:
        INIT_GEN = population.generation

    for genome in population.genomes.values():
        genome.fitness = 0

    terminate = False
    start = 0
    run_step = 0
    game_step = 0
    for model in MODELS: model.eval()
    while not terminate:
        gts = clock.perf_counter()
        step = 0
        reverse_mappings = tuple([
            {index: key for key, index in m.items()}
            for m in split_mappings
        ])

        states = _T(ENV.reset(mapping=cons_mapping)[0])
        step = 0
        DEBUG_STEP = 16
        done = False
        while not done:
            with torch.no_grad():
                DEBUG = step == DEBUG_STEP and (
                    (population.generation == INIT_GEN) 
                    or 
                    (population.generation % 50 == 0 and population.generation > 0)
                )
                if DEBUG:
                    print(f"\nstates =>\n{states}\n\tshape = {states.shape}")

                # Filter dead birds from calculation
                ts = clock.perf_counter()
                if FILTER_DEAD:
                    split_keys: list[list[int]] = [[] for _ in range(MODEL_NUM)]
                    split_indices: list[list[int]] = [[] for _ in range(MODEL_NUM)]
                    for index, dead in enumerate(~ENV.players.active):
                        if not dead:
                            found = False
                            for rmi, rm in enumerate(reverse_mappings):
                                if index in rm:
                                    split_keys[rmi].append(rm[index])
                                    split_indices[rmi].append(index)
                                    found = True
                                    break
                                index -= len(split_mappings[rmi])
                            if not found:
                                raise KeyError(f"Index {index} not found in reverse mappings")
                    for mi, indices in enumerate(split_indices):
                        if len(indices) == 0:
                            split_keys[mi] = list(split_mappings[mi].keys())
                            split_indices[mi] = list(split_mappings[mi].values())
                else:
                    # a shared model holds every genus' genomes, so it must be told which rows each call is for
                    split_keys = [list(m.keys()) if GENERA > 1 else None for m in split_mappings]
                    split_indices = [list(m.values()) for m in split_mappings]

                # Get actions and their log-probabilities # NOTE: Dense layers don't need batch dimension
                split_observations = torch.split(states, split_sizes, dim=0)
                if FILTER_DEAD:
                    split_observations = [obs[indices] for obs, indices in zip(split_observations, split_indices)]
                split_actions, split_log_probs = zip(*[
                    m.get_action(obs.unsqueeze(1), keys=split_keys[mi])
                    for mi, (m, obs) in enumerate(zip(MODELS, split_observations))
                ])
                split_actions = [a.squeeze(1) for a in split_actions]
                split_log_probs = [lp.squeeze(1) for lp in split_log_probs]

                # Pad dead bird actions
                padding = ENV.players.total - sum([act.shape[0] for act in split_actions])
                if padding > 0:
                    def fill_up(tensor: Tensor, indices_list: list[int], total: int):
                        fill = tensor.clone()
                        tensor = torch.zeros(
                            total, *tensor.shape[1:], device=DEVICE, dtype=DTYPE
                        )
                        tensor[indices_list] = fill
                        return tensor
                    split_actions = [
                        fill_up(act, indices, len(rm))
                        for act, indices, rm in zip(split_actions, split_indices, reverse_mappings)
                    ]
                    split_log_probs = [
                        fill_up(lp, indices, len(rm))
                        for lp, indices, rm in zip(split_log_probs, split_indices, reverse_mappings)
                    ]
                actions = torch.cat(split_actions, dim=0)
                log_probs = torch.cat(split_log_probs, dim=0)
                calc_time = clock.perf_counter() - ts
                if DEBUG:
                    print(f"\nsplit_states =>\n{split_observations[0]}\n\tshape = {split_observations[0].shape}")
                    print(f"\nsplit_actions =>\n{split_actions[0]}\n\tshape = {split_actions[0].shape}")
                    print(f"\nactions =>\n{actions}\n\tshape = {actions.shape}")

                # Get rewards
                _actions = MODELS[0].to_env(actions)
                next_states, rewards, _, done, _ = ENV.step(_actions)
                next_states, rewards = _T(next_states), _T(rewards)
                if DEBUG:
                    print(f"\nrewards =>\n{rewards}\n\tshape = {rewards.shape}")

                # Updated buffers
                terminate = trainer.update(states, actions, rewards, log_probs, done, done)

                # CMDLine rendering
                alive = ENV.players.active_total
                max_score = round(rewards.max().item(), 2)
                print(
                    f"\r{CM('Executing', Fore.GREEN)}: "
                    f"time_elapsed = {round(clock.perf_counter()-gts)}s, "
                    f"alive = {alive}, max_rew = {max_score}, "
                    f"sd={trainer.steps_done}, "
                    f"bl={trainer.primary.max_size()}     ", end=''
                )

                states = next_states

            # break if score gets large enough
            if done or terminate:
                terminate = True
                break

            step += 1
            run_step += 1

        print("\n------------------------------ Post debugging ------------------------------")
        print(f"Buffer multiplier = {MEMORY_SIZE}")
        print(f"Mapping = {trainer.episode_mapping}")
        print(f"Lengths = {trainer.episode_lengths}")
        print(f"Episodes Primary = {trainer.primary.episodes()}")
        print(f"Episodes Secondary = {trainer.secondary.episodes()}")
        print("----------------------------------------------------------------------------")

        for key, idx in cons_mapping.items():       # cursor index -> genome by the mapping, not by dict order
            population.genomes[key].fitness = float(ENV.players.fitness[idx])

        game_step += 1
    print(f"\n")
    for model in MODELS: model.train()

    _, file_no = population.save(FILE_NAME, FILE_DIR, FILE_NO, replace=population.generation != INIT_GEN)
    if population.generation == INIT_GEN:
        print(f"Saved initial population to file number '{file_no}'")
        print(f"Initial generation was set to {INIT_GEN}, now updated to {population.generation}")
        FILE_NO = file_no + 1


def genome_debug(algorithm: neat.rl.NEAT):
    population = algorithm.population
    COUNT = 10

    def fill(text: str, space: int):
        if not isinstance(text, str):
            text = str(text)
        amount = max(0, int(space - len(text)))
        return f" {' ' * amount}{text} "

    print(f"Population Summary:")
    ks = max(len(str(g.key)) for g in population.genomes.values())
    gs = 3
    fs = max(len(str(round(g.actual_fitness, 4))) for g in population.genomes.values())
    for i, genus in enumerate(population.genera):
        genomes = sorted([g for g in population.genomes.values() if g.genus == genus], key=lambda g: g.actual_fitness, reverse=True)[:COUNT]
        for genome in genomes:
            print(f"\t{fill(genome.key, ks)}|{fill(genome.genus, gs)}|{fill(round(genome.actual_fitness, 4), fs)}")
        if i < len(population.genera) - 1:
            print(f"\t...")


def test_best_network():
    """Watch the surviving elites play on a harder, longer level (rendered)."""
    elites = []
    for gn_id, specie in enumerate(POPULATION.species_set.species.values()):
        genomes = sorted(
            [g for g in specie.members.values() if g.fitness is not None and not np.isnan(g.fitness)],
            key=lambda g: g.fitness, reverse=True,
        )
        elites.extend(genomes[:TEST_ELITES])
        print(f"\nElites [{gn_id}]: {[g.key for g in genomes[:TEST_ELITES]]}")
    if not elites:
        print("No evaluated genomes to test")
        return
    POPULATION.crop([e.key for e in elites], DEVICE)

    env = Game(
        render_mode='human', horizon=TEST_GOAL, seq_len=None, decoy_frac=DECOY_FRAC, decoy_run=DECOY_RUN, cue_sigma=CUE_SIGMA,
        tag_sep=TAG_SEP, tag_sigma=TAG_SIGMA, drift=DRIFT, obs_noise=OBS_NOISE,
        target_speed=TARGET_SPEED, speed=AGENT_SPEED,
        fps=45
    )
    split_mappings = POPULATION.get_mapping()
    if len(POPULATION.genera) == 1: split_mappings = (split_mappings,)
    split_sizes = [len(m) for m in split_mappings]
    cons_mapping = POPULATION.get_mapping(consolidated=True)

    for model in MODELS: model.eval()
    try:
        for i in range(TEST_RUNS):
            print(f"Running test {i + 1}/{TEST_RUNS}")
            done = False
            states = _T(env.reset(mapping=cons_mapping)[0])
            while not done:
                with torch.no_grad():
                    split_states = torch.split(states, split_sizes, dim=0)
                    # a shared model (GENERA > 1) holds every genus: run each genus by its genome keys
                    split_actions = [
                        m.get_policy(obs.unsqueeze(1), keys=list(sm.keys()) if GENERA > 1 else None)
                        for m, obs, sm in zip(MODELS, split_states, split_mappings)
                    ]
                    actions = torch.cat(split_actions, dim=0).squeeze(1)
                next_states, _, _, done, _ = env.step(MODELS[0].to_env(actions))
                states = _T(next_states)
                env.render()
            print(f"Mean reward: best={env.players.fitness.max().item():.3f}, mean={env.players.fitness.mean().item():.3f}")
    except KeyboardInterrupt:
        print("\nTerminated")
    finally:
        for model in MODELS: model.train()


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "on"):
        return True
    if value.lower() in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"Cannot convert '{value}' to bool")


def load_population(load: str | None) -> None:
    """`load`: 'true' for the latest checkpoint, an integer for a file number, None/'false' to skip."""
    global FILE_NO
    if load is None or load.lower() in ("false", "none", "null", ""):
        return
    if load.lower() == "true":
        FILE_NO = None
        print("Will load the latest population checkpoint")
    else:
        try:
            FILE_NO = int(load)
        except ValueError:
            sys.exit(f"--load must be 'true', 'false' or a file number, got '{load}'")
        print(f"Will load population checkpoint from file number '{FILE_NO}'")
    POPULATION.load(FILE_NAME, FILE_DIR, FILE_NO)


def run(argv: list[str] | None = None, test: bool = True):
    """argv: CLI args (None = sys.argv); test: run the rendered test of the best genomes after training."""
    global ENV, POPULATION, FILE_NO, INIT_GEN, MODELS, INPUTS, OUTPUTS

    parser = argparse.ArgumentParser(description='Run NEAT algorithm on Signal Tracker game')
    parser.add_argument('--rm', type=str, default=None,
                        help='Render mode for the game (e.g., "human", None)')
    parser.add_argument('--load', type=str, default=None,
                        help='Load a previously trained model (True for latest, or specify file number)')
    parser.add_argument('--train', type=str, default='true',
                        help='Train the model (True/False or 1/0, default: True)')
    args = parser.parse_args(argv)

    ENV = Game(
        render_mode=args.rm, horizon=GOAL, seq_len=None, decoy_frac=DECOY_FRAC, decoy_run=DECOY_RUN, cue_sigma=CUE_SIGMA,
        tag_sep=TAG_SEP, tag_sigma=TAG_SIGMA, drift=DRIFT, obs_noise=OBS_NOISE,
        target_speed=TARGET_SPEED, speed=AGENT_SPEED,
        fps=None if args.rm != 'human' else 45
    )

    # Models are sized from the Env's spaces
    INPUTS = int(ENV.observation_space.shape[-1])
    OUTPUTS = 2 if DISCRETE else int(ENV.action_space.shape[-1])
    MODELS = build_models()
    if GENERA > 1: MODELS = MODELS * GENERA   # one shared model, one entry per genus
    print(MODELS[0])

    print("\ncreating population")
    if GENERA > 1:     # one model, GENERA genera of GENOMES genomes each
        POPULATION = neat.Population(GENOMES * GENERA, MODELS[0], CONFIG, init_reporter=True, genera=GENERA)
    else:
        POPULATION = neat.Population(GENOMES, MODELS[0], CONFIG, init_reporter=True)
        for model in MODELS[1:]:
            extra_population = neat.Population(GENOMES, model, CONFIG, init_reporter=True)
            POPULATION.absorb_population(extra_population)
    print(POPULATION)
    load_population(args.load)
    INIT_GEN = None             # set by the first evaluate() call
    FILE_NO = None

    if str_to_bool(args.train):
        print(f"Calculated hyperparameters: gamma={GAMMA:.4f}, kappa={KAPPA:.4f}, alpha={ALPHA:.4f}")
        trainer = neat.rl.PPO(
            POPULATION,
            schedulers=[
                scheduler.CosineAnnealing(CONFIG, 30, 0.5, ["weight_mutate_power", "weight_init_std"], True, True),
            ],
            device=DEVICE, dtype=DTYPE,
            log_sub_dir='signal_tracker/',
            log_name=(
                f"main-{unix_to_datetime_file(clock.time())}-_"
                f"e{EMBED_SIZE}-l{LAYERS}-c{COEFFICIENTS}-fx{FWD_EXP}-b{int(BIAS)}-prob{int(PROBABILISTIC)}-"
                f"g{round(GAMMA, 4)}-k{round(KAPPA, 4)}-a{round(ALPHA, 4)}-ao{ALPHA_ORDER}-rn{REW_NORM}-"
                f"p{round(LOSS_REG, 4)}-v{round(VAL_REG, 4)}-m{round(POL_REG, 4)}-"
                f"s{round(ENT_REG, 4)}-d{round(DIV_REG, 4)}-c{round(CPY_REG, 4)}-"
                f"mem{MEMORY_SIZE}-type{int(DISCRETE)}-gt{GAME_TYPE}"
            ),
            gamma=GAMMA, alpha=ALPHA, kappa=KAPPA, order=ALPHA_ORDER, normalize=REW_NORM,
            rew_reg=REW_REG, loss_reg=LOSS_REG, val_reg=VAL_REG, pol_reg=POL_REG, ent_reg=ENT_REG,
            div_reg=DIV_REG, cpy_reg=CPY_REG, validate=True, segr_size=None, use_entropy=True, use_critic=False,
            max_episodes=MEMORY_SIZE, max_steps=None,
        )
        print("Regularizations:")
        for name, value in vars(trainer).items():
            if "_reg" in name and isinstance(value, (int, float)):
                print(f"\t{name} => {value:.4f}")
        trainer.set_report_hook(genome_debug)

        print(f"starting evaluation: population={len(POPULATION.genomes)}")
        try:
            trainer.learn(
                # evaluate, ENV_CONFIG.max_frames, EPOCHS, ACCURACY_STEPS, ACCURACY_ERROR,
                evaluate, STEPS, EPOCHS, 1380,
                accuracy_error=0.1,
                accuracy_type='continuous' if not DISCRETE else 'discrete',
                verbose=True,
            )
        except KeyboardInterrupt:
            pass

    if test:
        test_best_network()


if __name__ == "__main__":
    run()
