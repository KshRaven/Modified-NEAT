import ModifiedNEAT as neat
import ModifiedNEAT.nn as mn
import torch
import torch.nn as nn
import os, sys
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

import ModifiedNEAT as neat
import ModifiedNEAT.nn as mn

from ModifiedNEAT.nn.base import Model
from ModifiedNEAT.optim import scheduler
from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.util.datetime import unix_to_datetime_file
from ModifiedNEAT.util.qol import manage_params

from game import Game

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


class BaseModel(Model):
    """Feed-forward policy network."""

    def __init__(
        self, inputs: int, outputs: int, dim_size: int, layers: int,
        activation: nn.Module = nn.SiLU(), probabilistic: bool = False, bias: bool = True,
        device: torch.device = "cpu", dtype: torch.dtype = torch.float32, **options,
    ):
        super().__init__()
        self.inputs             = inputs
        self.outputs            = outputs
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

        if activation is None: activation = nn.SiLU()
        if dtype is None: dtype = torch.float32

        self.lat_proj = mn.Sequential(
            mn.Linear(inputs, dim_size, True, device, dtype),
            activation,
            *sum([
                [
                    mn.Linear(dim_size, dim_size, bias, device, dtype),
                    activation,
                ]
                    for _ in range(layers)
            ], []),
        )
        self.pol_proj = mn.Sequential(
            mn.Linear(dim_size, dim_size, bias, device, dtype),
            activation,
            mn.Linear(dim_size, 2 * outputs, True, device, dtype),
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

    def get_action(self, state: Tensor, keys: int | list[int] | None = None, **options) -> tuple[Tensor, Tensor]:
        latent    = self.lat_proj(state, keys=keys)   # state: (genomes, batch, features)
        mean, std = self._get_mean_std(latent, keys=keys)
        dist: torch.distributions.Distribution = self.dist(mean, std)
        if self.probabilistic:
            action = dist.sample()
        else:
            action = mean
            if self.distribution == "discrete": action = action.argmax(-1)
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, state: Tensor, action: Tensor, keys: int | list[int] | None = None) -> tuple[Tensor, Tensor]:
        latent    = self.lat_proj(state, keys=keys)
        mean, std = self._get_mean_std(latent, keys=keys)
        dist: torch.distributions.Distribution = self.dist(mean, std)
        return dist.log_prob(action), dist.entropy()

    def get_policy(self, state: Tensor, keys: int | list[int] | None = None, **options) -> Tensor:
        """state: (genomes, batch, features) -> action: (genomes, batch, outputs)."""
        latent    = self.lat_proj(state, keys=keys)
        mean, std = self._get_mean_std(latent, keys=keys)
        action = mean
        if options.get('normal', self.probabilistic):
            action = action + (std * torch.randn_like(std))
        if self.distribution == 'discrete':
            action = torch.argmax(action, dim=-1)
        else:
            action = torch.tanh(action) # Ensure values are between (-1, +1) for acceleration and braking limits
        return action


# Run options
RENDER_MODE: str | None = "human"
LOAD: str | None        = None
TRAIN: bool             = True

# Environment
ENV_CONFIG_FILE: str | None = None  # f"{EXMP_DIR}/.config/main.json"
# Reward shaping: None keeps the config file / default, else "basic", "dhcr", "cell_tree" or "staged" (see README).
# The default state is the egocentric 'ego' state (13 inputs, 8 with ego_road=False); the model input size
# follows automatically. See README, section State.
REWARD_MODE: str | None = None

# Population
GENOMES = 200

# Model
DISCRETE        = False
OUTPUTS         = 3
EMBED_SIZE      = 96
LAYERS          = 1
FWD_EXP         = 1
BIAS            = True
ACTIVATION      = nn.Tanh()
PROBABILISTIC   = True
CLIP_MIN        = -13
CLIP_MAX        = 0
DISTRIBUTION    = "normal" if not DISCRETE else "discrete"

# Trainer
MEMORY_SIZE     = 10
GAMMA           = np.exp(np.log(0.1) / 256)
KAPPA           = 0.0
ALPHA           = np.exp(np.log(5.0) / (MEMORY_SIZE - 1))
ALPHA_ORDER     = 0
REW_NORM        = 3
REW_REG         = 1.00
LOSS_REG        = 0.90
POL_REG         = 0.33
ENT_REG         = 0.50
VAL_REG         = 0.00
DIV_REG         = 0.00
CPY_REG         = 0.00
EPOCHS          = None
ACCURACY_STEPS  = 256
ACCURACY_ERROR  = 0.1

# Evaluation
CULL_STD        = 1.0     # genomes more than this many std below the mean fitness are culled
CULL_MIN_EATERS = 0.10    # genomes that never ate are only culled once this share of the population did eat
RENDER_EVERY    = 20
DEBUG_STEP      = 10

# Test
TEST_COUNT      = 10
TEST_RUNS       = 20
TEST_POOL       = 20
TEST_LIVES      = 20
TEST_MAX_HIATUS = 200
TEST_FPS        = 30
TEST_MAX_FRAMES = 20000

# Files
FILE_NAME = (
    f"SnakeModel-E{EMBED_SIZE}_L{LAYERS}_FX{FWD_EXP}_"
    f"B{int(BIAS)}_A-{ACTIVATION.__class__.__name__}_R{CLIP_MIN}~{CLIP_MAX}"
)
FILE_DIR = "snake"
FILE_NO: int | None = None
INIT_GEN: int = 0

ENV_CONFIG = EnvironmentConfig(file=ENV_CONFIG_FILE)
if REWARD_MODE is not None:
    if REWARD_MODE not in ENV_CONFIG.reward.MODES:
        raise ValueError(f"REWARD_MODE must be one of {ENV_CONFIG.reward.MODES}, got {REWARD_MODE!r}")
    ENV_CONFIG.reward.mode = REWARD_MODE
ENV = Game(render_mode=RENDER_MODE, config=ENV_CONFIG)
INPUTS = ENV.observation_space.shape[-1]

CONFIG = neat.Config("snake", "../../storage/configs")
MULTIPLIER = 0.33
CONFIG.genome.init_type = "normal"
CONFIG.genome.weight_init_mean          = 0.0
CONFIG.genome.weight_init_std           = 1.00 * MULTIPLIER
CONFIG.genome.weight_min_value          = -math.inf
CONFIG.genome.weight_max_value          = +math.inf
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
CONFIG.reproduction.darwin_multiplier   = 0.33
CONFIG.reproduction.cross_multiplier    = 0.75
CONFIG.reproduction.preserve_elite      = False
CONFIG.species.compatibility_threshold  = math.inf
CONFIG.stagnation.max_stagnation        = 1
CONFIG.stagnation.species_elitism       = 2
CONFIG.save()
CONFIG.load(verbose=2)

MODEL = BaseModel(
    INPUTS, OUTPUTS, EMBED_SIZE, LAYERS,
    ACTIVATION, PROBABILISTIC, BIAS, DEVICE, DTYPE,
    clip_min=CLIP_MIN, clip_max=CLIP_MAX, distribution=DISTRIBUTION, fwd_exp=FWD_EXP,
)
POPULATION: neat.Population | None = None


def evaluate(population: neat.Population, **options):
    """Fitness evaluation: one episode with every genome."""
    global FILE_NO
    trainer: PPO = options["trainer"]
    mapping: dict[int, int] = population.get_mapping(consolidated=True)
    keys = list(mapping.keys())
    trainer.update_mapping(mapping)

    print(f"Generation {population.generation}")
    rewards_log = []
    with torch.no_grad():
        ENV.render_mode = "human" if population.generation % RENDER_EVERY == 0 else None
        states = ENV.reset(keys=keys)[0]
        done = False
        step = 0
        ENV.render()
        while not done:
            debug = step == DEBUG_STEP and population.generation == INIT_GEN
            states = torch.tensor(states, device=DEVICE, dtype=DTYPE)  # (genomes, features)
            if debug:
                print(f"\nstates {tuple(states.shape)}\n{states}")
            actions, log_probs = MODEL.get_action(states.unsqueeze(1), keys=keys)  # (genomes, batch=1, outputs)
            actions, log_probs = actions.squeeze(1), log_probs.squeeze(1)
            if debug:
                print(f"actions {tuple(actions.shape)}\n{actions}")
            next_states, rewards, _, done, _ = ENV.step(
                (torch.tanh(actions) if not DISCRETE else actions).cpu().numpy()
            )
            rewards = torch.tensor(rewards, device=DEVICE, dtype=DTYPE)  # (genomes, 1)
            if debug:
                print(f"rewards {tuple(rewards.shape)}\n{rewards}")
            rewards_log.append(rewards[..., 0])
            done = trainer.update(states, actions, rewards, log_probs, done, done)
            states = next_states
            ENV.render()

            print(
                f"\rLives={ENV.players.lives.mean().item()}, "
                f"Scores={ENV.players.scores.max().item()}, "
                f"Alive={ENV.players.active_total}, "
                f"Fitness={ENV.players.fitness.mean().item():.1f}, "
                f"Skill={ENV.snakes.shaper.skill:.2f}, "
                f"MaxSize={trainer.primary.max_size()}     ",
                end="",
            )
            step += 1
    print("\nEpisode finished")

    scores = torch.mean(torch.stack(rewards_log, dim=0), dim=0)
    for key, index in mapping.items():
        population.genomes[key].fitness = scores[index].item()

    # Cull the clear failures. (A genome that ATE is a success: the old rule `not scores < 1` removed exactly
    # those, the opposite of the car-racer rule `not used_brake`. Here a genome that never ate in any life is
    # culled, but only while enough of the population did eat, so the first generations are not wiped out.)
    fit_mean, fit_std = (func(ENV.players.fitness).item() for func in (np.mean, np.std))
    never_ate = ENV.players.true_scores < 1
    cull_idle = (~never_ate).mean() >= CULL_MIN_EATERS
    for key, index in mapping.items():
        if (cull_idle and never_ate[index]) or ENV.players.fitness[index] < fit_mean - (fit_std * CULL_STD):
            population.to_delete.append(key)

    _, FILE_NO = population.save_dict(
        name=FILE_NAME, directory=FILE_DIR, file_no=FILE_NO, replace=population.generation != INIT_GEN
    )


def test_best_network(count: int = TEST_COUNT, set_keys: list[int] | None = None):
    count = max(1, count)
    print("\n---------- TESTING ON SNAKE ----------")
    print(f"best_genome = {POPULATION.best_genome}")

    ENV.players.lives_total     = TEST_LIVES
    ENV.players.max_hiatus      = TEST_MAX_HIATUS
    ENV.render_mode             = "human"
    ENV.window.fps              = TEST_FPS
    ENV.max_frames              = TEST_MAX_FRAMES

    POPULATION.crop(list(POPULATION.genomes.keys())[:TEST_POOL])

    try:
        for run_idx in range(TEST_RUNS):
            keys = _select_keys(count, set_keys)
            mapping = {key: idx for idx, key in enumerate(keys)}
            reverse = {idx: key for key, idx in mapping.items()}
            print(f"\nTest {run_idx} | {len(keys)} genomes {keys[:10]}")

            with torch.no_grad():
                states = ENV.reset(keys=keys)[0]
                done = False
                ENV.render()
                step = 0
                while not done:
                    debug = run_idx == 0 and step == 0
                    states  = torch.tensor(states, device=DEVICE, dtype=DTYPE)
                    actions = MODEL.get_policy(states.unsqueeze(1), keys=keys).squeeze(1)
                    if debug:
                        print(f"states {tuple(states.shape)}, actions {tuple(actions.shape)}")
                    states, _, _, done, info = ENV.step(actions.cpu().numpy())
                    ENV.render()
                    if done:
                        print(f"\ninfo: {info}")
                    step += 1

                top = sorted(((k, ENV.players.scores[i].item()) for k, i in mapping.items()), key=lambda kv: kv[1], reverse=True)[:5]
                print(f"Top scores: {dict(top)}")
    except KeyboardInterrupt:
        print("\nTerminated")


def _select_keys(count: int, set_keys: list[int] | None) -> list[int]:
    """Pick genome keys for one test run, weighted by fitness; the best genome is always included."""
    if set_keys is not None:
        keys = list(set_keys)
        probs = None
    else:
        def sort_key(genome: neat.Genome):
            return genome.fitness if genome.fitness is not None else -math.inf, genome.key

        ranking = sorted(POPULATION.genomes.values(), key=sort_key, reverse=True)[:count + 10]
        fitness = [g.fitness for g in ranking if g.fitness is not None]
        probs = None
        if len(fitness) >= 2 and len(fitness) == len(ranking):
            fitness = np.array(fitness)
            try:
                probs = (fitness - fitness.min()) / (fitness.max() - fitness.min())
                probs = probs / probs.sum()
            except Exception:
                probs = None
        keys = [g.key for g in np.random.choice(ranking, size=min(count, len(ranking)), replace=False, p=probs)]

    if probs is None:
        random.shuffle(keys)
    best = POPULATION.best_genome
    if best is not None and best.key not in keys:
        keys.insert(0, best.key)
    return keys


def fix(value: float, default: float = 1.0) -> float:
    return default if np.isinf(value) or np.isnan(value) else value


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
    if load is None or load.lower() == "false":
        return
    if load.lower() == "true":
        POPULATION.load_dict(name=FILE_NAME, directory=FILE_DIR, file_no=None)
        print("Loaded latest checkpoint")
        return
    try:
        file_no = int(load)
    except ValueError:
        sys.exit(f"--load must be 'true', 'false' or a file number, got '{load}'")
    POPULATION.load_dict(name=FILE_NAME, directory=FILE_DIR, file_no=file_no)
    print(f"Loaded checkpoint {file_no}")


def run():
    global POPULATION, FILE_NO, INIT_GEN

    parser = argparse.ArgumentParser(description="Run NEAT on Snake")
    parser.add_argument("--render_mode", type=str, default=RENDER_MODE, help='"human" or None')
    parser.add_argument("--load", type=str, default=LOAD, help="'true' for latest, or a file number")
    parser.add_argument("--train", type=str, default=str(TRAIN), help="true/false")
    args = parser.parse_args()
    ENV.render_mode = args.render_mode

    print(MODEL)
    POPULATION = neat.Population(GENOMES, MODEL, CONFIG, init_rep=True)
    print(POPULATION)
    load_population(args.load)
    INIT_GEN = POPULATION.generation
    FILE_NO = None

    if str_to_bool(args.train):
        print(f"Calculated hyperparameters: gamma={GAMMA:.4f}, kappa={KAPPA:.4f}, alpha={ALPHA:.4f}")
        trainer = PPO(
            POPULATION,
            schedulers=[
                CosineAnnealing(CONFIG, 30, 0.5, ["weight_mutate_power", "weight_init_std"], True, True),
            ],
            device=DEVICE, dtype=DTYPE,
            log_sub_dir="snake/",
            log_name=(
                f"{unix_to_datetime_file(clock.time())}_"
                f"e{EMBED_SIZE}-l{LAYERS}-fx{FWD_EXP}-b{int(BIAS)}-prob{int(PROBABILISTIC)}-"
                f"g{round(GAMMA, 4)}-k{round(KAPPA, 4)}-a{round(ALPHA, 4)}-ao{ALPHA_ORDER}-rn{REW_NORM}-"
                f"p{round(LOSS_REG, 4)}-v{round(VAL_REG, 4)}-m{round(POL_REG, 4)}-"
                f"s{round(ENT_REG, 4)}-d{round(DIV_REG, 4)}-c{round(CPY_REG, 4)}-"
                f"mem{MEMORY_SIZE}-type{int(DISCRETE)}"
            ),
            gamma=GAMMA, alpha=ALPHA, kappa=KAPPA, order=ALPHA_ORDER, normalize=REW_NORM,
            rew_reg=REW_REG, loss_reg=LOSS_REG, val_reg=VAL_REG, pol_reg=POL_REG, ent_reg=ENT_REG,
            div_reg=DIV_REG, cpy_reg=CPY_REG, validate=True, segr_size=None, use_entropy=True, use_critic=False,
            max_episodes=MEMORY_SIZE,
        )
        print("Regularizations:")
        for name, value in vars(trainer).items():
            if "_reg" in name and isinstance(value, (int, float)):
                print(f"\t{name} => {value:.4f}")
        try:
            trainer.learn(
                evaluate, ENV_CONFIG.max_frames, EPOCHS, ACCURACY_STEPS, ACCURACY_ERROR,
                "continuous" if not DISCRETE else "discrete", verbose=True,
            )
        except KeyboardInterrupt:
            pass

    test_best_network()


if __name__ == "__main__":
    run()