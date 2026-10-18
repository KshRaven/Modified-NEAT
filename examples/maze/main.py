import argparse
import os
import sys
import time as clock
import warnings

import ModifiedNEAT as neat
import ModifiedNEAT.nn as mn
import numpy as np
import torch
import torch.nn as nn

from numba.core.errors import NumbaPerformanceWarning
from torch import Tensor

from ModifiedNEAT.optim import scheduler
from ModifiedNEAT.util.datetime import unix_to_datetime_file
from ModifiedNEAT.util.fancy_text import CM, Fore

EXMP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXMP_DIR not in sys.path:
    sys.path.insert(0, EXMP_DIR)

from game import EnvironmentConfig, Game

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
torch.set_printoptions(threshold=10)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
neat.set_device(DEVICE)
PROJECT_DIR = os.path.dirname(os.path.dirname(EXMP_DIR))
neat.util.storage.set_storage_location(f"{PROJECT_DIR}/storage/")

# Environment parameters
RENDER_MODE         = "human"
GRID_SIZE           = (16, 16)
CELL_SIZE           = 40
PLAYER_LIVES        = 20
RESET_ON_DEATH      = True
USE_EUCLIDEAN_DISTANCE = True
DISCRETE_STATES     = True
DISCRETE_ACTIONS    = True

# Model parameters
GENOMES         = 200
OUTPUTS         = 4
HIDDEN_SIZE     = 64
HIDDEN_LAYERS   = 2
BIAS            = True
ACTIVATION      = nn.SiLU()
PROBABILISTIC   = True
DISTRIBUTION    = "discrete" if DISCRETE_ACTIONS else "normal"
SDTYPE          = torch.long if DISCRETE_STATES else DTYPE

# NEAT parameters
WEIGHT_INIT_STD     = 0.75
WEIGHT_MUTATE_POWER = 0.25
WEIGHT_MUTATE_RATE  = 0.65
WEIGHT_REPLACE_RATE = 0.0
WEIGHT_ADD_PROB     = 0.0
WEIGHT_DEL_PROB     = 0.0
PARAM_EPSILON       = 0.0
MIN_SPECIES_SIZE    = GENOMES
PURGE               = 1
ELITISM             = 0.50
CLONE_THRESHOLD     = 0.50
SURVIVAL_THRESHOLD  = 0.10
CROSS_THRESHOLD     = 0.00
COMPATIBILITY_THRESHOLD = np.inf
MAX_STAGNATION      = 1
SPECIES_ELITISM     = 2
DARWIN_MULTIPLIER   = 0.33
CROSS_MULTIPLIER    = 0.75
PRESERVE_ELITE      = False

# Trainer parameters
MEMORY_SIZE     = 5
GAMMA           = np.exp(np.log(0.01) / 128)
KAPPA           = 0.0
ALPHA           = np.exp(np.log(4.0) / (MEMORY_SIZE - 1))
BETA            = 1.0
ALPHA_ORDER     = 0
BETA_ORDER      = 0
REW_NORM        = 4
REW_REG         = 1.00
PPO_REG         = 0.75
POL_REG         = 0.95
ENT_REG         = 0.65
CPY_REG         = 0.0
DIV_REG         = 0.0
SEGR_SIZE       = None
STEPS           = 2000
EPOCHS          = 300

FILE_NAME   = "MazeModel"
FILE_DIR    = "maze-runner/main"
FILE_NO     = None
INIT_GEN    = None
ENV         = None
MODEL       = None
CONFIG      = None
POPULATION  = None


def build_env_config() -> EnvironmentConfig:
    config = EnvironmentConfig()
    config.grid.grid_size = GRID_SIZE
    config.grid.cell_size = CELL_SIZE
    config.player.lives = PLAYER_LIVES
    config.agent.return_grid = False
    config.agent.discrete_states = DISCRETE_STATES
    config.agent.discrete_actions = DISCRETE_ACTIONS
    config.agent.reset_on_death = RESET_ON_DEATH
    config.agent.use_euclidean_distance = USE_EUCLIDEAN_DISTANCE
    return config


class FeedForwardModel(mn.Model):
    def __init__(self, inputs: int, outputs: int, vocab_size: int | None = None):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.vocab_size = vocab_size
        self.distribution = DISTRIBUTION
        self.probabilistic = PROBABILISTIC
        self.epsilon = PARAM_EPSILON

        self.input_projection = (
            mn.Sequential(
                mn.Embedding(vocab_size, HIDDEN_SIZE, None, DEVICE, DTYPE),
                nn.Flatten(start_dim=-2),
                mn.Linear(inputs * HIDDEN_SIZE, HIDDEN_SIZE, True, DEVICE, DTYPE),
            ) if vocab_size is not None else
            mn.Linear(inputs, HIDDEN_SIZE, True, DEVICE, DTYPE)
        )

        latent_layers = []
        size = HIDDEN_SIZE
        for _ in range(HIDDEN_LAYERS):
            latent_layers.extend([
                mn.LayerNorm(HIDDEN_SIZE, bias=BIAS, device=DEVICE, dtype=DTYPE),
                ACTIVATION,
                mn.Linear(size, HIDDEN_SIZE, BIAS, DEVICE, DTYPE),
            ])
            size = HIDDEN_SIZE
        self.latent = mn.Sequential(*latent_layers)
        self.pol_proj = mn.Sequential(*[
            mn.LayerNorm(HIDDEN_SIZE, bias=BIAS, device=DEVICE, dtype=DTYPE),
            ACTIVATION,
            mn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, BIAS, DEVICE, DTYPE),
            ACTIVATION,
            mn.Linear(HIDDEN_SIZE, 2 * outputs, True, DEVICE, DTYPE),
        ])

        # self.val_proj = mn.Sequential(*[
        #     mn.LayerNorm(HIDDEN_SIZE, bias=BIAS, device=DEVICE, dtype=DTYPE),
        #     ACTIVATION,
        #     mn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, BIAS, DEVICE, DTYPE),
        #     ACTIVATION,
        #     mn.Linear(HIDDEN_SIZE, 1, True, DEVICE, DTYPE),
        # ])

    def _encode(self, state: Tensor, keys=None, verbose: bool = False) -> Tensor:
        latent = self.input_projection(state, keys=keys, verbose=verbose)
        return self.latent(latent, keys=keys, verbose=verbose)

    def _mean_std(self, state: Tensor, keys=None) -> tuple[Tensor, Tensor]:
        latent = self._encode(state, keys=keys)
        values = self.pol_proj(latent, keys=keys)
        mean, raw_std = torch.chunk(values, 2, dim=-1)
        return mean, raw_std.abs() + self.epsilon

    def forward(self, state: Tensor, **options) -> Tensor:
        return self.get_policy(state, **options)

    def get_action(self, state: Tensor, keys=None, **options) -> tuple[Tensor, Tensor]:
        mean, std = self._mean_std(state, keys=keys)
        distribution = self.dist(mean, std)
        if self.probabilistic:
            action = distribution.sample()
        else:
            action = mean.argmax(-1) if self.distribution == "discrete" else mean
        return action, distribution.log_prob(action)

    def evaluate_action(self, state: Tensor, action: Tensor, keys=None, **options) -> tuple[Tensor, Tensor]:
        mean, std = self._mean_std(state, keys=keys)
        distribution = self.dist(mean, std)
        return distribution.log_prob(action), distribution.entropy()

    def get_policy(self, state: Tensor, keys=None, **options) -> Tensor:
        mean, std = self._mean_std(state, keys=keys)
        action = mean
        if options.get("normal", self.probabilistic):
            action = action + std * torch.randn_like(std)
        if self.distribution == "discrete":
            action = action.argmax(-1)
        return action


def configure_neat() -> neat.Config:
    config = neat.Config("sequential", ".config")
    config.genome.init_type = "normal"
    config.genome.weight_init_mean = 0.0
    config.genome.weight_init_std = WEIGHT_INIT_STD
    config.genome.weight_min_value = -np.inf
    config.genome.weight_max_value = np.inf
    config.genome.weight_mutate_power = WEIGHT_MUTATE_POWER
    config.genome.weight_mutate_rate = WEIGHT_MUTATE_RATE
    config.genome.weight_replace_rate = WEIGHT_REPLACE_RATE
    config.genome.weight_add_prob = WEIGHT_ADD_PROB
    config.genome.weight_del_prob = WEIGHT_DEL_PROB
    config.genome.param_epsilon = PARAM_EPSILON
    config.genome.single_structural_mutation = True
    config.reproduction.min_species_size = MIN_SPECIES_SIZE
    config.reproduction.purge = PURGE
    config.reproduction.elitism = ELITISM
    config.reproduction.clone_threshold = CLONE_THRESHOLD
    config.reproduction.survival_threshold = SURVIVAL_THRESHOLD
    config.reproduction.cross_threshold = CROSS_THRESHOLD
    config.species.compatibility_threshold = COMPATIBILITY_THRESHOLD
    config.stagnation.max_stagnation = MAX_STAGNATION
    config.stagnation.species_elitism = SPECIES_ELITISM
    config.reproduction.darwin_multiplier = DARWIN_MULTIPLIER
    config.reproduction.cross_multiplier = CROSS_MULTIPLIER
    config.reproduction.preserve_elite = PRESERVE_ELITE
    config.save()
    config.load(2)
    return config


def evaluate(population: neat.Population, **options):
    trainer = options["trainer"]
    global INIT_GEN, FILE_NO
    mapping = population.get_mapping(consolidated=True)
    trainer.update_mapping(mapping)
    if INIT_GEN is None:
        INIT_GEN = population.generation

    for genome in population.genomes.values():
        genome.fitness = 0

    MODEL.eval()
    ENV.render_mode = "human" if population.generation % 10 == 0 else None
    states = ENV.reset(keys=list(mapping.keys()))[0]
    done = False
    step = 0
    start = clock.perf_counter()
    ENV.render()
    while not done:
        with torch.no_grad():
            observations = torch.as_tensor(states, device=DEVICE, dtype=SDTYPE)
            actions, log_probs = MODEL.get_action(observations.unsqueeze(1))
            actions = actions.squeeze(1)
            log_probs = log_probs.squeeze(1)
            next_states, rewards, _, done, _ = ENV.step(actions.cpu().numpy())
            rewards = torch.as_tensor(rewards, device=DEVICE, dtype=DTYPE)
            terminate = trainer.update(observations, actions, rewards, log_probs, done, done)
            print(
                f"\r{CM('Executing', Fore.GREEN)}: "
                f"time_elapsed={round(clock.perf_counter() - start)}s, "
                f"alive={ENV.players.active_total}, max_rew={round(rewards.max().item(), 2)}, "
                f"sd={trainer.steps_done}, bl={trainer.primary.max_size()} ",
                end="",
            )
            states = next_states
            ENV.render()
            done = done or terminate
        step += 1
    print()

    fitnesses = ENV.players.fitness
    cutoff = fitnesses.mean() - (1.5 * fitnesses.std())
    for fitness, genome in zip(fitnesses, population.genomes.values()):
        if fitness < cutoff:
            population.to_delete.append(genome.key)
        genome.fitness = float(fitness)
    MODEL.train()
    _, file_no = population.save(FILE_NAME, FILE_DIR, FILE_NO, replace=population.generation != INIT_GEN)
    if population.generation == INIT_GEN:
        FILE_NO = file_no + 1


def run():
    global ENV, MODEL, CONFIG, FILE_NO, INIT_GEN, FILE_NAME, POPULATION
    parser = argparse.ArgumentParser(description="Run non-sequential PPO/NEAT on Maze-Runner")
    parser.add_argument("--rm", type=str, default=RENDER_MODE)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--train", type=str, default="true")
    args = parser.parse_args()

    def as_bool(value: str) -> bool:
        return value.lower() in ("true", "1", "yes", "on")

    ENV = Game(render_mode=args.rm, config=build_env_config())
    inputs = ENV.observation_space.shape[-1]
    vocab_size = int(ENV.agents.max_options or 1)
    MODEL = FeedForwardModel(
        inputs,
        OUTPUTS,
        vocab_size=vocab_size if DISCRETE_STATES else None,
    )
    FILE_NAME = f"MazeModel-I{inputs}-H{HIDDEN_SIZE}-L{HIDDEN_LAYERS}-B{int(BIAS)}"
    print(MODEL)
    CONFIG = configure_neat()
    POPULATION = neat.Population(GENOMES, MODEL, CONFIG, init_reporter=True)

    if args.load is not None and args.load.lower() == "true":
        POPULATION.load(FILE_NAME, FILE_DIR, None)
    elif args.load not in (None, "false", "False", "none", "null", ""):
        FILE_NO = int(args.load)
        POPULATION.load_dict(None, FILE_NAME, FILE_DIR, FILE_NO)

    print(f"Gamma={GAMMA:.4f}, Kappa={KAPPA:.4f}, Alpha={ALPHA:.4f}, Beta={BETA:.4f}")

    if as_bool(args.train):
        trainer = neat.rl.PPO(
            POPULATION,
            schedulers=[
                # scheduler.CosineAnnealing(CONFIG, 20, 0.1, "weight_mutate_power", False, True)
            ],
            device=DEVICE,
            dtype=DTYPE,
            log_sub_dir="maze_runner/",
            log_name=f"{unix_to_datetime_file(clock.time())}-main-e{HIDDEN_SIZE}-l{HIDDEN_LAYERS}",
            gamma=GAMMA, alpha=ALPHA, kappa=KAPPA, order=ALPHA_ORDER, normalize=REW_NORM,
            beta=BETA, beta_order=BETA_ORDER, rew_reg=REW_REG, loss_reg=PPO_REG,
            pol_reg=POL_REG, ent_reg=ENT_REG, div_reg=DIV_REG, cpy_reg=CPY_REG,
            validate=True, segr_size=SEGR_SIZE, max_steps=None,
            max_episodes=MEMORY_SIZE, use_entropy=True,
        )
        print("Regularizations:")
        for name, value in vars(trainer).items():
            if "_reg" in name:
                print(f"\t{name} => {value:.4f}")
        print("All params:")
        for name, value in vars(trainer).items():
            if isinstance(value, (int, float, bool)) and not name.startswith("_"):
                print(f"\t{name} => {value}")
        print(f"starting evaluation: population={len(POPULATION.genomes)}, grid={(ENV.grid.rows, ENV.grid.cols)}")
        try:
            trainer.learn(evaluate, STEPS, EPOCHS, 128,
                          accuracy_type="discrete" if DISCRETE_ACTIONS else "continuous",
                          verbose=True)
        except KeyboardInterrupt:
            pass
    else:
        print("Skipping training...")

    MODEL.eval()
    ENV.render_mode = "human"
    for _ in range(5):
        done = False
        states = ENV.reset(keys=GENOMES)[0]
        while not done:
            with torch.no_grad():
                observations = torch.as_tensor(states, device=DEVICE, dtype=SDTYPE)
                actions = MODEL.get_policy(observations.unsqueeze(1)).squeeze(1)
            states, _, _, done, _ = ENV.step(actions.cpu().numpy())
            ENV.render()
    MODEL.train()


if __name__ == "__main__":
    run()
