# https://neat-python.readthedocs.io/en/latest/xor_example.html

from SnakeGame import Game
from ModifiedNEAT.util.datetime import unix_to_datetime_file

import ModifiedNEAT as neat
import ModifiedNEAT.nn as mn
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
import random
import math
import numpy as np
import time as clock

from torch import Tensor
from typing import Union, Iterable
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float32
neat.set_device(DEVICE)
neat.util.storage.set_storage_location("../../storage/")


CONVOLUTIONAL = False

class Permute(nn.Module):
    def __init__(self, dims: int | Iterable[int]):
        super(Permute, self).__init__()
        if not isinstance(dims, Iterable):
            dims = [dims]
        self.dims = tuple(dims)

    def forward(self, input: Tensor) -> Tensor:
        extra = tuple(range(max(0, input.ndim-len(self.dims))))
        return input.permute(extra+self.dims)

    def extra_repr(self):
        return f"dims={self.dims}"


class BaseModel(mn.Model):
    def __init__(self, inputs: int, outputs: int, dim_size: int, kernel_size: int, fc_size: int, layers: int,
                 coefficients=1, activation: nn.Module = nn.SiLU(), probabilistic=False,
                 bias=True, device: torch.device = 'cpu', dtype: torch.dtype = torch.float32,
                 **options):
        super().__init__()
        # Attributes
        self.inputs         = inputs
        self.outputs        = outputs
        self.dim_size       = dim_size
        self.layers         = layers
        self.distribution   = options.get('distribution', 'normal')
        self.stride         = 1
        self.coefficients   = coefficients
        self.probabilistic  = probabilistic
        self.clip_min       = options.get('clip_min', -2)
        self.clip_max       = options.get('clip_max', -0)
        self.clip_range     = self.clip_max - self.clip_min

        # Build
        self.projection = mn.Sequential(*[
            mn.Polynomial(inputs, dim_size, coefficients, True, device, dtype),
            # mn.Conv2d(inputs, dim_size, kernel_size, padding=-1, bias=True, device=device, dtype=dtype),
            *sum([
                [
                    mn.LayerNorm(dim_size, bias=False, device=device, dtype=dtype),
                    # mn.GroupNorm(1, dim_size, bias=False, device=device, dtype=dtype),
                    activation,
                    mn.Polynomial(dim_size, dim_size, coefficients, bias, device, dtype),
                    # mn.Conv2d(dim_size, dim_size, kernel_size, padding=-1, bias=bias, device=device, dtype=dtype),
                ]
                for _ in range(layers)
            ], []),
        ])
        self.pol_proj = mn.Sequential(*[
            mn.LayerNorm(dim_size, bias=False, device=device, dtype=dtype),
            # mn.GroupNorm(1, dim_size, bias=False, device=device, dtype=dtype),
            activation,
            # Permute([-2, -1, -3]),
            # nn.Flatten(-3, -1),
            # nn.AdaptiveAvgPool1d(fc_size),
            # # mn.Linear(dim_size, dim_size, bias, device, dtype),
            # # mn.Polynomial(dim_size, 2*outputs, coefficients, True, device, dtype),
            mn.Linear(fc_size if CONVOLUTIONAL else dim_size,
                      outputs*(2 if self.distribution != 'discrete' else 1), True, device, dtype),
        ])

    def extra_repr(self) -> str:
        return f"probabilistic={self.probabilistic}, distro='{self.distribution}'"

    def forward(self, state: Tensor, keys: Union[int, list[int]] = None, **kwargs):
        return self.get_policy(state, keys=keys, **kwargs)

    def get_mean_std(self, latent: Tensor, keys: Union[int, list[int]] = None):
        mean_std = self.pol_proj(latent, keys=keys)
        if self.distribution != 'discrete':
            mean, log_std = torch.chunk(mean_std, 2, -1)
            # mean            = F.sigmoid(mean) * 4 + -2
            # std             = torch.pow(10, F.sigmoid(log_std) * self.clip_range + self.clip_min)
            std = torch.exp(log_std)
        else:
            mean = mean_std
            std = None
        return mean, std

    def get_action(self, state: Tensor, keys: Union[int, list[int]] = None) -> tuple[Tensor, Tensor]:
        # state = (state - 0.5) * 2
        squeeze = state.ndim == 4 if CONVOLUTIONAL else 2
        if squeeze:
            state = state.unsqueeze(1)
        latent      = self.projection(state, keys=keys)
        mean, std   = self.get_mean_std(latent, keys=keys)
        dist        = self.dist(mean, std)
        action      = torch.sigmoid(dist.sample() if self.probabilistic else mean) if self.distribution != 'discrete' else dist.sample()
        log_prob    = dist.log_prob(action)
        if squeeze:
            action = action.squeeze(1)
        return action, log_prob

    def evaluate_action(self, state: Tensor, action: Tensor, keys: Union[int, list[int]] = None):
        # state = (state - 0.5) * 2
        squeeze = state.ndim == 4 if CONVOLUTIONAL else 2
        if squeeze:
            state = state.unsqueeze(1)
        latent      = self.projection(state, keys=keys)
        mean, std   = self.get_mean_std(latent, keys=keys)
        dist        = self.dist(mean, std)
        log_prob    = dist.log_prob(action)
        entropy     = dist.entropy()
        return log_prob, entropy

    def get_policy(self, state: Tensor, keys: Union[int, list[int]] = None, **options) -> Tensor:
        # state = (state - 0.5) * 2
        squeeze = state.ndim == (4 if CONVOLUTIONAL else 2)
        if squeeze:
            state = state.unsqueeze(1)
        latent      = self.projection(state, keys=keys)
        mean, std   = self.get_mean_std(latent, keys=keys)
        dist        = self.dist(mean, std)
        action      = torch.sigmoid(dist.sample() if options.get('normal', self.probabilistic) else mean) if self.distribution != 'discrete' else dist.sample()
        if squeeze:
            action = action.squeeze(1)
        return action

    # def get_value(self, state: Tensor, keys: Union[int, list[int]] = None) -> Tensor:
    #     latent      = self.projection(state, keys=keys)
    #     value       = self.val_proj(latent, keys=keys)
    #     return value


ENV: Game | None = None
MODEL: BaseModel | None = None
FILE_NO: int | None = None
INIT_GEN: int = 0

SAFE_GENOMES = []

def eval_genomes(population: neat.Population, **options):
    """
    Run each genome against each other one time to determine the fitness.
    """
    global ENV, MODEL, FILE_NO, SAFE_GENOMES
    trainer: neat.NEAT = options['trainer']
    mapping: dict[int, int] = population.get_mapping(consolidated=True)
    _mapping = list(mapping.items())
    mapping = dict(_mapping)
    trainer.update_mapping(mapping)
    keys = list(mapping.keys())

    print(f"started generation {population.generation}")
    with torch.no_grad():
        states = ENV.reset(keys=keys)[0]
        done = False
        step = 0
        DEBUG_STEP = 10
        ENV.render()
        while not done:
            DEBUG = step == DEBUG_STEP and population.generation == INIT_GEN
            states = torch.tensor(states, device=DEVICE, dtype=DTYPE) # shape(genomes, features, *pixels)
            if DEBUG:
                print(f"states => \n{states} \n\tshape = {states.shape}")
            actions = MODEL.get_policy(states, keys=keys) # shape(genomes)
            if DEBUG:
                print(f"actions => \n{actions} \n\tshape = {actions.shape}")
            next_states, rewards, _, done, _ = ENV.step(actions.cpu().numpy())
            rewards = torch.tensor(rewards, device=DEVICE, dtype=DTYPE) # shape(genomes, features)
            if DEBUG:
                print(f"rewards => \n{rewards} \n\tshape = {rewards.shape}")
            # Updated buffers
            done = trainer.update(states, actions, rewards, done, done)
            states = next_states
            if step % 10 == 0:
                pass
            ENV.render()
            if population.generation % 20 == 0:
                ENV.clock.tick(40)
            print(f"\r"
                  f"Frames = {ENV.players.frames_done.max().item()}, "
                  f"Lives = {ENV.players.lives.mean().item()}, "
                  f"Scores={ENV.players.true_scores.max().item()}, "
                  f"Alive={ENV.players.active_total}, "
                  f"Fitness={ENV.players.fitness.mean().item():.4f}"
                  , end='')
            step += 1
    print("\ndone with env")

    scores = ENV.players.true_scores # torch.mean(torch.stack(stack, dim=0), dim=0)
    for key, index in mapping.items():
        genome = population.genomes[key]
        score = scores[index].item()
        genome.fitness = score
        if score <= 0 and key not in SAFE_GENOMES:
            population.to_delete.append(key)
        else:
            SAFE_GENOMES.append(key)
    SAFE_GENOMES = list(set(SAFE_GENOMES))

    _, FILE_NO = population.save_dict(
        name='original', directory='snake', file_no=FILE_NO, replace=population.generation != INIT_GEN
    )


def run_neat(population: neat.Population, epochs: int):
    population.run(eval_genomes, epochs, verbose=1)


def test_best_network(set_key: int = None):
    print("\n\n---------- RUNNING TEST ON SNAKE ----------")
    print(f"Obs => {ENV.observation_space.shape}")
    print(f"Act => {ENV.action_space.shape}")
    ENV.max_frames *= 10
    ENV.players.max_hiatus *= 10
    ENV.players.lives_total = 9

    for i in range(20):
        if set_key is None:
            genomes = list(POPULATION.genomes.values())
            def sort_key(genome: neat.Genome):
                fitness = genome.fitness
                return fitness if fitness is not None else -math.inf, genome.key
            ranking = sorted(genomes, key=sort_key, reverse=True)[:10]
            probs = [g.fitness for g in ranking if g.fitness is not None]
            if len(probs) >= 2:
                probs = np.array(probs)
                try:
                    probs = (probs - probs.min()) / (probs.max() - probs.min())
                    probs = probs / probs.sum()
                except Exception:
                    probs = None
            else:
                probs = None
            key0 = np.random.choice(ranking, size=None, replace=False, p=probs).key
        else:
            probs = None
            key0 = set_key

        print(f"Running on Genomes '{key0}'")
        keys = [key0]
        # if probs is None:
        #     random.shuffle(keys)
        print(f"Starting test no {i}")
        with torch.no_grad():

            states = ENV.reset(keys=keys)[0]
            done = False
            ENV.render()
            while not done:
                states = torch.tensor(states, device=DEVICE, dtype=DTYPE) # shape(genomes, features)
                actions = MODEL.get_policy(states, keys=keys) # shape(genomes)
                next_states, rewards, _, done, _ = ENV.step(actions.cpu().numpy())
                states = next_states
                ENV.render()
                ENV.clock.tick(60)
                print(f"\r"
                      f"Frames = {ENV.players.frames_done.max().item()}, "
                      f"Lives = {ENV.players.lives.mean().item()}, "
                      f"Scores={ENV.players.true_scores.max().item()}, "
                      f"Alive={ENV.players.active_total}, "
                      f"Fitness={ENV.players.fitness.mean().item():.4f}"
                      , end='')
            scores = {key: ENV.players.scores[index].item() for index, key in enumerate(keys)}
            print(
                f"\nDone with test:"
                f"\n\tScore -> {scores}"
            )


def fix(value: float, default: float = 1):
    if np.isinf(value) or np.isnan(value):
        return default
    else:
        return value


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    GENOMES     = 100
    WINDOW      = (30, 30)
    INIT_LEN    = 4
    BLOB_SIZE   = 15
    MAX_FRAMES  = 3000
    MAX_HIATUS  = 500
    TIMEOUT     = math.inf
    LIVES       = 5
    ENV = Game(
        WINDOW, blob=BLOB_SIZE,
        lives=LIVES, init_len=INIT_LEN, max_frames=MAX_FRAMES, max_hiatus=MAX_HIATUS, timeout=TIMEOUT,
        state_type='grid' if CONVOLUTIONAL else 'continuous',
    )

    CONFIG = neat.Config('original', 'snake')
    CONFIG.genome.init_type                 = 'normal'
    CONFIG.genome.weight_init_mean          = 0.0
    CONFIG.genome.weight_init_std           = 0.5
    CONFIG.genome.weight_min_value          = -math.inf
    CONFIG.genome.weight_max_value          = +math.inf
    CONFIG.genome.weight_mutate_power       = 5e-1
    CONFIG.genome.weight_mutate_rate        = 0.50
    CONFIG.genome.weight_replace_rate       = 0.00
    CONFIG.genome.weight_add_prob           = 0.00
    CONFIG.genome.weight_del_prob           = 0.00
    CONFIG.genome.single_structural_mutation = False
    CONFIG.genome.param_epsilon             = 1e-6
    CONFIG.reproduction.min_species_size    = GENOMES
    CONFIG.reproduction.purge               = 1
    CONFIG.reproduction.clone_threshold     = 0.05
    CONFIG.reproduction.survival_threshold  = 0.10
    CONFIG.reproduction.cross_threshold     = 0.00
    CONFIG.reproduction.elitism             = 0.30
    CONFIG.species.compatibility_threshold  = math.inf
    CONFIG.stagnation.max_stagnation        = 1
    CONFIG.stagnation.species_elitism       = 2
    CONFIG.reproduction.darwin_multiplier   = 0.50
    CONFIG.reproduction.cross_multiplier    = 0.50
    CONFIG.reproduction.preserve_elite      = False

    CONFIG.save()
    CONFIG.load(verbose=2)
    print(CONFIG)

    INPUTS          = 1 if CONVOLUTIONAL else ENV.observation_space.shape[-1]
    OUTPUTS         = 3
    EMBED_SIZE      = 32
    KERNEL_SIZE     = 3
    FC_SIZE         = 512
    LAYERS          = 2
    COEFFICIENTS    = 1
    ACTIVATION      = nn.SiLU()
    BIAS            = True
    PROBABILISTIC   = True
    CLIP_MIN        = -2
    CLIP_MAX        = 0.0
    DISTRIBUTION    = 'normal'

    MODEL = BaseModel(INPUTS, OUTPUTS, EMBED_SIZE, KERNEL_SIZE, FC_SIZE, LAYERS, COEFFICIENTS,
                      ACTIVATION, PROBABILISTIC, BIAS, DEVICE, DTYPE,
                      clip_min=CLIP_MIN, clip_max=CLIP_MAX, distribution=DISTRIBUTION)
    print(MODEL)

    POPULATION = neat.Population(GENOMES, MODEL, CONFIG, init_rep=True)
    print(POPULATION)

    POPULATION.load_dict(name='original', directory='snake', file_no=FILE_NO)
    INIT_GEN = POPULATION.generation

    # Training parameters
    EPOCHS          = 200
    MEMORY_SIZE     = 5
    GAMMA           = math.exp(math.log(0.33) / 512)
    KAPPA           = math.exp(math.log(0.33) / 128)
    ALPHA           = fix(np.exp(np.log(1.10) / (MEMORY_SIZE - 1)), 1.0)
    ALPHA_ORDER     = 2
    REW_NORM        = 2

    TRAINER = neat.NEAT(
        POPULATION,
        schedulers=[
            # neat.optim.scheduler.RandomAnnealing(config, 1e-1, 1e-0, 5, ['weight_init_std', 'weight_mutate_power'], True),
            neat.optim.scheduler.CosineAnnealing(CONFIG, 10, 0.1, 'weight_mutate_power', True, True),
            # neat.optim.scheduler.CosineAnnealing(config, 10, 0.1, 'weight_replace_rate', True, True),
            # neat.optim.scheduler.CosineAnnealing(config, 15, 0.05, 'weight_add_prob', True, True),
            # neat.optim.scheduler.CosineAnnealing(config, 15, 0.05, 'weight_del_prob', True, True),
        ],
        device=DEVICE, dtype=DTYPE,
        log_sub_dir='snake-original\\',
        log_name=f"{unix_to_datetime_file(clock.time())}_"
                 f"e{EMBED_SIZE}-l{LAYERS}--b{int(BIAS)}-"
                 f"g{round(GAMMA, 4)}-a{round(ALPHA, 4)}-ao{ALPHA_ORDER}-"
                 f"rn{REW_NORM}-p{round(0.0, 4)}",
        gamma=GAMMA, kappa=KAPPA, alpha=ALPHA, order=ALPHA_ORDER, normalize=REW_NORM,
        rew_reg=1.0, pol_reg=0.0, validate=True, groups=None,
        max_episodes=MEMORY_SIZE,
    )

    TRAINER.learn(eval_genomes, MAX_FRAMES * 2, EPOCHS, 256, 0.05, 'continuous', 3)

    test_best_network(set_key=None)
