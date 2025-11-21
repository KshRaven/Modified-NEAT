# https://neat-python.readthedocs.io/en/latest/xor_example.html

from PongGame import Game

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

from torch import Tensor
from typing import Union
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float32
neat.util.storage.set_storage_location("../../storage/")


class BaseModel(mn.Model):
    def __init__(self, inputs: int, outputs: int, dim_size: int, layers: int, coefficients=1, activation: nn.Module = nn.SiLU(),
                 probabilistic=False, bias=True, device: torch.device = 'cpu', dtype: torch.dtype = torch.float32, **options):
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
            # mn.Polynomial(inputs, dim_size, coefficients, True, device, dtype),
            mn.Linear(inputs, dim_size, True, device, dtype),
            *sum([
                [
                    # mn.LayerNorm(dim_size, bias=False, device=device, dtype=dtype),
                    activation,
                    # mn.Polynomial(dim_size, dim_size, coefficients, bias, device, dtype),
                    mn.Linear(dim_size, dim_size, bias, device, dtype),
                ]
                for _ in range(layers)
            ], []),
        ])
        self.pol_proj = mn.Sequential(*[
            # mn.Linear(dim_size, dim_size, bias, device, dtype),
            # mn.LayerNorm(dim_size, bias=False, device=device, dtype=dtype),
            activation,
            # mn.Polynomial(dim_size, 2*outputs, coefficients, True, device, dtype),
            mn.Linear(dim_size, 2*outputs, True, device, dtype),
        ])

    def extra_repr(self) -> str:
        return f"probabilistic={self.probabilistic}, distro='{self.distribution}'"

    def forward(self, state: Tensor, keys: Union[int, list[int]] = None, **kwargs):
        return self.get_policy(state, keys=keys, **kwargs)

    def get_mean_std(self, latent: Tensor, keys: Union[int, list[int]] = None):
        mean_std        = self.pol_proj(latent, keys=keys)
        mean, log_std   = torch.chunk(mean_std, 2, -1)
        mean            = F.sigmoid(mean) * 6 + -3
        std             = torch.pow(10, F.sigmoid(log_std) * self.clip_range + self.clip_min)
        return mean, std

    def get_action(self, state: Tensor, keys: Union[int, list[int]] = None) -> tuple[Tensor, Tensor]:
        latent      = self.projection(state, keys=keys)
        mean, std   = self.get_mean_std(latent, keys=keys)
        dist        = torch.distributions.Normal(mean, std)
        action      = torch.sigmoid((dist.sample() if self.probabilistic else mean) * torch.pi)
        log_prob    = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, state: Tensor, action: Tensor, keys: Union[int, list[int]] = None):
        latent      = self.projection(state, keys=keys)
        mean, std   = self.get_mean_std(latent, keys=keys)
        dist        = torch.distributions.Normal(mean, std)
        log_prob    = dist.log_prob(action)
        entropy     = dist.entropy()
        return log_prob, entropy

    def get_policy(self, state: Tensor, keys: Union[int, list[int]] = None, **options) -> Tensor:
        latent      = self.projection(state, keys=keys)
        mean, std   = self.get_mean_std(latent, keys=keys)
        dist        = torch.distributions.Normal(mean, std)
        action      = torch.sigmoid((dist.sample() if options.get('normal', self.probabilistic) else mean) * torch.pi)
        return action

    # def get_value(self, state: Tensor, keys: Union[int, list[int]] = None) -> Tensor:
    #     latent      = self.projection(state, keys=keys)
    #     value       = self.val_proj(latent, keys=keys)
    #     return value


ENV: Game | None = None
MODEL: BaseModel | None = None
FILE_NO: int | None = None
INIT_GEN: int = 0


def eval_genomes(population: neat.Population):
    """
    Run each genome against each other one time to determine the fitness.
    """
    global ENV, MODEL, FILE_NO
    mapping: dict[int, int] = population.get_mapping(consolidated=True)
    _mapping = list(mapping.items())
    # random.shuffle(_mapping)
    mapping = dict(_mapping)
    keys = list(mapping.keys())
    # genomes = list(population.genomes.items())
    # global MODEL
    # width, height = 700, 500
    # win = pygame.display.set_mode((width, height))
    # pygame.display.set_caption("Pong")

    print(f"started generation {population.generation}")
    stack = []
    with torch.no_grad():
        states = ENV.reset(keys=keys)[0]
        done = False
        step = 0
        DEBUG_STEP = 10
        ENV.render()
        while not done:
            DEBUG = step == DEBUG_STEP and population.generation == INIT_GEN
            states = torch.tensor(states, device=DEVICE, dtype=DTYPE) # shape(genomes, features)
            if DEBUG:
                print(f"states => \n{states} \n\tshape = {states.shape}")
            actions = MODEL.get_policy(states.unsqueeze(1), keys=keys).squeeze(1) # shape(genomes)
            if DEBUG:
                print(f"actions => \n{actions} \n\tshape = {actions.shape}")
            next_states, rewards, _, done, _ = ENV.step(actions.cpu().numpy())
            rewards = torch.tensor(rewards, device=DEVICE, dtype=DTYPE) # shape(genomes, features)
            if DEBUG:
                print(f"rewards => \n{rewards} \n\tshape = {rewards.shape}")
            stack.append(rewards[..., 0])
            states = next_states
            if step % 10 == 0:
                pass
            ENV.render()
            if population.generation % 20 == 0:
                ENV.clock.tick(40)
            print(f"\rLives = {ENV.players.lives.mean().item()}, "
                  f"Hits={(ENV.players.hits + ENV.players.scores).max().item()}, "
                  f"Alive={ENV.players.active_total}, "
                  f"Fitness={ENV.players.fitness.mean().item():.4f}"
                  , end='')
            step += 1
    print("\ndone with env")

    scores = torch.mean(torch.stack(stack, dim=0), dim=0)
    for key, index in mapping.items():
        genome = population.genomes[key]
        genome.fitness = scores[index].item()

    _, FILE_NO = population.save_dict(
        name='original', directory='pong', file_no=FILE_NO, replace=population.generation != INIT_GEN
    )


def run_neat(population: neat.Population, epochs: int):
    population.run(eval_genomes, epochs, verbose=None)


def test_best_network(set_keys: tuple[int, int] = None):
    print("\n\n---------- RUNNING TEST ON PONG ----------")
    ENV.goal = 150
    ENV.players.lives_total = 9

    for i in range(20):
        if set_keys is None:
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
            key0, key1 = [g.key for g in np.random.choice(ranking, size=2, replace=False, p=probs)]
        else:
            probs = None
            key0, key1 = set_keys

        print(f"Running on Genomes '{key0}' and Genomes '{key1}'")
        keys = [key0, key1]
        if probs is None:
            random.shuffle(keys)
        print(f"Starting test no {i}")
        with torch.no_grad():

            states = ENV.reset(keys=keys)[0]
            done = False
            ENV.render()
            while not done:
                states = torch.tensor(states, device=DEVICE, dtype=DTYPE) # shape(genomes, features)
                actions = MODEL.get_policy(states.unsqueeze(1), keys=keys).squeeze(1) # shape(genomes)
                next_states, rewards, _, done, _ = ENV.step(actions.cpu().numpy())
                states = next_states
                ENV.render()
                ENV.clock.tick(160)
                print(f"\rLives = {ENV.players.lives.mean().item()}, "
                      f"Hits={(ENV.players.hits + ENV.players.scores).max().item()}, "
                      f"Alive={ENV.players.active_total}, "
                      f"Fitness={ENV.players.fitness.mean().item():.4f}"
                      , end='')
            scores = {key: ENV.players.scores[index].item() for index, key in enumerate(keys)}
            hits = {key: ENV.players.hits[index].item() for index, key in enumerate(keys)}
            print(
                f"\nDone with test:"
                f"\n\tScore -> {scores}"
                f"\n\tHits -> {hits}"
            )


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    GENOMES = 100
    WINDOW = (800, 600)
    PADDLE = (10, 40)
    GOAL = 100
    LIVES = 10
    ENV = Game(WINDOW, GOAL, 3, LIVES, paddle_shape=PADDLE)

    CONFIG = neat.Config('original', 'pong')
    CONFIG.genome.init_type                 = 'normal'
    CONFIG.genome.weight_init_mean          = 0.0
    CONFIG.genome.weight_init_std           = 2.0
    CONFIG.genome.weight_min_value          = -math.inf
    CONFIG.genome.weight_max_value          = +math.inf
    CONFIG.genome.weight_mutate_power       = 2e-1
    CONFIG.genome.weight_mutate_rate        = 0.50
    CONFIG.genome.weight_replace_rate       = 0.00
    CONFIG.genome.weight_add_prob           = 0.00
    CONFIG.genome.weight_del_prob           = 0.00
    CONFIG.genome.single_structural_mutation = False
    CONFIG.genome.param_epsilon             = 1e-6
    CONFIG.reproduction.min_species_size    = GENOMES
    CONFIG.reproduction.purge               = 1
    CONFIG.reproduction.clone_threshold     = 0.05
    CONFIG.reproduction.survival_threshold  = 0.20
    CONFIG.reproduction.cross_threshold     = 0.00
    CONFIG.reproduction.elitism             = 30
    CONFIG.species.compatibility_threshold  = math.inf
    CONFIG.stagnation.max_stagnation        = 1
    CONFIG.stagnation.species_elitism       = 2
    CONFIG.reproduction.darwin_multiplier   = 0.50
    CONFIG.reproduction.cross_multiplier    = 0.50
    CONFIG.reproduction.preserve_elite      = False

    CONFIG.save()
    CONFIG.load(verbose=2)
    print(CONFIG)

    INPUTS          = 3
    OUTPUTS         = 3
    EMBED_SIZE      = 64
    LAYERS          = 3
    COEFFICIENTS    = 1
    ACTIVATION      = nn.SiLU()
    BIAS            = True
    PROBABILISTIC   = False
    CLIP_MIN        = -2
    CLIP_MAX        = 0

    MODEL = BaseModel(INPUTS, OUTPUTS, EMBED_SIZE, LAYERS, COEFFICIENTS,
                      ACTIVATION, PROBABILISTIC, BIAS, DEVICE, DTYPE,
                      clip_min=CLIP_MIN, clip_max=CLIP_MAX)
    print(MODEL)

    POPULATION = neat.Population(GENOMES, MODEL, CONFIG, init_rep=True)
    print(POPULATION)

    POPULATION.load_dict(name='original', directory='pong', file_no=FILE_NO)
    INIT_GEN = POPULATION.generation

    EPOCHS = 200

    # run_neat(POPULATION, EPOCHS)
    test_best_network(set_keys=None)
