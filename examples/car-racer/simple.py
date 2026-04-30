import ModifiedNEAT as neat
import ModifiedNEAT.nn as mn
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import warnings
import random
import math
import numpy as np
import argparse
import subprocess as sp
import pygame

from torch import Tensor
from typing import Union
from numba.core.errors import NumbaPerformanceWarning

EXMP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXMP_DIR not in sys.path:
    sys.path.insert(0, EXMP_DIR)

from game import Game, EnvironmentConfig

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
                    mn.LayerNorm(dim_size, bias=False, device=device, dtype=dtype),
                    activation,
                    # mn.Polynomial(dim_size, dim_size, coefficients, bias, device, dtype),
                    mn.Linear(dim_size, dim_size, bias, device, dtype),
                ]
                for _ in range(layers)
            ], []),
        ])
        self.pol_proj = mn.Sequential(*[
            # mn.Linear(dim_size, dim_size, bias, device, dtype),
            mn.LayerNorm(dim_size, bias=False, device=device, dtype=dtype),
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
        # mean            = F.sigmoid(mean) * 6 + -3
        # std             = torch.pow(10, F.sigmoid(log_std) * self.clip_range + self.clip_min)
        std = torch.exp(log_std)
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
        # dist        = torch.distributions.Normal(mean, std)
        # action      = torch.sigmoid((dist.sample() if options.get('normal', self.probabilistic) else mean) * torch.pi)
        if options.get('normal', self.probabilistic):
            action = mean + (std * torch.randn_like(std))
        else:
            action = mean
        if self.distribution == 'discrete':
            action = torch.argmax(action, dim=-1)
        else:
            action = torch.tanh(action)
        return action

    # def get_value(self, state: Tensor, keys: Union[int, list[int]] = None) -> Tensor:
    #     latent      = self.projection(state, keys=keys)
    #     value       = self.val_proj(latent, keys=keys)
    #     return value


ENV: Game | None = None
MODEL: BaseModel | None = None
POPULATION: neat.Population | None = None
FILE_NAME: str = "original"
FILE_NO: int | None = None
INIT_GEN: int = 0


def eval_genomes(population: neat.Population):
    """
    Run each genome against each other one time to determine the fitness.
    """
    global ENV, MODEL, FILE_NAME, FILE_NO
    mapping: dict[int, int] = population.get_mapping(consolidated=True)
    _mapping = list(mapping.items())
    # random.shuffle(_mapping)
    mapping = dict(_mapping)
    keys = list(mapping.keys())
    # genomes = list(population.genomes.items())
    # global MODEL

    print(f"started generation {population.generation}")
    stack = []
    with torch.no_grad():
        if population.generation % 10 == 0:
            ENV.render_mode = 'human'
        else:
            ENV.render_mode = None
        states = ENV.reset(keys=keys)[0]
        done = False
        step = 0
        DEBUG_STEP = 10
        ENV.render()
        while not done:
            DEBUG = step == DEBUG_STEP and population.generation == INIT_GEN
            states = torch.tensor(states, device=DEVICE, dtype=DTYPE) # shape(genomes, features)
            if DEBUG:
                print(f"\nstates => \n{states} \n\tshape = {states.shape}")
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

            print(f"\rLives = {ENV.players.lives.mean().item()}, "
                  f"Scores={ENV.players.scores.max().item()}, "
                  f"Alive={ENV.players.active_total}, "
                  f"Fitness={ENV.players.fitness.mean().item():.1f}"
                  , end='')
            step += 1
    print("\ndone with env")

    scores = torch.mean(torch.stack(stack, dim=0), dim=0)
    for key, index in mapping.items():
        genome = population.genomes[key]
        genome.fitness = scores[index].item()

    _, FILE_NO = population.save_dict(
        name=FILE_NAME, directory='racer', file_no=FILE_NO, replace=population.generation != INIT_GEN
    )


def run_neat(population: neat.Population, epochs: int):
    population.run(eval_genomes, epochs, verbose=None)


def test_best_network(*, count: int = 10, set_keys: list[int] = None, record: bool = False):
    count = max(1, count)

    print("\n\n---------- RUNNING TEST ON CAR-RACER ----------")
    print(f"best_genome = {POPULATION.best_genome}")
    ENV.players.lives_total = 10
    ENV.render_mode = 'human'
    ENV.window.fps = 60
    ENV.cars.usr_enganged = False
    ENV.min_laps = 4
    ENV.max_frames = 3000

    WIDTH, HEIGHT = ENV.window.width, ENV.window.height  # or screen.get_size()
    FPS = 20
    RECORD_DIR = f"{EXMP_DIR}/recordings"
    print(f"Recording @ {RECORD_DIR}")
    if not os.path.exists(RECORD_DIR):
        os.mkdir(RECORD_DIR)
    

    for i in range(25):
        ffmpeg_cmd = [
            "ffmpeg",
            "-loglevel", "quiet",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{WIDTH}x{HEIGHT}",
            "-r", str(FPS),
            "-i", "-",  # stdin
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-movflags", "+faststart",
            f"recordings/test_run_{i:03d}.mp4"
        ]

        if set_keys is not None:
            keys = set_keys
            probs = None
        else:
            genomes = list(POPULATION.genomes.values())
            def sort_key(genome: neat.Genome):
                fitness = genome.fitness
                return fitness if fitness is not None else -math.inf, genome.key
            ranking = sorted(genomes, key=sort_key, reverse=True)[:count + 10]
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
            keys = [g.key for g in np.random.choice(ranking, size=count, replace=False, p=probs)]

        print(f"Running on Genomes {keys[:10]}")
        if probs is None:
            random.shuffle(keys)
        best_genome = POPULATION.best_genome
        if best_genome is not None and best_genome.key not in keys:
            keys.insert(0, best_genome.key)
        mapping = {k: idx for idx, k in enumerate(keys)}
        reverse = {idx: k for k, idx in mapping.items()}
        print(f"\nStarting test no {i}")
        if record:
            process = sp.Popen(ffmpeg_cmd, stdin=sp.PIPE)
        with torch.no_grad():
            states = ENV.reset(keys=keys)[0]
            done = False
            ENV.render()
            while not done:
                states = torch.tensor(states, device=DEVICE, dtype=DTYPE) # shape(genomes, features)
                actions = MODEL.get_policy(states.unsqueeze(1), keys=keys).squeeze(1) # shape(genomes)
                next_states, rewards, _, done, info = ENV.step(actions.cpu().numpy())
                states = next_states
                ENV.render()
                ranking = np.argsort(-ENV.players.fitness)
                leaders = [reverse[idx] for idx in ranking]
                # print(
                #     f"\rLives = {ENV.players.lives.mean().item():.1f}, "
                #     f"Alive={ENV.players.active_total}, "
                #     f"{leaders[:5]}{' ' * 15}"
                #     , end=''
                # )

                frame = pygame.surfarray.array3d(ENV.window.screen)
                frame = np.transpose(frame, (1, 0, 2))  # Pygame → correct orientation

                if record:
                    process.stdin.write(frame.tobytes())

                if done:
                    print(f"\ninfo: {info}")
            scores = dict(
                sorted(
                    ((k, ENV.players.scores[idx].item()) for k, idx in mapping.items()),
                    key=lambda item: item[1],
                    reverse=True
                )[:5]
            )
            print(
                f"\nDone with test:"
                f"\n\tScore -> {scores}"
            )
        if record:
            process.stdin.close()
            process.wait()
            process.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run NEAT algorithm on CarRacer game')
    parser.add_argument('--render_mode', type=str, default="human", 
                        help='Render mode for the game (e.g., "human", None)')
    parser.add_argument('--load', type=str, default=None,
                        help='Load a previously trained model (True for latest, or specify file number)')
    parser.add_argument('--train', type=str, default='true',
                        help='Train the model (True/False or 1/0, default: True)')
    args = parser.parse_args()

    GENOMES = 200
    SHAPE = (6, 6)
    SIZE = 110
    LIVES = 10
    
    # Create environment configuration
    env_config = EnvironmentConfig(params={
        'shape': SHAPE,
        'size': SIZE,
        'render_mode': args.render_mode,
        'lives': LIVES,
        'max_frames': 2000,
        'max_hiatus': 200,
        'grid': {
            'size': SHAPE,
            'cell_size': SIZE,
            'track': {
                'curb': 0.05,
                'grass': 0.125,
                'gravel': 0.125,
            }
        },
        'window': {
            'fps': 60,
        }
    })
    
    # Create environment with configuration
    ENV = Game(config=env_config)

    CONFIG = neat.Config('racer', '../../storage/configs')
    CONFIG.genome.init_type                 = 'normal'
    CONFIG.genome.weight_init_mean          = 0.0
    CONFIG.genome.weight_init_std           = 1.5
    CONFIG.genome.weight_min_value          = -math.inf
    CONFIG.genome.weight_max_value          = +math.inf
    CONFIG.genome.weight_mutate_power       = 7e-1
    CONFIG.genome.weight_mutate_rate        = 0.60
    CONFIG.genome.weight_replace_rate       = 0.00
    CONFIG.genome.weight_add_prob           = 1.00
    CONFIG.genome.weight_del_prob           = 1e-9
    CONFIG.genome.single_structural_mutation = True
    CONFIG.genome.param_epsilon             = 1e-12
    CONFIG.reproduction.min_species_size    = GENOMES
    CONFIG.reproduction.purge               = 1
    CONFIG.reproduction.clone_threshold     = 0.05
    CONFIG.reproduction.survival_threshold  = 0.10
    CONFIG.reproduction.cross_threshold
    CONFIG.reproduction.elitism             = 0.30
    CONFIG.species.compatibility_threshold  = math.inf
    CONFIG.stagnation.max_stagnation        = 1
    CONFIG.stagnation.species_elitism       = 2
    CONFIG.reproduction.darwin_multiplier   = 0.33
    CONFIG.reproduction.cross_multiplier    = 0.50
    CONFIG.reproduction.preserve_elite      = False

    CONFIG.save()
    CONFIG.load(verbose=2)
    print(CONFIG)

    INPUTS          = 9
    OUTPUTS         = 2
    EMBED_SIZE      = 64
    LAYERS          = 3
    COEFFICIENTS    = 1
    ACTIVATION      = nn.SiLU()
    BIAS            = True
    PROBABILISTIC   = True
    FILE_NAME       = f"RacerModel-E{EMBED_SIZE}_L{LAYERS}_C{COEFFICIENTS}_"\
                      f"A-{ACTIVATION.__class__.__name__}_"\
                      f"B{int(BIAS)}_P{int(PROBABILISTIC)}"

    MODEL = BaseModel(INPUTS, OUTPUTS, EMBED_SIZE, LAYERS, COEFFICIENTS,
                      ACTIVATION, PROBABILISTIC, BIAS, DEVICE, DTYPE)
    print(MODEL)

    POPULATION = neat.Population(GENOMES, MODEL, CONFIG, init_rep=True)
    print(POPULATION)

    # Handle --load argument
    if args.load is not None:
        if args.load.lower() == 'true':
            # Load the latest checkpoint
            POPULATION.load_dict(name=FILE_NAME, directory='racer', file_no=None)
            print(f"Loaded latest population checkpoint")
        else:
            # Load a specific file number
            try:
                file_no = int(args.load)
                POPULATION.load_dict(name=FILE_NAME, directory='racer', file_no=file_no)
                print(f"Loaded population checkpoint from file {file_no}")
            except ValueError:
                if args.load.lower() != 'false':
                    print(f"Error: --load argument must be 'true' or a valid file number, got '{args.load}'")
                    exit(1)
    
    INIT_GEN = POPULATION.generation
    FILE_NO = None  # Reset FILE_NO for saving subsequent generations

    # Parse --train argument
    def str_to_bool(value: str) -> bool:
        if isinstance(value, bool):
            return value
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            return False
        else:
            raise ValueError(f"Cannot convert '{value}' to boolean")
    
    try:
        should_train = str_to_bool(args.train)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    EPOCHS = None

    if should_train:
        print(f"\n{'='*50}")
        print("Starting training...")
        print(f"{'='*50}\n")
        run_neat(POPULATION, EPOCHS)
    else:
        print(f"\n{'='*50}")
        print("Skipping training...")
        print(f"{'='*50}\n")

    test_best_network(count=4)
