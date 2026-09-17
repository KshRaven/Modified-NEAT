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

from ModifiedNEAT.nn.base import Model
from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.util.datetime import unix_to_datetime_file
from ModifiedNEAT.util.qol import manage_params
from torch import Tensor
from typing import Union
from numba.core.errors import NumbaPerformanceWarning

EXMP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXMP_DIR not in sys.path:
    sys.path.insert(0, EXMP_DIR)

from game import Game

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
torch.set_printoptions(threshold=10)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float32
neat.set_device(DEVICE)
print(f"Using PyTorch device: '{DEVICE}'. NEAT device: '{neat.get_device()}'")
neat.util.storage.set_storage_location("../../storage/") # Project directory


class BaseModel(Model):
    def __init__(
        self, inputs: int, outputs: int, dim_size: int, layers: int, coefficients=1, 
        activation=nn.SiLU(), probabilistic=False, bias=True, 
        device: torch.device | str = 'cpu', dtype: torch.dtype = torch.float32, **options
    ):
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
        self.normalize      = options.get('normalize', True)
        self.clip_min       = options.get('clip_min', -2)
        self.clip_max       = options.get('clip_max', -0)
        self.clip_range     = self.clip_max - self.clip_min
        self.lower          = 10 ** self.clip_min
        self.upper          = 10 ** self.clip_max

        # Build
        self.lat_proj = mn.Sequential(*[
            mn.Linear(inputs, dim_size, True, device, dtype),
            *sum([
                [
                    mn.LayerNorm(dim_size, bias=True, device=device, dtype=dtype),
                    activation,
                    mn.Linear(dim_size, dim_size, bias, device, dtype),
                ]
                for _ in range(layers)
            ], []),
        ])
        self.pol_proj = mn.Sequential(*[
            mn.LayerNorm(dim_size, bias=True, device=device, dtype=dtype),
            activation,
            mn.Linear(dim_size, 2*outputs, True, device, dtype),
        ])

    def extra_repr(self) -> str:
        return f"probabilistic={self.probabilistic}, distro='{self.distribution}'"

    def forward(self, state: Tensor, keys: Union[int, list[int]] = None, **kwargs):
        return self.get_policy(state, keys=keys, **kwargs)

    def get_mean_std(self, latent: Tensor, keys: Union[int, list[int]] = None) -> tuple[Tensor, Tensor]:
        mean_std        = self.pol_proj(latent, keys=keys)
        mean, log_std   = torch.chunk(mean_std, 2, -1)
        if self.normalize:
            mean = torch.tanh(mean) * max(1.0, self.upper)
            std = torch.pow(10, self.clip_min + self.clip_range * torch.sigmoid(log_std))
        else:
            std = torch.exp(log_std)
        return mean, std

    # def get_action(self, state: Tensor, keys: Union[int, list[int]] = None) -> tuple[Tensor, Tensor]:
    #     latent      = self.projection(state, keys=keys)
    #     mean, std   = self.get_mean_std(latent, keys=keys)
    #     dist        = torch.distributions.Normal(mean, std)
    #     action      = torch.sigmoid((dist.sample() if self.probabilistic else mean) * torch.pi)
    #     log_prob    = dist.log_prob(action)
    #     return action, log_prob

    # def evaluate_action(self, state: Tensor, action: Tensor, keys: Union[int, list[int]] = None):
    #     latent      = self.projection(state, keys=keys)
    #     mean, std   = self.get_mean_std(latent, keys=keys)
    #     dist        = torch.distributions.Normal(mean, std)
    #     log_prob    = dist.log_prob(action)
    #     entropy     = dist.entropy()
    #     return log_prob, entropy

    def get_policy(self, state: Tensor, keys: Union[int, list[int]] = None, **options) -> Tensor:
        latent      = self.lat_proj(state, keys=keys)
        mean, std   = self.get_mean_std(latent, keys=keys)
        # dist        = torch.distributions.Normal(mean, std)
        # action      = torch.sigmoid((dist.sample() if options.get('normal', self.probabilistic) else mean) * torch.pi)
        action = mean
        if options.get('normal', self.probabilistic):
            action = torch.normal(mean, std)
        if self.distribution == 'discrete':
            action = torch.argmax(action, dim=-1)
        else:
            action = torch.tanh(action * torch.pi)
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


# GAME SETTINGS
SPAWN_WIDTH         = 200
GAP_OFFSET          = 30
GAP_SIZE            = (200, 200)
PIPE_Y_VELOCITY     = 0
FULL_STATES         = False
DELAY               = 0
DISCRETE            = False

# Model properties
GENOMES             = 100
INPUTS              = (5 + (2 if PIPE_Y_VELOCITY else 0) if FULL_STATES else 3)
OUTPUTS             = 1 if not DISCRETE else 2
EMBED_SIZE          = 32
COEFFICIENTS        = 1
LAYERS              = 2
BIAS                = True
PROBABILISTIC       = True
TEST_ACTIVATION     = nn.SiLU()
CLIP_MIN            = -3
CLIP_MAX            = -1
DISTRIBUTION        = 'normal' if not DISCRETE else 'discrete'
MODEL_NUM           = 3

# Trainer properties
MEMORY_SIZE         = 5
GAMMA               = fix(np.exp(np.log(0.01) / 128), 0.0)
KAPPA               = 0.0 # fix(np.exp(np.log(0.01) / 128), 0.0)
ALPHA               = fix(np.exp(np.log(3.0) / (MEMORY_SIZE - 1)), 1.0)
ALPHA_ORDER         = 4
REW_NORM            = 0
POL_REG             = 0.75
STD_REG             = 0.25

# Training 
GOAL        = 20
STEPS       = GOAL * 100
EPOCHS      = 100
FILTER_DEAD = True

MODELS = [
    BaseModel(
        INPUTS, OUTPUTS, EMBED_SIZE, LAYERS, COEFFICIENTS,
        activation, PROBABILISTIC, BIAS, DEVICE, DTYPE,
        clip_min=CLIP_MIN, clip_max=CLIP_MAX, distribution=DISTRIBUTION,
    )
    for activation in [TEST_ACTIVATION for _ in range(MODEL_NUM)]
]

GAME_TYPE = 0 if not FULL_STATES else (1 if PIPE_Y_VELOCITY == 0 else 2)
BASE_NAME: str = (
    f"GT{GAME_TYPE}_DC{int(DISCRETE)}_T{MODEL_NUM}_"
    f"E{EMBED_SIZE}_L{LAYERS}_C{COEFFICIENTS}_"
    f"A-{TEST_ACTIVATION.__class__.__name__}_B{int(BIAS)}"
    f"{f'_R-{CLIP_MIN}~{CLIP_MAX}' if PROBABILISTIC else ''}"
)
FILE_NAME: str = f"FlappyBirdModel-{BASE_NAME}"
FILE_DIR: str | None = "flappy-bird/main"
FILE_NO: int | None = None
INIT_GEN: int | None = None

print(f"\ncreating config")
CONFIG = neat.Config('main', '.config')

CONFIG.genome.init_type                     = 'normal'
CONFIG.genome.weight_init_mean              = 0.0
CONFIG.genome.weight_init_std               = 1.0
CONFIG.genome.weight_min_value              = -np.inf
CONFIG.genome.weight_max_value              = +np.inf
CONFIG.genome.weight_mutate_power           = 5e-1
CONFIG.genome.weight_mutate_rate            = 0.65
CONFIG.genome.weight_replace_rate           = 0.0
CONFIG.genome.weight_add_prob               = 0.0
CONFIG.genome.weight_del_prob               = 0.0
CONFIG.genome.single_structural_mutation    = False

CONFIG.reproduction.min_species_size        = GENOMES
CONFIG.reproduction.purge                   = 1
CONFIG.reproduction.clone_threshold         = 0.00
CONFIG.reproduction.survival_threshold      = 0.15
CONFIG.reproduction.cross_threshold         = 0.00
CONFIG.reproduction.elitism                 = 0.33
CONFIG.species.compatibility_threshold      = np.inf
CONFIG.stagnation.max_stagnation            = 1
CONFIG.stagnation.species_elitism           = 2
CONFIG.reproduction.darwin_multiplier       = 0.50
CONFIG.reproduction.cross_multiplier        = 0.50
CONFIG.reproduction.preserve_elite          = False
CONFIG.save()
CONFIG.load(2)

ENV: Game | None = None


def evaluate(population: neat.Population, **options):
    trainer: neat.rl.NEAT = options['trainer']
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
    run_step = 0
    game_step = 0
    for model in MODELS: model.eval()
    while not terminate:
        gts = clock.perf_counter()
        reverse_mappings = tuple([
            {index: key for key, index in m.items()}
            for m in split_mappings
        ])

        states = ENV.reset()[0] # States are already PyTorch tensors
        step = 0
        DEBUG_STEP = 10
        done = False
        while not done:
            with torch.no_grad():
                DEBUG = step == DEBUG_STEP and population.generation == INIT_GEN
                
                # Get Inputs ~ send bird location, top pipe location and bottom pipe location etc
                if DEBUG:
                    print(f"\nstates =>\n{states}\n\tshape = {states.shape}")

                # Filter dead birds from calculation
                ts = clock.perf_counter()
                if FILTER_DEAD:
                    split_keys: list[list[int]] = [[] for _ in range(MODEL_NUM)]
                    split_indices: list[list[int]] = [[] for _ in range(MODEL_NUM)]
                    for index, dead in enumerate(ENV.birds.dead):
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
                    split_keys: list[list[int]] = [list(m.keys()) for m in split_mappings]
                    split_indices: list[list[int]] = [list(m.values()) for m in split_mappings]

                # Get actions # NOTE: Dense layers don't need batch dimension
                split_observations = torch.split(states, split_sizes, dim=0)
                split_observations = [obs[indices] for obs, indices in zip(split_observations, split_indices)]
                split_actions = [
                    m.get_policy(obs.unsqueeze(1), keys=split_keys[mi]).squeeze(1)
                    for mi, (m, obs) in enumerate(zip(MODELS, split_observations))
                ]

                # Pad dead bird actions
                padding = ENV.birds.bird_num - sum([act.shape[0] for act in split_actions])
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
                actions = torch.cat(split_actions, dim=0)
                calc_time = clock.perf_counter() - ts
                if DEBUG:
                    print(f"\nactions =>\n{actions}\n\tshape = {actions.shape}")

                # Get rewards
                next_states, rewards, _, done, _ = ENV.step(actions)
                if DEBUG:
                    print(f"\nrewards =>\n{rewards}\n\tshape = {rewards.shape}")
                    # v = MODEL.get_value(observations[:len(reverse_mapping0)].unsqueeze(1), ).squeeze(1)
                    # print(f"\nvalues =>\n{v}\n\tshape = {v.shape}")
                    # del v

                # Updated buffers
                terminate = trainer.update(states, actions, rewards, done, done)

                # CMDLine rendering
                alive = round(ENV.birds.active_num)
                max_score = round(rewards.max().item(), 2)
                print(
                    f"\r{CM('Executing', Fore.GREEN)}: "
                    f"time_elapsed = {round(clock.perf_counter()-gts)}s, "
                    f"alive = {alive}, max_rew = {max_score}, "
                    f"ct={calc_time:.2e}, sd={trainer.steps_done}, "
                    f"bl={trainer.primary.max_size()}", end=''
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

        l_lim, u_lim = 10, ENV.floor.y * 1.05 # Delete genomes that go beyond the game's y-limits
        for idx, (score, genome) in enumerate(zip(ENV.birds.get_reward(), population.genomes.values())):
            genome.fitness = score.item()
            y_position = ENV.birds.y[idx]
            died_beyond_limits = y_position <= l_lim or y_position >= u_lim
            if died_beyond_limits:
                population.to_delete.append(genome.key)

        game_step += 1
    print(f"\n")

    _, file_no = population.save_dict(FILE_NAME, FILE_DIR, FILE_NO, replace=population.generation != INIT_GEN)
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
    ks = max(1, np.max(np.log10([g.key for g in population.genomes.values()])).item()) + 1
    gs = 3
    fs = max(1, np.max(np.log10([g.actual_fitness for g in population.genomes.values()])).item()) + 4
    for i, genus in enumerate(population.genera):
        genomes = sorted([g for g in population.genomes.values() if g.genus == genus], key=lambda g: g.fitness, reverse=True)[:COUNT]
        for genome in genomes:
            print(f"\t{fill(genome.key, ks)}|{fill(genome.genus, gs)}|{fill(round(genome.actual_fitness, 4), fs)}")
        if i < len(population.genera) - 1:
            print(f"\t...")


def run():
    parser = argparse.ArgumentParser(description='Run NEAT algorithm on Flappy Bird game')
    parser.add_argument('--rm', type=str, default=None, 
                        help='Render mode for the game (e.g., "human", None)')
    parser.add_argument('--load', type=str, default=None,
                        help='Load a previously trained model (True for latest, or specify file number)')
    parser.add_argument('--train', type=str, default='true',
                        help='Train the model (True/False or 1/0, default: True)')
    args = parser.parse_args()

    global ENV, FILE_NO
    print(MODELS[0])

    # Create the population, which is the top-level object for a NEAT run.
    print(f"\ncreating population")
    POPULATION = neat.Population(GENOMES, MODELS[0], CONFIG, init_reporter=True)
    for model in MODELS[1:]:
        extra_population = neat.Population(GENOMES, model, CONFIG, init_reporter=True)
        POPULATION.absorb_population(extra_population)
    if args.load is not None:
        if args.load.lower() == 'true':
            FILE_NO = None
            print(f"Will load the latest population checkpoint")
            POPULATION.load_dict(None, FILE_NAME, FILE_DIR, FILE_NO)
        else:
            try:
                FILE_NO = int(args.load)
                print(f"Will load population checkpoint from file number '{FILE_NO}'")
                POPULATION.load_dict(None, FILE_NAME, FILE_DIR, FILE_NO)
            except ValueError:
                if args.load.lower() not in ['false', 'none', 'null', '']:
                    print(f"Error: --load argument must be a boolean, valid positive integer, null or empty; Got '{args.load}'")
                    exit(1)

    ENV = Game(
        POPULATION.size, goal=GOAL, seq_len=None,
        height=800, width=800, full_state=FULL_STATES, pipe_y_velocity=PIPE_Y_VELOCITY,
        spawn_width=SPAWN_WIDTH, tick=None, gap_offset=GAP_OFFSET, gap_size=GAP_SIZE,
        delay=DELAY, type2count=GENOMES, type2offset=0, device=DEVICE, dtype=DTYPE,
        render_mode=args.rm
    )

    def str_to_bool(value: str) -> bool:
        if isinstance(value, bool):
            return value
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            return False
        else:
            raise ValueError(f"Cannot convert '{value}' to boolean")

    if str_to_bool(args.train):
        trainer = neat.rl.NEAT(
            POPULATION,
            schedulers=[],
            device=DEVICE, dtype=DTYPE,
            log_sub_dir='flappy_bird/',
            log_name=f"{unix_to_datetime_file(clock.time())}-main_"
                     f"e{EMBED_SIZE}-c{COEFFICIENTS}-l{LAYERS}-b{int(BIAS)}-"
                     f"g{round(GAMMA, 4)}-k{round(KAPPA, 4)}-a{round(ALPHA, 4)}-ao{ALPHA_ORDER}-"
                     f"rn{REW_NORM}-p{round(POL_REG, 4)}-s{round(STD_REG, 4)}-mem{MEMORY_SIZE}-"
                     f"delay{DELAY}-type{int(DISCRETE)}-gt{GAME_TYPE}",
            gamma=GAMMA, alpha=ALPHA, kappa=KAPPA, order=ALPHA_ORDER, normalize=REW_NORM,
            rew_reg=1.0, pol_reg=POL_REG, std_reg=STD_REG, validate=True, segr_size=10,
            max_episodes=MEMORY_SIZE,
        )
        trainer.set_report_hook(genome_debug)

        print(f"starting evaluation: population={len(POPULATION.genomes)}")
        try:
            trainer.learn(
                evaluate, STEPS, EPOCHS, 256, 0.1,
                'binary' if not DISCRETE else 'discrete', True
            )
        except KeyboardInterrupt:
            pass

    elites_limit = 10
    elites_available = any([
        g.fitness is not None and not np.isnan(g.fitness) 
        for g in POPULATION.genomes.values()
    ])
    if elites_available:
        elites = []
        np.random.randn()
        for gn_id, specie in enumerate(POPULATION.species_set.species.values()):
            genomes = sorted(
                [
                    g for g in specie.members.values()
                    if g.fitness is not None and not np.isnan(g.fitness)
                ], 
                key=lambda g: g.fitness, reverse=True
            )
            elites.extend(genomes[:elites_limit])
            print(f"\nElites [{gn_id}]: {[g.key for g in genomes[:10]]}")
        POPULATION.crop([e.key for e in elites], DEVICE)

    last_genus = POPULATION.genera[-1]
    last_genus_size = len([g for g in POPULATION.genomes.values() if g.genus == last_genus])
    print(f"Last genus size: {last_genus_size}")
    env = Game(
        POPULATION.size, goal=100, seq_len=None,
        height=800, width=1200, full_state=FULL_STATES, pipe_y_velocity=PIPE_Y_VELOCITY * 1.85,
        spawn_width=SPAWN_WIDTH, tick=None, gap_offset=GAP_OFFSET, gap_size=GAP_SIZE,
        delay=DELAY, render_mode='human',
        type2count=last_genus_size, type2offset=50, device=DEVICE, dtype=DTYPE
    )
    split_mappings = POPULATION.get_mapping()
    if len(POPULATION.genera) == 1: split_mappings = (split_mappings,)
    split_sizes = [len(m) for m in split_mappings]
    cons_mapping = POPULATION.get_mapping(consolidated=True)
    
    with torch.no_grad():
        for i in range(5):
            done = False
            step = 0
            states = env.reset()[0]
            ts = clock.perf_counter()
            while not done:
                # Get states
                states = states.unsqueeze(1)

                # Get actions
                split_states = torch.split(states, split_sizes, dim=0)
                split_actions = [
                    m.get_policy(obs).squeeze(1)
                    for m, obs in zip(MODELS, split_states)
                ]
                actions = torch.cat(split_actions, dim=0)

                # Get rewards
                next_states, rewards, _, done, _ = env.step(actions)
                states = next_states
                
                env.render()
                step += 1


if __name__ == '__main__':
    run()
