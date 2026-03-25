
from game import Game
from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.nn.base import Model
# from ModifiedNEAT.nn.modules.sub import Linear, Conv1d, Transpose, ResidualBlock, Sequential, GroupNorm, ConverBase, SequenceEncoding
# from ModifiedNEAT.nn.modules import Reformer
from ModifiedNEAT.util.datetime import unix_to_datetime_file
from ModifiedNEAT.util.qol import manage_params
from Models import AutoEncoder

import ModifiedNEAT as neat
import ModifiedNEAT.nn as mn
import torch
import torch.nn as nn
# import torch.nn.functional as F

from torch import Tensor
from numba.typed import List, Dict
from numba.core.errors import NumbaPerformanceWarning
from typing import Union

import time as clock
import numpy as np
import warnings
import argparse
import os

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
torch.set_printoptions(threshold=10)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE  = torch.float64
neat.set_device(DEVICE)
print(f"Using torch device: '{DEVICE}'. neat device: '{neat.device()}'")


class BaseModel(Model):
    def __init__(self, inputs: int, outputs: int, dim_size: int, layers: int, coefficients=1, activation=nn.SiLU(),
                 probabilistic=False, bias=True, device: torch.device | str = 'cpu', dtype: torch.dtype = torch.float32, **options):
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
        self.clip_min       = options.get('clip_min', -4)
        self.clip_max       = options.get('clip_max', 0)
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

    def get_mean_std(self, latent: Tensor, keys: Union[int, list[int]] = None) -> tuple[Tensor, Tensor]:
        mean_std        = self.pol_proj(latent, keys=keys)
        mean, log_std   = torch.chunk(mean_std, 2, -1)
        # mean            = F.sigmoid(mean) * 6 + -3
        # std             = torch.pow(10, F.sigmoid(log_std) * self.clip_range + self.clip_min)
        std = torch.exp(log_std) if self.probabilistic else None
        return mean, std

    def get_policy(self, state: Tensor, keys: Union[int, list[int]] = None, **options) -> Tensor:
        latent      = self.projection(state, keys=keys)
        mean, std   = self.get_mean_std(latent, keys=keys)
        action      = torch.sigmoid(
            (mean + (std * torch.randn_like(std)))
            if options.get('normal', self.probabilistic) else
            mean
        )
        return action


class BaseModelSequential(Model):
    def __init__(self, max_seq_len: int, inputs: int, outputs: int, dim_size: int, layers: int,
                 heads: int, kv_heads: int | None = None, differential: int | bool = False, fwd_exp=1,
                 activation: mn.NeatModule | nn.Module = nn.SiLU(), probabilistic=False, bias=True,
                 device: torch.device | str = 'cpu', dtype: torch.dtype = torch.float32, **options):
        super().__init__()
        # Attributes
        self.max_seq_len    = max_seq_len
        self.inputs         = inputs
        self.outputs        = outputs
        self.dim_size       = dim_size
        self.layers         = max(1, layers)
        self.distribution   = manage_params(options, ['dist', 'distribution'], 'mult_var_normal')
        self.probabilistic  = probabilistic
        self.clip_min       = manage_params(options, 'clip_min', -5)
        self.clip_max       = manage_params(options, 'clip_max', -1)
        self.clip_range     = self.clip_max - self.clip_min
        self.constant       = manage_params(options, 'constant', 10_000)
        self.residual       = manage_params(options, 'residual', True)

        # Build
        ff = mn.Sequential(*[
            mn.RMSNorm(dim_size, device=device, dtype=dtype),
            # mn.LayerNorm(dim_size, bias=True, device=device, dtype=dtype),
            activation,
            mn.Linear(dim_size, dim_size, bias, device, dtype)
        ])
        self.projection = mn.Sequential(*[
            mn.BufferEmbedding(inputs, dim_size, True, 'continuous', device, dtype),
            # mn.BufferEncoding(max_seq_len, dim_size, True, 'continuous', device, dtype),
        ])
        self.transformer = mn.TransformerBase(
            max_seq_len, dim_size, layers, heads, kv_heads, differential, True, bias, device, dtype,
            residual=self.residual, normalize=True, epsilon=1e-9, fwd_exp=fwd_exp, constant=self.constant,
            activation=activation, # ff=ff,
        )
        self.pol_proj = mn.Sequential(*[
            mn.RMSNorm(dim_size, device=device, dtype=dtype),
            # mn.LayerNorm(dim_size, bias=True, device=device, dtype=dtype),
            activation,
            mn.Linear(dim_size, 2*outputs, True, device, dtype)
        ])

    def extra_repr(self) -> str:
        return f"probabilistic={self.probabilistic}, distro='{self.distribution}'"

    def forward(self, state: Tensor, keys: Union[int, list[int]] = None, **kwargs):
        return self.get_policy(state, keys=keys, **kwargs)

    def get_mean_std(self, latent: Tensor, keys: Union[int, list[int]] = None):
        mean_std        = self.pol_proj(latent, keys=keys)
        mean, log_std   = torch.chunk(mean_std, 2, -1)
        # mean            = F.tanh(mean) # * 4 + -2
        # std             = torch.pow(10, F.sigmoid(log_std) * self.clip_range + self.clip_min)
        std = torch.exp(log_std)
        return mean, std

    def _get(self, tensor: Tensor, keys: int | list[int] | None = None, verbose: int | bool = False):
        tensor = self.projection(tensor, keys=keys) # Embed | Encode the tensor
        tensor = self.transformer(tensor, keys=keys, single=True, verbose=verbose) # Pass through Tx blocks
        return tensor.squeeze(-2) # Remove the seq_len dimension

    def get_policy(self, state: Tensor, keys: Union[int, list[int]] = None, **options) -> Tensor:
        assert 5 >= state.ndim >= 3
        squeeze = state.ndim == 3 # Assumes no batch dim: (genomes, seq_len, features)
        if squeeze:
            state = state.unsqueeze(1)
        unflatten = state.ndim == 5 # Assumes inputs are pre-stacked: (genomes, records, batch_size, seq_len, features)
        batch_shape = state.shape[1:3]
        if unflatten:
            state = state.flatten(1, 2)
        # Expected shape (genomes, batch_size, seq_len, features)
        latent      = self._get(state, keys=keys)
        mean, std   = self.get_mean_std(latent, keys=keys)
        # distribution = self.dist(mean, std if options.get('normal', self.probabilistic) else None)
        # action      = distribution.sample()
        if options.get('normal', self.probabilistic):
            action = mean + (std * torch.randn_like(std))
        else:
            action = mean
        if self.distribution == 'discrete':
            action = torch.argmax(action, dim=-1)
        else:
            action = torch.sigmoid(action)
        # Expected shape (genomes, *batch_size, *features)
        if squeeze:
            action = action.squeeze(1)
        if unflatten:
            action = action.unflatten(1, batch_shape)
        return action


def fix(value: float, default: float = 1):
    if np.isinf(value) or np.isnan(value):
        return default
    else:
        return value


# GAME SETTINGS
SPAWN_WIDTH         = 200
GAP_OFFSET          = 30
GAP_SIZE            = (200, 200)
PIPE_Y_VELOCITY     = 3
FULL_STATES         = True
DELAY               = 0
USE_AE              = False
SEQUENTIAL          = True
DISCRETE            = True

# AutoEncoder properties
MAX_SEQ_LEN     = 6
A_INPUTS        = (5 + (2 if PIPE_Y_VELOCITY else 0) if FULL_STATES else 3)
A_OUTPUTS       = 4
A_OFFSET        = 0
DIM_SIZE        = 32
KERNEL_SIZE     = 3
STRIDE          = 1
S_LAYERS        = 0
T_LAYERS        = 2
F_LAYERS        = 0
A_HEADS         = 1
A_KV_HEADS      = None
A_DIFFERENTIAL  = False
BIAS            = False
PROBABILISTIC   = False
SAVE_NAME = f"ae_ml{MAX_SEQ_LEN}-i{A_INPUTS}-o{A_OUTPUTS}-d{DIM_SIZE}-k{KERNEL_SIZE}-s{STRIDE}-"\
            f"sl{S_LAYERS}-tl{T_LAYERS}-fl{F_LAYERS}-h{A_HEADS}-kv{A_KV_HEADS}-"\
            f"diff{A_DIFFERENTIAL}-b{BIAS}-prob{PROBABILISTIC}-off{A_OFFSET}"


AUTOENCODER = AutoEncoder(
    MAX_SEQ_LEN, A_INPUTS, DIM_SIZE, KERNEL_SIZE, S_LAYERS, T_LAYERS, F_LAYERS, A_HEADS, A_KV_HEADS,
    A_DIFFERENTIAL, BIAS, DEVICE, DTYPE,
    outputs=A_OUTPUTS, stride=STRIDE, probabilistic=PROBABILISTIC, out_bias=False,
)
if USE_AE:
    AUTOENCODER.load(SAVE_NAME, None, 'autoencoders', 'flappy-bird\\test', True)
    print(AUTOENCODER)
AUTOENCODER.eval()
AUTOENCODER.requires_grad_(False)
AUTOENCODER.single_mode(True)


# Model properties
GENOMES             = 100
SEQ_LEN             = MAX_SEQ_LEN // 1
INPUTS              = A_OUTPUTS if USE_AE else A_INPUTS # SEQ_LEN // (STRIDE ** S_LAYERS) * A_OUTPUTS # 3 if not FULL_STATES else 5
OUTPUTS             = 1 if not DISCRETE else 2
EMBED_SIZE          = 16
COEFFICIENTS        = 1
LAYERS              = 1
HEADS               = 1
KV_HEADS            = None
FWD_EXP             = 2
DIFFERENTIAL        = False
SKIP_CONNECTION     = True
ENABLE_BIAS         = True
PROBABILISTIC       = False
CONSTANT            = 100
MEMORY_SIZE         = 5
GAMMA               = np.exp(np.log(0.01) / 128)
ALPHA               = fix(np.exp(np.log(3.00) / (MEMORY_SIZE - 1)), 1.0)
KAPPA               = 0.0 # fix(np.exp(np.log(0.10) / 4), 0.0)
ALPHA_ORDER         = 2
REW_NORM            = 0
TEST_ACTIVATION     = nn.SiLU()
CLIP_MIN            = -5
CLIP_MAX            = -0
DISTRIBUTION        = 'normal' if not DISCRETE else 'discrete'
MODEL_NUM           = 4

POL_REG         = 0.75
STD_REG         = 0.75

MODELS = [
    BaseModelSequential(
        SEQ_LEN, INPUTS, OUTPUTS, EMBED_SIZE, LAYERS, HEADS, KV_HEADS, DIFFERENTIAL, FWD_EXP,
        activation, PROBABILISTIC, ENABLE_BIAS, DEVICE, DTYPE,
        clip_min=CLIP_MIN, clip_max=CLIP_MAX, distribution=DISTRIBUTION,
        constant=CONSTANT, residual=SKIP_CONNECTION,
    )
    if SEQUENTIAL else
    BaseModel(
        INPUTS, OUTPUTS, EMBED_SIZE, LAYERS, COEFFICIENTS,
        activation, PROBABILISTIC, ENABLE_BIAS, DEVICE, DTYPE,
        clip_min=CLIP_MIN, clip_max=CLIP_MAX, distribution=DISTRIBUTION,
    )
    for activation in [TEST_ACTIVATION for _ in range(MODEL_NUM)]
]

GAME_TYPE = 0 if not FULL_STATES else (1 if PIPE_Y_VELOCITY == 0 else 2)
BASE_NAME: str = (
    f"GT{GAME_TYPE}_AE{int(USE_AE)}_SQ{int(SEQUENTIAL)}_DC{int(DISCRETE)}_T{MODEL_NUM}"
    f"SeqL{SEQ_LEN}_E{EMBED_SIZE}_L{LAYERS}_C{COEFFICIENTS}_A-{TEST_ACTIVATION.__class__.__name__}_"
    f"H{HEADS}_K{KV_HEADS}_F{FWD_EXP}_D{int(DIFFERENTIAL)}_R{int(SKIP_CONNECTION)}_"
    f"Con{CONSTANT}_B{int(ENABLE_BIAS)}_P{int(PROBABILISTIC)}"
)
FILE_NAME: str = f"FlappyBirdModel-{BASE_NAME}"
FILE_DIR: str | None = None
FILE_NO: int | None = None
INIT_GEN: int | None = None

RUNS = 1
GOAL = 20
STEPS = GOAL * 100 * RUNS
EPOCHS = 150
FILTER_DEAD = True

print(f"\ncreating config")
CONFIG_DIR = f"{os.path.curdir}/configs"
CONFIG_NAME = "flappy_bird-ae"
CONFIG = neat.Config(CONFIG_NAME, CONFIG_DIR)
CONFIG.load(verbose=2)
CONFIG.save(debug=2)

ENV: Game | None = None


def evaluate(population: neat.Population, **options):
    trainer: neat.rl.NEAT = options['trainer']
    split_mappings = population.get_mapping()
    if len(population.genera) == 1: split_mappings = (split_mappings,)
    split_sizes = [len(m) for m in split_mappings]
    cons_mapping = population.get_mapping(consolidated=True)
    trainer.update_mapping(cons_mapping)
    # BUFFER = torch.zeros(SEQ_LEN, population.pop_size, INPUTS).to(DEVICE, DTYPE)
    global INIT_GEN, FILE_NO
    if INIT_GEN is None:
        INIT_GEN = population.generation

    for genome in population.genomes.values():
        genome.fitness = 0

    terminate = False
    start = 0
    run_step = 0
    game_step = 0
    DEBUG_STEP = SEQ_LEN - 1
    for model in MODELS: model.eval()
    while not terminate:
        gts = clock.perf_counter()
        step = 0
        reverse_mappings = tuple([
            {index: key for key, index in m.items()}
            for m in split_mappings
        ])

        states = ENV.reset()[0]
        done = False
        while not done:
            with torch.no_grad():
                DEBUG_DATA = step == DEBUG_STEP and population.generation == INIT_GEN

                # Get Inputs ~ send bird location, top pipe location and bottom pipe location
                # and determine from network whether to jump or not
                if USE_AE:
                    # states = AUTOENCODER(states.to(DEVICE, DTYPE), single=False)[:, -1 - DELAY]
                    states = AUTOENCODER(states.to(DEVICE, DTYPE))
                if DEBUG_DATA:
                    print(f"\nobservations =>\n{states}\n\tshape = {states.shape}")

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
                                # Use code below to debug in case filtering raises an error
                                print(f"\nPopulation size {population.size}"
                                      f"\nReverse mapping \n{reverse_mappings}"
                                      f"\nIndex = {index}, birds_shape = {ENV.birds.dead.shape}")
                                raise KeyError()
                    for mi, indices in enumerate(split_indices):
                        if len(indices) == 0:
                            split_keys[mi] = list(split_mappings[mi].keys())
                            split_indices[mi] = list(split_mappings[mi].values())
                    if DEBUG_DATA:
                        print(f"keys =>\n{split_keys[0]}")
                        print(f"indices =>\n{split_indices[0]}")
                else:
                    pass

                # Get actions
                split_observations = torch.split(states, split_sizes, dim=0)
                if FILTER_DEAD:
                    split_observations = [obs[indices] for obs, indices in zip(split_observations, split_indices)]
                split_actions = [
                    m.get_policy(obs.unsqueeze(1), keys=None if not FILTER_DEAD else split_keys[mi]).squeeze(1)
                    for mi, (m, obs) in enumerate(zip(MODELS, split_observations))
                ]
                if DEBUG_DATA:
                    obs_debug = split_observations[0]
                    keys_debug = None
                    if FILTER_DEAD:
                        obs_debug = obs_debug[split_indices[0]]
                        keys_debug = split_keys[0]
                    MODELS[0].get_policy(obs_debug.unsqueeze(1), keys=keys_debug, verbose=True)
                    print(f"actions =>\n{split_actions[0]}\n\tshape = {split_actions[0].shape}")
                    # print(f"probs =>\n{probs}\n\tshape = {probs.shape}")

                # Pad dead bird actions
                if FILTER_DEAD:
                    padding = ENV.birds.bird_num - sum([act.shape[0] for act in split_actions])
                    if padding > 0:
                        def fill_up(tensor: Tensor, indices_list: list[int], total: int):
                            fill = tensor.clone()
                            tensor = torch.zeros(
                                total, *tensor.shape[1:], device=DEVICE, dtype=DTYPE if not DISCRETE else torch.long
                                )
                            tensor[indices_list] = fill
                            return tensor
                        split_actions = [
                            fill_up(act, indices, len(rm))
                            for act, indices, rm in zip(split_actions, split_indices, reverse_mappings)
                        ]
                actions = torch.cat(split_actions, dim=0)
                if DEBUG_DATA:
                    print(f"filled actions =>\n{actions}\n\tshape = {actions.shape}")
                    print(f"mapping sizes = {[len(m) for m in reverse_mappings]}")
                    print(f"filling shapes = {[t.shape for t in split_actions]}")
                    if DISCRETE:
                        print(f"actions count {dict([(idx, (actions == idx).sum().item()) for idx in range(2)])}")
                if DISCRETE:
                    # Environment does not have discrete option so actions sent be reverted to expected shape and type
                    raw_actions = (0 ** actions).float().unsqueeze(-1)
                    if DEBUG_DATA:
                        print(f"un-discretized actions =>\n{actions}\n\tshape = {actions.shape}")
                else:
                    raw_actions = actions
                calc_time = clock.perf_counter() - ts

                # Get rewards
                next_states, rewards, _, done, _ = ENV.step(raw_actions)
                if DEBUG_DATA:
                    print(f"rewards =>\n{rewards}\n\tshape = {rewards.shape}")
                    # v = MODEL.get_value(observations[:len(reverse_mapping0)].unsqueeze(1), ).squeeze(1)
                    # print(f"values =>\n{v}\n\tshape = {v.shape}")
                    # del v

                # Updated buffers
                terminate = trainer.update(states, actions, rewards, done, done)

                alive = round(ENV.birds.active_num)
                max_score = round(rewards.max().item(), 2)
                if alive > 0:
                    best_index = torch.argmax(ENV.birds.score).cpu().item()
                    best_key = None
                    for rmi, rm in enumerate(reverse_mappings):
                        if best_index in rm:
                            best_key = rm[best_index]
                            break
                        best_index -= len(split_mappings[rmi])
                print(f"\r{CM('Executing', Fore.GREEN)}: time_elapsed = {round(clock.perf_counter()-gts)}s, "
                      f"alive = {alive}, max_rew = {max_score}, best_key={best_key}, ct={calc_time:.2e}, sd={trainer.steps_done} "
                      f"bl={trainer.primary.max_size()}", end='')

                # Render display (disabled during training to prevent window from popping up)
                # Uncomment below to enable rendering every 10 generations
                # if population.generation % 10 == 0:
                #     env.render()
                if False and population.generation % 10 == 0:
                    ENV.render()

                states = next_states

            # break if score gets large enough
            if done or terminate:
                # pickle.dump(population.genomes, open(".\\best.pickle", "wb"))
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

        l_lim, u_lim = 10, ENV.floor.y * 1.05
        # print(f"u lim = {u_lim}, l lim = {l_lim}")
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
    # trainer.save('flappy_bird', replace=population.generation != INIT_GEN)


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
    parser = argparse.ArgumentParser(description='Run NEAT algorithm on Pong game')
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
                print(f"WIll load population checkpoint from file number '{FILE_NO}'")
                POPULATION.load_dict(None, FILE_NAME, FILE_DIR, FILE_NO)
            except ValueError:
                if args.load.lower() not in ['false', 'none', 'null', '']:
                    print(f"Error: --load argument must be a boolean, valid positive integer, null or empty; Got '{args.load}'")
                    exit(1)

    ENV = Game(
        POPULATION.size, goal=GOAL, seq_len=SEQ_LEN if SEQUENTIAL else None,
        height=800, width=800, full_state=FULL_STATES, pipe_y_velocity=PIPE_Y_VELOCITY,
        spawn_width=SPAWN_WIDTH, tick=None, gap_offset=GAP_OFFSET, gap_size=GAP_SIZE,
        delay=DELAY, type2count=GENOMES, type2offset=0, device=DEVICE, dtype=DTYPE,
        render_mode=args.rm
    )
    # print(f"Anti Count = {game.birds.}")

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
            schedulers=[
                # neat.optim.scheduler.RandomAnnealing(config, 1e-1, 1e+1, 3, ['weight_init_std', 'weight_mutate_power'], True),
                # neat.optim.scheduler.CosineAnnealing(config, 10, 0.1, 'weight_mutate_rate', True, True),
                # neat.optim.scheduler.CosineAnnealing(config, 10, 0.1, 'weight_replace_rate', True, True),
                # neat.optim.scheduler.CosineAnnealing(config, 15, 0.05, 'weight_add_prob', True, True),
                # neat.optim.scheduler.CosineAnnealing(config, 15, 0.05, 'weight_del_prob', True, True),
            ],
            device=DEVICE, dtype=DTYPE,
            log_sub_dir='flappy_bird/',
            log_name=f"{unix_to_datetime_file(clock.time())}_"
                     f"e{EMBED_SIZE}-c{COEFFICIENTS}-m{SEQ_LEN}-l{LAYERS}-b{int(ENABLE_BIAS)}-h{HEADS}-"
                     f"prob{int(PROBABILISTIC)}-"
                     f"g{round(GAMMA, 4)}-a{round(ALPHA, 4)}-ao{ALPHA_ORDER}-"
                     f"rn{REW_NORM}-p{round(POL_REG, 4)}-sm{1}-mem{MEMORY_SIZE}-"
                     f"delay{DELAY}-type{int(DISCRETE)}",
            gamma=GAMMA, alpha=ALPHA, kappa=KAPPA, order=ALPHA_ORDER, normalize=REW_NORM,
            rew_reg=1.0, pol_reg=POL_REG, std_reg=STD_REG, validate=True, segr_size=10,
            max_episodes=MEMORY_SIZE,
        )
        trainer.set_report_hook(genome_debug)

        print(f"starting evaluation: population={len(POPULATION.genomes)}")
        # trainer.load(name='flappy_bird', file_no=None)
        try:
            trainer.learn(evaluate, STEPS, EPOCHS, 1024, 0.1,
                          'binary' if not DISCRETE else 'discrete', True)
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
        POPULATION.size, goal=100, seq_len=SEQ_LEN if SEQUENTIAL else None,
        height=800, width=1200, full_state=FULL_STATES, pipe_y_velocity=PIPE_Y_VELOCITY * 1.85,
        spawn_width=SPAWN_WIDTH, tick=None, gap_offset=GAP_OFFSET, gap_size=GAP_SIZE,
        delay=DELAY, render_mode='human',
        type2count=last_genus_size, type2offset=50, device=DEVICE, dtype=DTYPE
    )
    split_mappings = POPULATION.get_mapping()
    if len(POPULATION.genera) == 1: split_mappings = (split_mappings,)
    split_sizes = [len(m) for m in split_mappings]
    cons_mapping = POPULATION.get_mapping(consolidated=True)
    for i in range(5):
        done = False
        step = 0
        states = env.reset()[0]
        ts = clock.perf_counter()
        while not done:
            # Get states
            if USE_AE:
                # states = AUTOENCODER(states.to(DEVICE, DTYPE), single=False)[:, [-1 - DELAY]]
                states = AUTOENCODER(states.to(DEVICE, DTYPE)).unsqueeze(1)
            else:
                states = states.unsqueeze(1)

            # Get actions
            with torch.no_grad():
                split_observations = torch.split(states, split_sizes, dim=0)
                split_actions = [
                    m.get_policy(split_observations[mi]).squeeze(1)
                    for mi, m in enumerate(MODELS)
                ]
            actions = torch.cat(split_actions, dim=0)
            if DISCRETE:
                actions = (0 ** actions).float().unsqueeze(-1)

            # Get rewards
            next_states, rewards, _, done, _ = env.step(actions)

            # Get next states
            states = next_states

            # Render
            env.render()

            print(f"\rTime elapsed: {clock.perf_counter() - ts:.2f}s, "
                  f"Alive = {env.birds.active_num} "
                  f"Score = {env.score} ",
                  end='')
        print(f" ")


if __name__ == '__main__':
    run()
