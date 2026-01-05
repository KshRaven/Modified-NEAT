
from game import Game
from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.nn.base import Model
from ModifiedNEAT.nn.modules.sub import Linear, Conv1d, Transpose, ResidualBlock, Sequential, GroupNorm, ConverBase, SequenceEncoding
# from ModifiedNEAT.nn.modules import Reformer
from ModifiedNEAT.util.datetime import unix_to_datetime_file
from ModifiedNEAT.util.qol import manage_params
from Models import AutoEncoder

import ModifiedNEAT as neat
import ModifiedNEAT.nn as mn
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from numba.typed import List, Dict
from numba.core.errors import NumbaPerformanceWarning
from typing import Union

import time as clock
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
torch.set_printoptions(threshold=10)

DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'
DTYPE  = torch.float32


class BaseModel(Model):
    def __init__(self, inputs: int, outputs: int, dim_size: int, layers: int, coefficients=1, activation=nn.SiLU(),
                 probabilistic=False, bias=True, device: torch.device = 'cpu', dtype: torch.device = torch.float32, **options):
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
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32, **options):
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

        # Build
        self.projection = mn.Sequential(*[
            mn.BufferEmbedding(inputs, dim_size, True, 'continuous', device, dtype),
            # mn.BufferEncoding(max_seq_len, dim_size, True, 'continuous', device, dtype),
        ])
        self.transformer = mn.TransformerBase(
            max_seq_len, dim_size, layers, heads, kv_heads, differential, True, bias, device, dtype,
            residual=True, normalize=True, epsilon=1e-9, fwd_exp=fwd_exp, constant=self.constant
        )
        self.pol_proj = mn.Sequential(*[
            mn.LayerNorm(dim_size, bias=True, device=device, dtype=dtype),
            activation,
            mn.Linear(dim_size, 2*(outputs ** (2 if self.distribution == 'discrete' else 1)), True, device, dtype)
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
SPAWN_WIDTH     = 200
GAP_OFFSET      = 30
GAP_SIZE        = (200, 200)
PIPE_Y_VELOCITY = 3
FULL_STATES     = True
DELAY           = 0
USE_AE          = False
SEQUENTIAL      = True

# AutoEncoder properties
MAX_SEQ_LEN     = 3
A_INPUTS        = (5 + (2 if PIPE_Y_VELOCITY else 0) if FULL_STATES else 3)
A_OUTPUTS       = 4
A_OFFSET        = 0
DIM_SIZE        = 32
KERNEL_SIZE     = 3
STRIDE          = 1
S_LAYERS        = 0
T_LAYERS        = 2
F_LAYERS        = 0
HEADS           = 1
KV_HEADS        = None
DIFFERENTIAL    = False
BIAS            = False
PROBABILISTIC   = True
SAVE_NAME = f"ae_ml{MAX_SEQ_LEN}-i{A_INPUTS}-o{A_OUTPUTS}-d{DIM_SIZE}-k{KERNEL_SIZE}-s{STRIDE}-"\
            f"sl{S_LAYERS}-tl{T_LAYERS}-fl{F_LAYERS}-h{HEADS}-kv{KV_HEADS}-"\
            f"diff{DIFFERENTIAL}-b{BIAS}-prob{PROBABILISTIC}-off{A_OFFSET}"


AUTOENCODER = AutoEncoder(
    MAX_SEQ_LEN, A_INPUTS, DIM_SIZE, KERNEL_SIZE, S_LAYERS, T_LAYERS, F_LAYERS, HEADS, KV_HEADS,
    DIFFERENTIAL, BIAS, DEVICE, DTYPE,
    outputs=A_OUTPUTS, stride=STRIDE, probabilistic=PROBABILISTIC, out_bias=False,
)
if USE_AE:
    AUTOENCODER.load(SAVE_NAME, None, 'autoencoders', 'flappy-bird\\test', True)
AUTOENCODER.eval()
AUTOENCODER.requires_grad_(False)
AUTOENCODER.single_mode(True)


# Model properties
GENOMES             = 100
SEQ_LEN             = MAX_SEQ_LEN // 1
INPUTS              = A_OUTPUTS if USE_AE else A_INPUTS # SEQ_LEN // (STRIDE ** S_LAYERS) * A_OUTPUTS # 3 if not FULL_STATES else 5
OUTPUTS             = 1
EMBED_SIZE          = 32
COEFFICIENTS        = 1
LAYERS              = 1
FWD_EXP             = 2
ENABLE_BIAS         = True
PROBABILISTIC       = False
CONSTANT            = 100
MEMORY_SIZE         = 5
GAMMA               = np.exp(np.log(0.10) / 128)
ALPHA               = fix(np.exp(np.log(2.50) / (MEMORY_SIZE - 1)), 1.0)
KAPPA               = 0.0 # fix(np.exp(np.log(0.10) / 4), 0.0)
ALPHA_ORDER         = 0
REW_NORM            = 3
LOSS_REG            = 0.
ACTIVATION          = nn.Tanh()
CLIP_MIN            = -5
CLIP_MAX            = -0
DISTRIBUTION        = 'normal'

if SEQUENTIAL:
    MODELS = [
        BaseModelSequential(
            SEQ_LEN, INPUTS, OUTPUTS, EMBED_SIZE, LAYERS, HEADS, KV_HEADS, DIFFERENTIAL, FWD_EXP,
            activation, PROBABILISTIC, ENABLE_BIAS, DEVICE, DTYPE,
            clip_min=CLIP_MIN, clip_max=CLIP_MAX, distribution=DISTRIBUTION, constant=CONSTANT,
        )
        for activation in [ACTIVATION, nn.ReLU(), nn.SiLU()]
    ]
else:
    MODELS = [
        BaseModel(
            INPUTS, OUTPUTS, EMBED_SIZE, LAYERS, COEFFICIENTS,
            activation, PROBABILISTIC, ENABLE_BIAS, DEVICE, DTYPE,
            clip_min=CLIP_MIN, clip_max=CLIP_MAX, distribution=DISTRIBUTION,
        )
        for activation in [ACTIVATION, nn.ReLU(), nn.SiLU()]
    ]

MODEL0, MODEL1, MODEL2 = MODELS

INIT_GEN: int = None

RUNS = 1
GOAL = 20
STEPS = GOAL * 100 * RUNS
EPOCHS = 150

print(f"creating config")
config = neat.Config('flappy_bird')

config.genome.init_type                 = 'normal'
config.genome.weight_init_mean          = 0.0
config.genome.weight_init_std           = 1.5
config.genome.weight_min_value          = -np.inf
config.genome.weight_max_value          = +np.inf
config.genome.weight_mutate_power       = 6e-1
config.genome.weight_mutate_rate        = 0.50
config.genome.weight_replace_rate       = 0.00
config.genome.weight_add_prob           = 0.00
config.genome.weight_del_prob           = 0.00
config.genome.single_structural_mutation = False
config.genome.param_epsilon             = 1e-9
config.reproduction.min_species_size    = GENOMES
config.reproduction.purge               = 1
config.reproduction.clone_threshold     = 0.05
config.reproduction.survival_threshold  = 0.20
config.reproduction.cross_threshold     = 0.00
config.reproduction.elitism             = 30
config.species.compatibility_threshold  = np.inf
config.stagnation.max_stagnation        = 1
config.stagnation.species_elitism       = 2
config.reproduction.darwin_multiplier   = 0.25
config.reproduction.cross_multiplier    = 0.25
config.reproduction.preserve_elite      = False
config.save()
config.load(2)


def evaluate(population: neat.Population, **options):
    trainer: neat.rl.NEAT = options['trainer']
    mapping0, mapping1, mapping2 = population.get_mapping()
    cons_mapping = population.get_mapping(consolidated=True)
    trainer.update_mapping(cons_mapping)
    # BUFFER = torch.zeros(SEQ_LEN, population.pop_size, INPUTS).to(DEVICE, DTYPE)
    global INIT_GEN
    if INIT_GEN is None:
        INIT_GEN = population.generation

    for genome in population.genomes.values():
        genome.fitness = 0

    terminate = False
    start = 0
    run_step = 0
    game_step = 0
    DEBUG_STEP = SEQ_LEN - 1
    MODEL0.train()
    while not terminate:
        env = Game(
            population.size, goal=GOAL, seq_len=SEQ_LEN if SEQUENTIAL else None,
            height=800, width=800, full_state=FULL_STATES, pipe_y_velocity=PIPE_Y_VELOCITY,
            spawn_width=SPAWN_WIDTH, tick=None, gap_offset=GAP_OFFSET, gap_size=GAP_SIZE,
            delay=DELAY,
            type2count=len(mapping2), type2offset=0, device=DEVICE, dtype=DTYPE
        )
        # print(f"Anti Count = {game.birds.}")

        gts = clock.perf_counter()
        step = 0
        reverse_mapping0 = {index: key for key, index in mapping0.items()}
        reverse_mapping1 = {index: key for key, index in mapping1.items()}
        reverse_mapping2 = {index: key for key, index in mapping2.items()}

        states = env.reset()[0]
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
                keys0, keys1, keys2 = [], [], []
                indices0, indices1, indices2 = [], [], []
                for index, dead in enumerate(env.birds.dead):
                    if not dead:
                        if index in reverse_mapping0:
                            keys0.append(reverse_mapping0[index])
                            indices0.append(index)
                        elif index-len(mapping0) in reverse_mapping1:
                            keys1.append(reverse_mapping1[index-len(mapping0)])
                            indices1.append(index-len(mapping0))
                        elif index-(len(mapping0)+len(mapping1)) in reverse_mapping2:
                            keys2.append(reverse_mapping2[index-(len(mapping0)+len(mapping1))])
                            indices2.append(index-(len(mapping0)+len(mapping1)))
                        else:
                            print(f"\nPopulation size {population.size}"
                                  f"\nReverse mapping \n{reverse_mapping0} \n{reverse_mapping1}"
                                  f"\nIndex = {index}, birds_shape = {env.birds.dead.shape}")
                            raise KeyError()
                if len(indices0) == 0:
                    keys0 = list(mapping0.keys())
                    indices0 = list(mapping0.values())
                if len(indices1) == 0:
                    keys1 = list(mapping1.keys())
                    indices1 = list(mapping1.values())
                if len(indices2) == 0:
                    keys2 = list(mapping2.keys())
                    indices2 = list(mapping2.values())
                if DEBUG_DATA:
                    print(f"keys =>\n{keys0}")
                    print(f"indices =>\n{indices0}")

                # Get actions
                observations0, observations1, observations2 = torch.split(
                    states, [len(mapping0), len(mapping1), len(mapping2)], dim=0
                )
                actions0 = MODEL0.get_policy(observations0[indices0].unsqueeze(1), keys=keys0)
                actions1 = MODEL1.get_policy(observations1[indices1].unsqueeze(1), keys=keys1)
                actions2 = MODEL2.get_policy(observations2[indices2].unsqueeze(1), keys=keys2)
                # shape(seq_len=1, genomes, features_out)
                actions0, actions1, actions2 = actions0.squeeze(1), actions1.squeeze(1), actions2.squeeze(1)
                if DEBUG_DATA:
                    MODEL0.get_policy(observations0[indices0].unsqueeze(1), keys=keys0, verbose=True)
                    print(f"actions =>\n{actions0}\n\tshape = {actions0.shape}")
                    # print(f"probs =>\n{probs}\n\tshape = {probs.shape}")

                # Pad dead bird actions
                if True:
                    padding = env.birds.bird_num - actions0.shape[0] - actions1.shape[0] - actions2.shape[0]
                    if padding > 0:
                        def fill_up(tensor: Tensor, indices: list[int], total: int):
                            fill = tensor.clone()
                            tensor = torch.zeros(
                                total, *tensor.shape[1:], device=DEVICE, dtype=DTYPE
                                )
                            tensor[indices] = fill
                            return tensor
                        actions0 = fill_up(actions0, indices0, len(reverse_mapping0))
                        actions1 = fill_up(actions1, indices1, len(reverse_mapping1))
                        actions2 = fill_up(actions2, indices2, len(reverse_mapping2))
                    actions = torch.cat([actions0, actions1, actions2[..., :OUTPUTS]], dim=0)
                if DEBUG_DATA:
                    print(f"filled actions =>\n{actions}\n\tshape = {actions.shape}")
                calc_time = clock.perf_counter() - ts

                # Get rewards
                next_states, rewards, _, done, _ = env.step(actions)
                if DEBUG_DATA:
                    print(f"rewards =>\n{rewards}\n\tshape = {rewards.shape}")
                    # v = MODEL.get_value(observations[:len(reverse_mapping0)].unsqueeze(1), ).squeeze(1)
                    # print(f"values =>\n{v}\n\tshape = {v.shape}")
                    # del v

                # Updated buffers
                terminate = trainer.update(states, actions, rewards, done, done)

                alive = round(env.birds.active_num)
                max_score = round(rewards.max().item(), 2)
                if alive > 0:
                    best_index = torch.argmax(env.birds.score).cpu().item()
                    if best_index in reverse_mapping0:
                        best_key = reverse_mapping0[best_index]
                    elif best_index in reverse_mapping1:
                        best_key = reverse_mapping1[best_index]
                    elif best_index in reverse_mapping2:
                        best_key = reverse_mapping2[best_index]
                    else:
                        # raise KeyError()
                        best_key = None
                print(f"\r{CM('Executing', Fore.GREEN)}: time_elapsed = {round(clock.perf_counter()-gts)}s, "
                      f"alive = {alive}, max_rew = {max_score}, best_key={best_key}, ct={calc_time:.2e}, sd={trainer.steps_done} "
                      f"bl={trainer.primary.max_size()}", end='')

                # Render display
                if population.generation % 10 == 0:
                    env.render()

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

        l_lim, u_lim = 10, env.floor.y * 1.05
        # print(f"u lim = {u_lim}, l lim = {l_lim}")
        for idx, (score, genome) in enumerate(zip(env.birds.get_reward(), population.genomes.values())):
            genome.fitness = score.item()
            y_position = env.birds.y[idx]
            died_beyond_limits = y_position <= l_lim or y_position >= u_lim
            if died_beyond_limits:
                population.to_delete.append(genome.key)

        game_step += 1
    print(f"\n")

    population.save_dict('flappy_bird', replace=population.generation != INIT_GEN)
    # trainer.save('flappy_bird', replace=population.generation != INIT_GEN)


def genome_debug(algorithm: neat.rl.NEAT):
    population = algorithm.population
    COUNT = 5

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
    # MODEL.single_mode(True)
    # Configuration

    # Create the population, which is the top-level object for a NEAT run.
    print(f"creating population")
    population = neat.Population(GENOMES, MODEL0, config, init_reporter=True)
    population1 = neat.Population(GENOMES, MODEL1, config, init_reporter=True)
    population2 = neat.Population(GENOMES, MODEL2, config, init_reporter=True)
    population.absorb_population(population1)
    population.absorb_population(population2)
    # population.load_dict(name='flappy_bird', file_no=None)

    TRAIN = False
    if TRAIN:
        trainer = neat.rl.NEAT(
            population,
            schedulers=[
                # neat.optim.scheduler.RandomAnnealing(config, 1e-1, 1e+1, 3, ['weight_init_std', 'weight_mutate_power'], True),
                # neat.optim.scheduler.CosineAnnealing(config, 10, 0.1, 'weight_mutate_rate', True, True),
                # neat.optim.scheduler.CosineAnnealing(config, 10, 0.1, 'weight_replace_rate', True, True),
                # neat.optim.scheduler.CosineAnnealing(config, 15, 0.05, 'weight_add_prob', True, True),
                # neat.optim.scheduler.CosineAnnealing(config, 15, 0.05, 'weight_del_prob', True, True),
            ],
            device=DEVICE, dtype=DTYPE,
            log_sub_dir='flappy_bird\\',
            log_name=f"{unix_to_datetime_file(clock.time())}_"
                     f"e{EMBED_SIZE}-c{COEFFICIENTS}-m{SEQ_LEN}-l{LAYERS}-b{int(ENABLE_BIAS)}-h{HEADS}-"
                     f"prob{int(PROBABILISTIC)}-"
                     f"g{round(GAMMA, 4)}-a{round(ALPHA, 4)}-ao{ALPHA_ORDER}-"
                     f"rn{REW_NORM}-p{round(LOSS_REG, 4)}-sm{1}-mem{MEMORY_SIZE}-"
                     f"delay{DELAY}",
            gamma=GAMMA, alpha=ALPHA, kappa=KAPPA, order=ALPHA_ORDER, normalize=REW_NORM,
            rew_reg=1.0, pol_reg=0.0, validate=True, groups=None,
            max_episodes=MEMORY_SIZE,
        )
        trainer.set_report_hook(genome_debug)

        print(f"starting evaluation: population={len(population.genomes)}")
        # trainer.load(name='flappy_bird', file_no=None)
        try:
            trainer.learn(evaluate, STEPS, EPOCHS, 1024, 0.1, 'binary', 3)
        except KeyboardInterrupt:
            pass
    else:
        population.load_dict(name='flappy_bird', file_no=None)

    env = Game(
        population.size, goal=50, seq_len=SEQ_LEN if SEQUENTIAL else None,
        height=800, width=800, full_state=FULL_STATES, pipe_y_velocity=PIPE_Y_VELOCITY,
        spawn_width=SPAWN_WIDTH, tick=None, gap_offset=GAP_OFFSET, gap_size=GAP_SIZE,
        delay=DELAY,
        type2count=GENOMES, type2offset=0, device=DEVICE, dtype=DTYPE
    )
    mapping0, mapping1, mapping2 = population.get_mapping()
    cons_mapping = population.get_mapping(consolidated=True)
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
                observations0, observations1, observations2 = torch.split(
                    states, [len(mapping0), len(mapping1), len(mapping2)], dim=0
                )
                actions0 = MODEL0.get_policy(observations0)
                actions1 = MODEL1.get_policy(observations1)
                actions2 = MODEL2.get_policy(observations2)
                actions0, actions1, actions2 = \
                    actions0.squeeze(1), actions1.squeeze(1), actions2.squeeze(1)
            actions = torch.cat([actions0, actions1, actions2[..., :OUTPUTS]], dim=0)

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

    # while True:
    #     evaluate(population, trainer=trainer)

    # for p in population.get(winner):
    #     print(p)

    # show final stats
    # print('\nBest genome:\n{!s}'.format(winner.key))


if __name__ == '__main__':
    print(MODEL0)
    run()
