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
from ModifiedNEAT.optim import scheduler
from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.util.datetime import unix_to_datetime_file
from ModifiedNEAT.util.qol import manage_params
from torch import Tensor, device as TDEVICE, dtype as TDTYPE
from torch.nn import Module
from typing import Union
from numba.core.errors import NumbaPerformanceWarning

EXMP_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Example path = {EXMP_DIR}")
if EXMP_DIR not in sys.path:
    sys.path.insert(0, EXMP_DIR)

from game import Game

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
torch.set_printoptions(threshold=10)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float32
neat.set_device(DEVICE)
print(f"Using PyTorch device: '{DEVICE}'. NEAT device: '{neat.get_device()}'")
PROJECT_DIR = os.path.dirname(os.path.dirname(EXMP_DIR))
neat.util.storage.set_storage_location(f"{PROJECT_DIR}/storage/")
print(f"Example path = {neat.util.storage.STORAGE_DIR}")


class BaseModelSeq(mn.Model):
    def __init__(
        self, max_seq_len: int, inputs: int, outputs: int, dim_size: int, 
        layers: int, heads: int, kv_heads: int | None = None,
        activation: Module = None, probabilistic=False, bias=True, 
        device: TDEVICE = 'cpu', dtype: TDTYPE = None, **options
    ):
        super().__init__()
        # Attributes
        self.max_seq_len        = max_seq_len
        self.inputs             = inputs
        self.outputs            = outputs
        self.dim_size           = dim_size
        self.layers             = layers
        self.distribution       = options.get('distribution', 'normal')
        self.probabilistic      = probabilistic
        self.normalize          = options.get('normalize', True)
        self.clip_min           = options.get('clip_min', -2)
        self.clip_max           = options.get('clip_max', -0)
        self.clip_range         = self.clip_max - self.clip_min
        self.lower              = 10 ** self.clip_min
        self.upper              = 10 ** self.clip_max
        self.heads              = heads
        self.kv_heads           = kv_heads
        self.use_swiglu: bool   = options.get('use_swiglu', False)
        self.attn_bias: bool    = options.get('attn_bias', bias)
        self.fwd_exp: int       = options.get('fwd_exp', 2)
        self.constant: int      = options.get('constant', 1000)
        self.differential: bool | int = options.get('differential', False)
        self.epsilon            = options.get('epsilon', 1e-6)
        
        if activation is None:
            activation = nn.SiLU()
        if dtype is None:
            dtype = torch.float32
        
        # Build
        feed_fwd = None if self.use_swiglu else mn.Sequential(
            # mn.LayerNorm(dim_size, bias=True, device=device, dtype=dtype),
            # mn.RMSNorm(dim_size, device=device, dtype=dtype),
            mn.Linear(dim_size, dim_size, bias, device, dtype),
            activation, 
            mn.Linear(dim_size, dim_size, self.attn_bias, device, dtype),
        )
        transformer = mn.TransformerBase(
            max_seq_len, dim_size, layers, heads, kv_heads,
            differential=self.differential, causal_mask=True, 
            bias=self.attn_bias, ff=feed_fwd, 
            auto_single=True, fwd_exp=self.fwd_exp,
            residual=True, normalize=True, constant=self.constant,
            device=device, dtype=dtype
        )
        self.lat_proj = mn.Sequential(
            mn.Linear(inputs, dim_size, True, device, dtype),
            transformer,
        )
        self.pol_proj = mn.Sequential(*[
            # mn.LayerNorm(dim_size, bias=True, device=device, dtype=dtype),
            # mn.RMSNorm(dim_size, device=device, dtype=dtype),
            mn.Linear(dim_size, dim_size, bias, device, dtype),
            nn.Tanh(),
            mn.Linear(dim_size, 2*outputs, True, device, dtype),
        ])
       
    @property 
    def transformer(self) -> mn.TransformerBase:
        return self.lat_proj.modules_list[1]

    def extra_repr(self) -> str:
        return f"probabilistic={self.probabilistic}, distro='{self.distribution}'"

    def forward(self, state: Tensor, keys: Union[int, list[int]] = None, **kwargs):
        return self.get_policy(state, keys=keys, **kwargs)

    def _get_mean_std(self, latent: Tensor, keys: Union[int, list[int]] = None):
        mean_std   = self.pol_proj(latent, keys=keys)
        # mean, log_std   = torch.chunk(mean_std, 2, -1)
        # if self.normalize:
        #     mean = torch.tanh(mean) * max(1, self.upper)
        #     std = torch.pow(10, self.clip_min + self.clip_range * torch.sigmoid(log_std))
        # else:
        #     std = torch.exp(log_std)
        mean, _std = torch.chunk(mean_std, 2, -1)
        std = torch.abs(_std)
        return mean, std

    def _get_std(self, latent: Tensor, keys: Union[int, list[int]] = None, raw: bool = False):
        mean_std   = self.pol_proj(latent, keys=keys)
        _, log_std = torch.chunk(mean_std, 2, -1)
        if self.normalize:
            std = torch.pow(10, self.clip_min + self.clip_range * torch.sigmoid(log_std))
            if raw: std = torch.log10(std)
        else:
            if raw: std = log_std
            else: std = torch.exp(log_std)
        return std

    def get_action(self, state: Tensor, keys: int | list[int] = None, **options) -> tuple[Tensor, Tensor]:
        assert state.ndim == 4, f"Latent shape should be (genomes, batch/processes, seq_len, features) but got {state.shape}"
        latent      = self.lat_proj(state, keys=keys, verbose=options.get('verbose', False))
        assert latent.ndim == 4, f"Latent shape should be (genomes, batch/processes, seq_len=1, features) but got {latent.shape}"
        latent      = latent.squeeze(-2) # Since single token mode should be enabled
        mean, std   = self._get_mean_std(latent, keys=keys)
        dist: torch.distributions.Distribution = self.dist(mean, std)
        action = dist.sample()
        # if self.distribution != "discrete": action = torch.tanh(action) # NOTE: Breaks PPO loss with infinities
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, state: Tensor, action: Tensor, keys: int | list[int] = None) -> tuple[Tensor, Tensor | None]:
        latent    = self.lat_proj(state, keys=keys, verbose=False)
        latent    = latent.squeeze(-2)
        mean, std = self._get_mean_std(latent, keys=keys)
        dist: torch.distributions.Distribution = self.dist(mean, std)
        # if self.distribution != "discrete": action = torch.atanh(action * (1 - self.epsilon))
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy

    def get_policy(self, state: Tensor, keys: Union[int, list[int]] = None, **options) -> Tensor:
        assert state.ndim == 4, f"Latent shape should be (genomes, batch/processes, seq_len, features) but got {state.shape}"
        latent      = self.lat_proj(state, keys=keys, verbose=options.get('verbose', False))
        assert latent.ndim == 4, f"Latent shape should be (genomes, batch/processes, seq_len=1, features) but got {latent.shape}"
        latent      = latent.squeeze(-2) # Since single token mode should be enabled
        mean, std   = self._get_mean_std(latent, keys=keys)
        action = mean
        if options.get('normal', self.probabilistic):
            action = action + (std * torch.randn_like(std))
        if self.distribution == 'discrete':
            action = torch.argmax(action, dim=-1)
        else:
            action = torch.tanh(action)
        return action
    
    def get_mean(self, state: Tensor, keys: Union[int, list[int], None] = None):
        assert state.ndim == 4 # state(1, batch_size, seq_len, features)
        latent = self.lat_proj(state, keys=keys)
        assert latent.ndim == 4
        latent = latent.squeeze(-2) # Remove seq_len dim
        mean   = self._get_mean_std(latent, keys=keys)[0] # , raw=True)[0]
        return mean
    
    def get_std(self, state: Tensor, keys: Union[int, list[int], None] = None):
        assert state.ndim == 4 # state(1, batch_size, seq_len, features)
        latent = self.lat_proj(state, keys=keys)
        assert latent.ndim == 4
        latent = latent.squeeze(-2) # Remove seq_len dim
        std    = self._get_std(latent, keys=keys, raw=True)
        return std
    
    def enable_cache(self):
        self.transformer.force_cache(True)
    
    def disable_cache(self):
        self.transformer.force_cache(False)
    
    def reset_cache(self):
        self.transformer.empty_cache()


def fix(value: float, default: float = 1):
    if np.isinf(value) or np.isnan(value):
        return default
    else:
        return value


# GAME SETTINGS
SPAWN_WIDTH         = 200
GAP_OFFSET          = 30
GAP_SIZE            = (210, 210)
PIPE_Y_VELOCITY     = 3
FULL_STATES         = True
DELAY               = 0
DISCRETE            = False

# Model properties
GENOMES             = 100
MAX_SEQ_LEN         = 8
SEQ_LEN             = MAX_SEQ_LEN // 1
SEQ_LEN_TRAIN       = min(max(1, 1), SEQ_LEN)
SEQ_LEN_EVAL        = min(max(SEQ_LEN, 1), SEQ_LEN)
INPUTS              = (5 + (2 if PIPE_Y_VELOCITY else 0) if FULL_STATES else 3)
OUTPUTS             = 1 if not DISCRETE else 2
EMBED_SIZE          = 32
LAYERS              = 1
HEADS               = 1
KV_HEADS            = None
FWD_EXP             = 1
DIFFERENTIAL        = False
BIAS                = True
PROBABILISTIC       = False
PROJ_NORM           = False
CONSTANT            = 1000
TEST_ACTIVATION     = nn.SiLU()
SWIGLU              = True
CLIP_MIN            = -4
CLIP_MAX            = +0
DISTRIBUTION        = 'normal' if not DISCRETE else 'discrete'
MODEL_NUM           = 3

# Trainer properties
MEMORY_SIZE         = 5
GAMMA               = fix(np.exp(np.log(0.01) / 128), 0.0)
KAPPA               = 0.0 # fix(np.exp(np.log(0.01) / 128), 0.0)
ALPHA               = fix(np.exp(np.log(3.0) / (MEMORY_SIZE - 1)), 1.0)
ALPHA_ORDER         = 0
REW_NORM            = 3
REW_REG             = 1.0
LOSS_REG            = 0.67
POL_REG             = 0.67
ENT_REG             = 0.10
DIV_REG             = 0.01
CPY_REG             = 0 # 0.75

# Training 
GOAL        = 20
STEPS       = GOAL * 100
EPOCHS      = 500
FILTER_DEAD = False # TODO: Might remove since it might bring about truncated tensors with dequed kv cache, or make optional

MODELS = [
    BaseModelSeq(
        MAX_SEQ_LEN, INPUTS, OUTPUTS, EMBED_SIZE, LAYERS, HEADS, KV_HEADS,
        activation, PROBABILISTIC, BIAS, DEVICE, DTYPE,
        clip_min=CLIP_MIN, clip_max=CLIP_MAX, distribution=DISTRIBUTION,
        constant=CONSTANT, differential=DIFFERENTIAL, attn_bias=BIAS, 
        use_swiglu=SWIGLU, fwd_exp=FWD_EXP, normalize=PROJ_NORM,
    )
    for activation in [TEST_ACTIVATION for _ in range(MODEL_NUM)]
]

GAME_TYPE = 0 if not FULL_STATES else (1 if PIPE_Y_VELOCITY == 0 else 2)
BASE_NAME: str = (
    f"GT{GAME_TYPE}_DC{int(DISCRETE)}_T{MODEL_NUM}_"
    f"MSL{MAX_SEQ_LEN}_E{EMBED_SIZE}_L{LAYERS}_H{HEADS}_K{KV_HEADS}_"
    f"A-{TEST_ACTIVATION.__class__.__name__}_B{int(BIAS)}_"
    f"F{FWD_EXP}_D{int(DIFFERENTIAL)}_Con{CONSTANT}_"
    f"S{int(SWIGLU)}{f'_R-{CLIP_MIN}~{CLIP_MAX}' if PROBABILISTIC else ''}"
)
FILE_NAME: str = f"FlappyBirdModelSeq-{BASE_NAME}"
FILE_DIR: str | None = "flappy-bird/sequential"
FILE_NO: int | None = None
INIT_GEN: int | None = None

print(f"\ncreating config")
CONFIG = neat.Config('sequential', '.config')

CONFIG.genome.init_type                     = 'normal'
CONFIG.genome.weight_init_mean              = 0.0
CONFIG.genome.weight_init_std               = 1.0
CONFIG.genome.weight_min_value              = -np.inf
CONFIG.genome.weight_max_value              = +np.inf
CONFIG.genome.weight_mutate_power           = 0.6
CONFIG.genome.weight_mutate_rate            = 0.50
CONFIG.genome.weight_replace_rate           = 0.0
CONFIG.genome.weight_add_prob               = 0.0
CONFIG.genome.weight_del_prob               = 0.0
CONFIG.genome.single_structural_mutation    = True
CONFIG.reproduction.min_species_size        = GENOMES
CONFIG.reproduction.purge                   = 1
CONFIG.reproduction.elitism                 = 0.33
CONFIG.reproduction.clone_threshold         = 0.50
CONFIG.reproduction.survival_threshold      = 0.20
CONFIG.reproduction.cross_threshold         = 0.00
CONFIG.species.compatibility_threshold      = np.inf
CONFIG.stagnation.max_stagnation            = 1
CONFIG.stagnation.species_elitism           = 2
CONFIG.reproduction.darwin_multiplier       = 0.33
CONFIG.reproduction.cross_multiplier        = 0.75
CONFIG.reproduction.preserve_elite          = False
CONFIG.save()
CONFIG.load(2)

ENV: Game | None = None
SELECTOR = torch.arange(SEQ_LEN, device=DEVICE, dtype=DTYPE).long()


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
    for model in MODELS: 
        model.eval()
        model.enable_cache()
    while not terminate:
        gts = clock.perf_counter()
        step = 0
        reverse_mappings = tuple([
            {index: key for key, index in m.items()}
            for m in split_mappings
        ])

        states = ENV.reset()[0] # TODO: Implement keys paramter
        for model in MODELS: model.reset_cache()
        step = 0
        DEBUG_STEP = 10
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
                    
                # Get actions with forced cache enabled during training
                # States could be instances or sequences of shape (genomes, *seq_len, inputs)
                split_states = torch.split(states.unsqueeze(1).select(-2, -1).unsqueeze(-2), split_sizes, dim=0)
                # split_states = torch.split(states.unsqueeze(1).index_select(-2, SELECTOR[-3:]), split_sizes, dim=0)
                split_actions, split_log_probs = zip(*[
                    m.get_action(obs, keys=None, verbose=2 if DEBUG and i == 0 else False)
                    for i, (m, obs) in enumerate(zip(MODELS, split_states))
                ])
                # Actions shape (genomes, inputs)
                actions = torch.cat(split_actions, dim=0).squeeze(1) # Group and remove batch
                log_probs = torch.cat(split_log_probs, dim=0).squeeze(1)
                if DEBUG:
                    print(f"\nsplit_states =>\n{split_states[0]}\n\tshape = {split_states[0].shape}")
                    print(f"\nsplit_actions =>\n{split_actions[0]}\n\tshape = {split_actions[0].shape}")
                    print(f"\nactions =>\n{actions}\n\tshape = {actions.shape}")

                # Get rewards
                next_states, rewards, _, done, _ = ENV.step(actions)
                if DEBUG:
                    print(f"\nrewards =>\n{rewards}\n\tshape = {rewards.shape}")

                # Updated buffers
                terminate = trainer.update(states, actions, rewards, log_probs, done, done)

                # CMDLine rendering
                alive = round(ENV.birds.active_num)
                max_score = round(rewards.max().item(), 2)
                print(
                    f"\r{CM('Executing', Fore.GREEN)}: "
                    f"time_elapsed = {round(clock.perf_counter()-gts)}s, "
                    f"alive = {alive}, max_rew = {max_score}, "
                    f"sd={trainer.steps_done}, "
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

        l_lim, u_lim = 10, ENV.floor.y * 1.05
        for idx, (score, genome) in enumerate(zip(ENV.birds.get_reward(), population.genomes.values())):
            genome.fitness = score.item()
            y_position = ENV.birds.y[idx]
            died_beyond_limits = y_position <= l_lim or y_position >= u_lim
            if died_beyond_limits:
                population.to_delete.append(genome.key)

        game_step += 1
    print(f"\n")
    for model in MODELS:
        model.train()
        model.disable_cache()
    
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
    parser = argparse.ArgumentParser(description='Run NEAT algorithm on Flappy Bird game (Sequential)')
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
            POPULATION.load(FILE_NAME, FILE_DIR, FILE_NO)
        else:
            try:
                FILE_NO = int(args.load)
                print(f"Will load population checkpoint from file number '{FILE_NO}'")
                POPULATION.load(FILE_NAME, FILE_DIR, FILE_NO)
            except ValueError:
                if args.load.lower() not in ['false', 'none', 'null', '']:
                    print(f"Error: --load argument must be a boolean, valid positive integer, null or empty; Got '{args.load}'")
                    exit(1)

    ENV = Game(
        POPULATION.size, goal=GOAL, seq_len=SEQ_LEN,
        height=800, width=800, full_state=FULL_STATES, pipe_y_velocity=PIPE_Y_VELOCITY,
        spawn_width=SPAWN_WIDTH, tick=None, gap_offset=GAP_OFFSET, gap_size=GAP_SIZE,
        delay=DELAY, type2count=GENOMES, type2offset=0, device=DEVICE, dtype=DTYPE,
        render_mode=args.rm,
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
        # trainer = neat.rl.NEAT(
        trainer = neat.rl.PPO(
            POPULATION,
            # ghosts={g: 0 for g in POPULATION.genera[1:]},
            schedulers=[
                scheduler.CosineAnnealing(CONFIG, 20, 0.5, 'weight_mutate_power', False, True),
                scheduler.BinaryAnnealing(CONFIG, 20, 0.10, 'cross_threshold'),
            ],
            device=DEVICE, dtype=DTYPE,
            log_sub_dir='flappy_bird/',
            log_name=f"{unix_to_datetime_file(clock.time())}-sequential_"
                     f"seq{SEQ_LEN}_e{EMBED_SIZE}-l{LAYERS}_h{HEADS}-kv{KV_HEADS}-b{int(BIAS)}"
                     f"g{round(GAMMA, 4)}-k{round(KAPPA, 4)}-a{round(ALPHA, 4)}-ao{ALPHA_ORDER}-"
                     f"rn{REW_NORM}-p{round(POL_REG, 4)}-s{round(ENT_REG, 4)}-d{round(DIV_REG, 4)}-c{round(CPY_REG, 4)}-"
                     f"mem{MEMORY_SIZE}-delay{DELAY}-type{int(DISCRETE)}-gt{GAME_TYPE}",
            gamma=GAMMA, alpha=ALPHA, kappa=KAPPA, order=ALPHA_ORDER, normalize=REW_NORM,
            rew_reg=REW_REG, loss_reg=LOSS_REG, pol_reg=POL_REG, ent_reg=ENT_REG, div_reg=DIV_REG, cpy_reg=CPY_REG,
            validate=True, segr_size=None, use_entropy=True,
            max_episodes=MEMORY_SIZE, max_steps=None,
        )
        print("Regularizations:")
        for l, v in vars(trainer).items(): 
            if "_reg" in l: print(f"\t{l} => {v:.4f}")
        # print(f"Ghost mapping = {trainer.ghosts}")
        trainer.set_report_hook(genome_debug)

        print(f"starting evaluation: population={len(POPULATION.genomes)}")
        try:
            trainer.learn(
                evaluate, STEPS, EPOCHS, 1380,
                accuracy_error=0.1,
                accuracy_type='binary' if not DISCRETE else 'discrete', 
                verbose=True,
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
        POPULATION.size, goal=100, seq_len=SEQ_LEN,
        height=800, width=1200, full_state=FULL_STATES, pipe_y_velocity=PIPE_Y_VELOCITY * 1.50, velocity=7,
        spawn_width=SPAWN_WIDTH, tick=None, gap_offset=GAP_OFFSET, gap_size=tuple(int(s * 0.95) for s in GAP_SIZE),
        delay=DELAY, render_mode='human',
        type2count=last_genus_size, type2offset=0, device=DEVICE, dtype=DTYPE
    )
    
    split_mappings = POPULATION.get_mapping()
    if len(POPULATION.genera) == 1: split_mappings = (split_mappings,)
    split_sizes = [len(m) for m in split_mappings]
    cons_mapping = POPULATION.get_mapping(consolidated=True)
    
    # Disable cache for post-training evaluation to use full sequences
    for model in MODELS: model.eval(); model.disable_cache()
    
    for i in range(5):
        done = False
        step = 0
        states = env.reset()[0]
        for model in MODELS: model.reset_cache()
        while not done:            
            # states[..., 6:] = 0.0
            
            # Get actions
            with torch.no_grad():
                # split_states = torch.split(states.unsqueeze(1).index_select(-2, SELECTOR[-SEQ_LEN_EVAL:]), split_sizes, dim=0)
                split_states = torch.split(states.unsqueeze(1), split_sizes, dim=0)
                split_actions = [
                    m.get_policy(obs)
                    for m, obs in zip(MODELS, split_states)
                ]
            actions = torch.cat(split_actions, dim=0).squeeze(1)

            # Get rewards
            next_states, rewards, _, done, _ = env.step(actions)
            states = next_states
            
            env.render()
            step += 1
            
    for model in MODELS: model.train()


if __name__ == '__main__':
    run()
