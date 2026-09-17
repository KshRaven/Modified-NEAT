"""
Maze-Runner (Game) sequential PPO runner.

Layout follows the Car-Racer `sequential.py` example (env setup / evaluate / run
structure) but trains with the evolutionary PPO trainer (`ppo.py`, `neat.rl.PPO`)
the way the Flappy-Bird `sequential_ppo.py` example does, so the model here
implements the methods PPO actually calls: `get_action` (rollout, returns
action + log_prob) and `evaluate_action` (re-scored during the learn step).

Two model variants are provided, selected off `config.agent.return_grid`
(the flag lives only on `AgentConfig`, per `env.py`'s docstring):

    - BaseModelSeq  : discrete_states / vector observations -> small causal
                      transformer latent + policy head. Same shape/contract
                      as the Car-Racer / Flappy-Bird sequential models.
    - ConvModelSeq  : return_grid=True -> the observation is a stack of
                      per-cell grid planes (genomes, C, H, W). Rather than a
                      fixed torch CNN, this mirrors `ConvNetwork` from
                      `binary_refactored.ipynb`: the grid is spatially
                      flattened to a (C, H*W) sequence and pushed through the
                      same evolvable input-projection + `layers`-deep body of
                      `mn.Conv1d` + SiLU blocks. An optional small transformer
                      (`mn.TransformerBase`) can be inserted right after that
                      conv stack, operating on the still-spatially-flattened
                      -but-not-fully-flattened (dim_size, seq_len) tensor --
                      i.e. treating each of the `seq_len` grid cells as one
                      token of width `dim_size` -- before the tensor is fully
                      flattened into the final fc layers (here, the same
                      shared `pol_proj` policy head BaseModelSeq uses). This
                      hook is off by default (`use_transformer=False`) since
                      the ticket calls it a later addition; flip it on with
                      `--use_transformer true`.

NOTE ON ASSUMPTIONS: `agent.py` / `grid.py` (which define the exact grid-plane
layout for `return_grid` observations and `Direction`'s int encoding) were not
available at the time of writing. `ConvModelSeq` assumes the grid observation
has shape (channels, rows, cols); adjust `GRID_CHANNELS` below if `agent.py`
stacks a different number of planes (e.g. walls/traps/goal/self/others). Conv1d
is used (rather than Conv2d) specifically to match `ConvNetwork`'s evolvable
building blocks as they exist in `binary_refactored.ipynb` -- this trades away
2D spatial locality for consistency with that reference architecture.
Everything else (buffers, PPO update signature, evaluate-loop shape handling)
mirrors the two reference files exactly.
"""

import ModifiedNEAT as neat
import ModifiedNEAT.nn as mn
import torch
import torch.nn as nn
import os, sys
import time as clock
import numpy as np
import warnings
import argparse
import math

from ModifiedNEAT.optim import scheduler
from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.util.datetime import unix_to_datetime_file
from torch import Tensor, device as TDEVICE, dtype as TDTYPE
from torch.nn import Module
from typing import Union
from numba.core.errors import NumbaPerformanceWarning

EXMP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXMP_DIR not in sys.path:
    sys.path.insert(0, EXMP_DIR)

from game import Game, EnvironmentConfig

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
torch.set_printoptions(threshold=10)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float32
neat.set_device(DEVICE)
print(f"Using PyTorch device: '{DEVICE}'. NEAT device: '{neat.get_device()}'")
PROJECT_DIR = os.path.dirname(os.path.dirname(EXMP_DIR))
neat.util.storage.set_storage_location(f"{PROJECT_DIR}/storage/")


def fix(value: float, default: float = 1):
    if np.isinf(value) or np.isnan(value):
        return default
    return value


# ----------------------------------------------------------------------------
# Shared policy-head / distribution / cache mixin used by both model variants
# ----------------------------------------------------------------------------
class _PolicyMixin:
    """
    Everything downstream of the "latent" (post-encoder) representation: the
    policy head, sampling, log-prob evaluation, and cache passthroughs that
    `ModifiedNEAT.rl.PPO` and the evaluate loop require. Both BaseModelSeq
    (transformer encoder) and ConvModelSeq (CNN encoder) mix this in so PPO
    only ever needs `get_action` / `evaluate_action` / `get_policy`.
    """

    def extra_repr(self) -> str:
        return f"probabilistic={self.probabilistic}, distro='{self.distribution}'"

    # def dist(self, mean: Tensor, std: Tensor) -> torch.distributions.Distribution:
    #     if self.distribution == 'discrete':
    #         return torch.distributions.Categorical(logits=mean)
    #     return torch.distributions.Normal(mean, std.clamp_min(1e-6))

    def _get_mean_std(self, latent: Tensor, keys: Union[int, list[int]] = None):
        mean_std = self.pol_proj(latent, keys=keys)
        mean, _std = torch.chunk(mean_std, 2, -1)
        std = _std.abs() + self.epsilon
        return mean, std

    def forward(self, state: Tensor, keys: Union[int, list[int]] = None, **kwargs):
        return self.get_policy(state, keys=keys, **kwargs)

    def get_action(self, state: Tensor, keys: int | list[int] = None, **options) -> tuple[Tensor, Tensor]:
        latent = self._encode(state, keys=keys, verbose=options.get('verbose', False))
        mean, std = self._get_mean_std(latent, keys=keys)
        dist = self.dist(mean, std)
        if not self.probabilistic:
            action  = mean
            if self.distribution == "discrete": action = action.argmax(-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, state: Tensor, action: Tensor, keys: int | list[int] = None) -> tuple[Tensor, Tensor | None]:
        latent = self._encode(state, keys=keys, verbose=False)
        mean, std = self._get_mean_std(latent, keys=keys)
        dist = self.dist(mean, std)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy

    def get_policy(self, state: Tensor, keys: Union[int, list[int]] = None, **options) -> Tensor:
        latent = self._encode(state, keys=keys, verbose=options.get('verbose', False))
        mean, std = self._get_mean_std(latent, keys=keys)
        action = mean
        if options.get('normal', self.probabilistic):
            action = action + (std * torch.randn_like(std))
        if self.distribution == 'discrete':
            action = torch.argmax(action, dim=-1)
        return action

    def enable_cache(self):
        if hasattr(self, 'transformer'):
            self.transformer.force_cache(True)

    def disable_cache(self):
        if hasattr(self, 'transformer'):
            self.transformer.force_cache(False)

    def reset_cache(self):
        if hasattr(self, 'transformer'):
            self.transformer.empty_cache()


# ----------------------------------------------------------------------------
# Vector-observation model (discrete_states / return_grid=False)
# ----------------------------------------------------------------------------
class BaseModelSeq(_PolicyMixin, mn.Model):
    def __init__(
        self, max_seq_len: int, inputs: int, outputs: int, dim_size: int,
        layers: int, heads: int, kv_heads: int | None = None,
        activation: Module = None, probabilistic=False, bias=True,
        device: TDEVICE = 'cpu', dtype: TDTYPE = None, **options
    ):
        super().__init__()
        self.max_seq_len   = max_seq_len
        self.inputs        = inputs
        self.outputs       = outputs
        self.dim_size      = dim_size
        self.layers        = layers
        self.distribution  = options.get('distribution', 'discrete')
        self.probabilistic = probabilistic
        self.heads         = heads
        self.kv_heads      = kv_heads
        self.use_swiglu    = options.get('use_swiglu', False)
        self.attn_bias     = options.get('attn_bias', bias)
        self.fwd_exp       = options.get('fwd_exp', 2)
        self.constant      = options.get('constant', 1000)
        self.differential  = options.get('differential', False)
        self.epsilon: float = options.get('epsilon', 1e-9)

        if activation is None:
            activation = nn.SiLU()
        if dtype is None:
            dtype = torch.float32

        feed_fwd = None if self.use_swiglu else mn.Sequential(
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
        ) if max_seq_len is not None else mn.Identity()
        self.lat_proj = mn.Sequential(
            mn.Linear(inputs, dim_size, True, device, dtype),
            transformer,
        )
        self.pol_proj = mn.Sequential(*[
            mn.Linear(dim_size, dim_size, bias, device, dtype),
            nn.Tanh(),
            mn.Linear(dim_size, 2 * outputs, True, device, dtype),
        ])

    @property
    def transformer(self) -> mn.TransformerBase:
        return self.lat_proj.modules_list[1]

    def _encode(self, state: Tensor, keys: Union[int, list[int]] = None, verbose: bool = False) -> Tensor:
        assert state.ndim >= 3, f"Expected (genomes, batch, seq_len, features), got {state.shape}"
        latent = self.lat_proj(state, keys=keys, verbose=verbose)
        return latent.squeeze(-2)  # single-token mode


# ----------------------------------------------------------------------------
# Grid-observation (return_grid=True) convolutional model -- mirrors
# `ConvNetwork` from binary_refactored.ipynb
# ----------------------------------------------------------------------------
class ConvModelSeq(_PolicyMixin, mn.Model):
    """
    Evolvable convolutional encoder for AgentConfig(return_grid=True)
    observations of shape (genomes, batch, C, H, W), built the same way as
    `ConvNetwork` in `binary_refactored.ipynb`: the grid is spatially
    flattened to a (channels, seq_len=H*W) sequence, then run through
    `mn.Conv1d` + SiLU input-projection and a `layers`-deep evolvable conv
    body (kernel_size, padding='same' via padding=-1) -- identical shapes and
    order to the notebook's sequence, just without its final classification
    head.

    Optionally (`use_transformer=True`), a small `mn.TransformerBase` is
    inserted right after that conv body, before the tensor is fully
    flattened: the (dim_size, seq_len) conv output is transposed to
    (seq_len, dim_size) so every grid cell becomes one token, processed by
    the transformer, then transposed back. This is the "tx after
    convolutional/residual layers, on the partially-flattened tensor, just
    before the final fc layers" hook the ticket asks for -- wired in but
    disabled by default.

    The final fc step here only projects down to a `dim_size`-wide latent
    (rather than straight to `outputs`, like the notebook's ConvNetwork
    does); that latent is handed to the same evolvable `pol_proj` policy
    head BaseModelSeq uses, so PPO's `get_action` / `evaluate_action` /
    `get_policy` work unmodified against either model.
    """

    def __init__(
        self, channels: int, dim_size: int, kernel_size: int, seq_len: int, layers: int,
        outputs: int, groups: int = 1, use_transformer: bool = False, tx_layers: int = 1,
        heads: int = 1, kv_heads: int | None = None, bias: bool = True,
        probabilistic=False, device: TDEVICE = 'cpu', dtype: TDTYPE = None, **options
    ):
        super().__init__()
        self.channels        = channels
        self.dim_size        = dim_size
        self.kernel_size     = kernel_size
        self.seq_len         = seq_len          # H * W, the spatially-flattened grid length
        self.layers          = layers
        self.outputs         = outputs
        self.groups          = groups           # stored for parity with ConvNetwork; unused for now (see its TODO)
        self.use_transformer = use_transformer
        self.distribution    = options.get('distribution', 'discrete')
        self.probabilistic   = probabilistic
        self.epsilon: float  = options.get('epsilon', 1e-9)
        self.vocab_size: int | None = options.get('vocab_size', None)

        if dtype is None: dtype = torch.float32

        # Input projection + evolvable conv body -- same shape/order as ConvNetwork:
        # in=(genomes, *batch, channels, *grid) -> (genomes, *batch, dim_size, *grid)
        sequence = [
            (
                mn.Conv2d(channels, dim_size, 1, bias=True, device=device, dtype=dtype)
                if self.vocab_size is None else
                mn.Sequential(
                    mn.Embedding(self.vocab_size, dim_size, None, device, dtype),
                    mn.Permute((0, 1, 4, 2, 3)),
                )
            ),
            nn.SiLU(),
            mn.Conv2d(dim_size, dim_size, kernel_size, stride=2, padding=-1, bias=bias, device=device, dtype=dtype),
            nn.SiLU(),
            mn.MaxPool2d(kernel_size, stride=2, padding=-1),
        ]
        for _ in range(layers):
            sequence.extend([
                mn.Conv2d(dim_size, dim_size, kernel_size, padding=-1, bias=bias, device=device, dtype=dtype),
                nn.SiLU(),
            ])
        self.conv = mn.Sequential(*sequence)

        # Optional transformer over the partially-flattened (dim_size, *grid) conv output,
        # inserted after the conv/residual stack and before the final fc layers.
        self.transformer_block = mn.TransformerBase(
            seq_len, dim_size, tx_layers, heads, kv_heads,
            causal_mask=False, bias=bias, auto_single=False,
            residual=True, normalize=True,
            device=device, dtype=dtype,
        ) if use_transformer else None

        # Final flatten + fc down to a dim_size-wide latent (ConvNetwork instead flattens straight
        # to `outputs`; here that last projection is deferred to the shared `pol_proj` head below).
        # print(f"Fc Size = {math.sqrt(seq_len)} / {2}^{2} ")
        self.out_proj = mn.Sequential(
            mn.Transpose(-3, -1), # Take embedding dim to last dim
            nn.Flatten(start_dim=-3),
            mn.Linear(dim_size * int(math.sqrt(seq_len) // (2 ** 2)) ** 2, dim_size, bias=bias, device=device, dtype=dtype),
            nn.SiLU(),
        )
        self.pol_proj = mn.Sequential(*[
            mn.Linear(dim_size, dim_size, bias, device, dtype),
            nn.Tanh(),
            mn.Linear(dim_size, 2 * outputs, True, device, dtype),
        ])

    def _encode(self, state: Tensor, keys: Union[int, list[int]] = None, verbose: bool = False) -> Tensor:
        # state: (genomes, batch, C, H, W) -> spatially flatten to (genomes, batch, C, H*W)
        assert state.ndim >= 4, f"Expected (genomes, batch, C, H, W), got {state.shape}"
        latent = self.conv(state, keys=keys, verbose=verbose)   # (genomes, batch, dim_size, *grid)
        # if self.transformer_block is not None:
        #     latent = *flaten(latent, -2, -1)
        #     latent = latent.transpose(-2, -1)                 # (genomes, batch, grid_sequence, dim_size)
        #     latent = self.transformer_block(latent, keys=keys, single=False, verbose=verbose)        
        # TODO: Ensure no single mode
        # print(f"Pre-Out Latent = {latent.shape}")
        latent = self.out_proj(latent, keys=keys)                        
        return latent

    def enable_cache(self): pass

    def disable_cache(self): pass

    def reset_cache(self): pass
            
            
class SequenceBuilder(Module):
    def __init__(self, max_seq_len: int | None, dim: int):
        super().__init__()
        assert max_seq_len is None or max_seq_len > 0
        self.max_seq_len = max_seq_len
        self.dim         = dim
        self.sequence: Tensor | None = None
        self.indices: Tensor | None = None
        
    def reset(self):
        self.sequence = None
        
    def forward(self, tensor: Tensor):
        if self.max_seq_len is None: 
            return tensor
        tensor = tensor.unsqueeze(self.dim)
        # Initialize
        if self.sequence is None:
            self.sequence = tensor.repeat_interleave(self.max_seq_len, self.dim) # (genomes, *dims, seq_len, features)
            self.indices = torch.arange(self.max_seq_len)
            torch.index_fill(self.sequence, self.dim, self.indices[:-1], 0.)
        # Shift
        else:
            torch.index_select(self.sequence, self.dim, self.indices[:-1])[:] = torch.index_select(self.sequence, self.dim, self.indices[1:])
            torch.index_select(self.sequence, self.dim, self.indices[-1])[:] = tensor
        return self.sequence


# ----------------------------------------------------------------------------
# Small 8x8 maze config for starters
# ----------------------------------------------------------------------------
def build_env_config(return_grid: bool, discrete_states: bool = False, discrete_actions: bool = True) -> EnvironmentConfig:
    config = EnvironmentConfig()
    # Small grid to start with -- 8x8.
    config.grid.grid_size           = (16, 16)
    config.grid.cell_size           = 40
    config.player.lives             = 5
    config.agent.return_grid        = return_grid
    config.agent.discrete_states    = discrete_states
    config.agent.discrete_actions   = discrete_actions   # 4-way movement (Direction enum)
    config.agent.reset_on_death     = False
    return config


ENV: Game | None = None

# ---- Model properties ----
DISCRETE_STATES = True
DISCRETE_ACTIONS = False
GENOMES         = 200
MAX_SEQ_LEN     = 8          # one grid-cell decision per step; no lookback needed
OUTPUTS         = 4          # 4-way discrete movement (Direction enum)
EMBED_SIZE      = 32
LAYERS          = 1
HEADS           = 1
KV_HEADS        = None
BIAS            = True
DIFFERENTIAL    = False
SWIGLU          = True
ACTIVATION      = nn.SiLU()
PROBABILISTIC   = True
DISTRIBUTION    = 'discrete' if DISCRETE_ACTIONS else 'normal'
SDTYPE = torch.long if DISCRETE_STATES else DTYPE


def evaluate(population: 'neat.Population', **options):
    trainer: 'neat.rl.PPO' = options['trainer']
    global INIT_GEN, FILE_NO

    cons_mapping = population.get_mapping(consolidated=True)
    keys = list(cons_mapping.keys())
    trainer.update_mapping(cons_mapping)
    if INIT_GEN is None:
        INIT_GEN = population.generation

    for genome in population.genomes.values():
        genome.fitness = 0

    MODEL.eval()
    MODEL.enable_cache()

    gts = clock.perf_counter()
    if population.generation % 10 == 0:
        ENV.render_mode = 'human'
    else:
        ENV.render_mode = None

    states = ENV.reset(keys=keys)[0]
    MODEL.reset_cache()

    done = False
    step = 0
    DEBUG_STEP = 0
    ENV.render()
    seq_builder = SequenceBuilder(max_seq_len=MAX_SEQ_LEN, dim=-2)
    while not done:
        with torch.no_grad():
            DEBUG = step == DEBUG_STEP and population.generation == INIT_GEN
            states = torch.tensor(states, device=DEVICE, dtype=SDTYPE)

            if DEBUG:
                print(f"\nstates =>\n{states}\n\tshape = {states.shape}")
                
            # Observation feeds a single "token"/frame per step: (genomes, batch=1, [seq_len=1,] *features)
            if ENV.config.agent.return_grid:
                obs = states.unsqueeze(1)              # (genomes, batch, C, H, W)
            else:
                states = seq_builder(states)  # (genomes, seq_len, features)
                obs = states.unsqueeze(1)  # (genomes, batch, seq_len, features)
                obs = obs.select(-2, -1).unsqueeze(-2)

            if DEBUG:
                print(f"\nobs =>\n{obs}\n\tshape = {obs.shape}")

            actions, log_probs = MODEL.get_action(
                obs, keys=None, 
                verbose= 2 if DEBUG else False,
            )
            actions = actions.squeeze(1)
            log_probs = log_probs.squeeze(1)

            if DEBUG:
                print(f"\nactions =>\n{actions}\n\tshape = {actions.shape}")

            next_states, rewards, _, done, info = ENV.step(actions.cpu().numpy())
            rewards = torch.tensor(rewards, device=DEVICE, dtype=DTYPE)

            terminate = trainer.update(states, actions, rewards, log_probs, done, done)

            print(
                f"\r{CM('Executing', Fore.GREEN)}: "
                f"time_elapsed={round(clock.perf_counter() - gts)}s, "
                f"alive={ENV.players.active_total}, "
                f"max_rew={round(rewards.max().item(), 2)}, "
                f"sd={trainer.steps_done}, "
                f"bl={trainer.primary.max_size()} ",
                end=''
            )

            states = next_states
            ENV.render()
            done = done or terminate

        step += 1
    print()

    fitnesses = ENV.players.fitness
    mf, sf = fitnesses.mean(), fitnesses.std()
    cutoff = mf - (1.0 * sf)
    for idx, (fitness, genome) in enumerate(zip(fitnesses, population.genomes.values())):
        # if fitness < cutoff: population.to_delete.append(genome.key)
        genome.fitness = fitness.item()

    MODEL.disable_cache()

    _, file_no = population.save(FILE_NAME, FILE_DIR, FILE_NO, replace=population.generation != INIT_GEN)
    if population.generation == INIT_GEN:
        print(f"Saved initial population to file number '{file_no}'")
        FILE_NO = file_no + 1


def run():
    global ENV, MODEL, GENOMES, FILE_NAME, FILE_DIR, FILE_NO, INIT_GEN

    parser = argparse.ArgumentParser(description='Run PPO/NEAT on the Maze-Runner env (Sequential)')
    parser.add_argument('--rm', type=str, default="human",
                        help='Render mode for the game (e.g. "human", None)')
    parser.add_argument('--load', type=str, default=None,
                        help='Load a previously trained model (True for latest, or a file number)')
    parser.add_argument('--train', type=str, default='true',
                        help='Train the model (True/False or 1/0, default: True)')
    parser.add_argument('--return_grid', type=str, default='false',
                        help='Use grid observations + the convolutional model instead of the vector model')
    parser.add_argument('--use_transformer', type=str, default='false',
                        help='(return_grid only) insert a small transformer after the conv body, '
                             'before the final fc layers')
    args = parser.parse_args()

    def str_to_bool(value: str) -> bool:
        if isinstance(value, bool):
            return value
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            return False
        raise ValueError(f"Cannot convert '{value}' to boolean")

    RETURN_GRID = True # str_to_bool(args.return_grid)

    # ---- Game settings: 8x8 starter maze ----
    env_config = build_env_config(RETURN_GRID, DISCRETE_STATES, DISCRETE_ACTIONS)
    ENV = Game(render_mode=args.rm, config=env_config)

    if RETURN_GRID:
        # state_shape   = ENV.observation_space.shape  # (C, H, W)
        test_state    = ENV.agents.get_state()
        state_shape   = test_state.shape
        GRID_CHANNELS = state_shape[-3] # state_shape[0]
        SEQ_LEN_GRID  = math.prod(state_shape[-2:]) # state_shape[1] * state_shape[2]   # H * W, flattened grid length
        vocab_size    = int(ENV.agents.max_options or 1)
        print(f"Channels = {GRID_CHANNELS}, Seq Len = {SEQ_LEN_GRID}")
        KERNEL_SIZE   = 3
        CONV_LAYERS   = 3
        GROUPS        = 1
        USE_TX        = str_to_bool(args.use_transformer)
        MODEL = ConvModelSeq(
            GRID_CHANNELS, EMBED_SIZE, KERNEL_SIZE, SEQ_LEN_GRID, CONV_LAYERS,
            OUTPUTS, GROUPS, USE_TX, tx_layers=1, heads=HEADS, kv_heads=KV_HEADS,
            bias=BIAS, probabilistic=PROBABILISTIC, device=DEVICE, dtype=DTYPE,
            distribution=DISTRIBUTION, vocab_size=vocab_size if DISCRETE_STATES else None,
        )
        BASE_NAME = (
            f"MazeModelConv-C{GRID_CHANNELS}_E{EMBED_SIZE}_K{KERNEL_SIZE}_L{CONV_LAYERS}_"
            f"TX{int(USE_TX)}_B{int(BIAS)}"
        )
    else:
        test_state    = ENV.agents.get_state()
        vocab_size    = int(ENV.agents.max_options or 1)
        INPUTS = ENV.observation_space.shape[-1]
        MODEL = BaseModelSeq(
            MAX_SEQ_LEN, INPUTS, OUTPUTS, EMBED_SIZE, LAYERS, HEADS, KV_HEADS,
            ACTIVATION, PROBABILISTIC, BIAS, DEVICE, DTYPE,
            distribution=DISTRIBUTION, differential=DIFFERENTIAL,
            attn_bias=BIAS, use_swiglu=SWIGLU,
            vocab_size=vocab_size if DISCRETE_STATES else None,
        )
        BASE_NAME = (
            f"MazeModelSeq-I{INPUTS}_MSL{MAX_SEQ_LEN}_E{EMBED_SIZE}_L{LAYERS}_H{HEADS}_"
            f"A-{ACTIVATION.__class__.__name__}_B{int(BIAS)}_S{int(SWIGLU)}"
        )
    print(MODEL)

    FILE_NAME = f"{BASE_NAME}"
    FILE_DIR  = "maze-runner/sequential"
    FILE_NO   = None
    INIT_GEN  = None

    # ---- NEAT/species config ----
    CONFIG = neat.Config('sequential', '.config')
    CONFIG.genome.init_type                  = 'normal'
    CONFIG.genome.weight_init_mean           = 0.0
    CONFIG.genome.weight_init_std            = 0.15
    CONFIG.genome.weight_min_value           = -np.inf
    CONFIG.genome.weight_max_value           = +np.inf
    CONFIG.genome.weight_mutate_power        = 0.008
    CONFIG.genome.weight_mutate_rate         = 0.65
    CONFIG.genome.weight_replace_rate        = 0.0
    CONFIG.genome.weight_add_prob            = 0.25
    CONFIG.genome.weight_del_prob            = 0.01
    CONFIG.genome.param_epsilon              = 1e-12
    CONFIG.genome.single_structural_mutation = True
    CONFIG.reproduction.min_species_size     = GENOMES
    CONFIG.reproduction.purge                = 1
    CONFIG.reproduction.elitism              = 0.33
    CONFIG.reproduction.clone_threshold      = 0.50
    CONFIG.reproduction.survival_threshold   = 0.10
    CONFIG.reproduction.cross_threshold      = 0.00
    CONFIG.species.compatibility_threshold   = np.inf
    CONFIG.stagnation.max_stagnation         = 1
    CONFIG.stagnation.species_elitism        = 2
    CONFIG.reproduction.darwin_multiplier    = 0.40
    CONFIG.reproduction.cross_multiplier     = 0.75
    CONFIG.reproduction.preserve_elite       = False
    CONFIG.save()
    CONFIG.load(2)

    POPULATION = neat.Population(GENOMES, MODEL, CONFIG, init_reporter=True)
    if args.load is not None:
        if args.load.lower() == 'true':
            FILE_NO = None
            POPULATION.load(FILE_NAME, FILE_DIR, FILE_NO)
            print("Loaded latest population checkpoint")
        else:
            try:
                FILE_NO = int(args.load)
                POPULATION.load_dict(None, FILE_NAME, FILE_DIR, FILE_NO)
                print(f"Loaded population checkpoint from file {FILE_NO}")
            except ValueError:
                if args.load.lower() not in ('false', 'none', 'null', ''):
                    print(f"Error: --load must be 'true'/'false' or a file number, got '{args.load}'")
                    exit(1)

    # ---- Trainer (PPO) properties ----
    MEMORY_SIZE  = 5
    GAMMA        = fix(np.exp(np.log(0.01) / 128), 0.0)
    KAPPA        = 0.0
    ALPHA        = fix(np.exp(np.log(10.0) / (MEMORY_SIZE - 1)), 1.0)
    BETA         = float(np.exp(np.log(10.0) / max(1, MEMORY_SIZE - 1)))
    ALPHA_ORDER  = 0
    BETA_ORDER   = 0
    REW_NORM     = 3
    REW_REG      = 1.0
    PPO_REG      = 0.0
    POL_REG      = 0.0
    ENT_REG      = 0.0
    CPY_REG      = 0.0
    DIV_REG      = 0.0
    SEGR_SIZE    = None
    print(f"Gamma={GAMMA:.4f}, Kappa={KAPPA:.4f}, Alpha={ALPHA:.4f}, Beta={BETA:.4f}, ")

    STEPS  = 2000 # 8 * 8 * 10   # rollout steps per epoch, scaled to the 8x8 grid
    EPOCHS = 1_000

    if str_to_bool(args.train):
        trainer = neat.rl.PPO(
            POPULATION,
            schedulers=[
                scheduler.CosineAnnealing(CONFIG, 20, 0.1, 'weight_mutate_power', False, True),
                # scheduler.BinaryAnnealing(CONFIG, 20, 0.10, 'cross_threshold'),
            ],
            device=DEVICE, dtype=DTYPE,
            log_sub_dir='maze_runner/',
            log_name=f"{unix_to_datetime_file(clock.time())}-sequential_"
                     f"e{EMBED_SIZE}-grid{int(RETURN_GRID)}-"
                     f"g{round(GAMMA, 4)}-k{round(KAPPA, 4)}-a{round(ALPHA, 4)}-ao{ALPHA_ORDER}-"
                     f"rn{REW_NORM}-l{round(PPO_REG, 4)}-p{round(POL_REG, 4)}-e{round(ENT_REG, 4)}-d{round(DIV_REG, 4)}-"
                     f"mem{MEMORY_SIZE}",
            gamma=GAMMA, alpha=ALPHA, kappa=KAPPA, order=ALPHA_ORDER, normalize=REW_NORM,
            beta=BETA, beta_order=BETA_ORDER,
            rew_reg=REW_REG, loss_reg=PPO_REG, pol_reg=POL_REG, ent_reg=ENT_REG, div_reg=DIV_REG, cpy_reg=CPY_REG,
            validate=True, segr_size=SEGR_SIZE, max_steps=None, max_episodes=MEMORY_SIZE, use_entropy=True,
        )
        print("Regularizations:")
        for l, v in vars(trainer).items(): 
            if "_reg" in l: 
                print(f"\t{l} => {v:.4f}")
        print("All params:")
        for l, v in vars(trainer).items(): 
            if isinstance(v, (int, float, bool)) and l[0] != '_': 
                print(f"\t{l} => {v}")

        print(f"starting evaluation: population={len(POPULATION.genomes)}, grid={(ENV.grid.rows, ENV.grid.cols)}, return_grid={RETURN_GRID}")
        try:
            trainer.learn(evaluate, STEPS, EPOCHS, 128, accuracy_type="discrete" if DISCRETE_ACTIONS else "continuous", verbose=True)
        except KeyboardInterrupt:
            pass
    else:
        print("Skipping training...")

    # ---- Post-training playback ----
    MODEL.eval()
    MODEL.disable_cache()
    ENV.render_mode = 'human'
    for _ in range(5):
        done = False
        states = ENV.reset(keys=GENOMES)[0]
        MODEL.reset_cache()
        while not done:
            with torch.no_grad():
                states_t = torch.tensor(states, device=DEVICE, dtype=SDTYPE)
                obs = states_t.unsqueeze(1) if RETURN_GRID else states_t.unsqueeze(1).unsqueeze(-2)
                actions = MODEL.get_policy(obs).squeeze(1)
            states, _, _, done, _ = ENV.step(actions.cpu().numpy())
            ENV.render()
    MODEL.train()


if __name__ == '__main__':
    run()