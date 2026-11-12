import os, sys
import torch
import torch.nn as nn
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
EXMP_ROOT = os.path.dirname(EXMP_DIR)   # examples/ (holds the shared 'misc' extensions)
if EXMP_ROOT not in sys.path: sys.path.insert(0, EXMP_ROOT)

import ModifiedNEAT as neat
import ModifiedNEAT.nn as mn

from ModifiedNEAT.optim import scheduler
from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.util.datetime import unix_to_datetime_file
from ModifiedNEAT.util.qol import manage_params
from ModifiedNEAT.nn.modules.reporting import differential_attention_logger

from game import Game
from misc import BinaryStepper
from misc import NoiseRegPPO

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


class Block(mn.NeatModule):
    def __init__(
        self, inputs: int, outputs: int, dim_size: int, coefficients: int,
        normalize: bool = False, residual: bool = True,
        activation: nn.Module | None = None, bias: bool = True,
        device: torch.device = "cpu", dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            inp=inputs, out=outputs, dim=dim_size, coeffs=coefficients, norm=normalize,
            res=residual, actv=activation.__class__.__name__ if not None else str(None), bias=bias
        )
        self.inputs     = inputs
        self.outputs    = outputs
        self.dim_size   = dim_size
        self.normalize  = normalize
        self.residual   = residual
        self.activation = activation if activation is not None else nn.SiLU()
        self.bias       = bias
        self.device     = device
        self.dtype      = dtype

        self.pre_norm = mn.LayerNorm(inputs, bias=False, device=device, dtype=dtype) if normalize else None
        self.inp_proj = mn.Polynomial(inputs, dim_size, coefficients, bias, device, dtype)
        self.out_proj = mn.Polynomial(dim_size, outputs, coefficients, bias, device, dtype)
        self.res_proj = (
            mn.Identity()
            if inputs == outputs else
            mn.Polynomial(inputs, outputs, 1, False, device, dtype)
        ) if residual else None

    def forward(self, tensor: Tensor, keys: int | list[int] | None = None, **kwargs):
        residue = tensor
        if self.normalize: tensor = self.pre_norm(tensor, keys=keys)
        tensor = self.out_proj(self.activation(self.inp_proj(tensor, keys=keys)), keys=keys)
        if self.residual: tensor = tensor + self.res_proj(residue, keys=keys)
        return tensor


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
        self.coefficients: int  = options.get('coefficients', 1)
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
        self.heads              = heads
        self.kv_heads           = kv_heads
        self.use_swiglu: bool   = options.get('use_swiglu', False)
        self.attn_bias: bool    = options.get('attn_bias', bias)
        self.fwd_exp: int       = options.get('fwd_exp', 1)
        self.constant: int      = options.get('constant', 1000)
        self.differential: bool | int = options.get('differential', False)
        self.diff_mode: str     = options.get('diff_mode', 'energy')     # 'energy' | 'softmax' (see Attention)
        self.branch_mode: str   = options.get('branch_mode', 'free')     # 'free' | 'tied'
        self.share_kv: bool     = options.get('share_kv', False)         # exact cache == full sequence (no pol_reg needed)
        self.epsilon            = options.get('epsilon', 1e-9)
        self.steps: int         = options.get('steps', 1)         # resolution levels per continuous output (see stepper.py)
        self.bipolar: bool      = options.get('bipolar', True)    # env actions live in [-1, 1]
        self.stepper: BinaryStepper | None = None
        if self.distribution != 'discrete':
            self.stepper = BinaryStepper(outputs, self.steps, self.bipolar)
        self.act_mult           = 1 if self.stepper is None else self.stepper.bits   # raw outputs per env output = log2 levels
        self.raw_outputs        = outputs * self.act_mult
        
        if activation is None:
            activation = nn.SiLU()
        if dtype is None:
            dtype = torch.float32
        
        # Build
        # NOTE: SwiGLU seems to work well for backpropagated networks but not for NEAT weight crossover & mutation
        # feed_fwd = None if self.use_swiglu else mn.Sequential(
        #     # mn.LayerNorm(dim_size, bias=bias, device=device, dtype=dtype),
        #     # mn.RMSNorm(dim_size, device=device, dtype=dtype),
        #     mn.Linear(dim_size, dim_size * self.fwd_exp, bias, device, dtype),
        #     activation, 
        #     mn.Linear(dim_size * self.fwd_exp, dim_size, self.attn_bias, device, dtype),
        #     # activation, 
        # )
        feed_fwd = None if self.use_swiglu else Block(
            dim_size, dim_size, dim_size * self.fwd_exp, self.coefficients,
            self.normalize, True, activation, bias, device, dtype
        )
        transformer = mn.TransformerBase(
            max_seq_len, dim_size, layers, heads, kv_heads,
            differential=self.differential, causal_mask=True, 
            diff_mode=self.diff_mode, branch_mode=self.branch_mode, share_kv=self.share_kv,
            bias=self.attn_bias, ff=feed_fwd, 
            auto_single=True, fwd_exp=self.fwd_exp, proj_coeffs=self.coefficients,
            residual=True, constant=self.constant, normalize=self.normalize,
            device=device, dtype=dtype
        )
        self.lat_proj = mn.Sequential(
            # mn.Linear(inputs, dim_size, True, device, dtype),
            mn.Polynomial(inputs, dim_size, self.coefficients, True, device, dtype),
            # activation, 
            transformer,
        )
        self.pol_proj = mn.Sequential(*[
            (
                mn.LayerNorm(dim_size, bias=False, device=device, dtype=dtype) 
                if self.normalize else 
                mn.Identity()
            ),
            # mn.Linear(dim_size, dim_size, bias, device, dtype),
            mn.Polynomial(dim_size, dim_size, self.coefficients, bias, device, dtype),
            activation,
            # mn.Linear(dim_size, 2 * outputs, True, device, dtype),
            mn.Polynomial(dim_size, 2 * self.raw_outputs, self.coefficients, True, device, dtype),
        ])
        # self.val_proj = mn.Sequential(*[
        #     mn.LayerNorm(dim_size, bias=bias, device=device, dtype=dtype),
        #     # mn.RMSNorm(dim_size, device=device, dtype=dtype),
        #     mn.Linear(dim_size, dim_size, bias, device, dtype),
        #     nn.SiLU(),
        #     mn.Linear(dim_size, dim_size, bias, device, dtype),
        #     nn.SiLU(),
        #     mn.Linear(dim_size, 1, bias, device, dtype),
        # ])
       
    @property 
    def transformer(self) -> mn.TransformerBase:
        return self.lat_proj.modules_list[-1]

    def extra_repr(self) -> str:
        return f"probabilistic={self.probabilistic}, distro='{self.distribution}'"

    def forward(self, state: Tensor, keys: Union[int, list[int]] = None, **kwargs):
        return self.get_policy(state, keys=keys, **kwargs)

    def _get_mean_std(self, latent: Tensor, keys: Union[int, list[int]] = None):
        self.transformer.layers[0].self_attention.forced_cache
        mean_std      = self.pol_proj(latent, keys=keys)
        mean, log_std = torch.chunk(mean_std, 2, -1)
        # if self.normalize:
        #     mean = torch.tanh(mean) * max(1, self.upper)
        #     std = torch.pow(10, self.clip_min + self.clip_range * torch.sigmoid(log_std))
        # else:
        #     std = torch.exp(log_std)
        # mean, _std = torch.chunk(mean_std, 2, -1)
        # std = torch.abs(_std)
        std = torch.exp(log_std.clamp(self.clip_min, self.clip_max))
        return mean, std

    def _get_std(self, latent: Tensor, keys: Union[int, list[int]] = None, raw: bool = False):
        mean_std      = self.pol_proj(latent, keys=keys)
        mean, log_std = torch.chunk(mean_std, 2, -1)
        # _, _std = torch.chunk(mean_std, 2, -1)
        # if self.normalize:
        #     std = torch.pow(10, self.clip_min + self.clip_range * torch.sigmoid(log_std))
        #     if raw: std = torch.log10(std)
        # else:
        #     if raw: std = log_std
        #     else: std = torch.exp(log_std)
        # std = torch.abs(_std)
        std = torch.exp(log_std.clamp(self.clip_min, self.clip_max))
        return std

    def _reduce(self, tensor: Tensor) -> Tensor:
        """(..., outputs * bits) -> (..., outputs): joint over the bits of one output."""
        if self.stepper is None:
            return tensor
        return tensor.unflatten(-1, (self.outputs, self.act_mult)).sum(-1)

    def to_env(self, actions: Tensor) -> Tensor:
        """Raw (pre-tanh) actions, as stored in the rollout buffer -> value for ENV.step. 1 level == plain tanh."""
        return actions if self.stepper is None else self.stepper(actions)

    def get_action(self, state: Tensor, keys: int | list[int] = None, **options) -> tuple[Tensor, Tensor]:
        assert state.ndim >= 4, f"Latent shape should be (genomes, *batch/processes, seq_len, features) but got {state.shape}"
        latent      = self.lat_proj(state, keys=keys, verbose=options.get('verbose', False))
        assert latent.ndim >= 4, f"Latent shape should be (genomes, batch/processes, seq_len=1, features) but got {latent.shape}"
        latent      = latent.squeeze(-2) # Since single token mode should be enabled
        mean, std   = self._get_mean_std(latent, keys=keys)
        dist: torch.distributions.Distribution = self.dist(mean, std)
        if self.probabilistic:
            action = dist.sample()
        else:
            action = mean
            if self.distribution == "discrete": action = action.argmax(-1)
        log_prob = self._reduce(dist.log_prob(action))
        return action, log_prob

    def evaluate_action(self, state: Tensor, action: Tensor, keys: int | list[int] = None) -> tuple[Tensor, Tensor | None]:
        latent    = self.lat_proj(state, keys=keys, verbose=False)
        latent    = latent.squeeze(-2)
        mean, std = self._get_mean_std(latent, keys=keys)
        dist: torch.distributions.Distribution = self.dist(mean, std)
        # if self.distribution != "discrete": action = torch.atanh(action * (1 - self.epsilon))
        log_prob = self._reduce(dist.log_prob(action))
        entropy = self._reduce(dist.entropy())
        return log_prob, entropy

    def get_policy(self, state: Tensor, keys: Union[int, list[int]] = None, **options) -> Tensor:
        assert state.ndim >= 4, f"Latent shape should be (genomes, *batch/processes, seq_len, features) but got {state.shape}"
        latent      = self.lat_proj(state, keys=keys, verbose=options.get('verbose', False))
        assert latent.ndim >= 4, f"Latent shape should be (genomes, batch/processes, seq_len=1, features) but got {latent.shape}"
        latent      = latent.squeeze(-2) # Since single token mode should be enabled
        mean, std   = self._get_mean_std(latent, keys=keys)
        action = mean
        if options.get('normal', self.probabilistic):
            action = action + (std * torch.randn_like(std))
        if self.distribution == 'discrete':
            action = torch.argmax(action, dim=-1)
        # NOTE: PRE-tanh on purpose (same space as the rollout buffer written from get_action(), which the trainer's
        # policy accuracy compares against). Callers convert once with MODELS[0].to_env(actions) (= tanh for LEVELS=1) before ENV.step.
        return action

    # def get_value(self, state: Tensor, keys: Union[int, list[int]] = None, **options) -> Tensor:
    #     assert state.ndim >= 4, f"Latent shape should be (genomes, *batch/processes, seq_len, features) but got {state.shape}"
    #     latent = self.lat_proj(state, keys=keys, verbose=options.get('verbose', False))
    #     assert latent.ndim >= 4, f"Latent shape should be (genomes, batch/processes, seq_len=1, features) but got {latent.shape}"
    #     latent = latent.squeeze(-2) # Since single token mode should be enabled
    #     value  = self.val_proj(latent, keys=keys)
    #     return value
    
    def get_mean(self, state: Tensor, keys: Union[int, list[int], None] = None):
        assert state.ndim >= 4 # state(1, batch_size, seq_len, features)
        latent = self.lat_proj(state, keys=keys)
        assert latent.ndim >= 4
        latent = latent.squeeze(-2) # Remove seq_len dim
        mean   = self._get_mean_std(latent, keys=keys)[0] # , raw=True)[0]
        return mean
    
    def get_std(self, state: Tensor, keys: Union[int, list[int], None] = None):
        assert state.ndim >= 4 # state(1, batch_size, seq_len, features)
        latent = self.lat_proj(state, keys=keys)
        assert latent.ndim >= 4
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


# ENVIRONMENT SETTINGS (Signal Tracker: a hidden target readable only on clean frames; see game/env.py)
HORIZON             = 200    # steps per episode
EPISODES            = 4      # episodes per training update (STEPS)
DECOY_FRAC          = 0.5    # fraction of decoy frames (cue = uniform noise): sequence noise between instances
DECOY_RUN           = 4.0    # mean decoy burst length (longer = needs a longer memory)
CUE_SIGMA           = 0.05   # reading noise of clean frames
TAG_SEP             = 2.0    # how well the tag separates clean frames from decoys (smaller = harder)
TAG_SIGMA           = 0.25
DRIFT               = 0.0    # common-mode drift shared by every frame (what differential attention subtracts)
OBS_NOISE           = 0.0    # plain per-feature noise (instance noise)
TARGET_SPEED        = 0.04
AGENT_SPEED         = 0.12
FULL_STATES         = True
DELAY               = 0
DISCRETE            = False

# Model properties
GENOMES             = 100
MAX_SEQ_LEN         = 16
SEQ_LEN             = MAX_SEQ_LEN // 1
SEQ_LEN_TRAIN       = min(max(1, 1), SEQ_LEN)
SEQ_LEN_EVAL        = min(max(SEQ_LEN, 1), SEQ_LEN)
INPUTS              = None   # set in run() from ENV.observation_space
OUTPUTS             = None   # set in run() from ENV.action_space
EMBED_SIZE          = 32
COEFFICIENTS        = 1
LAYERS              = 1
HEADS               = 1
KV_HEADS            = None
FWD_EXP             = 1
LEVELS              = 1000      # resolution levels per continuous output: raw outputs per action = bit_length(LEVELS) (1 = plain tanh)
DIFFERENTIAL        = 1
DIFF_MODE           = 'softmax'  # 'energy' = subtract before softmax (== one wider attention) | 'softmax' = subtract the maps
SHARE_KV            = True       # K/V from input tokens only: cached rollout == full-sequence pass exactly
BRANCH_MODE         = 'tied'     # 'free' = independent q/k per branch | 'tied' = shared projection + per-branch gains
BIAS                = True
PROBABILISTIC       = False
PROJ_NORM           = True
CONSTANT            = 1000
TEST_ACTIVATION     = nn.SiLU()
SWIGLU              = False
CLIP_MIN            = -13
CLIP_MAX            = +0
DISTRIBUTION        = 'normal' if not DISCRETE else 'discrete'
MODEL_NUM           = 1
GENERA              = 2   # >1: ONE model shared by this many genera (Population(genera=N), GENOMES per genus); 1: MODEL_NUM separate models
if GENERA > 1: MODEL_NUM = 1
SHARED              = GENERA > 1
CACHE               = True

# Trainer properties
MEMORY_SIZE         = 5
GAMMA               = fix(np.exp(np.log(0.01) / 128), 0.0)
KAPPA               = 0.0 # fix(np.exp(np.log(0.01) / 128), 0.0)
ALPHA               = fix(np.exp(np.log(1.35) / (MEMORY_SIZE - 1)), 1.0)
ALPHA_ORDER         = 0
REW_NORM            = 3
REW_REG             = 1.00
LOSS_REG            = 0.00
VAL_REG             = 0.00
POL_REG             = 0.95 # if (PROBABILISTIC or (CACHE and not SHARE_KV)) else 0.0   # SHARE_KV: parity is exact, no regularizer
ENT_REG             = 0.00
DIV_REG             = 0.01
NOISE_REG           = 0.00   # fitness weight of low measured attention noise (0 = no pressure)
NOISE_SIGMA         = 0.0 # 0.01   # probe: relative input perturbation
NOISE_LOG           = True   # measure + log noise even when NOISE_REG == 0 (baseline comparison)
CPY_REG             = 0.00

# Post-training test
TEST_RUNS    = 5
TEST_GOAL    = 400
TEST_ELITES  = 10      # best genomes per genus kept for the test

# Training 
GOAL        = HORIZON
STEPS       = HORIZON * EPISODES
EPOCHS      = 300
FILTER_DEAD = False # TODO: Might remove since it might bring about truncated tensors with dequed kv cache, or make optional

MODELS: list = []     # built in run() once the Env (and so INPUTS / OUTPUTS) exists


def build_models() -> list:
    """Models are sized from the Env's observation / action spaces, so they are created after the Env."""
    return [
        BaseModelSeq(
            MAX_SEQ_LEN, INPUTS, OUTPUTS, EMBED_SIZE, LAYERS, HEADS, KV_HEADS,
            activation, PROBABILISTIC, BIAS, DEVICE, DTYPE,
            clip_min=CLIP_MIN, clip_max=CLIP_MAX, distribution=DISTRIBUTION,
            constant=CONSTANT, attn_bias=BIAS, 
            differential=DIFFERENTIAL, diff_mode=DIFF_MODE, branch_mode=BRANCH_MODE, share_kv=SHARE_KV,
            use_swiglu=SWIGLU, fwd_exp=FWD_EXP, normalize=PROJ_NORM, steps=LEVELS, bipolar=True,
            coefficients=COEFFICIENTS,
        )
        for activation in [TEST_ACTIVATION for _ in range(MODEL_NUM)]
    ]


def shared_call(model, method: str, obs: Tensor, mapping: dict[int, int], **kw) -> list[Tensor]:
    """
    One call of a model shared by several genera. `obs` is (rows, batch, ...) in `mapping` (consolidated) row order;
    the model wants its rows in its own order, so rows are gathered by genome key, run once (a KV cache needs a fixed
    batch of genomes) and scattered back to the consolidated order.
    """
    keys = sorted(mapping, key=lambda k: model.mapping[k])
    rows = torch.as_tensor([mapping[k] for k in keys], device=obs.device, dtype=torch.long)
    out = getattr(model, method)(obs.index_select(0, rows), keys=keys, **kw)
    out = out if isinstance(out, (tuple, list)) else (out,)
    result = []
    for r in out:
        full = r.new_zeros((len(mapping), *r.shape[1:]))
        full[rows] = r
        result.append(full)
    return result


GAME_TYPE = int(round(DECOY_FRAC * 10))   # decoy fraction in tenths, for the file / log names
BASE_NAME: str = (
    f"GT{GAME_TYPE}_DC{int(DISCRETE)}_T{MODEL_NUM}{f'G{GENERA}' if GENERA > 1 else ''}_"
    f"MSL{MAX_SEQ_LEN}_E{EMBED_SIZE}_L{LAYERS}_"
    f"H{HEADS}_K{KV_HEADS}_C{COEFFICIENTS}_"
    f"A-{TEST_ACTIVATION.__class__.__name__}_B{int(BIAS)}_"
    f"F{FWD_EXP}_D{int(DIFFERENTIAL)}_Con{CONSTANT}_"
    f"S{int(SWIGLU)}{f'_R-{CLIP_MIN}~{CLIP_MAX}' if PROBABILISTIC else ''}"
    f"{'_skv' if SHARE_KV else ''}"
    f"{f'_Lv{LEVELS}' if LEVELS != 1 else ''}"
    f"{f'_{DIFF_MODE}-{BRANCH_MODE}' if DIFFERENTIAL and (DIFF_MODE, BRANCH_MODE) != ('energy', 'free') else ''}"
)
FILE_NAME: str = f"SignalTrackerModelSeq-{BASE_NAME}"
FILE_DIR: str | None = "signal-tracker/sequential"
FILE_NO: int | None = None
INIT_GEN: int | None = None

print(f"\ncreating config")
CONFIG = neat.Config('signal-tracker-sequential', '.config')

MULTIPLIER = 1.0 # 0.33
CONFIG.genome.init_type                     = 'normal'
CONFIG.genome.weight_init_mean              = 0.0
CONFIG.genome.weight_init_std               = 1.0 * MULTIPLIER
CONFIG.genome.weight_min_value              = -np.inf
CONFIG.genome.weight_max_value              = +np.inf
CONFIG.genome.weight_mutate_power           = 0.50 * MULTIPLIER
CONFIG.genome.weight_mutate_rate            = 0.60
CONFIG.genome.weight_replace_rate           = 0.00
CONFIG.genome.weight_add_prob               = 0.00
CONFIG.genome.weight_del_prob               = 0.00
CONFIG.genome.single_structural_mutation    = True
CONFIG.genome.param_epsilon                 = 0e-9
CONFIG.reproduction.min_species_size        = GENOMES
CONFIG.reproduction.purge                   = 1
CONFIG.reproduction.elitism                 = 0.33
CONFIG.reproduction.clone_threshold         = 0.50
CONFIG.reproduction.survival_threshold      = 0.10
CONFIG.reproduction.cross_threshold         = 0.00
CONFIG.species.compatibility_threshold      = np.inf
CONFIG.stagnation.max_stagnation            = 1
CONFIG.stagnation.species_elitism           = 2
CONFIG.reproduction.darwin_multiplier       = 0.20 # 0.33
CONFIG.reproduction.cross_multiplier        = 0.80
CONFIG.reproduction.preserve_elite          = False
CONFIG.save()
CONFIG.load(verbose=2)

ENV: Game | None = None
POPULATION: neat.Population | None = None
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
        if CACHE: model.enable_cache()
    while not terminate:
        gts = clock.perf_counter()
        step = 0
        reverse_mappings = tuple([
            {index: key for key, index in m.items()}
            for m in split_mappings
        ])

        states = _T(ENV.reset(mapping=cons_mapping)[0])
        for model in MODELS: 
            if CACHE: model.reset_cache()
        step = 0
        DEBUG_STEP = 0 # 16
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
                if SHARED:   # ONE model, every genus in one call
                    obs = states.unsqueeze(1).select(-2, -1).unsqueeze(-2) if CACHE else states.unsqueeze(1)
                    split_states = (obs,)
                    a_all, lp_all = shared_call(MODELS[0], 'get_action', obs, cons_mapping, verbose=2 if DEBUG else False)
                    split_actions, split_log_probs = (a_all,), (lp_all,)
                elif CACHE: split_states = torch.split(states.unsqueeze(1).select(-2, -1).unsqueeze(-2), split_sizes, dim=0)
                else:     split_states = torch.split(states.unsqueeze(1), split_sizes, dim=0)
                # split_states = torch.split(states.unsqueeze(1).index_select(-2, SELECTOR[-3:]), split_sizes, dim=0)
                if not SHARED:
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
                _actions = MODELS[0].to_env(actions)
                next_states, rewards, _, done, _ = ENV.step(_actions)
                next_states, rewards = _T(next_states), _T(rewards)
                if DEBUG:
                    print(f"\nrewards =>\n{rewards}\n\tshape = {rewards.shape}")

                # Updated buffers
                terminate = trainer.update(states, actions, rewards, log_probs, done, done)

                # CMDLine rendering
                alive = ENV.players.active_total
                max_score = round(rewards.max().item(), 2)
                print(
                    f"\r{CM('Executing', Fore.GREEN)}: "
                    f"time_elapsed = {round(clock.perf_counter()-gts)}s, "
                    f"alive = {alive}, max_rew = {max_score}, "
                    f"sd={trainer.steps_done}, "
                    f"bl={trainer.primary.max_size()}     ", end=''
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

        for key, idx in cons_mapping.items():       # cursor index -> genome by the mapping, not by dict order
            population.genomes[key].fitness = float(ENV.players.fitness[idx])

        game_step += 1
    print(f"\n")
    for model in MODELS: 
        model.train(); 
        if CACHE: model.disable_cache()
    
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
    ks = max(len(str(g.key)) for g in population.genomes.values())
    gs = 3
    fs = max(len(str(round(g.actual_fitness, 4))) for g in population.genomes.values())
    for i, genus in enumerate(population.genera):
        genomes = sorted([g for g in population.genomes.values() if g.genus == genus], key=lambda g: g.actual_fitness, reverse=True)[:COUNT]
        for genome in genomes:
            print(f"\t{fill(genome.key, ks)}|{fill(genome.genus, gs)}|{fill(round(genome.actual_fitness, 4), fs)}")
        if i < len(population.genera) - 1:
            print(f"\t...")


def test_best_network():
    """Watch the surviving elites play on a harder, longer level (rendered)."""
    elites = []
    for gn_id, specie in enumerate(POPULATION.species_set.species.values()):
        genomes = sorted(
            [g for g in specie.members.values() if g.fitness is not None and not np.isnan(g.fitness)],
            key=lambda g: g.fitness, reverse=True,
        )
        elites.extend(genomes[:TEST_ELITES])
        print(f"\nElites [{gn_id}]: {[g.key for g in genomes[:TEST_ELITES]]}")
    if not elites:
        print("No evaluated genomes to test")
        return
    POPULATION.crop([e.key for e in elites], DEVICE)

    env = Game(
        render_mode='human', horizon=TEST_GOAL, seq_len=SEQ_LEN, decoy_frac=DECOY_FRAC, decoy_run=DECOY_RUN, cue_sigma=CUE_SIGMA,
        tag_sep=TAG_SEP, tag_sigma=TAG_SIGMA, drift=DRIFT, obs_noise=OBS_NOISE,
        target_speed=TARGET_SPEED, speed=AGENT_SPEED,
        fps=45
    )
    split_mappings = POPULATION.get_mapping()
    if len(POPULATION.genera) == 1: split_mappings = (split_mappings,)
    split_sizes = [len(m) for m in split_mappings]
    cons_mapping = POPULATION.get_mapping(consolidated=True)

    for model in MODELS: model.eval()
    try:
        for i in range(TEST_RUNS):
            print(f"Running test {i + 1}/{TEST_RUNS}")
            done = False
            states = _T(env.reset(mapping=cons_mapping)[0])
            while not done:
                with torch.no_grad():
                    split_states = torch.split(states, split_sizes, dim=0)
                    if SHARED:
                        split_actions = shared_call(MODELS[0], 'get_policy', states.unsqueeze(1), cons_mapping)
                    else:
                        split_actions = [m.get_policy(obs.unsqueeze(1)) for m, obs in zip(MODELS, split_states)]
                    actions = torch.cat(split_actions, dim=0).squeeze(1)
                next_states, _, _, done, _ = env.step(MODELS[0].to_env(actions))
                states = _T(next_states)
                env.render()
            print(f"Mean reward: best={env.players.fitness.max().item():.3f}, mean={env.players.fitness.mean().item():.3f}")
    except KeyboardInterrupt:
        print("\nTerminated")
    finally:
        for model in MODELS: model.train()


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
    global FILE_NO
    if load is None or load.lower() in ("false", "none", "null", ""):
        return
    if load.lower() == "true":
        FILE_NO = None
        print("Will load the latest population checkpoint")
    else:
        try:
            FILE_NO = int(load)
        except ValueError:
            sys.exit(f"--load must be 'true', 'false' or a file number, got '{load}'")
        print(f"Will load population checkpoint from file number '{FILE_NO}'")
    POPULATION.load(FILE_NAME, FILE_DIR, FILE_NO)


def run(argv: list[str] | None = None, test: bool = True):
    """argv: CLI args (None = sys.argv); test: run the rendered test of the best genomes after training."""
    global ENV, POPULATION, FILE_NO, INIT_GEN, MODELS, INPUTS, OUTPUTS

    parser = argparse.ArgumentParser(description='Run NEAT algorithm on Signal Tracker game')
    parser.add_argument('--rm', type=str, default=None,
                        help='Render mode for the game (e.g., "human", None)')
    parser.add_argument('--load', type=str, default=None,
                        help='Load a previously trained model (True for latest, or specify file number)')
    parser.add_argument('--train', type=str, default='true',
                        help='Train the model (True/False or 1/0, default: True)')
    args = parser.parse_args(argv)

    ENV = Game(
        render_mode=args.rm, horizon=GOAL, seq_len=SEQ_LEN, decoy_frac=DECOY_FRAC, decoy_run=DECOY_RUN, cue_sigma=CUE_SIGMA,
        tag_sep=TAG_SEP, tag_sigma=TAG_SIGMA, drift=DRIFT, obs_noise=OBS_NOISE,
        target_speed=TARGET_SPEED, speed=AGENT_SPEED,
        fps=None if args.rm != 'human' else 45
    )

    # Models are sized from the Env's spaces
    INPUTS = int(ENV.observation_space.shape[-1])
    OUTPUTS = 2 if DISCRETE else int(ENV.action_space.shape[-1])
    MODELS = build_models()
    print(MODELS[0])

    print("\ncreating population")
    if GENERA > 1:     # one model, GENERA genera of GENOMES genomes each
        POPULATION = neat.Population(GENOMES * GENERA, MODELS[0], CONFIG, init_reporter=True, genera=GENERA)
    else:
        POPULATION = neat.Population(GENOMES, MODELS[0], CONFIG, init_reporter=True)
        for model in MODELS[1:]:
            extra_population = neat.Population(GENOMES, model, CONFIG, init_reporter=True)
            POPULATION.absorb_population(extra_population)
    print(POPULATION)
    load_population(args.load)
    INIT_GEN = None             # set by the first evaluate() call
    FILE_NO = None

    if str_to_bool(args.train):
        print(f"Calculated hyperparameters: gamma={GAMMA:.4f}, kappa={KAPPA:.4f}, alpha={ALPHA:.4f}")
        trainer = NoiseRegPPO(
            POPULATION,
            schedulers=[
                scheduler.CosineAnnealing(CONFIG, 20, 0.25, ['weight_mutate_power', 'weight_init_std'], True, True),
                scheduler.BinaryAnnealing(CONFIG, 20, 0.10, 'cross_threshold'),
                # *([scheduler.BinaryAnnealing(CONFIG, 5, 0.99, 'pol_reg')] if POL_REG > 0 else []),   # a scheduler would re-enable pol_reg
                # scheduler.CosineAnnealing(CONFIG, 15, 0.67, 'loss_reg', True, True, phase=0),
                # scheduler.CosineAnnealing(CONFIG, 15, 0.50, 'pol_reg', True, True, phase=0),
                # scheduler.CosineAnnealing(CONFIG, 15, 0.75, 'ent_reg', True, True, phase=0),
            ],
            device=DEVICE, dtype=DTYPE,
            log_sub_dir='signal_tracker/',
            log_name=f"sequential-{unix_to_datetime_file(clock.time())}-"
                     f"seq{SEQ_LEN}-e{EMBED_SIZE}-l{LAYERS}-h{HEADS}-kv{KV_HEADS}-d{int(DIFFERENTIAL)}-"
                     f"c{COEFFICIENTS}-fx{FWD_EXP}-b{int(BIAS)}-prob{int(PROBABILISTIC)}-"
                     f"g{round(GAMMA, 4)}-k{round(KAPPA, 4)}-a{round(ALPHA, 4)}-ao{ALPHA_ORDER}-rn{REW_NORM}-"
                     f"p{round(LOSS_REG, 4)}-v{round(VAL_REG, 4)}-m{round(POL_REG, 4)}-"
                     f"s{round(ENT_REG, 4)}-d{round(DIV_REG, 4)}-c{round(CPY_REG, 4)}-"
                     f"mem{MEMORY_SIZE}-delay{DELAY}-type{int(DISCRETE)}-gt{GAME_TYPE}",
            gamma=GAMMA, alpha=ALPHA, kappa=KAPPA, order=ALPHA_ORDER, normalize=REW_NORM,
            rew_reg=REW_REG, loss_reg=LOSS_REG, val_reg=VAL_REG, pol_reg=POL_REG, ent_reg=ENT_REG,
            div_reg=DIV_REG, cpy_reg=CPY_REG,
            noise_reg=NOISE_REG, noise_sigma=NOISE_SIGMA, noise_log=NOISE_LOG,   # examples/misc/noise_reg.py
            validate=True, segr_size=None, use_entropy=True, use_critic=False,
            max_episodes=MEMORY_SIZE, max_steps=1024,
        )
        print("Regularizations:")
        for name, value in vars(trainer).items():
            if "_reg" in name and isinstance(value, (int, float)):
                print(f"\t{name} => {value:.4f}")
        print(f"\tnoise_reg => {trainer.noise_reg:.4f}")
        trainer.set_report_hook(genome_debug)
        if (DIFFERENTIAL or NOISE_REG > 0 or NOISE_LOG) and differential_attention_logger is not None:
            # Compact stats (leak, beta, fitness correlation); charts are grouped in Tensorboard > Custom Scalars
            trainer.add_log_source('diff_attn', differential_attention_logger(population=True, detail=False))

        print(f"starting evaluation: population={len(POPULATION.genomes)}")
        try:
            trainer.learn(
                evaluate, STEPS, EPOCHS,
                accuracy_error=0.2,
                accuracy_type='continuous' if not DISCRETE else 'discrete',
                verbose=True,
                batching='record', record_batch_size=32, batch_size=1024, # NOTE: Reduce/Change if GPU/CPU memory cannot hold
            )
        except KeyboardInterrupt:
            pass

    if test:
        test_best_network()


if __name__ == "__main__":
    run()
