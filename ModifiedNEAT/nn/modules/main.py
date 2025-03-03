
from ModifiedNEAT.nn.base import Model
from ModifiedNEAT.nn.modules.base import NeatModule, Linear, RMSNorm
from ModifiedNEAT.nn.modules.sub import BufferEmbedding, BufferEncoding, TransformerBase
from ModifiedNEAT.util.qol import manage_params

from torch import Tensor
from typing import Union, Iterable

import torch.nn as nn
import torch.nn.functional as F
import torch


class Transformer(NeatModule):
    def __init__(
            self, inputs: int, outputs: int, embed_size: int, max_seq_len: int, layers: int,
            heads: int = None, kv_heads: int = None, differential=True, dropout: int = None,
            bias=False, device = torch.device('cpu'), dtype: torch.dtype = torch.float32, **options):
        super(Transformer, self).__init__()
        self.distribution       = manage_params(options, 'distribution', 'normal')
        self.fwd_exp            = manage_params(options, 'fwd_exp', None)
        self.epsilon            = manage_params(options, 'epsilon', 1e-8)
        self.constant           = manage_params(options, 'constant', 10000)
        self.affine             = manage_params(options, 'affine', True)
        self.causal_mask        = manage_params(options, 'causal_mask', True)
        self.primary_activation = manage_params(options, 'pri_actv', nn.SiLU())
        self.secondary_activation = manage_params(options, 'sec_actv', None)

        # ModifiedNEAT
        self.embedder    = BufferEmbedding(inputs, embed_size, bias, device, dtype)
        self.encoder     = BufferEncoding(max_seq_len, embed_size, bias, device, dtype)
        self.transformer = TransformerBase(
            max_seq_len, embed_size, layers, heads, kv_heads, self.fwd_exp, differential,
            self.constant, self.epsilon, self.affine, self.causal_mask, dropout, bias, device, dtype
        )
        self.dec_norm   = RMSNorm(embed_size, self.epsilon, self.affine, device, dtype)
        output_dim      = outputs if self.distribution != 'discrete' else 2 ** outputs
        self.decode     = Linear(embed_size, output_dim, bias, device, dtype)
        if dropout is None:
            dropout = 0
        self.dropout    = nn.Dropout(dropout)

        # STATE
        self.device = device
        self.dtype  = dtype
        self.eval()

        # ATTRIBUTES
        self.max_seq_len = max_seq_len

    @property
    def genomes_total(self):
        return self.decode.genome_num

    def forward(self, tensor: Tensor, pos_idx: int = None, keys: Union[int, Iterable[int]] = None,
                verbose: int = None, get=False, single=False):
        if pos_idx is not None:
            tensor = tensor[:, :pos_idx+1]
        if verbose:
            print(f"\nTransformer Input =>\n{tensor}\n\tdim = {tensor.shape}")

        tensor = self.embedder(tensor, keys=keys, verbose=verbose)
        if self.primary_activation is not None:
            tensor = self.primary_activation(tensor)
        tensor = self.dropout(self.encoder(tensor, keys=keys))
        tensor = self.transformer(tensor, keys=keys, verbose=verbose, get=get, single=single)
        tensor = self.decode(self.dec_norm(tensor, keys=keys), keys=keys)
        if self.secondary_activation is not None:
            tensor = self.secondary_activation(tensor)
        if verbose:
            print(f"\nTransformer Output =>\n{tensor}\n\tdim = {tensor.shape}")
        if self.distribution == 'discrete':
            tensor = torch.argmax(tensor, -1)
        return tensor

    def get_attention(self):
        a, v = [], []
        for ai, vi in self.transformer.get_attention():
            a.append(ai)
            v.append(vi)
        return a, v

    def infer(self, tensor: Tensor, pos_idx: int = None, keys: Union[int, Iterable[int]] = None, verbose=False):
        pos_idx = self.max_seq_len + pos_idx if pos_idx is not None and pos_idx < 0 else pos_idx
        if pos_idx is None:
            pos_idx = self.max_seq_len-1
        sequence_dim = -2 if self.distribution == 'discrete' else -1
        tokens_current = min(tensor.shape[sequence_dim], pos_idx+1)
        tensor = tensor[..., :tokens_current+1, :]
        for idx in range(tokens_current):
            current_idx = tokens_current+idx
            token = self.forward(tensor, current_idx, keys, verbose)
            tensor = torch.cat((tensor, token), dim=sequence_dim)
        return tensor


class Reformer(Model, NeatModule):
    def __init__(
            self, inputs: int, pol_out: int, val_out: int, embed_size: int, max_seq_len: int, layers: int, heads: int,
            kv_heads: int = None, differential=True, dropout: float = 0.1, bias=False, feedback=False,
            device: torch.device = 'cpu', dtype: torch.dtype = torch.float32, **options):
        super(Reformer, self).__init__()
        self.pri_actv           = manage_params(options, 'pri_actv', nn.SiLU())
        self.sec_actv           = manage_params(options, 'sec_actv', nn.Tanh())
        self.affine             = manage_params(options, 'affine', True)
        self.distribution       = manage_params(options, ['distribution', 'dist'], 'normal')
        self.epsilon            = manage_params(options, 'epsilon', 1e-8)
        options['sec_actv'] = None
        self.probabilistic      = True

        # ModifiedNEAT
        self.feedback_feat = (pol_out + val_out) if feedback else None
        self.feedback_gain = RMSNorm(pol_out+val_out, self.epsilon, self.affine, device, dtype) if feedback else None
        self.pol_proj = Transformer(
            inputs, embed_size, embed_size, max_seq_len, layers, heads, kv_heads, differential, dropout,
            bias, device, dtype, **options,
        )
        self.mean     = Linear(embed_size, pol_out, bias, device, dtype)
        self.log_std  = Linear(embed_size, pol_out, bias, device, dtype)
        self.val_proj = Transformer(
            inputs, embed_size, embed_size, max_seq_len, layers, heads, kv_heads, differential, dropout,
            bias, device, dtype, **options,
        )
        self.decode   = Linear(embed_size, val_out, bias, device, dtype)

        # STATE
        self.device = device
        self.dtype = dtype
        self.eval()

        # ATTRIBUTES
        self.seq_len = max_seq_len
        self.single = False

    def get_feedback(self, state: Tensor, keys: Union[int, Iterable[int]] = None):
        if self.feedback_gain is not None:
            state[..., -self.feedback_feat:] = self.feedback_gain(state[..., -self.feedback_feat:], keys=keys)
        return state

    @property
    def genomes_total(self):
        return self.pol_proj.genomes_num

    def get_latent(
            self, model: Transformer, state: Tensor, keys: Union[int, Iterable[int]] = None, idx: int = None,
            verbose=False, get=False, single=False
    ):
        latent = model.forward(self.get_feedback(state, keys=keys), idx, keys, verbose, get, single)
        if self.pri_actv is not None:
            latent = self.pri_actv(latent)
        return latent

    def get_mean(
            self, latent: Tensor, keys: Union[int, Iterable[int]] = None
    ) -> Tensor:
        mean = self.mean(latent, keys=keys)
        if self.probabilistic:
            mean = F.sigmoid(mean)
        elif self.sec_actv is not None:
            mean = self.sec_actv(mean)
        return self.reduce(mean)

    def get_std(
            self, latent: Tensor, keys: Union[int, Iterable[int]] = None
    ) -> Tensor:
        std = self.log_std(latent, keys=keys)
        if self.probabilistic:
            std = torch.exp(-9.21 + F.sigmoid(std) * 6.91)
        elif self.sec_actv is not None:
            std = self.sec_actv(torch.exp(std))
        return self.reduce(std)

    def get_action(
            self, state: Tensor, keys: Union[int, Iterable[int]] = None, get=False, single=False
    ) -> tuple[Tensor, Tensor]:
        latent      = self.get_latent(self.pol_proj, state, keys, get=get, single=self.single)
        mean        = self.get_mean(latent, keys=keys)
        std         = self.get_std(latent, keys=keys) if self.distribution != 'discrete' else None
        dist        = self.dist(mean, std)
        action      = dist.sample()
        log_prob    = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(
            self, state: Tensor, action: Tensor, keys: Union[int, Iterable[int]] = None, get=False, single=False
    ) -> [Tensor, Union[Tensor, None]]:
        latent      = self.get_latent(self.pol_proj, state, keys, get=get, single=self.single)
        mean        = self.get_mean(latent, keys=keys)
        std         = self.get_std(latent, keys=keys) if self.distribution != 'discrete' else None
        dist        = self.dist(mean, std)
        log_prob    = dist.log_prob(action)
        entropy     = dist.entropy()
        return log_prob, entropy

    def get_policy(
            self, state: Tensor, keys: Union[int, Iterable[int]] = None, get=False, single=False, **options
    ) -> Tensor:
        pos_idx = manage_params(options, ['pos_idx', 'idx'], None)
        verbose = manage_params(options, 'verbose', None)
        latent  = self.get_latent(self.pol_proj, state, keys, pos_idx, verbose, get=get, single=single)
        mean    = self.get_mean(latent, keys=keys)
        std     = self.get_std(latent, keys=keys) if self.distribution != 'discrete' else None
        dist    = self.dist(mean, std)
        action  = dist.sample()
        return action

    def get_value(
            self, state: Tensor, keys: Union[int, Iterable[int]] = None, get=False, single=False
    ) -> Tensor:
        latent  = self.get_latent(self.val_proj, state, keys, get=get, single=self.single)
        value   = self.decode(latent, keys=keys)
        return self.reduce(value)

    def forward(
            self, observations: Tensor, keys: Union[int, Iterable[int]] = None, get=False, **options
    ):
        actions = self.get_policy(observations, keys=keys, get=get, single=self.single, **options)
        return actions

    def infer(
            self, tensor: Tensor, pos_idx: int = None, genome_mask: Tensor = None,
            keys: Union[int, Iterable[int]] = None, verbose: int = None, get=False, single=True
    ):
        pos_idx = self.seq_len+pos_idx if pos_idx and pos_idx < 0 else pos_idx
        tensor = self.forward(tensor, pos_idx=pos_idx, mask=genome_mask, keys=keys,
                              verbose=verbose, get=get, single=single)
        return tensor

    def single_mode(self, enable=False):
        self.single = enable

    def reduce(self, tensor: Tensor):
        if self.single:
            tensor = tensor.squeeze(-2)
        return tensor

    @staticmethod
    def _addindent(s_, numSpaces):
        s = s_.split("\n")
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * " ") + line for line in s]
        s = "\n".join(s)
        s = first + "\n" + s
        return s

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = self._addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str
