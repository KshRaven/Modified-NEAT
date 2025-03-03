
from ModifiedNEAT.nn.base import Model, NeatModule
from ModifiedNEAT.nn.modules import BufferEncoding, BufferEmbedding, TransformerBase
from ModifiedNEAT.nn.modules import Sequential, Linear, Transpose, Ignore
from ModifiedNEAT.nn.modules import Conv1d, Conv2d, Conv3d, ResidualBlock, ConverBase
from ModifiedNEAT.nn.modules import LayerNorm, RMSNorm, GroupNorm, BatchNorm
from ModifiedNEAT.nn.modules.base import get_conv
from ModifiedNEAT.nn.modules.util import get_tensor_info
from ModifiedNEAT.util.qol import manage_params
from ModifiedNEAT.util.fancy_text import CM, Fore

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


# class Reformer(Model, NeatModule):
#     def __init__(
#             self, inputs: int, pol_out: int, val_out: int, embed_size: int, max_seq_len: int, layers: int, heads: int,
#             kv_heads: int = None, differential=True, dropout: float = 0.1, bias=False, feedback=False,
#             device: torch.device = 'cpu', dtype: torch.dtype = torch.float32, **options):
#         super(Reformer, self).__init__()
#         self.pri_actv           = manage_params(options, 'pri_actv', nn.SiLU())
#         self.sec_actv           = manage_params(options, 'sec_actv', nn.Tanh())
#         self.affine             = manage_params(options, 'affine', True)
#         self.distribution       = manage_params(options, ['distribution', 'dist'], 'normal')
#         self.epsilon            = manage_params(options, 'epsilon', 1e-8)
#         options['sec_actv'] = None
#         self.probabilistic      = True
#
#         # ModifiedNEAT
#         self.feedback_feat = (pol_out + val_out) if feedback else None
#         self.feedback_gain = RMSNorm(pol_out+val_out, self.epsilon, self.affine, device, dtype) if feedback else None
#         self.pol_proj = Transformer(
#             inputs, embed_size, embed_size, max_seq_len, layers, heads, kv_heads, differential, dropout,
#             bias, device, dtype, **options,
#         )
#         self.mean     = Linear(embed_size, pol_out, bias, device, dtype)
#         self.log_std  = Linear(embed_size, pol_out, bias, device, dtype)
#         self.val_proj = Transformer(
#             inputs, embed_size, embed_size, max_seq_len, layers, heads, kv_heads, differential, dropout,
#             bias, device, dtype, **options,
#         )
#         self.decode   = Linear(embed_size, val_out, bias, device, dtype)
#
#         # STATE
#         self.device = device
#         self.dtype = dtype
#         self.eval()
#
#         # ATTRIBUTES
#         self.seq_len = max_seq_len
#         self.single = False
#
#     def get_feedback(self, state: Tensor, keys: Union[int, Iterable[int]] = None):
#         if self.feedback_gain is not None:
#             state[..., -self.feedback_feat:] = self.feedback_gain(state[..., -self.feedback_feat:], keys=keys)
#         return state
#
#     @property
#     def genomes_total(self):
#         return self.pol_proj.genomes_num
#
#     def get_latent(
#             self, model: Transformer, state: Tensor, keys: Union[int, Iterable[int]] = None, idx: int = None,
#             verbose=False, get=False, single=False
#     ):
#         latent = model.forward(self.get_feedback(state, keys=keys), idx, keys, verbose, get, single)
#         if self.pri_actv is not None:
#             latent = self.pri_actv(latent)
#         return latent
#
#     def get_mean(
#             self, latent: Tensor, keys: Union[int, Iterable[int]] = None
#     ) -> Tensor:
#         mean = self.mean(latent, keys=keys)
#         if self.probabilistic:
#             mean = F.sigmoid(mean)
#         elif self.sec_actv is not None:
#             mean = self.sec_actv(mean)
#         return self.reduce(mean)
#
#     def get_std(
#             self, latent: Tensor, keys: Union[int, Iterable[int]] = None
#     ) -> Tensor:
#         std = self.log_std(latent, keys=keys)
#         if self.probabilistic:
#             std = torch.exp(-9.21 + F.sigmoid(std) * 6.91)
#         elif self.sec_actv is not None:
#             std = self.sec_actv(torch.exp(std))
#         return self.reduce(std)
#
#     def get_action(
#             self, state: Tensor, keys: Union[int, Iterable[int]] = None, get=False, single=False
#     ) -> tuple[Tensor, Tensor]:
#         latent      = self.get_latent(self.pol_proj, state, keys, get=get, single=self.single)
#         mean        = self.get_mean(latent, keys=keys)
#         std         = self.get_std(latent, keys=keys) if self.distribution != 'discrete' else None
#         dist        = self.dist(mean, std)
#         action      = dist.sample()
#         log_prob    = dist.log_prob(action)
#         return action, log_prob
#
#     def evaluate_action(
#             self, state: Tensor, action: Tensor, keys: Union[int, Iterable[int]] = None, get=False, single=False
#     ) -> [Tensor, Union[Tensor, None]]:
#         latent      = self.get_latent(self.pol_proj, state, keys, get=get, single=self.single)
#         mean        = self.get_mean(latent, keys=keys)
#         std         = self.get_std(latent, keys=keys) if self.distribution != 'discrete' else None
#         dist        = self.dist(mean, std)
#         log_prob    = dist.log_prob(action)
#         entropy     = dist.entropy()
#         return log_prob, entropy
#
#     def get_policy(
#             self, state: Tensor, keys: Union[int, Iterable[int]] = None, get=False, single=False, **options
#     ) -> Tensor:
#         pos_idx = manage_params(options, ['pos_idx', 'idx'], None)
#         verbose = manage_params(options, 'verbose', None)
#         latent  = self.get_latent(self.pol_proj, state, keys, pos_idx, verbose, get=get, single=single)
#         mean    = self.get_mean(latent, keys=keys)
#         std     = self.get_std(latent, keys=keys) if self.distribution != 'discrete' else None
#         dist    = self.dist(mean, std)
#         action  = dist.sample()
#         return action
#
#     def get_value(
#             self, state: Tensor, keys: Union[int, Iterable[int]] = None, get=False, single=False
#     ) -> Tensor:
#         latent  = self.get_latent(self.val_proj, state, keys, get=get, single=self.single)
#         value   = self.decode(latent, keys=keys)
#         return self.reduce(value)
#
#     def forward(
#             self, observations: Tensor, keys: Union[int, Iterable[int]] = None, get=False, **options
#     ):
#         actions = self.get_policy(observations, keys=keys, get=get, single=self.single, **options)
#         return actions
#
#     def infer(
#             self, tensor: Tensor, pos_idx: int = None, genome_mask: Tensor = None,
#             keys: Union[int, Iterable[int]] = None, verbose: int = None, get=False, single=True
#     ):
#         pos_idx = self.seq_len+pos_idx if pos_idx and pos_idx < 0 else pos_idx
#         tensor = self.forward(tensor, pos_idx=pos_idx, mask=genome_mask, keys=keys,
#                               verbose=verbose, get=get, single=single)
#         return tensor
#
#     def single_mode(self, enable=False):
#         self.single = enable
#
#     def reduce(self, tensor: Tensor):
#         if self.single:
#             tensor = tensor.squeeze(-2)
#         return tensor
#
#     @staticmethod
#     def _addindent(s_, numSpaces):
#         s = s_.split("\n")
#         # don't do anything for single-line stuff
#         if len(s) == 1:
#             return s_
#         first = s.pop(0)
#         s = [(numSpaces * " ") + line for line in s]
#         s = "\n".join(s)
#         s = first + "\n" + s
#         return s
#
#     def __repr__(self):
#         # We treat the extra repr like the sub-module, one item per line
#         extra_lines = []
#         extra_repr = self.extra_repr()
#         # empty string will be split into list ['']
#         if extra_repr:
#             extra_lines = extra_repr.split("\n")
#         child_lines = []
#         for key, module in self._modules.items():
#             mod_str = repr(module)
#             mod_str = self._addindent(mod_str, 2)
#             child_lines.append("(" + key + "): " + mod_str)
#         lines = extra_lines + child_lines
#
#         main_str = self._get_name() + "("
#         if lines:
#             # simple one-liner info, which most builtin Modules will use
#             if len(extra_lines) == 1 and not child_lines:
#                 main_str += extra_lines[0]
#             else:
#                 main_str += "\n  " + "\n  ".join(lines) + "\n"
#
#         main_str += ")"
#         return main_str


class Conver(NeatModule):
    def __init__(
            self, inputs: int, outputs: int, max_seq_len: int, dim_size: int, kernel_size: int, layers: int,
            norm_groups: int, channels: Union[int, list[int]] = None,
            heads: int = None, kv_heads: int = None, differential=True,
            bias=False, device = torch.device('cpu'), dtype: torch.dtype = torch.float32, **options):
        super(Conver, self).__init__()
        if channels is None:
            channels = dim_size
        self.enc_layers         = manage_params(options, ['enc_layers', 'encoder_layers'], 2) + 1
        self.dec_layers         = manage_params(options, ['dec_layers', 'decoder_layers'], 2) + 1
        self.distribution       = manage_params(options, 'distribution', 'normal')
        self.causal_mask        = manage_params(options, 'causal_mask', True)
        self.epsilon            = manage_params(options, 'epsilon', 1e-6)
        self.affine             = manage_params(options, 'affine', True)
        self.probabilistic      = manage_params(options, ['prob', 'probabilistic'], False) and self.distribution != 'discrete'
        self.dec_actv           = manage_params(options, ['dec_actv', 'decoder_activation'], nn.SiLU())
        self.lower_clip         = manage_params(options, 'lower_clip', -20)
        self.upper_clip         = manage_params(options, 'upper_clip', 20)
        self.feedback           = manage_params(options, 'feedback', False)

        # ATTRIBUTES
        self.inputs = inputs if not self.feedback else inputs - self.feedback
        self.outputs = outputs

        # BUILD
        options['image_ndim'] = 1
        Convolution = get_conv((max_seq_len,))
        self.feedback_gain = Sequential(
            Transpose(),
            Convolution(self.feedback, dim_size, kernel_size, stride=1, padding=-1,
                        padding_mode=manage_params(options, 'padding_mode', 'zeros'),
                        device=device, dtype=dtype),
            ResidualBlock(dim_size, self.feedback, kernel_size, norm_groups, bias, device, dtype, **options),
            Transpose(),
        ) if self.feedback and manage_params(options, 'feedback_norm', True) else None
        self.feedback_dropout = Ignore(manage_params(options, ['feedback_drop', 'feedback_dropout'], 0))
        self.encoder = Sequential(
            Transpose(),
            Conv1d(inputs, dim_size, kernel_size, 1, -1,
                   padding_mode=manage_params(options, 'padding_mode', 'zeros'),
                   bias=bias, device=device, dtype=dtype),
            *[
                ResidualBlock(dim_size, dim_size, kernel_size, norm_groups, bias, device, dtype, **options)
                for _ in range(self.enc_layers)
            ]
        )
        self.transformer = ConverBase(
            (max_seq_len,), dim_size, kernel_size, norm_groups, layers, heads, kv_heads, differential, self.causal_mask,
            bias, device, dtype, **options
        )
        decoder = []
        for layer_idx in range(self.dec_layers):
            decoder.append(
                GroupNorm(norm_groups, dim_size, self.epsilon, self.affine, bias, device, dtype)
            )
            if self.dec_actv is not None:
                decoder.append(self.dec_actv)
            if layer_idx != self.dec_layers - 1:
                decoder.append(
                    Conv1d(dim_size, dim_size, kernel_size, padding=-1,
                           padding_mode=manage_params(options, 'padding_mode', 'zeros'),
                           bias=bias, device=device, dtype=dtype)
                )
            else:
                decoder.append(
                    Conv1d(dim_size, outputs if not self.probabilistic else outputs*2, 1, 1,
                           padding_mode=manage_params(options, 'padding_mode', 'zeros'),
                           bias=bias, device=device, dtype=dtype)
                )
                decoder.append(Transpose(-1, -2))
        self.decoder = Sequential(*decoder)
        self.primary_activation = manage_params(options, 'pri_actv', None)
        self.secondary_activation = manage_params(options, 'sec_actv', None)
        self.selector = torch.arange(max_seq_len, device=device, dtype=dtype)

        # STATE
        self.device = device
        self.dtype  = dtype
        self.train()
        self.requires_grad_(False)

    def handle_feedback(self, tensor: Tensor, keys: Union[int, list[int]] = None):
        if self.feedback:
            tensor, feedback = torch.split(tensor, self.inputs, -1)
            assert feedback.shape[-1] == self.feedback
            feedback = self.feedback_dropout(feedback)
            if self.feedback_gain is not None:
                feedback = feedback * self.feedback_gain(feedback, keys=keys)
            tensor = torch.cat((tensor, feedback), -1)
        return tensor

    def forward(self, tensor: Tensor, keys: Union[int, list[int]] = None, pos_idx: int = None,
                verbose: int = None, get=False, single=False, context: Tensor = None, noise: Tensor = None, logits=False):
        if verbose:
            print(get_tensor_info(tensor, f'{self.__class__.__name__} Input', verbose))
        squeeze = tensor.ndim == 3
        if squeeze:
            tensor = tensor.unsqueeze(0)
        if pos_idx is not None:
            position = min(tensor.shape[-2], pos_idx+1)
            tensor = tensor.index_select(-2, self.selector[:position])

        # Expected input shape (genomes, batch_size, seq_len, features), therefore transpose it
        tensor = self.encoder(self.handle_feedback(tensor, keys=keys), keys=keys) # , verbose=verbose)
        tensor = self.transformer(tensor, keys=keys, context=context, pos_idx=pos_idx, verbose=verbose, get=get, single=single)
        # Expected transform shape (genomes, batch_size, dim_size, seq_len), therefore transpose it
        tensor = self.decoder(tensor, keys=keys)

        if squeeze:
            tensor = tensor.squeeze(0)
        if self.probabilistic:
            # Get the distribution
            mean, log_variance = torch.chunk(tensor, 2, dim=-1)
            log_variance = torch.clamp(log_variance, self.lower_clip, self.upper_clip)
            std = log_variance.exp().sqrt()
            if noise is None:
                noise = torch.randn_like(std, device=std.device, dtype=std.dtype)
            tensor = mean + std * noise
        if self.distribution == 'discrete' and not logits:
            tensor = torch.argmax(tensor, -1)
        else:
            if self.secondary_activation is not None:
                tensor = self.secondary_activation(tensor)
        if verbose:
            print(get_tensor_info(tensor, f'{self.__class__.__name__} Output', verbose))

        return tensor

    def get_attention(self):
        a, v = [], []
        for ai, vi in self.transformer.get_attention():
            a.append(ai)
            v.append(vi)
        return a, v

    def infer(self, inputs: Tensor, keys: Union[int, list[int]] = None, pos_idx: int = None, get=False, single=True, verbose=False):
        with torch.no_grad():
            if single:
                pos_idx = -1
            if pos_idx is not None:
                seq_len = inputs.shape[-2]
                pos_idx = seq_len + pos_idx if pos_idx < 0 else pos_idx
                assert 0 < pos_idx < seq_len
            outputs = self.forward(inputs, keys, pos_idx, verbose, get, single)
            return outputs


class Reformer(Model):
    def __init__(

            self, inputs: int, pol_out: int, val_out: int, max_seq_len: int, dim_size: int, layers: int,
            heads: int = None, kv_heads: int = None, differential=True,
            kernel_size=1, norm_groups=1, channels: Union[int, list[int]] = None,
            bias=False, device = torch.device('cpu'), dtype: torch.dtype = torch.float32, **options):
        super(Reformer, self).__init__()
        # ATTRIBUTES
        self.pol_size       = pol_out
        self.val_size       = val_out
        self.seq_len        = max_seq_len
        self.dim_size       = dim_size
        self.distribution   = manage_params(options, ['distribution', 'dist'], 'normal')
        self.epsilon        = manage_params(options, 'epsilon', 1e-8)
        self.probabilistic  = manage_params(options, ['probabilistic', 'prob'], False)
        self.clip_std_min   = manage_params(options, ['clip_min'], None)
        self.clip_std_max   = manage_params(options, ['clip_max'], None)
        self.clip_std       = self.clip_std_min is not None or self.clip_std_max is not None
        self.bias_enabled   = bias

        # BUILD
        self.pri_actv = manage_params(options, 'pri_actv', nn.SiLU())
        self.sec_actv = manage_params(options, 'sec_actv', None)
        options['distribution'] = options['dist'] = 'normal'
        options['sec_actv'] = None
        options['probabilistic'] = options['prob'] = False
        self.pol_proj = Conver(
            inputs, dim_size, max_seq_len, dim_size, kernel_size, layers, norm_groups, channels,
            heads, kv_heads, differential, bias, device, dtype, **options
        )
        self.mean_log_std = Linear(dim_size, pol_out, bias, device, dtype)
        self.val_proj = Conver(
            inputs, dim_size, max_seq_len, dim_size, kernel_size, layers, norm_groups, channels,
            heads, kv_heads, differential, bias, device, dtype, **options
        )
        self.decode = Linear(dim_size, val_out, bias, device, dtype)
        self.mean_actv = manage_params(options, 'mean_actv', None)
        self.std_actv = manage_params(options, 'std_actv', None)

        # STATE
        self.single_pass    = False
        self.logits_pass    = True
        self.idx: int       = None
        self.device         = device
        self.dtype          = dtype
        self.train()

    def single_mode(self, enable=False):
        self.single_pass = enable

    def logits_mode(self, disable=False):
        self.logits_pass = not disable

    def position_idx_constant(self, index: int = None):
        self.idx = index

    def reduce(self, tensor: Tensor):
        if self.single_pass:
            tensor = tensor.squeeze(-2)
        # if not self.logits_pass and self.distribution == 'discrete':
        #     tensor = tensor.argmax(-1, False)
        return tensor

    def get_latent(self, module: Conver, state: Tensor, keys: Union[int, list[int]], pos_idx: int = None, verbose=False, get=False, single=False):
        latent = module.forward(state, keys, pos_idx, verbose, get, single, logits=self.logits_pass)
        if self.pri_actv is not None:
            latent = self.pri_actv(latent)
        return latent

    def get_mean(self, latent: Tensor, keys: Union[int, Iterable[int]] = None) -> Tensor:
        mean = latent[..., :self.pol_size]
        if self.mean_actv is not None:
            mean = self.mean_actv(mean)
        return self.reduce(mean)

    def get_std(self, latent: Tensor, keys: Union[int, Iterable[int]] = None) -> Union[Tensor, None]:
        if self.distribution != 'discrete':
            log_std = latent[..., -self.pol_size:]
            if self.clip_std:
                log_std = F.hardtanh(log_std, self.clip_std_min, self.clip_std_max)
            std = torch.exp(log_std)
            if self.std_actv is not None:
                std = self.std_actv(std)
            return self.reduce(std)
        else:
            return None

    def get_action(self, state: Tensor, keys: Union[int, Iterable[int]] = None) -> tuple[Tensor, Tensor]:
        latent      = self.get_latent(self.pol_proj, state, keys, self.idx, single=self.single_pass)
        source      = self.mean_log_std(latent, keys=keys)
        mean        = self.get_mean(source, keys)
        std         = self.get_std(source, keys)
        dist        = self.dist(mean, std, None, None)
        action      = dist.sample()
        log_prob    = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, state: Tensor, action: Tensor, keys: Union[int, Iterable[int]] = None) -> [Tensor, Union[Tensor, None]]:
        latent      = self.get_latent(self.pol_proj, state, keys, self.idx, single=self.single_pass)
        source      = self.mean_log_std(latent, keys=keys)
        mean        = self.get_mean(source, keys)
        std         = self.get_std(source, keys)
        dist        = self.dist(mean, std, None, None)
        log_prob    = dist.log_prob(action)
        entropy     = dist.entropy()
        return log_prob, entropy

    def get_policy(self, state: Tensor, keys: Union[int, Iterable[int]] = None, **options) -> Tensor:
        pos_idx = manage_params(options, ['pos_idx', 'idx'], None)
        verbose = manage_params(options, 'verbose', None)
        get     = manage_params(options, 'get', False)
        single  = manage_params(options, 'single', self.single_pass)
        latent  = self.get_latent(self.pol_proj, state, keys, pos_idx, verbose, get, single)
        source  = self.mean_log_std(latent, keys=keys)
        mean    = self.get_mean(source, keys)
        std     = self.get_std(source, keys)
        if verbose:
            print(f"\n{CM('Mean =>', Fore.LIGHTCYAN_EX)}\n{mean}, \n\tdim = {mean.shape}")
            if std is not None:
                print(f"\n{CM('Std =>', Fore.LIGHTCYAN_EX)}\n{std}, \n\tdim = {std.shape}")
        dist    = self.dist(mean, std, None, None)
        action  = dist.sample()
        if single:
            action = action.squeeze(-2)
        return action

    def get_value(self, state: Tensor, keys: Union[int, Iterable[int]] = None, **options) -> Tensor:
        pos_idx = manage_params(options, ['pos_idx', 'idx'], None)
        verbose = manage_params(options, 'verbose', None)
        get     = manage_params(options, 'get', False)
        single  = manage_params(options, 'single', self.single_pass)
        latent  = self.get_latent(self.val_proj, state, keys, pos_idx, verbose, get, single)
        value   = self.decode(latent, keys=keys)
        return self.reduce(value)

    @staticmethod
    def randomize(tensor: Tensor, noise: Tensor = None):
        if noise is None:
            return tensor * torch.randn_like(tensor)
        else:
            return tensor * noise

    def learn(self, inputs: Tensor, keys: Union[int, Iterable[int]] = None, verbose: int = None):
        return self.forward(inputs, keys, verbose=verbose)

    def forward(self, inputs: Tensor, keys: Union[int, Iterable[int]] = None, **options):
        pos_idx = manage_params(options, ['pos_idx', 'idx'], None)
        verbose = manage_params(options, 'verbose', None)
        get     = manage_params(options, 'get', False)
        single  = manage_params(options, 'single', False)
        randomize = manage_params(options, 'randomize', True)
        noise   = manage_params(options, 'noise', None)

        latent  = self.get_latent(self.pol_proj, inputs, keys, pos_idx, verbose, get, single)
        source  = self.mean_log_std(latent, keys=keys)
        mean    = self.get_mean(source, keys)
        std     = self.get_std(source, keys)

        if randomize:
            outputs = mean + self.randomize(std, noise)
        else:
            outputs = mean
        return outputs

    def infer(self, inputs: Tensor, keys: Union[int, Iterable[int]] = None, pos_idx: int = None, verbose: int = None, get=False, single=True):
        with torch.no_grad():
            if single:
                pos_idx = -1
            if pos_idx is not None:
                seq_len = inputs.shape[-2]
                pos_idx = seq_len + pos_idx if pos_idx < 0 else pos_idx
                assert 0 < pos_idx < seq_len
            inputs = self.forward(inputs, keys=keys, pos_idx=pos_idx, verbose=verbose, get=get, single=single)
            return inputs

    def policy_params(self):
        params = list(self.pol_proj.parameters())
        params.extend(list(self.mean_log_std.parameters()))
        return params

    def value_params(self):
        params = list(self.val_proj.parameters())
        params.extend(list(self.decode.parameters()))
        return params
