
from ..base import Model, NeatModule
from . import SequenceEncoding, BufferEncoding, BufferEmbedding, TransformerBase
from . import Sequential, Linear, Transpose, Ignore
from . import Conv1d, Conv2d, Conv3d, ResidualBlock, ConverBase
from . import LayerNorm, RMSNorm, GroupNorm, BatchNorm
from .base import get_conv
from .util import get_tensor_info
from ...util.qol import manage_params
from ...util.fancy_text import CM, Fore

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
        # self.distribution       = manage_params(options, 'distribution', 'normal')
        # self.fwd_exp            = manage_params(options, 'fwd_exp', None)
        # self.epsilon            = manage_params(options, 'epsilon', 1e-8)
        # self.constant           = manage_params(options, 'constant', 10000)
        # self.affine             = manage_params(options, 'affine', True)
        # self.causal_mask        = manage_params(options, 'causal_mask', True)
        # self.primary_activation = manage_params(options, 'pri_actv', nn.SiLU())
        # self.secondary_activation = manage_params(options, 'sec_actv', None)
        #
        # # ModifiedNEAT
        # self.embedder    = BufferEmbedding(inputs, embed_size, bias, device, dtype)
        # self.encoder     = BufferEncoding(max_seq_len, embed_size, bias, device, dtype)
        # self.transformer = TransformerBase(
        #     max_seq_len, embed_size, layers, heads, kv_heads, self.fwd_exp, differential,
        #     self.constant, self.epsilon, self.affine, self.causal_mask, dropout, bias, device, dtype
        # )
        # self.dec_norm   = RMSNorm(embed_size, self.epsilon, self.affine, device, dtype)
        # output_dim      = outputs if self.distribution != 'discrete' else 2 ** outputs
        # self.decode     = Linear(embed_size, output_dim, bias, device, dtype)
        # if dropout is None:
        #     dropout = 0
        # self.dropout    = nn.Dropout(dropout)
        #
        # # STATE
        # self.device = device
        # self.dtype  = dtype
        # self.eval()
        #
        # # ATTRIBUTES
        # self.max_seq_len = max_seq_len
        raise NotImplementedError()

    # @property
    # def genomes_total(self):
    #     return self.decode.genome_num
    #
    # def forward(self, tensor: Tensor, pos_idx: int = None, keys: Union[int, Iterable[int]] = None,
    #             verbose: int = None, get=False, single=False):
    #     if pos_idx is not None:
    #         tensor = tensor[:, :pos_idx+1]
    #     if verbose:
    #         print(f"\nTransformer Input =>\n{tensor}\n\tdim = {tensor.shape}")
    #
    #     tensor = self.embedder(tensor, keys=keys, verbose=verbose)
    #     if self.primary_activation is not None:
    #         tensor = self.primary_activation(tensor)
    #     tensor = self.dropout(self.encoder(tensor, keys=keys))
    #     tensor = self.transformer(tensor, keys=keys, verbose=verbose, get=get, single=single)
    #     tensor = self.decode(self.dec_norm(tensor, keys=keys), keys=keys)
    #     if self.secondary_activation is not None:
    #         tensor = self.secondary_activation(tensor)
    #     if verbose:
    #         print(f"\nTransformer Output =>\n{tensor}\n\tdim = {tensor.shape}")
    #     if self.distribution == 'discrete':
    #         tensor = torch.argmax(tensor, -1)
    #     return tensor
    #
    # def get_attention(self):
    #     a, v = [], []
    #     for ai, vi in self.transformer.get_attention():
    #         a.append(ai)
    #         v.append(vi)
    #     return a, v
    #
    # def infer(self, tensor: Tensor, pos_idx: int = None, keys: Union[int, Iterable[int]] = None, verbose=False):
    #     pos_idx = self.max_seq_len + pos_idx if pos_idx is not None and pos_idx < 0 else pos_idx
    #     if pos_idx is None:
    #         pos_idx = self.max_seq_len-1
    #     sequence_dim = -2 if self.distribution == 'discrete' else -1
    #     tokens_current = min(tensor.shape[sequence_dim], pos_idx+1)
    #     tensor = tensor[..., :tokens_current+1, :]
    #     for idx in range(tokens_current):
    #         current_idx = tokens_current+idx
    #         token = self.forward(tensor, current_idx, keys, verbose)
    #         tensor = torch.cat((tensor, token), dim=sequence_dim)
    #     return tensor


class Conver(NeatModule):
    def __init__(
            self, inputs: int, outputs: int, max_seq_len: int, dim_size: int, kernel_size: int, layers: int,
            norm_groups: int, channels: Union[int, list[int]] = None,
            heads: int = None, kv_heads: int = None, differential=True,
            bias=False, device = torch.device('cpu'), dtype: torch.dtype = torch.float32, **options):
        super(Conver, self).__init__()
        if channels is None:
            channels = dim_size
        self.enc_layers         = manage_params(options, ['enc_layers', 'encoder_layers'], 0)
        self.dec_layers         = manage_params(options, ['dec_layers', 'decoder_layers'], 0) + 1
        self.distribution       = manage_params(options, 'distribution', 'normal')
        self.causal_mask        = manage_params(options, 'causal_mask', True)
        self.epsilon            = manage_params(options, 'epsilon', 1e-9)
        self.affine             = manage_params(options, 'affine', True)
        self.probabilistic      = manage_params(options, ['prob', 'probabilistic'], False) and self.distribution != 'discrete'
        self.lower_clip         = manage_params(options, 'lower_clip', -20)
        self.upper_clip         = manage_params(options, 'upper_clip', 20)
        self.feedback           = manage_params(options, 'feedback', False)
        self.init_kernel_size   = manage_params(options, 'init_kernel_size', kernel_size)
        self.trans_kernel_size  = manage_params(options, 'trans_kernel_size', 1)

        # ATTRIBUTES
        self.inputs = inputs if not self.feedback else inputs - self.feedback
        self.outputs = outputs

        # BUILD
        self.pri_actv = self.activation = manage_params(options, ['activation', 'actv', 'pri_actv'], nn.SiLU())
        self.sec_actv = manage_params(options, 'sec_actv', None)
        options['image_ndim'] = 1
        Convolution = get_conv((max_seq_len,))
        self.feedback_gain = Sequential(
            Transpose(),
            Convolution(self.feedback, dim_size, 1, stride=1, padding=-1,
                        padding_mode=manage_params(options, 'padding_mode', 'zeros'),
                        bias=True, device=device, dtype=dtype),
            GroupNorm(dim_size, dim_size, self.epsilon, self.affine, False, device, dtype),
            Convolution(dim_size, self.feedback, 1, stride=1, padding=-1,
                        padding_mode=manage_params(options, 'padding_mode', 'zeros'),
                        bias=bias, device=device, dtype=dtype),
            nn.Tanh(),
            Transpose(),
        ) if self.feedback and manage_params(options, 'feedback_gain', False) else None
        self.feedback_dropout = Ignore(manage_params(options, ['feedback_drop', 'feedback_dropout'], 0))
        self.encoder = Sequential(
            Transpose(),
            Conv1d(inputs, dim_size, self.init_kernel_size, 1, -1,
                   padding_mode=manage_params(options, 'padding_mode', 'zeros'),
                   bias=True, device=device, dtype=dtype),
            *[
                ResidualBlock(dim_size, dim_size, kernel_size, norm_groups, bias, device, dtype, **options)
                for _ in range(self.enc_layers)
            ],
        )
        self.transformer = ConverBase(
            (max_seq_len,), dim_size, self.trans_kernel_size, norm_groups, layers, heads, kv_heads, differential,
            self.causal_mask, False, device, dtype, **options
        )
        decoder = []
        for layer_idx in range(self.dec_layers):
            if layer_idx != self.dec_layers - 1:
                decoder.append(
                    ResidualBlock(dim_size, dim_size, 1, norm_groups, bias, device, dtype, **options)
                )
            else:
                decoder.extend([
                    GroupNorm(norm_groups, dim_size, self.epsilon, self.affine, False, device, dtype),
                    self.activation,
                    Conv1d(
                        dim_size, outputs, 1, 1,
                        padding_mode=manage_params(options, 'padding_mode', 'zeros'),
                        bias=bias, device=device, dtype=dtype
                    ),

                ])
                if self.sec_actv is not None:
                    decoder.append(self.sec_actv)
                decoder.append(Transpose(-1, -2))
        self.decoder = Sequential(*decoder)
        self.selector = torch.arange(max_seq_len, device=device, dtype=torch.int)

        # STATE
        self.device = device
        self.dtype  = dtype
        self.train()
        self.requires_grad_(False)

    def handle_feedback(self, tensor: Tensor, keys: Union[int, list[int]] = None):
        if self.feedback:
            tensor, feedback = torch.split(tensor, self.inputs, -1)
            try:
                assert feedback.shape[-1] == self.feedback
            except Exception as e:
                print(f"feedback = {feedback.shape}, feedback_dims = {self.feedback}")
                raise e
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
            if self.sec_actv is not None:
                tensor = self.sec_actv(tensor)
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
        self.affine         = manage_params(options, 'affine', True)
        self.probabilistic  = manage_params(options, ['probabilistic', 'prob'], True)
        self.clip_std_min   = manage_params(options, ['clip_min'], None)
        self.clip_std_max   = manage_params(options, ['clip_max'], None)
        self.clip_std       = self.clip_std_min is not None or self.clip_std_max is not None
        self.bias_enabled   = bias

        # BUILD
        self.pri_actv = self.activation = manage_params(options, ['activation', 'actv', 'pri_actv'], nn.SiLU())
        self.sec_actv = manage_params(options, 'sec_actv', None)
        options['distribution'] = options['dist'] = 'normal'
        options['sec_actv'] = None
        options['probabilistic'] = options['prob'] = False
        self.pol_proj = Conver(
            inputs, dim_size, max_seq_len, dim_size, kernel_size, layers, norm_groups, channels,
            heads, kv_heads, differential, bias, device, dtype, **options
        )
        self.mean_log_std = Sequential(
            RMSNorm(dim_size, self.epsilon, self.affine, device, dtype),
            self.pri_actv,
            Linear(dim_size, pol_out*(2 if self.probabilistic else 1), True, device, dtype),
        )
        if manage_params(options, 'get_values', False):
            self.val_proj = Conver(
                inputs, dim_size, max_seq_len, dim_size, kernel_size, layers, norm_groups, channels,
                heads, kv_heads, differential, bias, device, dtype, **options
            )
            self.decode = Linear(dim_size, val_out, True, device, dtype)
        else:
            self.val_proj = None
            self.decode = None
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
        if self.probabilistic and self.distribution != 'discrete':
            log_std = latent[..., -self.pol_size:]
            if self.std_actv is not None:
                log_std = self.std_actv(log_std)
            if self.clip_std:
                if not isinstance(self.std_actv, nn.Sigmoid):
                    log_std = torch.clamp(log_std, self.clip_std_min, self.clip_std_max)
                else:
                    log_std = log_std * (self.clip_std_max-self.clip_std_min) + self.clip_std_min
            std = torch.pow(10.0, log_std)
            return self.reduce(std)
        else:
            return None

    def get_action(self, state: Tensor, keys: Union[int, Iterable[int]] = None) -> tuple[Tensor, Tensor]:
        latent      = self.get_latent(self.pol_proj, state, keys, self.idx, single=self.single_pass)
        source      = self.mean_log_std(latent, keys=keys)
        mean        = self.get_mean(source, keys)
        std         = self.get_std(source, keys)
        dist        = self.dist(mean, std, None, None)
        action      = dist.sample() if self.probabilistic else mean
        if self.sec_actv is not None:
            action = self.sec_actv(action)
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
            print(get_tensor_info(mean, f"{self.__class__.__name__} Mean", verbose))
            if std is not None:
                print(get_tensor_info(std, f"{self.__class__.__name__} StdDev", verbose))
        dist    = self.dist(mean, std, None, None)
        action  = dist.sample() if self.probabilistic else mean
        if self.sec_actv is not None:
            action = self.sec_actv(action)
        if single:
            action = action.squeeze(-2)
        return action

    def get_value(self, state: Tensor, keys: Union[int, Iterable[int]] = None, **options) -> Tensor:
        if self.val_proj is not None:
            pos_idx = manage_params(options, ['pos_idx', 'idx'], None)
            verbose = manage_params(options, 'verbose', None)
            get     = manage_params(options, 'get', False)
            single  = manage_params(options, 'single', self.single_pass)
            latent  = self.get_latent(self.val_proj, state, keys, pos_idx, verbose, get, single)
            value   = self.decode(latent, keys=keys)
            return self.reduce(value)
        else:
            raise NotImplementedError(f"Value modules were not initialized")

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
        randomize = manage_params(options, 'randomize', False)
        noise   = manage_params(options, 'noise', None)

        latent  = self.get_latent(self.pol_proj, inputs, keys, pos_idx, verbose, get, single)
        source  = self.mean_log_std(latent, keys=keys)
        mean    = self.get_mean(source, keys)
        std     = self.get_std(source, keys)

        if randomize:
            outputs = mean + self.randomize(std, noise)
        else:
            outputs = mean
        if self.sec_actv is not None:
            outputs = self.sec_actv(outputs)
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
