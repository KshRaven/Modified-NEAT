from typing import Union

from build.nn.base import handle_input_mask, handle_output_mask, handle_input_dims, handle_output_dims, Model
from build.models.base import Linear, LayerNorm
from build.models.sub import BufferEncoding, BufferEmbedding, MultiHeadSelfAttention
from build.util.qol import manage_params

from torch import Tensor

import torch.nn as nn
# import torch.nn.functional as F
import torch


class MiniFormer(nn.Module):
    def __init__(
            self, inputs: int, outputs: int, embed_size: int, max_seq_len: int, layers: int, heads: int,
                 kv_heads: int = None, dropout: float = 0.1, bias=False,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32, **options):
        super(MiniFormer, self).__init__()
        self.distribution       = manage_params(options, 'distribution', 'discrete')
        self.epsilon            = manage_params(options, 'epsilon', 1e-10)
        self.constant           = manage_params(options, 'constant', 10000)
        self.primary_activation = manage_params(options, 'pri_actv', nn.ReLU())
        self.secondary_activation = manage_params(options, 'sec_actv', None)

        # BUILD
        self.embed      = BufferEmbedding(inputs, embed_size, bias, device, dtype, self.primary_activation)
        self.encode     = BufferEncoding(max_seq_len, embed_size, bias, device, dtype, None)
        self.att_norm   = nn.ModuleList([
            LayerNorm(embed_size, self.epsilon, True, bias, device, dtype)
        ])
        self.attention  = nn.ModuleList([
            MultiHeadSelfAttention(max_seq_len, embed_size, heads, kv_heads, self.constant, True,
                                   bias, device, dtype, self.primary_activation)
            for _ in range(layers)
        ])
        self.dec_norm   = LayerNorm(embed_size, self.epsilon, True, bias, device, dtype)
        output_dim      = outputs if self.distribution != 'discrete' else 2 ** outputs
        self.decode     = Linear(embed_size, output_dim, bias, device, dtype, self.secondary_activation, False)
        self.dropout    = nn.Dropout(dropout)

        # STATE
        self.device = device
        self.dtype  = dtype
        self.eval()

        # ATTRIBUTES
        self.seq_len = max_seq_len

    @property
    def genomes_total(self):
        return self.decode.genomes_num

    def forward(self, inputs: Tensor, idx: int = None, genome_mask: Tensor = None, verbose: int = None):
        tensor, squeeze = handle_input_dims(inputs, self.genomes_total)
        tensor, genome_mask = handle_input_mask(tensor, genome_mask, -2)
        if idx is not None:
            tensor = tensor[:, :idx+1]
        if verbose:
            print(f"\nMiniFormer Input =>\n{tensor}\n\tdim = {tensor.shape}")
        tensor = self.dropout(self.encode(self.embed(tensor, mask=genome_mask, verbose=verbose), verbose=verbose))
        for x, (norm, block) in enumerate(zip(self.att_norm, self.attention)):
            query = tensor if idx is None else tensor[:, idx:idx+1]
            tensor = self.dropout(block(norm(tensor), context=query, mask=genome_mask, verbose=verbose if x == 0 else False) + query)
        tensor = self.decode(self.dec_norm(tensor), mask=genome_mask)
        if squeeze:
            tensor = tensor.squeeze(0)
        if verbose:
            print(f"\nMiniFormer Output =>\n{tensor}\n\tdim = {tensor.shape}")
        tensor = handle_output_mask(tensor, genome_mask, -2)
        outputs = handle_output_dims(tensor, squeeze, self.genomes_total)
        if self.distribution == 'discrete':
            outputs = torch.argmax(outputs, -1)
        return outputs

    def infer(self, tensor: Tensor, pos_idx: int = None, genome_mask: Tensor = None, verbose=False):
        idx = self.seq_len+pos_idx if pos_idx and pos_idx < 0 else pos_idx
        tensor = self.forward(tensor, idx, genome_mask, verbose)
        return tensor


class Reformer(Model):
    def __init__(
            self, inputs: int, pol_out: int, val_out: int, embed_size: int, max_seq_len: int, layers: int, heads: int,
            kv_heads: int = None, dropout: float = 0.1, bias=False,
            device: torch.device = 'cpu', dtype: torch.dtype = torch.float32, **options):
        super(Reformer, self).__init__()
        self.secondary_activation = manage_params(options, 'sec_actv', None)

        # BUILD
        self.distribution = manage_params(options, ['distribution', 'dist'], 'discrete')
        options['distribution'] = 'normal'
        options['sec_actv'] = manage_params(options, 'pri_actv', None)
        self.pol_proj = MiniFormer(
            inputs, embed_size, embed_size, max_seq_len, layers, heads, kv_heads, dropout,
            bias, device, dtype, **options,
        )
        self.mean     = Linear(embed_size, pol_out, bias, device, dtype, self.secondary_activation)
        self.log_std  = Linear(embed_size, pol_out, bias, device, dtype, nn.Sigmoid())
        self.val_proj = MiniFormer(
            inputs, embed_size, embed_size, max_seq_len, layers, heads, kv_heads, dropout,
            bias, device, dtype, **options,
        )
        self.decode   = Linear(embed_size, val_out, bias, device, dtype, None)

        # STATE
        self.device = device
        self.dtype = dtype
        self.eval()

        # ATTRIBUTES
        self.seq_len = max_seq_len

    @property
    def genomes_total(self):
        return self.decode.genomes_num

    def dist(self, m: Tensor, s: Tensor):
        if self.distribution == 'discrete':
            distribution = torch.distributions.Categorical(torch.softmax(m, -1))
        elif self.distribution == 'normal':
            distribution = torch.distributions.Normal(m, s)
        elif self.distribution == 'mult_var_normal':
            cov = torch.diag_embed(s**2)
            distribution = torch.distributions.MultivariateNormal(m, cov)
        else:
            raise NotImplementedError(f"Unsupported distribution")
        return distribution

    def get_mean(self, latent: Tensor) -> Tensor:
        return self.mean(latent)

    def get_std(self, latent: Tensor) -> Tensor:
        return torch.pow(10, -6 + self.log_std(latent) * 4.33)

    def get_action(self, state: Tensor) -> tuple[Tensor, Tensor]:
        latent      = self.pol_proj(state)
        mean        = self.get_mean(latent)
        std         = self.get_std(latent) if self.distribution != 'discrete' else None
        dist        = self.dist(mean, std)
        action      = dist.sample()
        log_prob    = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, state: Tensor, action: Tensor) -> [Tensor, Union[Tensor, None]]:
        latent      = self.pol_proj(state)
        mean        = self.get_mean(latent)
        std         = self.get_std(latent) if self.distribution != 'discrete' else None
        dist        = self.dist(mean, std)
        log_prob    = dist.log_prob(action)
        entropy     = dist.entropy()
        return log_prob, entropy

    def get_policy(self, state: Tensor, **options) -> Tensor:
        pos_idx = manage_params(options, ['pos_idx', 'idx'], None)
        mask    = manage_params(options, ['genome_mask', 'mask'], None)
        verbose = manage_params(options, 'verbose', None)
        latent  = self.pol_proj(state)
        mean    = self.get_mean(latent)
        std     = self.get_std(latent) if self.distribution != 'discrete' else None
        dist    = self.dist(mean, std)
        action  = dist.sample()
        if pos_idx:
            action = action[..., pos_idx, :, :] # shape(batch_size, seq_len, genomes, features
        return action

    def get_value(self, state: Tensor) -> Tensor:
        latent = self.val_proj(state)
        value = self.decode(latent)
        return value

    def forward(self, observations: Tensor, **options):
        actions = self.get_policy(observations, **options)
        return actions

    def infer(self, tensor: Tensor, pos_idx: int = None, genome_mask: Tensor = None, verbose: int = None):
        pos_idx = self.seq_len+pos_idx if pos_idx and pos_idx < 0 else pos_idx
        tensor = self.forward(tensor, pos_idx=pos_idx, mask=genome_mask, verbose=verbose)
        return tensor

