
from build.models.base import NeatModule, NeatParameter, Linear, RMSNorm
from build.nn.genome import Genome
from build.util.fancy_text import CM, Fore
from build.util.qol import manage_params

from torch import Tensor, device as DEVICE, dtype as DTYPE
from typing import Union, Iterable
from numpy import ndarray as CPUArray

import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np


class InvalidValueError(ValueError):
    pass


"""
Vanilla
"""


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, inp_features: int = None, depth=1, fwd_exp=4,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(TokenEmbedding, self).__init__()
        # BUILD
        self.embedding  = nn.Embedding(vocab_size, embed_size, device=device, dtype=dtype)
        self.reduction  = None if inp_features is None else nn.Sequential(
            nn.Linear(inp_features, embed_size*fwd_exp, False, device, dtype),
            *[nn.Linear(embed_size*fwd_exp, embed_size*fwd_exp, False, device, dtype) for _ in range(depth)],
            nn.Linear(embed_size*fwd_exp, 1, False, device, dtype)
        )

        # ATTRIBUTES
        self.embed_size = embed_size

        # STATES
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, debug=False):
        # # Check if input is a tensor
        # if not isinstance(tensor, Tensor):
        #     raise InvalidValueError(f"Input Sequence entered is not a Tensor.")
        # Expands input to embedding space; [records, sequence] to [records, sequence, embed_size]
        # or (batch_size, seq_len, features) to (batch_size, seq_len, features, embed_size)
        tensor = self.embedding(tensor)
        if self.reduction is not None:
            # Convert (batch_size, seq_len, features, embed_size) to (batch_size, seq_len, embed_size)
            tensor = self.reduction(torch.transpose(tensor, -1, -2)).squeeze(-1)
        if debug:
            print(debug)
            print(f"\nEmbedded Data =>\n{tensor}\n\tdim = {tensor.shape}")

        return tensor


class SinusoidalEncoding(nn.Module):
    def __init__(self, seq_length: int, embed_size: int, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(SinusoidalEncoding, self).__init__()
        # BUILD - [records, sequence, embed_size], EMBEDDING - [sequence, embed_size]
        self.positional_encoding = self._generate_encoding(seq_length, embed_size).\
            to(device=device, dtype=dtype)
        # EMBEDDING - [1, sequence, embed_size]
        self.positional_encoding = self.positional_encoding.unsqueeze(0)

        # ATTRIBUTES
        self.max_seq_length = seq_length
        self.embed_size     = embed_size

        # STATES
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, debug: bool = False):
        # Get the dimension shape of the input
        records, seq_length, embed_size = tensor.size()
        # Expanding positional encoding to shape of input
        positional_encoding = self.positional_encoding.expand(records, -1, embed_size)
        # Add encoding to tensor
        tensor = tensor + positional_encoding[:, :seq_length]
        if debug:
            print(f"\nPositional Encoding =>\n{positional_encoding}\n\tdim = {positional_encoding.shape}")
            print(f"\nEncoded Sequences =>\n{tensor}\n\tdim = {tensor.shape}")

        return tensor

    @staticmethod
    def _generate_encoding(max_seq_length: int, embed_size: int, constant=10000.0):
        encoding = torch.zeros(max_seq_length, embed_size)
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.multiply(
                torch.arange(0, embed_size, 2),
                (-torch.log(torch.tensor(constant)) / embed_size)
            )
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding


class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_size: int, fwd_exp: int = None, bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(MultiLayerPerceptron, self).__init__()
        if fwd_exp is None:
            fwd_exp = 4
        # hidden_size = 4 * embed_size
        # hidden_size = int(2 * hidden_size / 3)
        # if fwd_exp is not None:
        #     hidden_size = int(fwd_exp * hidden_size)
        # hidden_size = mult * ((hidden_size + mult - 1) // mult)
        hidden_size = fwd_exp * embed_size

        # BUILD
        self.inp_proj = nn.Linear(embed_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(hidden_size, embed_size, bias=bias, device=device, dtype=dtype)
        self.activation = nn.SiLU()

    def forward(self, tensor: Tensor):
        tensor = self.out_proj(self.activation(self.inp_proj(tensor)))
        return tensor


"""
Timeseries
"""


class BufferEmbedding(NeatModule):
    def __init__(self, inputs: int, embed_size: int, bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(BufferEmbedding, self).__init__()
        # BUILD
        self.embedding  = Linear(inputs, embed_size, bias, device, dtype)

        # ATTRIBUTES
        self.input_dim  = inputs
        self.embed_size = embed_size

        # STATES
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, keys: Union[int, list[int]] = None, verbose: int = None):
        # Expand input to embedding space; [batch_size, sequence, features] to [batch_size, sequence, embed_size]
        # print(f"forward={self.embedding.weights.data.shape, tensor.shape}")
        tensor = self.embedding(tensor, keys=keys)
        if verbose:
            print(f"\nEmbedded tensor =>\n{tensor}\n\tdim = {tensor.shape}")

        return tensor


class BufferEncoding(NeatModule):
    def __init__(self, max_seq_len: int, embed_size: int, bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(BufferEncoding, self).__init__()
        # BUILD
        self.positions      = torch.arange(max_seq_len, device=device, dtype=dtype).unsqueeze(0).unsqueeze(1).unsqueeze(-1) / (max_seq_len-1)
        # shape(genomes, batch_size, seq_len, features)
        self.encoding       = Linear(1, embed_size, bias, device, dtype)
        self.activation     = nn.SiLU()

        # ATTRIBUTES
        self.max_seq_len    = max_seq_len
        self.embed_size     = embed_size

        # STATES
        self.device  = device
        self.dtype   = dtype

    def forward(self, tensor: Tensor, keys: Union[int, list[int]] = None, verbose: int = None):
        # tensor = (genomes, batch_size, seq_len, embed_size)
        genomes, _, seq_len, _ = tensor.shape
        # Expanding positional encoding to shape of input
        positions = self.positions[:, :, :seq_len]
        if verbose and verbose >= 2:
            print(f"\nPositions =>\n{positions}\n\tdim = {positions.shape}")
        positional_encoding = self.activation(self.encoding(positions.expand(genomes, 1, seq_len, 1), keys=keys))
        # Add encoding to tensor
        tensor = tensor + positional_encoding
        if verbose:
            print(f"\nPositional Encoding =>\n{positional_encoding}\n\tdim = {positional_encoding.shape}")
            print(f"\nEncoded tensor =>\n{tensor}\n\tdim = {tensor.shape}")

        return tensor


class RoPE(NeatModule):
    def __init__(self, seq_len: int, embed_size: int, heads: int, constant: int = 10000,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(RoPE, self).__init__()
        # BUILD - [genomes, batch_size, seq_len, head_dim / 2], EMBEDDING - [genomes, seq_len, head_dim / 2]
        self.complex_frequencies = self._generate_encoding(seq_len, embed_size // heads, constant, 0).to(device=device)
        self.complex_frequencies = self.complex_frequencies.unsqueeze(0).unsqueeze(1).unsqueeze(-2)
        # EMBEDDING - [genomes, 1, sequence, embed_size]
        self.select = torch.arange(seq_len).to(device=device, dtype=torch.int32)

        # ATTRIBUTES
        self.max_seq_length = seq_len
        self.embed_size     = embed_size
        self.head_dim       = embed_size // heads
        self.constant       = constant

        # STATES
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    @staticmethod
    def _generate_encoding(seq_length: int, head_dim: int, constant: float = 10000.0, verbose: int = None):
        # Dimensions of embedding must be even
        assert head_dim % 2 == 0, f"Head dimension must be divisible by 2"
        # Get theta where theta_i = 10000 ^ (-2 * (i-1) / embedding) for i = [1, 2, ..., dim / 2]; [head_dim / 2]
        theta = 1.0 / torch.pow(constant, torch.arange(0, head_dim, 2).float() / head_dim)
        # Get positions as m; [sequence]
        positions   = torch.arange(seq_length)
        # Multiply theta by each position; [sequence] outer* [head_dim / 2] -> [sequence, head_dim / 2]
        angles      = torch.outer(positions, theta).float()
        # We compute complex number in polar form c = R * exp(i * m * theta); [sequence, head_dim / 2]
        complex_f   = torch.polar(torch.ones_like(angles), angles)
        if verbose:
            print(f"\nTheta =>\n{theta}\n\tdim = {theta.shape}")
            print(f"\nPositions =>\n{positions}\n\tdim = {positions.shape}")
            print(f"\nAngles =>\n{angles}\n\tdim = {angles.shape}")
            print(f"\nComplex Frequencies init =>\n{complex_f}\n\tdim = {complex_f.shape}")

        return complex_f

    def update(self, genomes: dict[int, Genome], params: Union[Tensor, CPUArray] = None, verify=False):
        self.genome_num = len(genomes)
        # BUILD - [genomes, batch_size, seq_len, head_dim / 2], EMBEDDING - [seq_len, head_dim / 2]
        complex_frequencies = self._generate_encoding(
            self.max_seq_length, self.head_dim, self.constant).to(device=self.device)
        # [genomes, batch_size, sequence, head_dim / 2] -> [genomes, 1, sequence, 1, head_dim / 2]
        self.complex_frequencies = complex_frequencies.unsqueeze(0).unsqueeze(1).unsqueeze(-2)
        self.mapping = {genome.key: index for index, genome in enumerate(genomes.values())}

    def forward(self, tensor: Tensor, pos_idx: int = None, verbose: int = None):
        if pos_idx is None:
            pos_idx = 0
        seq_len = tensor.shape[-3]
        # [genomes, batch_size, sequence, heads, head_dim] -> [genomes, batch_size, sequence, heads, head_dim]
        complex_tensor = torch.view_as_complex(tensor.reshape(*tensor.shape[:-1], -1, 2))
        # [records, sequence, heads, head_dim] * [1, sequence, 1, head_dim / 2] = [records, sequence, heads, head_dim / 2]
        complex_frequencies = torch.index_select(self.complex_frequencies, -3, self.select[pos_idx:pos_idx+seq_len])
        rotated_tensor = complex_tensor * complex_frequencies
        # [records, sequence, heads, head_dim / 2] -> [records, sequence, heads, head_dim / 2, 2]
        split_tensor = torch.view_as_real(rotated_tensor)
        # [records, sequence, heads, head_dim / 2, 2] -> [records, sequence, heads, head_dim]
        # [records, sequence, heads, head_dim] -> [records, sequence, embed_size]
        tensor = split_tensor.reshape(*tensor.shape).type_as(tensor)
        if verbose and verbose >= 3:
            print(f"\nComplex Tensor=>\n{complex_tensor}\n\tdim = {complex_tensor.shape}")
            print(f"\nComplex Frequencies =>\n{complex_frequencies}\n\tdim = {complex_frequencies.shape}")
            print(f"\nRotated Tensor =>\n{rotated_tensor}\n\tdim = {rotated_tensor.shape}")
            print(f"\nSplit Tensor =>\n{split_tensor}\n\tdim = {split_tensor.shape}")
            print(f"\nEncoded Tensor =>\n{tensor}\n\tdim = {tensor.shape}")

        return tensor


class AttentionLambda(NeatModule):
    def __init__(self, heads: int, head_dim: int, layer_idx: int = None, epsilon=1e-8,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(AttentionLambda, self).__init__(heads=heads, head_dim=head_dim, layer_idx=layer_idx)
        if layer_idx is None:
            layer_idx = 0

        # BUILD
        self.query1 = NeatParameter((heads, head_dim), False, device, dtype)
        self.query2 = NeatParameter((heads, head_dim), False, device, dtype)
        self.key1   = NeatParameter((heads, head_dim), False, device, dtype)
        self.key2   = NeatParameter((heads, head_dim), False, device, dtype)
        self.init   = 0.8 - 0.6 * np.exp(-0.3 * layer_idx)
        self.eps    = epsilon

    def update_limit(self):
        for param in self.neat_parameters():
            remove_rg = param.data.requires_grad
            if remove_rg:
                param.data.requires_grad_(False)
            param.data[:] = torch.clamp(param.data, self.eps, 1)
            if remove_rg:
                param.data.requires_grad_(True)

    def forward(self, query: Tensor, key: Tensor, keys: Union[int, Iterable[int]] = None):
        # query:     (genomes, batch_size, q_len, heads, head_dim)
        # key:       (genomes, batch_size, k_len, heads, head_dim)
        # attention: (genomes, batch_size, heads, q_len, k_len)
        offset = 1
        return (
            torch.exp(torch.sum(
                self.expand(self.query1[keys], query, offset, keys) * self.expand(self.key1[keys], key, offset, keys), -2, True
            )) -
            torch.exp(torch.sum(
                self.expand(self.query2[keys], query, offset, keys) * self.expand(self.key2[keys], key, offset, keys), -2, True
            )) +
            self.init
            # self.expand(self.query1[keys], attention, offset)[..., :query_len, :key_len] *
            # self.expand(self.key1[keys], attention, offset)[..., :query_len, :key_len] +
            # self.expand(self.init[keys], attention, offset)[..., :query_len, :key_len]
        )


class Attention(NeatModule):
    def __init__(
            self, seq_len: int, embed_size: int, heads: int = None, kv_heads: int = None, differential=True,
            layer_idx: int = None, constant=10000.0, eps=1e-8, affine=True, causal_mask=True, bias=False,
            device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options):
        super(Attention, self).__init__()
        if heads is None:
            heads = 1
        if embed_size % heads != 0:
            raise ValueError(f"Embedding dimensions must be a multiple of heads num")
        if kv_heads is None:
            kv_heads = heads
        if heads % kv_heads != 0:
            raise ValueError(f"Query heads num must be a multiple of number of Key-Value heads num")

        inputs: int  = manage_params(options, 'inputs', None)
        outputs: int = manage_params(options, 'outputs', None)

        # ATTRIBUTES
        self.heads      = heads
        self.embed_size = embed_size
        self.head_dim   = embed_size // heads
        self.kv_heads   = kv_heads
        self.q_kv_ratio = heads // kv_heads
        self.seq_len    = seq_len
        self.differential = differential
        self.causal_mask = causal_mask
        self.constant   = constant

        # BUILD
        self.mult = 2 if differential else 1
        self.query_proj = Linear(embed_size if not inputs else inputs, heads*self.head_dim*self.mult, bias, device, dtype)
        self.key_proj   = Linear(embed_size if not inputs else inputs, kv_heads*self.head_dim*self.mult, bias, device, dtype)
        self.value_proj = Linear(embed_size if not inputs else inputs, kv_heads*self.head_dim, bias, device, dtype)
        self.out_proj   = Linear(embed_size, embed_size if not outputs else outputs, bias, device, dtype)
        self.rotary_embedding = RoPE(seq_len, embed_size*self.mult, heads, constant, device, dtype)
        self.softmax    = nn.Softmax(-1)
        self.norm       = RMSNorm(self.head_dim, eps, affine, device, dtype)
        self.diff_lambda = AttentionLambda(heads, self.head_dim, layer_idx, eps, device, dtype) if differential else None

        self.k_cache = torch.zeros(1, seq_len, kv_heads, self.head_dim*self.mult, device=device, dtype=dtype)
        self.v_cache = torch.zeros(1, seq_len, kv_heads, self.head_dim, device=device, dtype=dtype)

        # STATES
        self.device = device
        self.dtype  = dtype

    def adjust_cache_size(self, batch_size: int):
        cache_size = self.k_cache.shape[0]
        padding = max(0, batch_size - cache_size)
        if padding > 0:
            self.k_cache = torch.cat([self.k_cache, torch.zeros(padding, *self.k_cache.shape[1:], device=self.device, dtype=self.dtype)], 0)
            self.v_cache = torch.cat([self.v_cache, torch.zeros(padding, *self.v_cache.shape[1:], device=self.device, dtype=self.dtype)], 0)

    @staticmethod
    def repeat_kv(tensor: Tensor, ratio: int):
        batch_size, seq_len, genomes, kv_heads, head_dim = tensor.shape
        if ratio == 1:
            return tensor
        else:
            return tensor.unsqueeze(-2).expand(batch_size, seq_len, genomes, kv_heads, ratio, head_dim).\
                reshape(batch_size, seq_len, genomes, kv_heads * ratio, head_dim)

    def attention(self, query: Tensor, key: Tensor, mask: bool, verbose: int = None):
        # Get the attention score (energy)
        energy = torch.einsum("gbqhd,gbkhd->gbhqk", [query, key])
        # queries shape: (genomes, batch_size, query_len, heads, head_dim)
        # key shape:     (genomes, batch_size, key_len, heads, head_dim)
        # energy shape:  (genomes, batch_size, heads, query_len, key_len)
        if verbose:
            print(f"\n{CM('Energy =>', Fore.LIGHTYELLOW_EX)}"
                  f"\n{energy}, \n\tdim = {energy.shape}")

        if mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask_ = torch.ones_like(energy, dtype=torch.bool).triu(1)
            # Fill the upper triangle with -inf
            energy.masked_fill_(mask_, -torch.inf)
            if verbose:
                print(f"\n{CM('Masked Energy =>', Fore.LIGHTYELLOW_EX)}\n{energy}, \n\tdim = {energy.shape}")
                if verbose >= 2:
                    print(f"\n{CM('Mask =>', Fore.CYAN)}\n{mask_}, \n\tdim = {mask_.shape}")

        # Get the softmax of the energy
        # scores = energy / (self.head_dim ** (1 / 2))
        # scores = self.actv(energy / (self.head_dim ** (1 / 2)))
        scores = self.softmax(energy / np.sqrt(self.head_dim))

        return scores

    def forward(self, tensor: Tensor, context: Tensor = None, pos_idx: int = None,
                keys: Union[int, Iterable[int]] = None, verbose: int = None, get=False):
        if verbose:
            print(f'\n{CM("Executing Self Attention", Fore.LIGHTBLUE_EX)}')

        # Linearize Q, K, V
        query: Tensor   = self.query_proj(tensor if context is None else context, keys=keys)
        key: Tensor     = self.key_proj(tensor, keys=keys)
        value: Tensor   = self.value_proj(tensor, keys=keys)
        if verbose:
            print(CM('Post Linearization =>'))
            print(f"\n{CM('Query =>', Fore.LIGHTYELLOW_EX)}\n{CM(query, Fore.LIGHTRED_EX)}, \n\tdim = {query.shape}")
            print(f"\n{CM('Key =>', Fore.LIGHTYELLOW_EX)}\n{CM(key, Fore.LIGHTGREEN_EX)}, \n\tdim = {key.shape}")
            print(f"\n{CM('Value =>', Fore.LIGHTYELLOW_EX)}\n{CM(value, Fore.LIGHTBLUE_EX)}, \n\tdim = {value.shape}")

        genomes, batch_size = query.shape[:2]

        # Reshape Q, K, V for each rep head
        query   = query.view(genomes, batch_size, -1, self.heads, self.head_dim*self.mult)
        key     = key.view(genomes, batch_size, -1, self.kv_heads, self.head_dim*self.mult)
        value   = value.view(genomes, batch_size, -1, self.kv_heads, self.head_dim)
        if verbose:
            print(f"\n{CM('Q after Reshaping =>', Fore.LIGHTYELLOW_EX)}\n{query}, \n\tdim = {query.shape}")
            print(f"\n{CM('K after Reshaping =>', Fore.LIGHTYELLOW_EX)}\n{key}, \n\tdim = {key.shape}")

        # UPDATE CACHE
        if pos_idx is not None:
            # (batch_size, seq_len, kv_heads, head_dim)
            if batch_size != self.k_cache.shape[0]:
                self.adjust_cache_size(batch_size)

            # SET
            n, p = pos_idx+1-key.shape[1], pos_idx+1
            assert n > 0 and p <= self.seq_len
            self.k_cache[:batch_size, n:p] = key
            self.v_cache[:batch_size, n:p] = value

            # GET
            key   = self.k_cache[:batch_size, :pos_idx+1]
            value = self.v_cache[:batch_size, :pos_idx+1]

        # Duplicate K and V for kv heads num per query head
        key   = self.repeat_kv(key, self.q_kv_ratio)
        value = self.repeat_kv(value, self.q_kv_ratio)
        if verbose:
            print(f"\n{CM('Duplicated K =>', Fore.LIGHTYELLOW_EX)}\n{key}, \n\tdim = {key.shape}")
            print(f"\n{CM('Duplicated V =>', Fore.LIGHTYELLOW_EX)}\n{value}, \n\tdim = {value.shape}")

        # ROTARY EMBEDDING
        query = self.rotary_embedding(query, pos_idx, verbose)
        key   = self.rotary_embedding(key, None, verbose)
        if verbose:
            print(f"\n{CM('Q after Rotary Embedding =>', Fore.LIGHTYELLOW_EX)}\n{query}, \n\tdim = {query.shape}")
            print(f"\n{CM('K after Rotary Embedding =>', Fore.LIGHTYELLOW_EX)}\n{key}, \n\tdim = {key.shape}")

        mask = self.causal_mask and context is None
        if not self.differential:
            attention: Tensor = self.attention(query, key, mask, verbose)
        else:
            query1, query2  = query.split(self.head_dim, -1)
            key1, key2      = key.split(self.head_dim, -1)
            attention1      = self.attention(query1, key1, False, verbose)
            attention2      = self.attention(query2, key2, False, False)
            diff_lambda     = self.diff_lambda(query1, key1, keys=keys)
            if verbose and verbose >= 2:
                print(f"\n{CM('Lambda =>', Fore.LIGHTYELLOW_EX)}"
                      f"\n{torch.round(diff_lambda, decimals=4)}\n\tdim = {diff_lambda.shape}")
            attention       = attention1 - (diff_lambda * attention2)

            if mask:
                attention.masked_fill_(torch.ones_like(attention, dtype=torch.bool).triu(1), -torch.inf)
                if verbose and verbose >= 3:
                    print(f"\n{CM('Masked Energy =>', Fore.LIGHTYELLOW_EX)}\n{attention}, \n\tdim = {attention.shape}")

            # Get the softmax of the energy
            attention = self.softmax(attention)

        if verbose:
            print(f"\n{CM('Attention Scores =>', Fore.LIGHTYELLOW_EX)}"
                  f"\n{torch.round(attention, decimals=4)}\n\tdim = {attention.shape}")

        # Get the weighted sum of the values and reshape to remove heads
        att_value = self.norm(torch.einsum("gbhqv,gbvhd->gbqhd", [attention, value]), keys=keys)
        if self.differential:
            att_value = att_value * (1 - self.diff_lambda.init)
        # attention = (scores @ value).transpose(-2, -1).contiguous()
        # scores shape:    (genomes, batch_size, heads, query_len, value_len)
        # values shape:    (genomes, batch_size, value_len, heads, head_dim)
        # attention shape: (genomes, batch_size, query_len, heads, head_dim) then concat last 2 dim
        att_value = att_value.reshape(genomes, batch_size, -1, self.embed_size)
        # out_view shape:  (batch_size, query_len, embed_size)
        if verbose:
            print(f"\n{CM('Attention Values =>', Fore.LIGHTYELLOW_EX)}\n{att_value}\n\tdim = {att_value.shape}")

        # Apply weights
        tensor: Tensor = self.out_proj(att_value, keys=keys)
        if verbose:
            print(f"\n{CM('Output Projection =>', Fore.LIGHTCYAN_EX)}\n{tensor}\n\tdim = {tensor.shape}")

        if not get:
            return tensor
        else:
            return tensor, (attention, att_value)


class SwiGLUFeedForward(NeatModule):
    def __init__(self, embed_size: int, fwd_exp: int = None, bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(SwiGLUFeedForward, self).__init__()
        if fwd_exp is None:
            fwd_exp = 4
        # hidden_size = 4 * embed_size
        # hidden_size = int(2 * hidden_size / 3)
        # if fwd_exp is not None:
        #     hidden_size = int(fwd_exp * hidden_size)
        # hidden_size = mult * ((hidden_size + mult - 1) // mult)
        hidden_size = fwd_exp * embed_size

        # BUILD
        self.w1 = Linear(embed_size, hidden_size, bias, device, dtype)
        self.w2 = Linear(hidden_size, embed_size, bias, device, dtype)
        self.w3 = Linear(embed_size, hidden_size, bias, device, dtype)
        self.actv = nn.SiLU()

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        # (batch_size, seq_len, embed_size) -> (batch_size, seq_len, hidden_size)
        swish = self.actv(self.w1(tensor, keys=keys))
        # (batch_size, seq_len, embed_size) -> (batch_size, seq_len, hidden_size)
        tensor_ = self.w3(tensor, keys=keys)
        # (batch_size, seq_len, hidden_size) * (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        tensor = swish * tensor_
        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, embed_size / out)
        tensor = self.w2(tensor, keys=keys)
        return tensor


class TransformerBlock(NeatModule):
    def __init__(
            self, seq_len: int, embed_size: int, heads: int = None, kv_heads: int = None, fwd_exp=4, differential=True,
            layer_idx: int = None, constant=10000.0, eps=1e-8, affine=True, causal_mask=True, dropout: int = None,
            bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32,):
        super(TransformerBlock, self).__init__()
        if dropout is None:
            dropout = 0.0

        # BUILD
        self.attention   = Attention(seq_len, embed_size, heads, kv_heads, differential, layer_idx,
                                     constant, eps, affine, causal_mask, bias, device, dtype)
        self.att_norm    = RMSNorm(embed_size, eps, affine, device, dtype)
        self.feedforward = SwiGLUFeedForward(embed_size, fwd_exp, bias, device, dtype)
        self.ffd_norm    = RMSNorm(embed_size, eps, affine, device, dtype)
        self.dropout     = nn.Dropout(dropout)

        # ATTRIBUTES
        self.embed_size = embed_size
        self.heads      = heads
        self.epsilon    = eps
        self.head_dim   = embed_size // heads

        # STATES
        self.attention_tensor: Tensor = None
        self.attented_value: Tensor = None
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, context: Tensor = None, pos_idx: int = None,
                keys: Union[int, Iterable[int]] = None, verbose: int = None, get=False):
        # Normalize then get the attention
        attention = self.attention(
                self.att_norm(tensor, keys=keys), context=context, pos_idx=pos_idx, keys=keys, verbose=verbose, get=get)
        if get:
            attention, (self.attention_tensor, self.attented_value) = attention
        # Apply residual connection then dropout
        tensor      = self.dropout((tensor if context is None else context) + attention)
        # Pass through feed forward
        activation  = self.feedforward(self.ffd_norm(tensor, keys=keys), keys=keys)
        # Apply residual connection
        tensor      = self.dropout(tensor + activation)
        if verbose:
            print(f"\n{CM('Transformer Block Output =>', Fore.LIGHTYELLOW_EX)}\n{tensor}\n\tdim = {tensor.shape}")

        return tensor


class TransformerBase(NeatModule):
    def __init__(
            self, seq_len: int, embed_size: int, layers: int, heads: int = None, kv_heads: int = None, fwd_exp=4, 
            differential=True, constant=10000.0, eps=1e-8, affine=True, causal_mask=True, dropout: int = None,
            bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32,):
        super(TransformerBase, self).__init__()
        # BUILD
        self.layers: list[TransformerBlock] = nn.ModuleList()
        for layer_idx in range(layers):
            self.layers.append(
                TransformerBlock(
                    seq_len, embed_size, heads, kv_heads, fwd_exp, differential, layer_idx,
                    constant, eps, affine, causal_mask, dropout, bias, device, dtype
                )
            )

        # ATTRIBUTES
        self.seq_len    = seq_len
        self.embed_size = embed_size
        self.heads      = heads
        self.layer_num  = layers
        self.fwd_exp    = fwd_exp
        self.kv_heads   = kv_heads
        self.dropout    = dropout

        # STATE
        self.attention_tensors: list[Tensor] = None
        self.attented_values: list[Tensor] = None
        self.device     = device
        self.dtype      = dtype

    def forward(self, tensor: Tensor, context: Tensor = None, pos_idx: int = None,
                keys: Union[int, Iterable[int]] = None, verbose: int = None, get=False, single=False):
        # Pass through the encoder blocks
        for layer_idx, layer in enumerate(self.layers):
            single_fetch = single and layer_idx == len(self.layers) - 1
            if not single:
                set_context = context
            elif single_fetch:
                # Using last token index in sequence to get the next token
                # shape (genomes, batch_size, seq_len, embed_size)
                set_context = torch.select(tensor, -2, -1).unsqueeze(-2)
            else:
                set_context = None
            tensor = layer(tensor, context=set_context, pos_idx=pos_idx, keys=keys,
                           verbose=verbose if layer_idx == 0 else False, get=get)

        return tensor

    def get_attention(self):
        return [(layer.attention_tensor, layer.attented_value) for layer in self.layers]
