
from build.models.base import Linear
from build.util.fancy_text import CM, Fore

from torch import Tensor, device as DEVICE, dtype as DTYPE
# from typing import Union

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


"""
Timeseries
"""


class BufferEmbedding(nn.Module):
    def __init__(self, inputs: int, embed_size: int, bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32,
                 activation: nn.Module = None):
        super(BufferEmbedding, self).__init__()
        # BUILD
        self.embedding  = Linear(inputs, embed_size, bias, device, dtype, activation, False)

        # ATTRIBUTES
        self.input_dim  = inputs
        self.embed_size = embed_size

        # STATES
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, mask: Tensor = None, verbose: int = None):
        # Expand input to embedding space; [batch_size, sequence, features] to [batch_size, sequence, embed_size]
        tensor = self.embedding(tensor, mask=mask)
        if verbose:
            print(f"\nEmbedded tensor =>\n{tensor}\n\tdim = {tensor.shape}")

        return tensor


class BufferEncoding(nn.Module):
    def __init__(self, max_seq_len: int, embed_size: int, bias=False,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, activation: nn.Module = None):
        super(BufferEncoding, self).__init__()
        # BUILD
        self.positions      = torch.arange(max_seq_len, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) / (max_seq_len-1)
        # shape(batch_size, seq_len, genomes, features)
        self.encoding       = Linear(1, embed_size, bias, device, dtype, activation, False)

        # ATTRIBUTES
        self.max_seq_len    = max_seq_len
        self.embed_size     = embed_size

        # STATES
        self.device  = device
        self.dtype   = dtype

    def forward(self, tensor: Tensor, verbose: int = None):
        # tensor = (*batch_size, seq_len, genomes, features)
        seq_len = tensor.shape[1]
        # Expanding positional encoding to shape of input
        positional_encoding = self.encoding(self.positions[:, :seq_len].expand(*tensor.shape[:-1], 1))
        # Add encoding to tensor
        tensor = tensor + positional_encoding
        if verbose:
            print(f"\nPositional Encoding =>\n{positional_encoding}\n\tdim = {positional_encoding.shape}")
            print(f"\nEncoded tensor =>\n{tensor}\n\tdim = {tensor.shape}")

        return tensor


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, seq_length: int, embed_size: int, heads: int, constant: int = 10000,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(RotaryPositionalEmbedding, self).__init__()
        # BUILD - [records, sequence, embed_size], EMBEDDING - [sequence, embed_size]
        self.complex_frequencies = self._generate_encoding(seq_length, embed_size // heads, constant).\
            to(device=device)
        # EMBEDDING - [1, sequence, embed_size]
        self.select = torch.arange(seq_length).to(device=device, dtype=torch.int32)

        # ATTRIBUTES
        self.max_seq_length = seq_length
        self.embed_size     = embed_size

        # STATES
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    @staticmethod
    def _generate_encoding(seq_length: int, head_dim: int, constant: float = 10000.0):
        # Dimensions of embedding must be even
        assert head_dim % 2 == 0, f"Head dimension must be divisible by 2"
        # Get theta where theta_i = 10000 ^ (-2 * (i-1) / embedding) for i = [1, 2, ..., dim / 2]; [head_dim / 2]
        theta = torch.pow(constant, -2 * torch.arange(0, head_dim, 2).float() / head_dim)
        # Get positions as m; [sequence]
        positions   = torch.arange(seq_length)
        # Multiply theta by each position; [sequence] outer* [head_dim / 2] -> [sequence, head_dim / 2]
        angles      = torch.outer(positions, theta).float()
        # We compute complex number in polar form c = R * exp(i * m * theta); [sequence, head_dim / 2]
        complex_f   = torch.polar(torch.ones_like(angles), angles)
        # if True:
        #     print(f"\nTheta =>\n{theta}\n\tdim = {theta.shape}")
        #     print(f"\nPositions =>\n{positions}\n\tdim = {positions.shape}")
        #     print(f"\nAngles =>\n{angles}\n\tdim = {angles.shape}")
        #     print(f"\nComplex Frequencies init =>\n{complex_f}\n\tdim = {complex_f.shape}")

        return complex_f

    def forward(self, tensor: Tensor, debug: bool = False):
        seq_len = tensor.shape[1]
        # [records, sequence, heads, head_dim] -> [records, sequence, heads, head_dim / 2]
        complex_tensor = torch.view_as_complex(tensor.reshape(*tensor.shape[:-1], -1, 2))
        # [sequence, head_dim / 2] -> [1, sequence, 1, head_dim / 2]
        complex_frequencies = self.complex_frequencies.unsqueeze(0).unsqueeze(2)
        # [records, sequence, heads, head_dim] * [1, sequence, 1, head_dim / 2]
        # = [records, sequence, heads, head_dim / 2]
        rotated_tensor = complex_tensor * torch.index_select(complex_frequencies, 1, self.select[:seq_len])
        # [records, sequence, heads, head_dim / 2] -> [records, sequence, heads, head_dim / 2, 2]
        split_tensor = torch.view_as_real(rotated_tensor)
        # [records, sequence, heads, head_dim / 2, 2] -> [records, sequence, heads, head_dim]
        # [records, sequence, heads, head_dim] -> [records, sequence, embed_size]
        tensor = split_tensor.reshape(*tensor.shape).type_as(tensor)
        if debug:
            print(f"\nComplex Tensor=>\n{complex_tensor}\n\tdim = {complex_tensor.shape}")
            print(f"\nComplex Frequencies =>\n{complex_frequencies}\n\tdim = {complex_frequencies.shape}")
            print(f"\nRotated Tensor =>\n{rotated_tensor}\n\tdim = {rotated_tensor.shape}")
            print(f"\nSplit Tensor =>\n{split_tensor}\n\tdim = {split_tensor.shape}")
            print(f"\nEncoded Tensor =>\n{tensor}\n\tdim = {tensor.shape}")

        return tensor


class RMSNormalization(nn.Module):
    def __init__(self, embed_size: int, epsilon: float = 1e-6, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(RMSNormalization, self).__init__()
        # BUILD
        self.gamma = nn.Parameter(torch.ones(embed_size)).to(device=device, dtype=dtype)

        # ATTRIBUTES
        self.epsilon = epsilon

        # STATES
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def _norm(self, tensor: Tensor):
        # [records, sequence, embed_size]
        return tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) * self.epsilon)

    def forward(self, tensor: Tensor, debug=False):
        norm = self._norm(tensor.float()).type_as(tensor)
        if debug:
            print(f"\nNorm =>\n{norm}\n\tdim = {norm.shape}")
            print(f"\nGamma =>\n{self.gamma}\n\tdim = {self.gamma.shape}")
        return self.gamma * norm


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self, seq_len: int, embed_size: int, heads: int, kv_heads: int = None, constant=10000.0,
            causal_mask=True, bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32,
            activation: nn.Module = None):
        super(MultiHeadSelfAttention, self).__init__()
        if embed_size % heads != 0:
            raise ValueError(f"Embedding dimensions must be a multiple of heads num")
        if kv_heads is None:
            kv_heads = heads
        if heads % kv_heads != 0:
            raise ValueError(f"Query heads num must be a multiple of number of Key-Value heads num")

        # ATTRIBUTES
        self.heads      = heads
        self.embed_size = embed_size
        self.head_dim   = embed_size // heads
        self.kv_heads   = kv_heads
        self.q_kv_ratio = heads // kv_heads
        self.seq_len    = seq_len
        self.causal_mask = causal_mask
        self.constant   = constant

        # BUILD
        self.query_proj = Linear(embed_size, heads*self.head_dim, bias, device, dtype, activation, False)
        self.key_proj   = Linear(embed_size, kv_heads*self.head_dim, bias, device, dtype, activation, False)
        self.value_proj = Linear(embed_size, kv_heads*self.head_dim, bias, device, dtype, activation, False)
        self.out_proj   = Linear(embed_size, embed_size, bias, device, dtype, activation, False)
        # self.rotary_embedding = RotaryPositionalEmbedding(seq_len, embed_size, heads, constant, device, dtype)
        self.softmax    = nn.Softmax(-1)

        self.k_cache = torch.zeros(1, seq_len, kv_heads, self.head_dim, device=device, dtype=dtype)
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

    def forward(self, tensor: Tensor, context: Tensor = None, pos_idx: int = None, mask: Tensor = None, verbose: int = None):
        if verbose:
            print(f'\n{CM("Executing Self Attention", Fore.LIGHTBLUE_EX)}')

        batch_size, _, genomes, _ = tensor.shape

        # Linearize Q, K, V
        query   = self.query_proj(tensor if context is None else context, mask=mask)
        key     = self.key_proj(tensor, mask=mask)
        value   = self.value_proj(tensor, mask=mask)
        if verbose:
            print(CM('Post Linearization =>'))
            print(f"\n{CM('Query =>', Fore.BLUE)}\n{CM(query, Fore.LIGHTRED_EX)}, \n\tdim = {query.shape}")
            print(f"\n{CM('Key =>', Fore.BLUE)}\n{CM(key, Fore.LIGHTYELLOW_EX)}, \n\tdim = {key.shape}")
            print(f"\n{CM('Value =>', Fore.BLUE)}\n{CM(value, Fore.LIGHTBLUE_EX)}, \n\tdim = {value.shape}")

        # Reshape Q, K, V for each rep head
        query   = query.view(batch_size, -1, genomes, self.heads, self.head_dim)
        key     = key.view(batch_size, -1, genomes, self.kv_heads, self.head_dim)
        value   = value.view(batch_size, -1, genomes, self.kv_heads, self.head_dim)
        if verbose:
            print(f"\n{CM('Q after Reshaping =>', Fore.LIGHTYELLOW_EX)}\n{query}, \n\tdim = {query.shape}")
            print(f"\n{CM('K after Reshaping =>', Fore.LIGHTYELLOW_EX)}\n{key}, \n\tdim = {key.shape}")

        # # ROTARY EMBEDDING
        # query = self.rotary_embedding(query, pos_idx=pos_idx)
        # key   = self.rotary_embedding(key, pos_idx=pos_idx)
        # if verbose:
        #     print(f"\n{CM('Q after Rotary Embedding =>', Fore.LIGHTYELLOW_EX)}\n{query}, \n\tdim = {query.shape}")
        #     print(f"\n{CM('K after Rotary Embedding =>', Fore.LIGHTYELLOW_EX)}\n{key}, \n\tdim = {key.shape}")

        # UPDATE CACHE
        if pos_idx is not None:
            # (batch_size, seq_len, kv_heads, head_dim)
            if pos_idx == 0:
                self.adjust_cache_size(key.shape[0])

            # SET
            self.k_cache[:batch_size, pos_idx:pos_idx+1] = key
            self.v_cache[:batch_size, pos_idx:pos_idx+1] = value

            # GET
            key   = self.k_cache[:batch_size, :pos_idx+1]
            value = self.v_cache[:batch_size, :pos_idx+1]

        # Duplicate K and V for kv heads num per query head
        key   = self.repeat_kv(key, self.q_kv_ratio)
        value = self.repeat_kv(value, self.q_kv_ratio)
        if verbose:
            print(f"\n{CM('Duplicated K =>', Fore.LIGHTYELLOW_EX)}\n{key}, \n\tdim = {key.shape}")
            print(f"\n{CM('Duplicated V =>', Fore.LIGHTYELLOW_EX)}\n{value}, \n\tdim = {value.shape}")

        # Get the attention score (energy)
        energy = torch.einsum("bqghd,bkghd->bghqk", [query, key])
        # energy = query @ key.transpose(-1, -2)
        # queries shape: (batch_size, query_len, heads, head_dim)
        # key shape:     (batch_size, key_len, heads, head_dim)
        # energy shape:  (batch_size, heads, query_len, key_len)
        if verbose:
            print(
                f"\n{CM('Energy =>', Fore.LIGHTYELLOW_EX)}\n{energy}, \n\tdim = {energy.shape}")

        if self.causal_mask and context is None:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(energy, dtype=torch.bool).triu(1)
            # Fill the upper triangle with -inf
            energy.masked_fill_(mask, -torch.inf)
            if verbose:
                print(
                    f"\n{CM('Masked Energy =>', Fore.LIGHTYELLOW_EX)}\n{energy}, \n\tdim = {energy.shape}")

        # Get the softmax of the energy
        # scores = energy / (self.head_dim ** (1 / 2))
        # scores = self.actv(energy / (self.head_dim ** (1 / 2)))
        scores = self.softmax(energy / np.sqrt(self.head_dim))
        if verbose:
            print(f"\n{CM('Attention Scores =>', Fore.LIGHTYELLOW_EX)}"
                  f"\n{torch.round(scores, decimals=4)}\n\tdim = {scores.shape}")

        # Get the weighted sum of the values and reshape to remove heads
        attention = torch.einsum("bghqv,bvghd->bqghd", [scores, value])
        # attention = (scores @ value).transpose(-2, -1).contiguous()
        # scores shape:    (batch_size, heads, query_len, value_len)
        # values shape:    (batch_size, value_len, heads, head_dim)
        # attention shape: (batch_size, query_len, heads, head_dim) then concat last 2 dim
        attention = attention.reshape(batch_size, -1, genomes, self.embed_size)
        # out_view shape:  (batch_size, query_len, embed_size)
        if verbose:
            print(f"\n{CM('Attention Values =>', Fore.LIGHTYELLOW_EX)}\n{attention}\n\tdim = {attention.shape}")

        # Apply weights
        tensor = self.out_proj(attention, mask=mask)
        if verbose:
            print(f"\n{CM('Attented Values =>', Fore.LIGHTCYAN_EX)}\n{tensor}\n\tdim = {tensor.shape}")

        return tensor


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


class TransformerBlock(nn.Module):
    def __init__(self, seq_len: int, embed_size: int, heads: int, fwd_exp: int, kv_heads: int = None,
                 dropout: float = None, norm_eps=1e-12, constant=10000., bias=False,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(TransformerBlock, self).__init__()
        if dropout is None:
            dropout = 0.0

        # BUILD
        self.attention   = MultiHeadSelfAttention(seq_len, embed_size, heads, kv_heads, constant, bias, device, dtype)
        self.att_norm    = nn.LayerNorm(embed_size, norm_eps, bias=bias, device=device, dtype=dtype)
        # self.att_norm       = RMSNormalization(embed_size, norm_eps, device=device, dtype=dtype)
        self.feedforward = MultiLayerPerceptron(embed_size, fwd_exp, bias, device, dtype)
        self.ffd_norm    = nn.LayerNorm(embed_size, norm_eps, bias=bias, device=device, dtype=dtype)
        # self.ffd_norm       = RMSNormalization(embed_size, norm_eps, device=device, dtype=dtype)
        self.dropout     = nn.Dropout(dropout)

        # ATTRIBUTES
        self.embed_size = embed_size
        self.heads      = heads
        self.epsilon    = norm_eps
        self.head_dim   = embed_size // heads

        # STATES
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, pos_idx: int = None, debug: bool = False):
        # Normalize then get the multi-head attention (with rotary embedding)
        attention   = self.attention(self.att_norm(tensor), causal_mask=True, pos_idx=pos_idx, debug=debug)
        # Apply residual connection then dropout
        tensor      = self.dropout(tensor + attention)
        # Pass through feed forward
        activation  = self.feedforward(self.ffd_norm(tensor))
        # Apply residual connection
        tensor      = self.dropout(tensor + activation)
        if debug:
            print(f"\n{CM('Encoder Block Output =>', Fore.LIGHTYELLOW_EX)}\n{tensor}\n\tdim = {tensor.shape}")

        return tensor


class Transformer(nn.Module):
    def __init__(self, seq_len: int, embed_size: int, heads: int, layers: int, fwd_exp: int,
                 kv_heads: int = None, dropout: float = None, norm_eps: float = 1e-12, constant=10000., bias=False,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(Transformer, self).__init__()
        # BUILD
        self.layers = nn.ModuleList()
        for layer in range(layers):
            self.layers.append(
                TransformerBlock(seq_len, embed_size, heads, fwd_exp, kv_heads, dropout, norm_eps, constant, bias,
                                 device, dtype)
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
        self.device     = device
        self.dtype      = dtype

    def forward(self, tensor: Tensor, pos_idx: int = None, debug: bool = False):
        # Pass through the encoder blocks
        for layer_idx, layer in enumerate(self.layers):
            tensor = layer(tensor, pos_idx=pos_idx, debug=debug if layer_idx == 0 else False)

        return tensor
