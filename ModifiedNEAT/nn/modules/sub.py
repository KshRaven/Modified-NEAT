from __future__ import annotations

from ..base import NeatModule, NeatParameter
from .base import *
from ...util.fancy_text import CM, Fore
from ...util.qol import manage_params

from torch import Tensor, device as DEVICE, dtype as DTYPE
from typing import Union, Iterable

import torch
import torch.nn as nn
# import torch.nn.functional as F
import math


class InvalidValueError(ValueError):
    pass


# --------------------------------------------- #
# Residuals                                     #
# --------------------------------------------- #

class ResidualBlock(NeatModule):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, norm_groups: int,
                 bias=False, device='cpu', dtype=torch.float32, **options):
        super().__init__()
        # Options
        self.activation: nn.Module = manage_params(options, ['activation', 'actv'], nn.SiLU())
        self.hidden_size: int   = manage_params(options, ['hidden_size', 'dim_size', 'embed_size'],
                                                channels_out if channels_out % norm_groups == 0 else channels_in)
        self.stride: int        = manage_params(options, 'stride', 1)
        self.dilation: int      = manage_params(options, 'dilation', 1)
        self.padding: Union[int, tuple[int, ...]] = manage_params(
            options, 'padding', calc_padding(kernel_size, stride=self.stride, dilation=self.dilation))
        self.padding_mode: str  = manage_params(options, 'padding_mode', 'zeros')
        self.epsilon: float     = manage_params(options, ['epsilon', 'eps'], 1e-5)
        self.affine: bool     = manage_params(options, 'affine', True)
        self.image_ndim: bool     = manage_params(options, 'image_ndim', 2)

        # Build
        try:
            if self.image_ndim == 1:
                Convolution = Conv1d
            elif self.image_ndim == 2:
                Convolution = Conv2d
            elif self.image_ndim == 3:
                Convolution = Conv3d
            else:
                raise ValueError(f"Unsupported num of image dimension '{self.image_ndim}'")
            self.norm1 = GroupNorm(norm_groups, channels_in, self.epsilon, self.affine, False, device, dtype)
            self.conv1 = Convolution(channels_in, self.hidden_size, kernel_size, self.stride, self.padding, self.dilation,
                                     bias=bias, device=device, dtype=dtype)

            self.norm2 = GroupNorm(norm_groups, self.hidden_size, self.epsilon, self.affine, False, device, dtype)
            self.conv2 = Convolution(self.hidden_size, channels_out, kernel_size, self.stride, self.padding, self.dilation,
                                     bias=bias, device=device, dtype=dtype)

            if not (channels_in != channels_out or self.stride > 1 or self.dilation > 1):
                self.residual_layer = Identity()
            else:
                self.residual_layer = Convolution(channels_in, channels_out, 1, self.stride,
                                                  calc_padding(1, self.stride, self.dilation), self.dilation,
                                                  bias=bias, device=device, dtype=dtype)
        except Exception as e:
            print(CM((channels_in, channels_out, kernel_size, norm_groups, bias, device, dtype, options), Fore.LIGHTRED_EX))
            raise e

        # Attributes
        self.in_channels = channels_in
        self.out_channels = channels_out
        self.kernel_size = kernel_size
        self.norm_groups = norm_groups
        self.bias = bias

        # States
        self.device = device
        self.stype = dtype

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        # x: (Batch_Size, In_Channels, Height, Width)
        residue = tensor

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        tensor = self.conv1(self.activation(self.norm1(tensor, keys=keys)), keys=keys)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        tensor = self.conv2(self.activation(self.norm2(tensor, keys=keys)), keys=keys)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return tensor + self.residual_layer(residue, keys=keys)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}" \
               f"{display('stride', self.stride-1)}{display('dilation', self.dilation-1)}" \
               f"{display('pad', self.padding)}{display('pad_mode', self.padding_mode, self.padding)}" \
               f", bias={self.bias}, ng={self.norm_groups}, actv={self.activation.__class__.__name__})"


"""
Vanilla
"""


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, inp_features: int = None, depth=1, fwd_exp=4,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(TokenEmbedding, self).__init__()
        # ModifiedNEAT
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
        # ModifiedNEAT - [records, sequence, embed_size], EMBEDDING - [sequence, embed_size]
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

        # ModifiedNEAT
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
    def __init__(self, features: int, embed_size: int, bias=False, type='continuous',
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(BufferEmbedding, self).__init__()
        # BUILD
        if type == 'continuous':
            self.embedding = Linear(features, embed_size, bias, device, dtype)
        elif type == 'discrete':
            raise NotImplementedError(f"nn.Embedding not yet implemented for NEAT modules")
            self.padding_idx: int | None = manage_params(options, 'padding_idx', None)
            self.embedding = Embedding(features, embed_size, padding_idx=self.padding_idx, device=device, dtype=dtype)
        else:
            raise NotImplementedError(f"Unsupported embedding type: '{type}'")

        # ATTRIBUTES
        self.input_dim  = features
        self.embed_size = embed_size
        self.type       = type

        # STATES
        self.device = device
        self.dtype  = dtype

    def forward(self, tensor: Tensor, keys: Union[int, list[int]] = None, verbose: int = None):
        # Expand input to embedding space; [batch_size, sequence, *features] to [batch_size, sequence, embed_size]
        # print(f"forward={self.embedding.weights.data.shape, tensor.shape}")
        tensor = self.embedding(tensor, keys=keys)
        if verbose:
            print(get_tensor_info(tensor, "Embedded Tensor", verbose))

        return tensor


class BufferEncoding(NeatModule):
    def __init__(self, max_seq_len: int, embed_size: int, bias=True, type='discrete',
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(BufferEncoding, self).__init__()
        # BUILD
        assert max_seq_len >= 1
        if type == 'continuous':
            self.positions = torch.arange(max_seq_len, device=device, dtype=dtype).unsqueeze(0).unsqueeze(1).unsqueeze(-1)
            if max_seq_len > 1:
                self.positions /= (max_seq_len - 1)
        elif type == 'discrete':
            self.positions = torch.arange(max_seq_len, device=device, dtype=torch.int32).unsqueeze(0).unsqueeze(1)
        else:
            raise NotImplementedError(f"Unsupported encoding type: '{type}'")
        # shape(genomes, batch_size, seq_len, features)
        self.selector       = torch.arange(max_seq_len, device=device, dtype=torch.long)
        if type == 'continuous':
            self.encoding = Linear(1, embed_size, bias, device, dtype)
        elif type == 'discrete':
            raise NotImplementedError(f"nn.Embedding not yet implemented for NEAT modules")
            self.encoding = Embedding(max_seq_len, embed_size, device=device, dtype=dtype)
        # self.activation     = nn.SiLU()

        # ATTRIBUTES
        self.max_seq_len    = max_seq_len
        self.embed_size     = embed_size
        self.encoding_memory: Tensor | None = None

        # STATES
        self.device  = device
        self.dtype   = dtype

    def forward(self, tensor: Tensor, keys: Union[int, list[int]] = None, offset: tuple[int, int] = None, verbose: int = None, hold=False):
        if offset is None:
            offset = 0
        if verbose and verbose >= 2:
            print(get_tensor_info(tensor, "Unencoded tensor", verbose))
        # tensor = (genomes, batch_size, seq_len, embed_size)
        genomes, _, seq_len, _ = tensor.shape
        # Expanding positional encoding to shape of input
        positions = self.positions.expand(genomes, *self.positions.shape[1:])
        positions = torch.index_select(positions, -2, self.selector[offset:offset+seq_len])
        if verbose and verbose >= 2:
            print(get_tensor_info(positions, "Positions", verbose))
        positional_encoding: Tensor = self.encoding(positions, keys=keys)
        # positional_encoding = self.activation(positional_encoding)

        # Add encoding to tensor
        tensor = tensor + positional_encoding
        if verbose:
            print(get_tensor_info(positional_encoding, "Positional Encoding", verbose))
            print(get_tensor_info(tensor, "Encoded tensor", verbose))

        return tensor


class SequenceEncoding(NeatModule):
    def __init__(self, max_pixels: tuple[int, ...], embed_size: int, bias=True, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(SequenceEncoding, self).__init__()
        self.pixels_total  = int(math.prod(max_pixels))
        self.positions      = torch.arange(self.pixels_total, device=device, dtype=dtype).reshape(max_pixels).\
                                  unsqueeze(0).unsqueeze(1).unsqueeze(2)
        if self.pixels_total > 1:
            self.positions /= (self.pixels_total - 1)
        # BUILD
        if len(max_pixels) == 1:
            Convolution = Conv1d
        elif len(max_pixels) == 2:
            Convolution = Conv2d
        elif len(max_pixels) == 3:
            Convolution = Conv3d
        else:
            raise ValueError(f"Unsupported num of image dimension '{len(max_pixels)}'")
        # shape(genomes, batch_size, channels, *pixels)
        self.encoding       = Convolution(1, embed_size, 1, bias=bias, device=device, dtype=dtype)
        # self.activation     = nn.SiLU()

        # ATTRIBUTES
        self.max_pixels     = max_pixels
        self.ndim           = len(max_pixels)
        self.embed_size     = embed_size
        self.selector       = torch.arange(int(max(self.max_pixels)), device=device, dtype=torch.int32)
        self.encoding_memory: Tensor|None = None

        # STATES
        self.device  = device
        self.dtype   = dtype

    def forward(self, tensor: Tensor, keys: Union[int, list[int]] = None, offset: tuple[int, int] = None, verbose: int = None, hold=False):
        if offset is None:
            offset = [0 for _ in range(self.ndim)]
        if verbose and verbose >= 2:
            print(get_tensor_info(tensor, "Unencoded tensor", verbose))
        # tensor = (genomes, batch_size, channels, *pixels)
        (genomes, _, channels), pixels = tensor.shape[:3], tensor.shape[3:]
        assert len(pixels) == len(offset) == self.ndim

        # Expanding positional encoding to shape of input
        positions = self.positions.expand(genomes, *self.positions.shape[1:])
        for dim, off in zip(pixels[::-1], offset[::-1]):
            positions = torch.index_select(positions, -1, self.selector[off:off+dim])
        if verbose and verbose >= 2:
            print(get_tensor_info(positions, "Positions", verbose))
        positional_encoding: Tensor = self.encoding(positions, keys=keys)
        # positional_encoding = self.activation(positional_encoding)

        # Add encoding to tensor
        tensor = tensor + positional_encoding
        if verbose:
            print(get_tensor_info(positional_encoding, "Positional Encoding", verbose))
            print(get_tensor_info(tensor, "Encoded tensor", verbose))

        return tensor


# --------------------------------------------- #
# Attention                                     #
# --------------------------------------------- #

class RoPE(NeatModule):
    """
    Rotary Position Embedding (RoPE).

    Operates on tensors shaped (genomes, batch_size, seq_len, heads, head_dim).
    """

    def __init__(self, max_seq_len: int, embed_size: int, heads: int,
                 constant: int = 10_000,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.embed_size  = embed_size
        self.head_dim    = embed_size // heads
        self.heads       = heads
        self.constant    = constant
        self.device      = device
        self.dtype       = dtype

        # Use float32 for complex arithmetic when model dtype is lower precision
        self.conv_dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32

        # embeddings (seq_len, head_dim / 2)
        self.complex_frequencies = self._generate_encoding(max_seq_len, embed_size // heads, constant, 0)
        # embeddings (genomes=1, batch_size=1, seq_len, heads=1, head_dim / 2)
        self.complex_frequencies = self.complex_frequencies.to(device=device).unsqueeze(0).unsqueeze(1).unsqueeze(-2)
        self.select = torch.arange(max_seq_len, device=device, dtype=torch.long) # Sequence selector

    def extra_repr(self) -> str:
        return (
            f"max_seq_len={self.max_seq_len}, embed_size={self.embed_size}, "
            f"heads={self.heads}, constant={self.constant}"
        )

    # @staticmethod
    def _generate_encoding(self, seq_length: int, head_dim: int, constant: float = 10000.0, verbose: bool | int = None):
        # Dimensions of embedding must be even
        assert head_dim % 2 == 0, "head_dim must be divisible by 2"
        # Get theta where theta_i = 10000 ^ (-2 * (i-1) / embedding) for i = [1, 2, ..., dim / 2]; [head_dim / 2]
        theta    = 1.0 / torch.pow(constant, torch.arange(0, head_dim, 2).float() / head_dim)
        # Get positions as m; [sequence]
        positions = torch.arange(seq_length)
        # Multiply theta by each position; [sequence] outer* [head_dim / 2] -> [sequence, head_dim / 2]
        angles   = torch.outer(positions, theta).to(dtype=self.conv_dtype)
        # We compute complex number in polar form c = R * exp(i * m * theta); [sequence, head_dim / 2]
        cf = torch.polar(torch.ones_like(angles), angles)  # (seq_len, head_dim/2)
        if verbose:
            print(f"\nTheta =>\n{theta}\n\tdim = {theta.shape}")
            print(f"\nPositions =>\n{positions}\n\tdim = {positions.shape}")
            print(f"\nAngles =>\n{angles}\n\tdim = {angles.shape}")
            print(f"\nComplex Frequencies init =>\n{cf}\n\tdim = {cf.shape}")
        return cf

    def forward(self, tensor: Tensor, pos_idx: int = 0, verbose: int = None): # encode
        """
        Args:
            tensor:  (batch_size, seq_len, heads, head_dim)
            pos_idx: Starting position index.
        Returns:
            Same shape as input.
        """
        # NOTE: For uncached sequence, might want to start at 0.
        #       For instance use its absolute position in that sequence (be wary when the sequence is cached)
        # shape(genomes, batch_size, seq_len, heads, head_dim)
        seq_len = tensor.shape[-3]
        assert 0 < (pos_idx + seq_len) <= self.max_seq_len
        # [genomes, batch_size, sequence, heads, head_dim] -> [genomes, batch_size, sequence, heads, head_dim/2, 2]
        complex_tensor = torch.view_as_complex(tensor.view(*tensor.shape[:-1], -1, 2).to(self.conv_dtype))
        # [genomes, batch_size, sequence, heads, head_dim/2] * [1, 1, sequence, 1, head_dim/2] = [genomes, batch_size, sequence, heads, head_dim/2]
        complex_frequencies = torch.index_select(self.complex_frequencies, -3, self.select[pos_idx:pos_idx+seq_len])
        # for _ in range(max(0, complex_tensor.ndim - complex_frequencies.ndim)):
        #     complex_frequencies = complex_frequencies.unsqueeze(0)
        try:
            rotated_tensor = complex_tensor * complex_frequencies
        except Exception as e:
            print(get_tensor_info(self.complex_frequencies, 'Complex Frequencies Default Debugging', verbose))
            print(get_tensor_info(complex_frequencies, 'Complex Frequencies Debugging', verbose))
            print(get_tensor_info(complex_tensor, 'Complex Tensor Debugging', verbose))
            raise e
        # [genomes, batch_size, sequence, heads, head_dim / 2] -> [genomes, batch_size, sequence, heads, head_dim / 2, 2]
        split_tensor = torch.view_as_real(rotated_tensor).to(self.dtype)
        # [records, sequence, heads, head_dim / 2, 2] -> [records, sequence, heads, head_dim]
        # [records, sequence, heads, head_dim] -> [records, sequence, embed_size]
        tensor = split_tensor.reshape(*tensor.shape).type_as(tensor)
        if verbose: #  and verbose >= 3:
            print(get_tensor_info(complex_tensor, 'Complex Tensor', verbose))
            print(get_tensor_info(complex_frequencies, 'Complex Frequencies', verbose))
            print(get_tensor_info(rotated_tensor, 'Rotated Tensor', verbose))
            print(get_tensor_info(split_tensor, 'Split Tensor', verbose))
            print(get_tensor_info(tensor, 'Encoded Tensor', verbose))

        return tensor

    def decode(self, tensor: Tensor, pos_idx: int = 0, verbose: int = None):
        """Invert a RoPE-encoded tensor (multiply by complex conjugate)."""
        seq_len = tensor.shape[-3]

        # [..., head_dim] -> complex [..., head_dim/2]
        complex_tensor = torch.view_as_complex(
            tensor.view(*tensor.shape[:-1], -1, 2).to(self.conv_dtype)
        )

        complex_frequencies = torch.index_select(self.complex_frequencies, -3, self.select[pos_idx:pos_idx+seq_len])
        # for _ in range(max(0, complex_tensor.ndim - complex_frequencies.ndim)):
        #     complex_frequencies = complex_frequencies.unsqueeze(0)

        # Inverse rotation
        inverse_frequencies = torch.conj(complex_frequencies)

        try:
            restored_tensor = complex_tensor * inverse_frequencies
        except RuntimeError as e:
            print(f"")
            print(get_tensor_info(tensor, 'tensor', 1, Fore.LIGHTRED_EX))
            print(f"position_index = {pos_idx}")
            print(get_tensor_info(complex_tensor, 'complex_tensor', 1, Fore.LIGHTRED_EX))
            print(get_tensor_info(complex_frequencies, 'complex_frequencies', 1, Fore.LIGHTRED_EX))
            print(get_tensor_info(inverse_frequencies, 'inverse_frequencies', 1, Fore.LIGHTRED_EX))
            raise e

        split_tensor = torch.view_as_real(
            restored_tensor
        ).to(self.dtype)

        tensor = split_tensor.reshape(*tensor.shape).type_as(tensor)

        if verbose and verbose >= 3:
            print(get_tensor_info(complex_tensor, 'Encoded Complex Tensor', verbose))
            print(get_tensor_info(complex_frequencies, 'Complex Frequencies', verbose))
            print(get_tensor_info(inverse_frequencies, 'Inverse Frequencies', verbose))
            print(get_tensor_info(restored_tensor, 'Restored Tensor', verbose))
            print(get_tensor_info(split_tensor, 'Split Tensor', verbose))
            print(get_tensor_info(tensor, 'Decoded Tensor', verbose))

        return tensor


class Repositioning(NeatModule):
    """
    Guesses what the cached KV tensors would look like if shifted by `offset`
    positions (e.g. after a dequeue).  Useful for KV-cache management in
    multi-layer transformers beyond the first.
    """

    def __init__(
            self, max_seq_len: int, embed_size: int, layer_idx: int = None, bias: bool = True,
            rope: RoPE | None = None, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32,
            **options
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.embed_size  = embed_size
        self.layer_idx   = layer_idx if layer_idx is not None else 1
        self.bias        = bias
        self.rope        = rope
        self.epsilon     = manage_params(options, ['eps', 'epsilon'], 1e-9)
        self.affine      = manage_params(options, 'affine', True)
        self.device      = device
        self.dtype       = dtype

        self.embedder   = Embedding(max_seq_len, embed_size, None, device, dtype)
        self.norm       = RMSNorm(embed_size, self.epsilon, self.affine, device, dtype)
        self.converter  = Linear(embed_size, embed_size, bias, device, dtype)


    def forward(self, tensor: Tensor, offset: int, keys: int | Iterable[int] = None, verbose: int = None):
        """
        Args:
            tensor: (batch_size, seq_len, embed_size)
            offset: Number of positions to shift.
        Returns:
            Same shape as input.
        """
        if offset == 0: # No need for model to learn to guess the same values
            return tensor

        if self.layer_idx == 0 and self.rope is not None:
            # NOTE: This assumes input is most probably idx(1:max_seq_len-1) being shifted to idx(0:max_seq_len-2)
            #       that was being cached somewhere
            # TODO: Decode RoPE then get the next ascending positions' encoding
            raw_tensor = self.rope.decode(tensor, pos_idx=offset, verbose=verbose) # This should have no RoPE
            if verbose: #  and verbose >= 3:
                print(get_tensor_info(raw_tensor, 'Raw Tensor', verbose))
            repo_tensor = self.rope(raw_tensor, pos_idx=0)
        else:
            # NOTE: Guess possible future tensors with current tensor and offset, since with different positional
            #       embeddings the subsequent layer would have different tensors for queries keys and values
            shift = self.embedder(offset-1, keys=keys) # shape(genomes, dim_size)
            if verbose: #  and verbose >= 3:
                print(get_tensor_info(shift, f'Shift({offset})', verbose))
            # shift(genomes, dim_size) -> tensor(genomes, ..., dim_size)
            while shift.ndim < tensor.ndim:
                shift = shift.unsqueeze(1)
            repo_tensor = self.converter(self.norm(tensor + shift))
        if verbose: #  and verbose >= 3:
            print(get_tensor_info(repo_tensor, 'Repositioned Tensor', verbose))
        return repo_tensor


class AttentionLambda(NeatModule):
    """
    Learnable per-head scalar gains used by Differential Attention.

    Produces a tensor shaped (genomes, batch_size, heads, lambdas, 1, 1) that is
    broadcast over the (..., query_len, key_len) attention map.
    """

    u_lim = 0.8
    l_lim = 0.2

    def __init__(
        self, heads: int, head_dim: int, layer_idx: int = None,
        lambdas: int = 1, init_mean: float = 0., init_std: float = 0.5,
        max_gain: float = 2.0, affine: bool = True, epsilon: float = 1e-8,
        device: DEVICE = 'cpu', dtype: DTYPE = torch.float32
    ):
        super().__init__(
            heads=heads, head_dim=head_dim, coeffs=lambdas,
            base_limits=(self.u_lim, self.l_lim), base_affine=affine
        )

        self.heads      = heads
        self.head_dim   = head_dim
        self.layer_idx  = layer_idx if layer_idx is not None else 0
        self.lambdas    = lambdas
        self.max_gain   = max_gain
        self.base_affine = affine
        self.epsilon    = epsilon
        self.range      = abs(self.u_lim - self.l_lim)

        # self.q1 = NeatParameter((heads, head_dim, lambdas), requires_grad=False, device=device, dtype=dtype)
        # self.q2 = NeatParameter((heads, head_dim, lambdas), requires_grad=False, device=device, dtype=dtype)
        # self.k1 = NeatParameter((heads, head_dim, lambdas), requires_grad=False, device=device, dtype=dtype)
        # self.k2 = NeatParameter((heads, head_dim, lambdas), requires_grad=False, device=device, dtype=dtype)
        # self.init = (
        #     self.u_lim - (self.range * math.exp(-0.3 * self.layer_idx))
        #     if not affine else
        #     NeatParameter((heads, 1), requires_grad=False, device=device, dtype=dtype) # .unsqueeze(2)
        # )
        self.scalars = NeatParameter((heads, head_dim, lambdas), requires_grad=False, device=device, dtype=dtype)
        # shape (1, lambdas) — broadcasts with gain (heads, lambdas)
        self.exponents = (torch.arange(lambdas, device=device, dtype=dtype) + 1).unsqueeze(0).unsqueeze(1)
        self.multipliers = torch.pow(-1, self.exponents)

        self.mean = init_mean
        self.std  = init_std

        # Register a hook to modify gradients
        for param in self.neat_parameters():
            with torch.no_grad():
                param.data.normal_(init_mean, init_std)
        self.min_val = init_mean - init_std * 2
        self.max_val = init_mean + init_std * 2

        # with torch.no_grad():
        #     for p in [self.q1, self.q2, self.k1, self.k2]:
        #         p.data.normal_(init_mean, init_std)

    def extra_repr(self):
        return (
            f"heads={self.heads}, head_dim={self.head_dim}, coeffs={self.lambdas}, "
            f"base_limits={(self.u_lim, self.l_lim)}, base_affine={self.base_affine}"
        )

    def init_affine(self, keys: Union[int, list[int], None]):
        # if isinstance(self.init, NeatParameter):
        #     return self.l_lim + (self.range * torch.sigmoid(self.init[keys]))
        # else:
        #     return self.init
        return 0.0

    def post_attention_shift(self, keys: Union[int, list[int], None], offset: int | None = None, verbose: int = False):
        """Returns a scalar/tensor broadcastable with attended (batch, q_len, heads, head_dim)."""
        # # TODO: Fix the parameterized implementation of init
        # # return self.init if not isinstance(self.init, NeatParameter) else self.expand(self.init[keys], tensor, offset=offset, keys=keys)
        # if not isinstance(self.init, NeatParameter):
        #     shift = self.init
        # else:
        #     # reshape (heads, 1) -> (1, 1, heads, 1) to broadcast over (b, q, h, d)
        #     parameters = self.init[keys]
        #     shift = self.l_lim + (self.range * torch.sigmoid(parameters))
        #     shift = shift.unsqueeze(1).unsqueeze(2)
        shift = torch.tensor(0.0, device=self.scalars.device, dtype=self.scalars.dtype)
        if verbose:
            print(get_tensor_info(shift, 'Shift', verbose))
        return shift

    # def update_limit(self):
    #     for param in self.neat_parameters():
    #         with torch.no_grad():
    #             param.data[:] = torch.clamp(param.data, self.min_val, self.max_val)

    def forward(self, keys: Union[int, Iterable[int]] = None):
        """
        Returns:
            (1, heads, lambdas, 1, 1)
            Broadcasts over energy shaped (batch, heads, coeffs, q_len, k_len).
        """
        # query:     (batch_size, q_len, heads, head_dim)
        # key:       (batch_size, k_len, heads, head_dim)
        # attention: (batch_size, heads, q_len, k_len)
        # # TODO: Might want to verify the need of clamping the parameters
        # q1 = self.q1[keys] # * self.std # F.tanh(self.q1[keys], self.min_val, self.max_val)
        # k1 = self.k1[keys] # * self.std # F.tanh(self.k1[keys], self.min_val, self.max_val)
        # q2 = self.q2[keys] # * self.std # F.tanh(self.q2[keys], self.min_val, self.max_val)
        # k2 = self.k2[keys] # * self.std # F.tanh(self.k2[keys], self.min_val, self.max_val)
        # gain = (
        #     torch.exp(torch.sum(q1 * k1, dim=-2)) -
        #     torch.exp(torch.sum(q2 * k2, dim=-2))
        # )
        # gain = torch.sigmoid(gain) * self.max_gain
        # TODO: Most implementations of NEAT weight mutation do not scale well so exponential calculations are to be avoided
        gain = torch.clamp(torch.mean(torch.tanh(self.scalars[keys]) * self.max_gain, dim=-2), -self.max_gain, +self.max_gain)
        gain = (gain + self.max_gain) / 2
        assert torch.all((gain >= 0) & (gain <= self.max_gain))
        # self.exponents: (1, lambdas), self._init_affine(): (heads, 1) or scalar
        scalars = (
            # (base + self.biases(base, keys, 0)) * self.multipliers # ** self.exponents * self.multipliers
            ((gain + self.init_affine(keys)) ** self.exponents) * self.multipliers
        )
        # Returns shape (genomes, batch_size=1, heads, lambdas, query_len=1, key_len=1)
        return scalars.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)


class Attention(NeatModule):
    """
    Multi-head (grouped-query) self-attention with:
      - RoPE positional embeddings
      - Optional differential attention (λ-weighted multi-head combination)
      - Optional KV-cache with dequeue-based overflow handling
      - Pre-RMSNorm + residual connection
    """

    def __init__(
            self, max_seq_len: int, dim_size: int, heads: int = None, kv_heads: int = None,
            differential: int | bool = None, layer_idx: int = None, causal_mask: bool = True,
            bias: bool = False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options
    ):
        super().__init__()

        # ---- head / dim bookkeeping ----------------------------------------
        if heads is None:
            heads = 1
        if dim_size % heads != 0:
            raise ValueError("dim_size must be divisible by heads")
        if kv_heads is None:
            kv_heads = heads
        if heads % kv_heads != 0:
            raise ValueError("heads must be divisible by kv_heads")
        if isinstance(differential, bool):
            differential = int(differential)

        # ---- options -------------------------------------------------------
        self.auto_single    = manage_params(options, 'auto_single', False)
        self.constant       = manage_params(options, 'constant', 10_000)
        self.epsilon        = manage_params(options, ['eps', 'epsilon'], 1e-9)
        self.affine         = manage_params(options, 'affine', True)
        self.skip_connection = manage_params(options, ['skip_connection', 'residual'], True)
        self.normalize      = manage_params(options, 'normalize', True)
        self.cache          = manage_params(options, ['cache', 'kv_cache'], True)
        inputs: int         = manage_params(options, 'inputs', None)
        outputs: int        = manage_params(options, 'outputs', None)
        self.repo_enabled   = manage_params(options, 'reposition', True)

        # ---- attributes ----------------------------------------------------
        self.bias        = bias
        self.heads       = heads
        self.dim_size    = dim_size
        self.head_dim    = dim_size // heads
        self.kv_heads    = kv_heads
        self.q_kv_ratio  = heads // kv_heads
        self.max_seq_len = max_seq_len
        self.causal_mask = causal_mask
        self.differential      = differential
        self.mult              = 1 if not differential else 1 + int(differential)
        self.att_coeff_num     = None if not differential else 1 + int(differential)
        self.att_coeff_indices = None if not differential else torch.arange(differential, device=device) + 1
        self.device = device
        self.dtype  = dtype

        # ---- sub-modules ---------------------------------------------------
        in_dim  = dim_size if inputs  is None else inputs
        out_dim = dim_size if outputs is None else outputs

        # self.pre_norm = RMSNorm(dim_size, self.epsilon, self.affine, device, dtype) if self.normalize else None
        self.pre_norm = LayerNorm(dim_size, self.epsilon, self.affine, bias, device, dtype) if self.normalize else None

        self.query_proj = Linear(in_dim, heads * self.head_dim * self.mult, bias, device, dtype)
        self.key_proj   = Linear(in_dim, kv_heads * self.head_dim * self.mult, bias, device, dtype)
        self.value_proj = Linear(in_dim, kv_heads * self.head_dim, bias, device, dtype)
        self.out_proj   = Linear(dim_size, out_dim, bias, device, dtype)

        self.rotary_embedding = RoPE(self.max_seq_len, dim_size * self.mult, heads, self.constant, device, dtype)
        self.softmax    = nn.Softmax(dim=-1)
        # self.head_norm  = RMSNorm(self.head_dim, self.epsilon, self.affine, device, dtype) if self.normalize else None
        self.head_norm  = LayerNorm(self.head_dim, self.epsilon, self.affine, bias, device, dtype) if self.normalize else None
        self.diff_lambda = AttentionLambda(
            heads, self.head_dim, layer_idx, differential, 0.0, 0.1, 2.0, True, self.epsilon, device, dtype
        ) if differential else None

        # Repositioning helpers for KV-cache dequeue
        self.repo_k = Repositioning(
            max_seq_len, self.head_dim * self.mult, layer_idx, bias, self.rotary_embedding, device, dtype, **options
        ) if self.cache else None
        value_rope = (
            self.rotary_embedding if self.mult == 1 else 
            RoPE(self.max_seq_len, dim_size, heads, self.constant, device, dtype)
        ) # TODO: Does heads even have an effect on RoPE
        self.repo_v = Repositioning(
            max_seq_len, self.head_dim, layer_idx, bias, value_rope, device, dtype, **options
        ) if self.cache else None

        # ---- KV-cache state ------------------------------------------------
        self.cache_k: Tensor | None = None  # (genomes, batch_size, max_seq_len, heads, head_dim*mult)
        self.cache_v: Tensor | None = None  # (genomes, batch_size, max_seq_len, heads, head_dim)
        self.cache_size: int = 0 # TODO: Maybe make cache size a tensor that references each genome (and batch index maybe)
        self.forced_cache: bool = False

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
        
    def force_cache(self, enable: bool = True):
        self.forced_cache = enable
        if enable:
            if self.cache_k is None:
                self._init_cache(None, 1)
        else:
            # if not self.cache:
            # Save memory by clearing any cache, especially on GPU memory. NOTE: cuda.empty_cache()
            self.cache_k = self.cache_v = None
            self.cache_size = 0
        
    def empty_cache(self):
        """Reset cache to initial state."""
        if self.cache_k is None:
            self._init_cache(None, 1)
        if self.cache_k is not None:
            self.cache_k.zero_()
        if self.cache_v is not None:
            self.cache_v.zero_()
        self.cache_size = 0
    
    def _init_cache(self, genomes: int | None = None, batch_size: int = 1):
        """Resize cache if batch dimensions changed."""
        # TODO: Modify parameters entered to cater for extra dims
        if genomes is None:
            genomes = self.genome_num if self.genome_num is not None else 1
        self.cache_k = torch.zeros(
            genomes, batch_size, self.max_seq_len, self.kv_heads, self.head_dim * self.mult,
            device=self.device, dtype=self.dtype
        )
        self.cache_v = torch.zeros(
            genomes, batch_size, self.max_seq_len, self.kv_heads, self.head_dim,
            device=self.device, dtype=self.dtype
        )
        self.cache_size = 0
    
    def _update_cache(self, key: Tensor, value: Tensor, pos_idx: int, verbose: bool | int = False):
        """Update cache with optional deque-like behavior.
        
        Args:
            key: (genomes, batch_size, seq_len, kv_heads, head_dim * mult)
            value: (genomes, batch_size, seq_len, kv_heads, head_dim)
            pos_idx: If None, append to cache. If specified, place at position and truncate after.
        """
        g, b, s, _, _ = key.shape
        
        if s > self.max_seq_len:
            raise ValueError(
                f"Input tensor's sequence length is greater than module's max sequence length; "
                f"{s} from K({key.shape}) | V({value.shape}) > {self.max_seq_len}"
            )
        
        # Resize cache if needed
        if self.cache_k.shape[0] != g or self.cache_k.shape[1] != b:
            self._init_cache(g, b)
        
        if self.forced_cache:
            # Deque-like append: add new tensor after current cache position
            start_pos = self.cache_size
            end_pos   = start_pos + s
        else:
            # Position-indexed placement with truncation
            if pos_idx < 0 or pos_idx >= self.max_seq_len:
                raise ValueError(
                    f"Position index '{pos_idx}' must be within limits (0, {self.max_seq_len}-1)"
                )
            # Place tensor at position pos_idx
            start_pos = pos_idx
            end_pos   = pos_idx + s
            if end_pos > self.max_seq_len:
                raise ValueError(
                    f"Position index '{pos_idx}' with input sequence length '{s}' "
                    f"exceeds max sequence length '{self.max_seq_len}'"
                )
            
        # Shift excess
        excess = max(0, end_pos - self.max_seq_len)
        if verbose: print(f"Shifting cache by {excess} position(s)")
        if excess > 0:
            try:
                # TODO: Modify to use tensor.select to properly cater for ex dims
                if key.ndim >= 6: raise NotImplementedError()
                self.cache_k[:, :, :-excess] = self.repo_k(self.cache_k[:, :, excess:], excess, keys=None, verbose=verbose)
                self.cache_v[:, :, :-excess] = self.repo_v(self.cache_v[:, :, excess:], excess, keys=None)
            except Exception as e:
                print(f"excess = {excess}")
                print(f"K: {key.shape} -> {self.cache_k.shape}")
                print(f"V: {value.shape} -> {self.cache_v.shape}")
                raise e
            
        # Only store what fits
        start_pos -= excess
        end_pos   -= excess
        if verbose: print(f"Setting position {start_pos} to {end_pos}")
        # TODO: Modify to use tensor.select to properly cater for ex dims
        if key.ndim >= 6: raise NotImplementedError()
        self.cache_k[:, :, start_pos:end_pos] = key
        self.cache_v[:, :, start_pos:end_pos] = value
        
        self.cache_size = min(end_pos, self.max_seq_len)

    # ------------------------------------------------------------------
    # GQA helper
    # ------------------------------------------------------------------

    def repeat_kv(self, tensor: Tensor):
        """Expand key/value heads to match query heads (GQA)."""
        # tensor: (genomes, *extra_dims, seq_len, kv_heads, head_dim)
        if self.q_kv_ratio == 1:
            return tensor
        genomes, pre, kv, post = tensor.shape[0], tensor.shape[1:-2], tensor.shape[-2], tensor.shape[-1]
        return (
            tensor
            .unsqueeze(-2)
            .expand(genomes, *pre, kv, self.q_kv_ratio, post)
            .reshape(genomes, *pre, kv * self.q_kv_ratio, post)
        )

    # ------------------------------------------------------------------
    # Core attention computation
    # ------------------------------------------------------------------

    def attention(
        self, query: Tensor, key: Tensor, value: Tensor,
        keys: Union[int, Iterable[int]], mask: bool, verbose: int = None
    ):
        """
        Args:
            query:  (genomes, batch, q_len,  heads, head_dim [* mult])
            key:    (genomes, batch, kv_len, heads, head_dim [* mult])
            value:  (genomes, batch, kv_len, heads, head_dim)
        Returns:
            (scores, attended_values) with attended shaped (batch, q_len, heads, head_dim)
        """
        if self.differential:
            query = query.view(*query.shape[:-1], self.att_coeff_num, -1)
            key   = key.view(*key.shape[:-1], self.att_coeff_num, -1)

        # Get the attention score (energy)
        eqn = "...qhd,...khd->...hqk" if not self.differential else "...qhcd,...khcd->...hcqk"
        energy = torch.einsum(eqn, [query, key])
        # queries shape: (genomes, batch_size, query_len, heads, *coeffs, head_dim)
        # key shape:     (genomes, batch_size, key_len, heads, *coeffs, head_dim)
        # energy shape:  (genomes, batch_size, heads, *coeffs, query_len, key_len)
        if verbose:
            print(get_tensor_info(energy, 'Energy', verbose))

        if self.differential:
            lambdas: Tensor = self.diff_lambda(keys)
            if verbose and verbose >= 2:
                print(get_tensor_info(torch.round(lambdas, decimals=4), 'Lambdas', verbose+2))
                print(get_tensor_info(self.att_coeff_indices, 'Coeff Indices', verbose+2))
            # Index '-3' is the lambdas' coefficients dimension
            # Collapse the coefficient dimension: (g, b, h, c, q, k) -> (g, b, h, q, k)
            # energy = self.softmax(energy) #  / math.sqrt(self.head_dim))
            energy = torch.select(energy, dim=-3, index=0) + torch.sum(
                torch.index_select(energy, dim=-3, index=self.att_coeff_indices) * lambdas, dim=-3
            )

        if mask:
            q_len, k_len = energy.shape[-2:]
            diff = k_len - q_len
            mask_ = torch.ones_like(energy, dtype=torch.bool).triu(1 + diff)
            energy.masked_fill_(mask_, -torch.inf)
            if verbose and verbose >= 2:
                print(get_tensor_info(mask_, 'Mask', verbose))
                print(get_tensor_info(energy, 'Masked Energy', verbose))

        # Get the softmax of the energy: scores is always (g, b, h, q, k) at this point
        scores = self.softmax(energy / math.sqrt(self.head_dim))
        if verbose:
            print(get_tensor_info(torch.round(scores, decimals=4), 'Attention Score', verbose))

        # Get the weighted sum of the values and reshape to remove heads
        attention = torch.einsum("...hqv,...vhd->...qhd", [scores, value])
        if self.head_norm is not None:
            attention = self.head_norm(attention, keys=keys)
        # scores shape:    (genomes, batch_size, heads, query_len, value_len)
        # values shape:    (genomes, batch_size, value_len, heads, head_dim)
        # attention shape: (genomes, batch_size, query_len, heads, head_dim) then concat last 2 dim
        if self.differential:
            # attention = attention * (1 - self.diff_lambda.biases(attention, keys, 2))
            attention = attention * (1 - self.diff_lambda.post_attention_shift(keys, verbose=verbose))

        return scores, attention

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, prompt: Tensor = None,
                pos_idx: int | None = 0, verbose: int = None, get: bool = False):
        """
        Args:
            tensor:  (batch_size, seq_len, dim_size)
            prompt:  Optional query-only tensor of the same shape (cross-attention).
            pos_idx: Starting position for RoPE.  None → 0.
            get:     If True, also return the raw attention scores.
        Returns:
            output tensor (batch_size, seq_len, dim_size)
            or (output, scores) when get=True.
        """
        if self.auto_single and prompt is None:
            prompt = tensor.select(-2, -1).unsqueeze(-2)
        if pos_idx is not None:
            seq_len = tensor.shape[-2]
            pos_idx = (seq_len + pos_idx) if pos_idx < 0 else pos_idx
            assert 0 <= pos_idx < seq_len
        if pos_idx is None:
            pos_idx = 0
        if verbose:
            print(f'\n{CM("Executing Self Attention", Fore.LIGHTBLUE_EX)}')
            print(get_tensor_info(tensor, f'Input', verbose, Fore.LIGHTRED_EX))
            if prompt is not None:
                print(get_tensor_info(prompt, f'Prompt', verbose, Fore.LIGHTRED_EX))

        # Get residue and pre-normalize
        residue = tensor if prompt is None else prompt
        if self.pre_norm is not None:
            tensor = self.pre_norm(tensor, keys=keys)
            if prompt is not None:
                prompt = self.pre_norm(prompt, keys=keys)

        # Linearize Q, K, V
        query: Tensor   = self.query_proj(tensor if prompt is None else prompt, keys=keys)
        key: Tensor     = self.key_proj(tensor, keys=keys)
        value: Tensor   = self.value_proj(tensor, keys=keys)
        if verbose:
            print(CM('Post Linearization =>'))
            print(get_tensor_info(query, 'Query', verbose, Fore.LIGHTRED_EX))
            print(get_tensor_info(key, 'Key', verbose, Fore.LIGHTGREEN_EX))
            print(get_tensor_info(value, 'Value', verbose, Fore.LIGHTBLUE_EX))

        # Reshape Q, K, V for each rep head
        g, b, s, d = query.shape
        query   = query.view(g, b, s, self.heads, self.head_dim*self.mult)
        key     = key.view(g, b, -1, self.kv_heads, self.head_dim*self.mult)
        value   = value.view(g, b, -1, self.kv_heads, self.head_dim)
        if verbose:
            print(get_tensor_info(query, 'Q after Reshaping', verbose))
            print(get_tensor_info(key, 'K after Reshaping', verbose))

        # Apply Rotary Embeddings NOTE: Ensure when one-shot/single-shot prompting the correct position is provided
        key_len = key.shape[-3]
        if self.forced_cache:
            overflow = (self.cache_size + key_len) > self.max_seq_len
            rope_pos = self.cache_size if not overflow else (self.max_seq_len - key_len) # NOTE: dequeing
        else:
            rope_pos = pos_idx
        # pos_idx = min(self.max_seq_len - 1, max(0, pos_idx)) 

        q_pi = rope_pos if not self.forced_cache else rope_pos + (key_len - 1)
        k_pi = 0 if not self.forced_cache else rope_pos
        query = self.rotary_embedding(query, q_pi, verbose)
        key   = self.rotary_embedding(key, k_pi) # TODO: Only temporary for current use case
        if verbose:
            print(get_tensor_info(query, f'Q after Rotary Embedding({q_pi}, {s})', verbose))
            print(get_tensor_info(key, f'K after Rotary Embedding({k_pi}, {key_len})', verbose))

        # KV-cache
        if self.forced_cache and self.cache_k is not None: # TODO: Using forced_cache flag instead of cache for now
            self._update_cache(key, value, rope_pos, verbose)
            # Use cached key and value up to cache_pos
            size  = (pos_idx + s) if not self.forced_cache else self.cache_size
            key   = self.cache_k[:, :, :size]
            value = self.cache_v[:, :, :size]
            if verbose:
                print(f'Using cached K, V with cache_size={size}')
                print(get_tensor_info(key, 'Cached K', verbose))
                print(get_tensor_info(value, 'Cached V', verbose))

        # GQA expansion
        key   = self.repeat_kv(key)
        value = self.repeat_kv(value)
        if verbose:
            print(get_tensor_info(key, 'Duplicated K', verbose))
            print(get_tensor_info(value, 'Duplicated V', verbose))

        # Apply attention
        attention_scores, attention = self.attention(
            query, key, value, keys, self.causal_mask, verbose
        )
        attention = attention.reshape(g, b, s, self.dim_size)
        # out_view shape:  (genomes, batch_size, seq_len, dim_size)
        if verbose:
            print(get_tensor_info(attention, 'Attended Values', verbose))

        # Output projection + residual
        tensor: Tensor = self.out_proj(attention, keys=keys)
        if self.skip_connection:
            tensor = tensor + residue
        if verbose:
            print(get_tensor_info(tensor, 'Output Projection', verbose))

        # Store attention for debugging
        if not get:
            return tensor
        else:
            return tensor, attention_scores

    def extra_repr(self) -> str:
        return (f"*** "
                f"residual={self.skip_connection}, normalize={self.normalize}, "
                f"auto_single={self.auto_single}. causal_mask={self.causal_mask}"
                f"***")

    # def __repr__(self):
    #     return f"{self.__class__.__name__}({self.dim_size}, pixels={self.pixels}, kernel_size={self.kernel_size}, " \
    #            f"bias={self.bias}{display('hd', self._heads)}{display('kv_hd', self._kv_heads)}" \
    #            f"{display('diffs', self.att_coeff_num-1 if self.att_coeff_num else None)}" \
    #            f"{display('pad', self.padding)}{display('pad_mode', self.padding_mode, self.padding)}" \
    #            f"{display('pre_norm_ng', self.norm_groups)}{display('residual', self.skip_connection)}" \
    #            f"{display('as', self.auto_single)}"

# TODO: Convoluted attention is not updated yet

class ConvSelfAttention(NeatModule):
    def __init__(
            self, max_pixels: list[int], dim_size: int, kernel_size: int, heads: int = None, kv_heads: int = None,
            differential: int = None, layer_idx: int = None, causal_mask=False,
            bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options):
        super(ConvSelfAttention, self).__init__()
        self._heads = heads
        self._kv_heads = kv_heads
        if heads is None:
            heads = 1
        if dim_size % heads != 0:
            raise ValueError(f"Embedding dimensions must be a multiple of heads num")
        if kv_heads is None:
            kv_heads = heads
        if heads % kv_heads != 0:
            raise ValueError(f"Query heads num must be a multiple of number of Key-Value heads num")
        if isinstance(differential, bool):
            differential = int(differential)
        if not isinstance(max_pixels, (list, tuple)):
            raise ValueError(f"pixels parameter must be Iterable got '{type(max_pixels)}'")

        # OPTIONS
        self.auto_single = manage_params(options, 'auto_single', False)
        if self.auto_single:
            kernel_size = 1
        self.padding = manage_params(options, 'padding', calc_padding(kernel_size, stride=1, dilation=1))
        self.padding_mode = manage_params(options, 'padding_mode', 'zeros')
        self.constant = manage_params(options, 'constant', 10000)
        self.epsilon = manage_params(options, 'epsilon', 1e-9)
        self.affine = manage_params(options, 'affine', True)
        self.norm_groups = manage_params(options, 'norm_groups', 1)
        self.skip_connection = manage_params(options, ['skip_connection', 'residual'], True)

        # ATTRIBUTES
        self.heads      = heads
        self.dim_size   = dim_size
        self.head_dim   = dim_size // heads
        self.kv_heads   = kv_heads
        self.q_kv_ratio = heads // kv_heads
        self.pixels     = max_pixels
        self.differential = differential
        self.att_coeff_num = None if not differential else 1+differential
        self.att_coeff_indices = None if not differential else torch.arange(differential, device=device) + 1
        self.causal_mask = causal_mask
        self.pixels_total = math.prod(max_pixels)
        self.kernel_size = kernel_size
        self.bias = bias

        # MODULES
        self.pre_norm = GroupNorm(
            self.norm_groups, dim_size, self.epsilon, self.affine, False, device=device, dtype=dtype
        ) if self.norm_groups else None
        if len(max_pixels) == 1:
            Convolution = Conv1d
        elif len(max_pixels) == 2:
            Convolution = Conv2d
        elif len(max_pixels) == 3:
            Convolution = Conv3d
        else:
            raise ValueError(f"Unsupported num of image dimension '{len(max_pixels)}'")
        self.mult = 1+differential if differential else 1
        self.query_proj = Convolution(dim_size, heads * self.head_dim * self.mult, kernel_size=kernel_size,
                                      padding=self.padding, padding_mode=self.padding_mode,
                                      bias=bias, device=device, dtype=dtype)
        self.key_proj   = Convolution(dim_size, kv_heads * self.head_dim * self.mult, kernel_size=kernel_size,
                                      padding=self.padding, padding_mode=self.padding_mode,
                                      bias=bias, device=device, dtype=dtype)
        self.value_proj = Convolution(dim_size, kv_heads * self.head_dim, kernel_size=kernel_size,
                                      padding=self.padding, padding_mode=self.padding_mode,
                                      bias=bias, device=device, dtype=dtype)
        self.out_proj   = Convolution(dim_size, dim_size, kernel_size=kernel_size,
                                      padding=self.padding, padding_mode=self.padding_mode,
                                      bias=bias, device=device, dtype=dtype)
        self.rotary_embedding = RoPE(self.pixels_total, dim_size * self.mult, heads, self.constant, device, dtype)
        self.softmax    = nn.Softmax(-1)
        self.head_norm  = RMSNorm(self.head_dim, self.epsilon, self.affine, device, dtype)
        self.diff_lambda = AttentionLambda(
            heads, self.head_dim, layer_idx, differential, 0.0, 0.1, True, self.epsilon, device, dtype
        ) if differential else None

        # STATES
        self.device = device
        self.dtype  = dtype

    def repeat_kv(self, tensor: Tensor):
        genomes, batch_size, seq_len, kv_heads, head_dim = tensor.shape
        if self.q_kv_ratio == 1:
            return tensor
        else:
            return tensor.unsqueeze(-2).expand(genomes, batch_size, seq_len, kv_heads, self.q_kv_ratio, head_dim).\
                contiguous().view(genomes, batch_size, seq_len, kv_heads * self.q_kv_ratio, head_dim)

    @staticmethod
    def convert(image: Tensor, heads: int, head_dim: int, multiplier: int = 1):
        (g, b), c, p = image.shape[:2], image.shape[2] // multiplier, image.shape[3:]
        assert c == heads * head_dim
        # Return shape (genomes, batch_size, pixels, heads, head_dim)
        return image.view(g, b, c, -1).transpose(-1, -2).contiguous().view(g, b, -1, heads, head_dim*multiplier), (g, b, c, p)

    @staticmethod
    def revert(tensor: Tensor, batch_size: int, channels: int, pixels: list[int]):
        # Return shape (genomes, batch_size, channels, *pixels)
        return tensor.transpose(-1, -2).contiguous().view(-1, batch_size, channels, *pixels)

    def attention(self, query: Tensor, key: Tensor, value: Tensor, keys: Union[int, Iterable[int]], mask: bool, verbose: int = None):
        if self.differential:
            query   = query.view(*query.shape[:-1], self.att_coeff_num, -1)
            key     = key.view(*key.shape[:-1], self.att_coeff_num, -1)
        # Get the attention score (energy)
        energy = torch.einsum("...qhd,...khd->...hqk" if not self.differential else "...qhcd,...khcd->...hcqk", [query, key])
        # queries shape: (genomes, batch_size, query_len, heads, *coeffs, head_dim)
        # key shape:     (genomes, batch_size, key_len, heads, *coeffs, head_dim)
        # energy shape:  (genomes, batch_size, heads, *coeffs, query_len, key_len)
        if verbose:
            print(get_tensor_info(energy, 'Energy', verbose))

        if mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask_ = torch.ones_like(energy, dtype=torch.bool).triu(1)
            # Fill the upper triangle with -inf
            energy.masked_fill_(mask_, -torch.inf)
            if verbose and verbose >= 2 and not self.differential:
                print(get_tensor_info(mask_, 'Mask', verbose))
                print(get_tensor_info(energy, 'Masked Energy', verbose))

        # Get the softmax of the energy
        scores = self.softmax(energy / math.sqrt(self.head_dim))

        if self.differential:
            lambdas: Tensor = self.diff_lambda(keys)
            if verbose and verbose >= 2:
                print(get_tensor_info(torch.round(lambdas, decimals=4), 'Lambdas', verbose+2))
                print(get_tensor_info(self.att_coeff_indices, 'Coeff Indices', verbose+2))
            scores = torch.select(scores, dim=-3, index=0) + torch.sum(
                torch.index_select(scores, dim=-3, index=self.att_coeff_indices) * lambdas, dim=-3
            ) # * lambdas)

            if mask:
                scores.masked_fill_(torch.ones_like(scores, dtype=torch.bool).triu(1), -torch.inf)
                if verbose and verbose >= 2 and not self.differential:
                    print(get_tensor_info(scores, 'Differential Masked Energy', verbose))
            scores = self.softmax(scores)

        if verbose:
            print(get_tensor_info(torch.round(scores, decimals=4), 'Attention Score', verbose))

        # Get the weighted sum of the values and reshape to remove heads
        attention = self.head_norm(torch.einsum("...hqv,...vhd->...qhd", [scores, value]), keys=keys)
        # scores shape:    (genomes, batch_size, heads, query_len, value_len)
        # values shape:    (genomes, batch_size, value_len, heads, head_dim)
        # attention shape: (genomes, batch_size, query_len, heads, head_dim) then concat last 2 dim
        if self.differential:
            attention = attention * (1 - self.diff_lambda.biases(attention, keys, 2))

        return scores, attention

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, 
                pretext: Tensor = None, pos_idx: int = None, verbose: int = None, get=False):
        if self.auto_single and len(self.pixels) == 1 and pretext is None:
            pretext = tensor.select(-1, -1).unsqueeze(-1)
        if pos_idx is not None:
            pos_idx = self.pixels_total + pos_idx if pos_idx < 0 else pos_idx
            assert 0 < pos_idx < self.pixels_total
        if verbose:
            print(f'\n{CM("Executing Self Attention", Fore.LIGHTBLUE_EX)}')
            print(get_tensor_info(tensor, f'Input', verbose, Fore.LIGHTRED_EX))
            if pretext is not None:
                print(get_tensor_info(pretext, f'Pretext', verbose, Fore.LIGHTRED_EX))

        # Get residue and pre-normalize
        residue = tensor if pretext is None else pretext
        if self.pre_norm is not None:
            tensor = self.pre_norm(tensor, keys=keys)
            if pretext is not None:
                pretext = self.pre_norm(pretext, keys=keys)

        # Convolve Q, K, V
        query: Tensor   = self.query_proj(tensor if pretext is None else pretext, keys=keys)
        key: Tensor     = self.key_proj(tensor, keys=keys)
        value: Tensor   = self.value_proj(tensor, keys=keys)
        if verbose:
            print(CM('Post Linearization =>'))
            print(get_tensor_info(query, 'Query', verbose, Fore.LIGHTRED_EX))
            print(get_tensor_info(key, 'Key', verbose, Fore.LIGHTGREEN_EX))
            print(get_tensor_info(value, 'Value', verbose, Fore.LIGHTBLUE_EX))

        batch_size, q_seq_len = query.shape[:2]

        # Reshape Q, K, V for each rep head
        query, (g, b, c, p) = self.convert(query, self.heads, self.head_dim, self.mult)
        key     = self.convert(key, self.kv_heads, self.head_dim, self.mult)[0]
        value   = self.convert(value, self.kv_heads, self.head_dim)[0]
        if verbose:
            print(get_tensor_info(query, 'Q after Reshaping', verbose))
            print(get_tensor_info(key, 'K after Reshaping', verbose))

        # Apply Rotary Embeddings
        query = self.rotary_embedding(query, pos_idx if len(self.pixels) == 1 else None, verbose)
        key   = self.rotary_embedding(key, None)
        if verbose:
            print(get_tensor_info(query, 'Q after Rotary Embedding', verbose))
            print(get_tensor_info(key, 'K after Rotary Embedding', verbose))

        # Duplicate K and V for kv heads num per query head
        key   = self.repeat_kv(key)
        value = self.repeat_kv(value)
        if verbose:
            print(get_tensor_info(key, 'Duplicated K', verbose))
            print(get_tensor_info(value, 'Duplicated V', verbose))

        # Apply attention
        attention_scores, attention = self.attention(query, key.contiguous(), value.contiguous(), keys,
                                                     self.causal_mask and pretext is None, verbose)
        attention = attention.reshape(g, b, -1, self.dim_size)
        # out_view shape:  (genomes, batch_size, *pixels, channels)
        if verbose:
            print(get_tensor_info(attention, 'Attented Values', verbose))

        # Apply output projection
        try:
            # Apply weights
            tensor: Tensor = self.out_proj(self.revert(attention, b, c, p), keys=keys)
        except Exception as e:
            print(CM(f"batch_size={b}, channels={c}, pixels={p}, attention={attention.shape}"
                       f"\nquery={query.shape}, key={key.shape}, value={value.shape}\n", Fore.LIGHTRED_EX))
            raise e
        # Add the residue
        if self.skip_connection:
            tensor = tensor + residue
        if verbose:
            print(get_tensor_info(tensor, 'Output Projection', verbose))

        # Store attention for debugging
        if not get:
            return tensor
        else:
            return tensor, attention_scores

    # def __repr__(self):
    #     return f"{self.__class__.__name__}({self.dim_size}, pixels={self.pixels}, kernel_size={self.kernel_size}, " \
    #            f"bias={self.bias}{display('hd', self._heads)}{display('kv_hd', self._kv_heads)}" \
    #            f"{display('diffs', self.att_coeff_num-1 if self.att_coeff_num else None)}" \
    #            f"{display('pad', self.padding)}{display('pad_mode', self.padding_mode, self.padding)}" \
    #            f"{display('pre_norm_ng', self.norm_groups)}{display('residual', self.skip_connection)}" \
    #            f"{display('as', self.auto_single)}"


class ConvCrossAttention(NeatModule):
    def __init__(
            self, max_pixels: list[int], cross_max_pixels: list[int], dim_size: int, kernel_size: int,
            heads: int = None, kv_heads: int = None, differential: int = None, layer_idx: int = None, causal_mask=False,
            bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options):
        super(ConvCrossAttention, self).__init__()
        self._heads = heads
        self._kv_heads = kv_heads
        if heads is None:
            heads = 1
        if dim_size % heads != 0:
            raise ValueError(f"Embedding dimensions must be a multiple of heads num")
        if kv_heads is None:
            kv_heads = heads
        if heads % kv_heads != 0:
            raise ValueError(f"Query heads num must be a multiple of number of Key-Value heads num")
        if isinstance(differential, bool):
            differential = int(differential)
        if not isinstance(max_pixels, (list, tuple)):
            raise ValueError(f"pixels parameter must be Iterable got '{type(max_pixels)}'")
        if not isinstance(cross_max_pixels, (list, tuple)):
            raise ValueError(f"pixels parameter must be Iterable got '{type(cross_max_pixels)}'")

        # OPTIONS
        self.padding = manage_params(options, 'padding', calc_padding(kernel_size, stride=1, dilation=1))
        self.padding_mode = manage_params(options, 'padding_mode', 'zeros')
        self.constant = manage_params(options, 'constant', 10000)
        self.epsilon = manage_params(options, 'epsilon', 1e-8)
        self.affine = manage_params(options, 'affine', True)
        self.norm_groups = manage_params(options, 'norm_groups', None)
        self.skip_connection = manage_params(options, ['skip_connection', 'residual'], False)

        # ATTRIBUTES
        self.heads      = heads
        self.dim_size   = dim_size
        self.head_dim   = dim_size // heads
        self.kv_heads   = kv_heads
        self.q_kv_ratio = heads // kv_heads
        self.pixels     = max_pixels
        self.cross_max_pixels = cross_max_pixels
        self.differential = differential
        self.att_coeff_num = None if not differential else 1+differential
        self.att_coeff_indices = None if not differential else torch.arange(differential, device=device) + 1
        self.causal_mask = causal_mask
        self.pixels_total = max(math.prod(max_pixels), math.prod(cross_max_pixels))
        self.kernel_size = kernel_size
        self.bias = bias

        # ModifiedNEAT
        self.pre_norm = GroupNorm(
            self.norm_groups, dim_size, self.epsilon, self.affine, False, device=device, dtype=dtype
        ) if self.norm_groups else None
        Convolution, CrossConvolution = get_conv(max_pixels), get_conv(cross_max_pixels)
        self.mult = 1+differential if differential else 1
        self.query_proj = Convolution(dim_size, heads * self.head_dim * self.mult, kernel_size=kernel_size,
                                      padding=self.padding, padding_mode=self.padding_mode,
                                      bias=bias, device=device, dtype=dtype)
        self.key_proj   = CrossConvolution(dim_size, kv_heads * self.head_dim * self.mult, kernel_size=kernel_size,
                                           padding=self.padding, padding_mode=self.padding_mode,
                                           bias=bias, device=device, dtype=dtype)
        self.value_proj = CrossConvolution(dim_size, kv_heads * self.head_dim, kernel_size=kernel_size,
                                           padding=self.padding, padding_mode=self.padding_mode,
                                           bias=bias, device=device, dtype=dtype)
        self.out_proj   = Convolution(dim_size, dim_size, kernel_size=kernel_size,
                                      padding=self.padding, padding_mode=self.padding_mode,
                                      bias=bias, device=device, dtype=dtype)
        self.rotary_embedding = RoPE(self.pixels_total, dim_size * self.mult, heads, self.constant, device, dtype)
        self.softmax    = nn.Softmax(-1)
        self.head_norm  = RMSNorm(self.head_dim, self.epsilon, self.affine, device, dtype)
        self.diff_lambda = AttentionLambda(
            heads, self.head_dim, layer_idx, differential, 0.0, 0.1, True, self.epsilon, device, dtype
        ) if differential else None

        # STATES
        self.device = device
        self.dtype  = dtype

    def repeat_kv(self, tensor: Tensor):
        genomes, batch_size, seq_len, kv_heads, head_dim = tensor.shape
        if self.q_kv_ratio == 1:
            return tensor
        else:
            return tensor.unsqueeze(-2).expand(genomes, batch_size, seq_len, kv_heads, self.q_kv_ratio, head_dim).\
                reshape(batch_size, seq_len, kv_heads * self.q_kv_ratio, head_dim)

    @staticmethod
    def convert(image: Tensor, heads: int, head_dim: int, multiplier: int = 1):
        (g, b), c, p = image.shape[:2], image.shape[2] // multiplier, image.shape[3:]
        assert c == heads * head_dim
        # Return shape (genomes, batch_size, pixels, heads, head_dim)
        return image.view(g, b, c, -1).transpose(-1, -2).contiguous().view(g, b, -1, heads, head_dim*multiplier), (b, c, p)

    @staticmethod
    def revert(tensor: Tensor, batch_size: int, channels: int, pixels: list[int]):
        return tensor.transpose(-1, -2).view(-1, batch_size, channels, *pixels)

    def attention(self, query: Tensor, key: Tensor, value: Tensor, keys: Union[int, Iterable[int]], mask: bool, verbose: int = None):
        if self.differential:
            query   = query.view(*query.shape[:-1], self.att_coeff_num, -1)
            key     = key.view(*key.shape[:-1], self.att_coeff_num, -1)
        # Get the attention score (energy)
        energy = torch.einsum("...qhd,...khd->...hqk" if not self.differential else "...qhcd,...khcd->...hcqk", [query, key])
        # queries shape: (batch_size, query_len, heads, head_dim)
        # key shape:     (batch_size, key_len, heads, head_dim)
        # energy shape:  (batch_size, heads, query_len, key_len)
        if verbose:
            print(get_tensor_info(energy, 'Energy', verbose))

        if mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask_ = torch.ones_like(energy, dtype=torch.bool).triu(1)
            # Fill the upper triangle with -inf
            energy.masked_fill_(mask_, -torch.inf)
            if verbose and verbose >= 2 and not self.differential:
                print(get_tensor_info(mask_, 'Mask', verbose))
                print(get_tensor_info(energy, 'Masked Energy', verbose))

        # Get the softmax of the energy
        scores = self.softmax(energy / math.sqrt(self.head_dim))

        if self.differential:
            lambdas: Tensor = self.diff_lambda()
            if verbose and verbose >= 2:
                print(get_tensor_info(torch.round(lambdas, decimals=4), 'Lambdas', verbose+2))
                print(get_tensor_info(self.att_coeff_indices, 'Coeff Indices', verbose+2))
            scores = torch.select(scores, dim=-3, index=0) + torch.sum(
                torch.index_select(scores, dim=-3, index=self.att_coeff_indices) * lambdas, dim=-3
            ) # * lambdas)

            if mask:
                scores.masked_fill_(torch.ones_like(scores, dtype=torch.bool).triu(1), -torch.inf)
                if verbose and verbose >= 2 and not self.differential:
                    print(get_tensor_info(scores, 'Differential Masked Energy', verbose))
            scores = self.softmax(scores)

        if verbose:
            print(get_tensor_info(torch.round(scores, decimals=4), 'Attention Score', verbose))

        # Get the weighted sum of the values and reshape to remove heads
        attention = self.head_norm(torch.einsum("...hqv,...vhd->...qhd", [scores, value]), keys=keys)
        # scores shape:    (batch_size, heads, query_len, value_len)
        # values shape:    (batch_size, value_len, heads, head_dim)
        # attention shape: (batch_size, query_len, heads, head_dim) then concat last 2 dim
        if self.differential:
            attention = attention * (1 - self.diff_lambda.biases(attention, keys, 2))

        return scores, attention

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None,
                context: Tensor = None, pos_idx: int = None, verbose: int = None, get=False):
        if pos_idx is not None and len(self.pixels) == 1:
            pos_idx = self.pixels_total + pos_idx if pos_idx < 0 else pos_idx
            assert 0 < pos_idx < self.pixels_total
        if verbose:
            print(f'\n{CM("Executing Cross Attention", Fore.LIGHTBLUE_EX)}')

        residue = tensor
        if self.pre_norm is not None:
            tensor = self.pre_norm(tensor, keys=keys)
            if context is not None:
                context = self.pre_norm(context, keys=keys)

        # Convolve Q, K, V
        try:
            query: Tensor   = self.query_proj(tensor, keys=keys)
            key: Tensor     = self.key_proj(tensor if context is None else context, keys=keys)
            value: Tensor   = self.value_proj(tensor if context is None else context, keys=keys)
        except Exception as e:
            print("\n")
            print(CM(self.query_proj, Fore.LIGHTRED_EX))
            print(CM(self.key_proj, Fore.LIGHTRED_EX))
            print(CM(self.value_proj, Fore.LIGHTRED_EX))
            print(get_tensor_info(tensor, 'Tensor', 3))
            if context is not None:
                print(get_tensor_info(context, 'Context', 3))
            raise e
        if verbose:
            print(CM('Post Linearization =>'))
            print(get_tensor_info(query, 'Query', verbose, Fore.LIGHTRED_EX))
            print(get_tensor_info(key, 'Key', verbose, Fore.LIGHTGREEN_EX))
            print(get_tensor_info(value, 'Value', verbose, Fore.LIGHTBLUE_EX))

        batch_size, q_seq_len = query.shape[:2]

        # Reshape Q, K, V for each rep head
        query, (b, c, p) = self.convert(query, self.heads, self.head_dim, self.mult)
        key     = self.convert(key, self.kv_heads, self.head_dim, self.mult)[0]
        value   = self.convert(value, self.kv_heads, self.head_dim)[0]
        if verbose:
            print(get_tensor_info(query, 'Q after Reshaping', verbose))
            print(get_tensor_info(key, 'K after Reshaping', verbose))

        # Apply Rotary Embeddings
        query = self.rotary_embedding(query, pos_idx if len(self.pixels) == 1 else None, verbose)
        key   = self.rotary_embedding(key, None)
        if verbose:
            print(get_tensor_info(query, 'Q after Rotary Embedding', verbose))
            print(get_tensor_info(key, 'K after Rotary Embedding', verbose))

        # Duplicate K and V for kv heads num per query head
        key   = self.repeat_kv(key)
        value = self.repeat_kv(value)
        if verbose:
            print(get_tensor_info(key, 'Duplicated K', verbose))
            print(get_tensor_info(value, 'Duplicated V', verbose))

        attention_scores, attention = self.attention(
            query, key.contiguous(), value.contiguous(), keys,
            self.causal_mask and context is None and len(self.pixels) == 1, verbose
        )
        attention = attention.reshape(batch_size, -1, self.dim_size)
        # out_view shape:  (batch_size, query_len, embed_size)
        if verbose:
            print(get_tensor_info(attention, 'Attented Values', verbose))

        try:
            # Apply weights
            tensor: Tensor = self.out_proj(self.revert(attention, b, c, p), keys=keys)
        except Exception as e:
            print(CM(f"batch_size={b}, channels={c}, pixels={p}, attention={attention.shape}"
                       f"\nquery={query.shape}, key={key.shape}, value={value.shape}\n", Fore.LIGHTRED_EX))
            raise e
        if self.skip_connection:
            tensor = tensor + residue
        if verbose:
            print(get_tensor_info(tensor, 'Output Projection', verbose))

        if not get:
            return tensor
        else:
            return tensor, attention_scores

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dim_size}, pixels={self.pixels}, kernel_size={self.kernel_size}, " \
               f"bias={self.bias}{display('hd', self._heads)}{display('kvhd', self._kv_heads)}" \
               f"{display('diffs', self.att_coeff_num-1 if self.att_coeff_num else None)}" \
               f"{display('pad', self.padding)}{display('pad_mode', self.padding_mode, self.padding)}" \
               f"{display('pre_norm_ng', self.norm_groups)}{display('residual', self.skip_connection)}"


# --------------------------------------------- #
# Gated Linear Units                            #
# --------------------------------------------- #

class SwiGLU(NeatModule):
    def __init__(self, dim_size: int, bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options):
        super(SwiGLU, self).__init__()

        # ATTRIBUTES
        self.dim_size       = dim_size
        self.bias           = bias
        self.epsilon        = manage_params(options, 'epsilon', 1e-8)
        self.affine         = manage_params(options, 'affine', True)
        self.normalize      = manage_params(options, 'normalize', True)
        self.skip_connection = manage_params(options, ['skip_connection', 'residual'], True)
        self.fwd_exp        = manage_params(options, ['fwd_exp', 'forward_expansion'], 2)
        self.out_bias       = manage_params(options, 'out_bias', bias)

        # BUILD
        hidden_size = int(self.fwd_exp * dim_size)
        # self.pre_norm = RMSNorm(dim_size, self.epsilon, self.affine, device, dtype) if self.normalize else None
        self.pre_norm = LayerNorm(dim_size, self.epsilon, self.affine, bias, device, dtype) if self.normalize else None
        self.inp_proj = Linear(dim_size, hidden_size, bias, device, dtype)
        self.mul_proj = Linear(dim_size, hidden_size, bias, device, dtype)
        self.out_proj = Linear(hidden_size, dim_size, self.out_bias, device, dtype)
        self.activation = manage_params(options, ['actv', 'activation'], nn.SiLU())

        # STATES
        self.device = device
        self.dtype = dtype

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        residue = tensor
        if self.pre_norm is not None:
            tensor = self.pre_norm(tensor, keys=keys)
        tensor = self.out_proj(self.activation(self.inp_proj(tensor, keys=keys)) * self.mul_proj(tensor, keys=keys), keys=keys)
        if self.skip_connection:
            tensor = tensor + residue
        return tensor

    def extra_repr(self) -> str:
        return (f"*** "
                f"residual={self.skip_connection}, normalize={self.normalize}, "
                f"***")


class ConvSwiGLU(NeatModule):
    def __init__(self, dim_size: int, kernel_size: int,  bias=False,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options):
        super(ConvSwiGLU, self).__init__()

        # ATTRIBUTES
        self.auto_single = manage_params(options, 'auto_single', False)
        if self.auto_single:
            kernel_size = 1
        self.dim_size       = dim_size
        self.kernel_size    = kernel_size
        self.bias           = bias
        self.stride         = manage_params(options, 'stride', 1)
        self.dilation       = manage_params(options, 'dilation', 1)
        self.padding        = manage_params(options, 'padding', calc_padding(kernel_size, self.stride, self.dilation))
        self.padding_mode   = manage_params(options, 'padding_mode', 'zeros')
        self.epsilon        = manage_params(options, 'epsilon', 1e-8)
        self.affine         = manage_params(options, 'affine', True)
        self.norm_groups    = manage_params(options, 'norm_groups', None)
        self.skip_connection = manage_params(options, ['skip_connection', 'residual'], False)
        self.fwd_exp        = manage_params(options, ['fwd_exp', 'forward_expansion'], 1)
        self.image_ndim     = manage_params(options, 'image_ndim', 2)
        # hidden_size = 4 * embed_size
        # hidden_size = int(2 * hidden_size / 3)
        # if fwd_exp is not None:
        #     hidden_size = int(fwd_exp * hidden_size)
        # hidden_size = mult * ((hidden_size + mult - 1) // mult)
        hidden_size = self.fwd_exp * dim_size

        # BUILD
        self.pre_norm = GroupNorm(
            self.norm_groups, dim_size, self.epsilon, self.affine, False, device=device, dtype=dtype
        ) if self.norm_groups else None
        Convolution = get_conv(self.image_ndim)
        self.inp_proj = Convolution(dim_size, hidden_size, kernel_size, stride=self.stride, dilation=self.dilation,
                                    padding=self.padding, padding_mode=self.padding_mode,
                                    bias=bias, device=device, dtype=dtype)
        self.mul_proj = Convolution(dim_size, hidden_size, kernel_size, stride=self.stride, dilation=self.dilation,
                                    padding=self.padding, padding_mode=self.padding_mode,
                                    bias=bias, device=device, dtype=dtype)
        self.out_proj = Convolution(hidden_size, dim_size, kernel_size, stride=self.stride, dilation=self.dilation,
                                    padding=self.padding, padding_mode=self.padding_mode,
                                    bias=bias, device=device, dtype=dtype)
        self.activation = manage_params(options, ['actv', 'activation'], nn.SiLU())

        # STATES
        self.device = device
        self.dtype = dtype

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        residue = tensor
        if self.pre_norm is not None:
            tensor = self.pre_norm(tensor, keys=keys)
        tensor = self.out_proj(self.activation(self.inp_proj(tensor, keys=keys)) * self.mul_proj(tensor, keys=keys), keys=keys)
        if self.skip_connection:
            tensor = tensor + residue
        return tensor


# --------------------------------------------- #
# Transformers                                  #
# --------------------------------------------- #


class TransformerBlock(NeatModule):
    def __init__(
            self, max_seq_len: int, dim_size: int, heads: int = None, kv_heads: int = None,
            differential=False, layer_idx: int = None, causal_mask=True,
            bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options):
        super(TransformerBlock, self).__init__()

        # ATTRIBUTES
        self.max_seq_len = max_seq_len
        self.dim_size = dim_size
        self.heads      = heads
        self.kv_heads       = kv_heads
        self.differential   = differential
        fwd_func = manage_params(options, ['fwd_func', 'ff'], None)

        # BUILD
        # options['residual'] = True
        self.self_attention = Attention(
            max_seq_len, dim_size, heads, kv_heads, differential, layer_idx, causal_mask,
            bias, device, dtype, **options
        )
        self.feedforward = SwiGLU(dim_size, bias, device, dtype, **options) if fwd_func is None else fwd_func
        self.dropout = nn.Dropout(manage_params(options, 'dropout', 0))
        # TODO: Create cross attention module

        # STATES
        self.self_attention_tensor: Tensor = None
        # self.cross_attention_tensor: Tensor = None
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None,
                prompt: Tensor = None, context: Tensor = None,
                pos_idx: int = None, verbose: int = None, get=False):
        # Normalize then get the self attention
        attention = self.self_attention(
                tensor, keys=keys, prompt=prompt, pos_idx=pos_idx, verbose=verbose, get=get
        )
        if get:
            attention, self.self_attention_tensor = attention
        # Apply dropout
        tensor = self.dropout(attention)
        # Normalize then get the cross attention
        # if self.cross_attention is not None and context is not None:
        #     attention = self.cross_attention(
        #             tensor, keys=keys, context=context, pos_idx=pos_idx, verbose=verbose, get=get
        #     )
        #     if get:
        #         attention, self.cross_attention_tensor = attention
        #     # Apply dropout
        #     tensor = self.dropout(attention)
        # Pass through feed forward
        activation = self.feedforward(tensor, keys=keys)
        # Dropout
        tensor = self.dropout(activation)
        if verbose:
            print(get_tensor_info(tensor, f'{self.__class__.__name__} Output', verbose))

        return tensor


class TransformerBase(NeatModule):
    def __init__(
            self, max_seq_len: int, dim_size: int, layers: int, heads: int = None, kv_heads: int = None,
            differential: int = None, causal_mask=True,
            bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options):
        super(TransformerBase, self).__init__()

        # ATTRIBUTES
        self.max_seq_len    = max_seq_len
        self.dim_size       = dim_size
        self.layer_num      = layers
        self.heads          = heads
        self.kv_heads       = kv_heads
        self.differential   = differential
        self.dropout        = manage_params(options, 'dropout', 0)
        self.auto_single    = manage_params(options, 'auto_single', False)

        # BUILD
        options['auto_single'] = False # NOTE: Disables the auto-single directly used in the Attention module
        self.layers: list[TransformerBlock] = nn.ModuleList()
        for layer_idx in range(layers):
            # if layer_idx == layers-1:
            #     if self.auto_single:
            #         options['auto_single'] = self.auto_single
            self.layers.append(
                TransformerBlock(
                    max_seq_len, dim_size, heads, kv_heads, differential, layer_idx,
                    causal_mask, bias, device, dtype, **options
                )
            )

        # STATE
        self.device     = device
        self.dtype      = dtype

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None,
                prompt: Tensor = None, context: Tensor = None, pos_idx: int = None,
                verbose: int = None, get=False, single=False):
        ndim = tensor.ndim
        rem_batch = rem_seq = False
        if ndim == 2: # No batch and sequence dim
            rem_batch = rem_seq = True
            tensor = tensor.unsqueeze(1).unsqueeze(2)
        elif ndim == 3: # No batch or sequence dim (assume sequence is missing)
            rem_seq = True
            tensor = tensor.unsqueeze(2)
        # Pass through the encoder blocks
        for layer_idx, layer in enumerate(self.layers):
            # Single mode is only when on final layer and instance (seq_len=1) output is required
            single_fetch = (
                ((single or self.auto_single) and layer_idx == len(self.layers) - 1)
                or
                (self.auto_single and layer.self_attention.forced_cache)
            )
            # shape (genomes, batch_size, seq_len, dim_size)
            if single_fetch:
                assert tensor.ndim == 4
                # Using last token index in sequence to get the next token
                # shape (genomes, batch_size, seq_len, dim_size)
                set_prompt = torch.select(tensor, -2, -1).unsqueeze(-2)
            else:
                set_prompt = prompt
            tensor = layer(tensor, keys=keys, prompt=set_prompt, context=context,
                           pos_idx=pos_idx if not single_fetch else tensor.shape[-2]-1, # deque mode corrects the position regardless
                           verbose=verbose, get=get)
        if rem_seq:
            tensor = tensor.squeeze(2)
        if rem_batch:
            tensor = tensor.squeeze(1)
        return tensor

    def get_attention(self):
        return [(
            layer.self_attention_tensor, # layer.cross_attention_tensor
        ) for layer in self.layers]

    def force_cache(self, enable: bool = True):
        for layer in self.layers:
            layer.self_attention.force_cache(enable)
    
    def empty_cache(self):
        for layer in self.layers:
            layer.self_attention.empty_cache()


class ConverBlock(NeatModule):
    def __init__(self, max_pixels: tuple[int, ...],  dim_size: int, kernel_size: int, norm_groups: int,
                 heads: int = None, kv_heads: int = None, differential: int = None, layer_idx: int = None,
                 causal_mask=False, bias=False, device='cpu', dtype=torch.float32, **options):
        super(ConverBlock, self).__init__()

        # ATTRIBUTES
        self.max_pixels     = max_pixels
        self.cross_max_pixels = manage_params(options, ['cross_max_pixels', 'cross_pixels'], None)
        self.dim_size       = dim_size
        self.kernel_size    = kernel_size
        self.norm_groups    = norm_groups
        self.heads          = heads
        self.kv_heads       = kv_heads
        self.differential   = differential
        self.image_ndim     = len(max_pixels)

        # ModifiedNEAT
        options['norm_groups'] = norm_groups
        options['skip_connection'] = True
        self.self_attention = ConvSelfAttention(
            max_pixels, dim_size, kernel_size, heads, kv_heads, differential, layer_idx, causal_mask,
            bias, device, dtype, **options
        )
        self.cross_attention = ConvCrossAttention(
            max_pixels, self.cross_max_pixels, dim_size, kernel_size, heads, kv_heads, differential, layer_idx,
            False, bias, device, dtype, **options
        ) if self.cross_max_pixels is not None else None
        options['image_ndim'] = 1
        self.feedforward = ConvSwiGLU(dim_size, kernel_size, bias, device, dtype, **options)
        # if self.image_ndim == 1:
        #     Dropout = nn.Dropout1d
        # elif self.image_ndim == 2:
        #     Dropout = nn.Dropout2d
        # elif self.image_ndim == 3:
        #     Dropout = nn.Dropout3d
        # else:
        #     raise ValueError(f"Unsupported num of image dimension '{self.image_ndim}'")
        self.dropout = nn.Dropout(manage_params(options, 'dropout', 0))

        # STATES
        self.self_attention_tensor: Tensor = None
        self.cross_attention_tensor: Tensor = None
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, pretext: Tensor = None, context: Tensor = None,
                pos_idx: int = None, verbose: int = None, get=False):
        # Normalize then get the self attention
        attention = self.self_attention(
                tensor, keys=keys, pretext=pretext, pos_idx=pos_idx, verbose=verbose, get=get
        )
        if get:
            attention, self.self_attention_tensor = attention
        # Apply dropout
        tensor = self.dropout(attention)
        # Normalize then get the cross attention
        if self.cross_attention is not None and context is not None:
            attention = self.cross_attention(
                    tensor, keys=keys, context=context, pos_idx=pos_idx, verbose=verbose, get=get
            )
            if get:
                attention, self.cross_attention_tensor = attention
            # Apply dropout
            tensor = self.dropout(attention)
        # Pass through feed forward
        activation = self.feedforward(tensor, keys=keys)
        # Dropout
        tensor = self.dropout(activation)
        if verbose:
            print(get_tensor_info(tensor, f'{self.__class__.__name__} Output', verbose))

        return tensor


class ConverBase(NeatModule):
    def __init__(self, max_pixels: tuple[int, ...],  dim_size: int, kernel_size: int, norm_groups: int, layers: int,
                 heads: int = None, kv_heads: int = None, differential: int = None,
                 causal_mask=False, bias=False, device='cpu', dtype=torch.float32, **options):
        super(ConverBase, self).__init__()

        # ATTRIBUTES
        self.max_pixels     = max_pixels
        self.dim_size       = dim_size
        self.kernel_size    = kernel_size
        self.norm_groups    = norm_groups
        self.layer_num      = layers
        self.heads          = heads
        self.kv_heads       = kv_heads
        self.differential   = differential
        self.image_ndim     = len(max_pixels)
        self.dropout        = manage_params(options, 'dropout', 0)
        self.auto_single    = manage_params(options, 'auto_single', False)

        # ModifiedNEAT
        options['auto_single'] = False
        self.positional_encoding = SequenceEncoding(max_pixels, dim_size, True, device, dtype)
        self.layers: list[ConverBlock] = nn.ModuleList()
        for layer_idx in range(layers):
            if layer_idx == layers-1:
                options['auto_single'] = self.auto_single
                kernel_size = 1
                options['padding'] = -1
            self.layers.append(
                ConverBlock(
                    max_pixels, dim_size, kernel_size,
                    norm_groups, heads, kv_heads, differential, layer_idx,
                    causal_mask, bias, device, dtype, **options
                )
            )

        # STATE
        self.device     = device
        self.dtype      = dtype

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, pretext: Tensor = None, context: Tensor = None, pos_idx: int = None,
                verbose: int = None, get=False, single=False):
        # Pass through the encoder blocks
        for layer_idx, layer in enumerate(self.layers):
            # Single mode is only when on final layer, pixels span 1 dimension and tensor has 3 dimensions only
            single_fetch = (single or self.auto_single) and layer_idx == len(self.layers) - 1 and len(self.max_pixels) == 1 and tensor.ndim == 4
            # shape (batch_size, channels, *pixels)
            tensor = self.positional_encoding(tensor, keys=keys, offset=None, verbose=verbose)
            if single_fetch:
                # Using last token index in sequence to get the next token
                set_pretext = torch.select(tensor, -1, -1).unsqueeze(-1)
            else:
                set_pretext = pretext
                if set_pretext is not None:
                    set_pretext = self.positional_encoding(tensor, keys=keys, offset=None, verbose=verbose)
            tensor = layer(tensor, keys=keys, pretext=set_pretext, context=context, pos_idx=pos_idx,
                           verbose=verbose, get=get)

        return tensor

    def get_attention(self):
        return [(layer.self_attention_tensor, layer.cross_attention_tensor) for layer in self.layers]
