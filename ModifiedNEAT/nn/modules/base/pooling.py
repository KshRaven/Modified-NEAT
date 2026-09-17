from ...base import NeatModule
from .util import calc_padding

from torch import Tensor
from typing import Union, Iterable

import torch.nn.functional as F


class Pooling(NeatModule):
    """Base pooling layer supporting n-dimensional pooling (max/avg)."""

    def __init__(self, kernel_size: Union[int, Iterable[int]], stride=None,
                 padding: Union[int, Iterable[int]] = 0, ndim=1, mode='max'):
        # ====================== Input Handling ======================
        if isinstance(kernel_size, (int, float)):
            kernel_size = tuple([int(kernel_size) for _ in range(ndim)])
        elif len(kernel_size) != ndim:
            raise ValueError(f"Kernel dimensions '{kernel_size}' do not match pooling dimensions '{ndim}'")
        if not isinstance(kernel_size, tuple):
            kernel_size = tuple(kernel_size)

        # Stride is a single value shared across dims (matches Convolution's convention)
        stride = kernel_size[0] if stride is None else stride

        padding_value = 0
        if isinstance(padding, (int, float)):
            padding = [int(padding) for _ in range(ndim)]
        elif len(padding) != ndim:
            raise ValueError(f"Padding dimensions '{padding}' do not match pooling dimensions '{ndim}'")
        if not isinstance(padding, list):
            padding = list(padding)
        if any([pv < -1 or not isinstance(pv, int) for pv in padding]):
            raise ValueError(f"Invalid {self.__class__.__name__} padding '{padding}'")
        for pi, pv in enumerate(padding):
            if pv == -1:
                padding[pi] = calc_padding(kernel_size[pi], stride, 1)
        padding = tuple(padding)

        if mode not in ('max', 'avg'):
            raise ValueError(f"Unsupported pooling mode '{mode}'")

        super(Pooling, self).__init__(
            kernel_size=kernel_size, stride=stride, padding=padding, mode=mode
        )

        # Attributes
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.padding_value  = padding_value
        self.mode           = mode
        self.ndim           = ndim
        # Max-pool pads with -inf so padded cells never win the max; avg-pool pads with 0
        self.pad_value      = float('-inf') if mode == 'max' else 0.0

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, verbose: int = None):
        verbose = verbose is not None and verbose >= 3
        # 'keys' is accepted (but unused - pooling has no learnable parameters) purely so this
        # layer's call signature matches other NeatModules for Sequential/container compatibility.
        if verbose:
            print(f"Inputs = {tensor.shape}")
        # Expected shape = (genomes, batch_size, ..., channels, *seq_len)
        pool_dims = len(self.kernel_size)
        extra_shape, channels, input_seq_len = \
            tensor.shape[:-(pool_dims+1)], tensor.shape[-(pool_dims+1)], tensor.shape[-pool_dims:]

        # ====================== Padding Handling ======================
        if any([pv != 0 for pv in self.padding]):
            padding = tuple([])
            for pv in self.padding:
                padding += (pv, pv)
            if verbose:
                print(f"padding = {padding}")
            tensor = tensor.view(-1, channels, *input_seq_len)
            tensor = F.pad(tensor, padding, mode='constant', value=self.pad_value)
            padded_seq_len = tensor.shape[2:]
            tensor = tensor.contiguous().view(*extra_shape, channels, *padded_seq_len)

        padded_seq_len: tuple[int, ...] = tensor.shape[-pool_dims:]
        if verbose:
            print(f"Padded input: {tensor.shape}")

        # ====================== Output Size Calculation ======================
        output_seq_len = tuple([(padded_seq_len[di] - k) // self.stride + 1
                                 for di, k in enumerate(self.kernel_size)])

        if verbose:
            print(f"Calculated output size: {output_seq_len}")

        if any([k > psl for k, psl in zip(self.kernel_size, padded_seq_len)]):
            raise ValueError(f"Kernel size {self.kernel_size} exceeds padded input size {padded_seq_len}")
        if any([osl <= 0 for osl in output_seq_len]):
            raise ValueError(f"Invalid output size: {output_seq_len}")

        # ====================== Window Generation ======================
        windows = tensor
        # TODO: Support dimensional stride
        for i, k in enumerate(self.kernel_size):
            windows = windows.unfold(len(extra_shape)+1+i, k, self.stride)  # Unfold along spatial dimensions

        if verbose:
            print(f"Windows tensor shape: {windows.shape}")

        # ====================== Reduction ======================
        # windows = (..., channels, *out_pixel_dims, *kernel_dims)
        reduce_dims = tuple(range(-pool_dims, 0))
        tensor = windows.amax(dim=reduce_dims) if self.mode == 'max' else windows.mean(dim=reduce_dims)

        if verbose:
            print(f"Final output shape: {tensor.shape}")

        return tensor


class MaxPooling(Pooling):
    """Base max pooling layer."""

    def __init__(self, kernel_size: Union[int, Iterable[int]], stride: int = 1,
                 padding: Union[int, Iterable[int]] = 0, ndim=1):
        super(MaxPooling, self).__init__(kernel_size, stride, padding, ndim, mode='max')


class MaxPool1d(MaxPooling):
    """1D Max pooling."""

    def __init__(self, kernel_size: Union[int, tuple[int]], stride: int = 1, padding=0):
        super(MaxPool1d, self).__init__(kernel_size, stride, padding, ndim=1)


class MaxPool2d(MaxPooling):
    """2D Max pooling."""

    def __init__(self, kernel_size: Union[int, tuple[int, int]], stride: int = 1, padding=0):
        super(MaxPool2d, self).__init__(kernel_size, stride, padding, ndim=2)


class MaxPool3d(MaxPooling):
    """3D Max pooling."""

    def __init__(self, kernel_size: Union[int, tuple[int, int, int]], stride: int = 1, padding=0):
        super(MaxPool3d, self).__init__(kernel_size, stride, padding, ndim=3)


class AvgPooling(Pooling):
    """Base average pooling layer."""

    def __init__(self, kernel_size: Union[int, Iterable[int]], stride=None,
                 padding: Union[int, Iterable[int]] = 0, ndim=1):
        super(AvgPooling, self).__init__(kernel_size, stride, padding, ndim, mode='avg')


class AvgPool1d(AvgPooling):
    """1D Average pooling."""

    def __init__(self, kernel_size: Union[int, tuple[int]], stride=None, padding=0):
        super(AvgPool1d, self).__init__(kernel_size, stride, padding, ndim=1)


class AvgPool2d(AvgPooling):
    """2D Average pooling."""

    def __init__(self, kernel_size: Union[int, tuple[int, int]], stride=None, padding=0):
        super(AvgPool2d, self).__init__(kernel_size, stride, padding, ndim=2)


class AvgPool3d(AvgPooling):
    """3D Average pooling."""

    def __init__(self, kernel_size: Union[int, tuple[int, int, int]], stride=None, padding=0):
        super(AvgPool3d, self).__init__(kernel_size, stride, padding, ndim=3)

# TODO: AdaptiveAvgPooling