from ...base import NeatModule, NeatParameter

from torch import Tensor
from typing import Union, Iterable

import torch
import numbers


class LayerNorm(NeatModule):
    """Layer normalization."""
    
    def __init__(self, normalized_shape: Union[int, Iterable[int]], eps=1e-8, elementwise_affine=True,
                 bias=True, device=torch.device('cpu'), dtype=torch.float32):
        if isinstance(normalized_shape, (numbers.Integral, numbers.Real)):
            normalized_shape = [int(normalized_shape)]
        normalized_shape = tuple(normalized_shape)
        super(LayerNorm, self).__init__(shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine,
                                        bias=elementwise_affine and bias)
        self.normalized_shape = tuple(normalized_shape)
        self.dim = tuple([i for i in range(-len(self.normalized_shape), 0)])
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.weights: NeatParameter = None
        self.biases: NeatParameter = None
        if self.elementwise_affine:
            self.weights = NeatParameter(normalized_shape, False, device, dtype)
            if bias:
                self.biases = NeatParameter(normalized_shape, False, device, dtype)

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        mean        = torch.mean(tensor, dim=self.dim, keepdim=True)
        variance    = torch.var(tensor, dim=self.dim, keepdim=True, unbiased=False)
        tensor      = (tensor - mean) / torch.sqrt(variance + self.eps)

        if self.elementwise_affine:
            tensor = tensor * self.expand(self.weights[keys], tensor, keys=keys)
            if self.biases is not None:
                tensor = tensor + self.expand(self.biases[keys], tensor, keys=keys)

        return tensor


class RMSNorm(NeatModule):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, normalized_shape: Union[int, Iterable[int]], eps=1e-8,
                 elementwise_affine=True, device=torch.device('cpu'), dtype=torch.float32):
        if isinstance(normalized_shape, (numbers.Integral, numbers.Real)):
            normalized_shape = [int(normalized_shape)]
        normalized_shape = tuple(normalized_shape)
        super(RMSNorm, self).__init__(shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.normalized_shape = normalized_shape
        self.dim = tuple([i for i in range(-len(self.normalized_shape), 0)])
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.weights: NeatParameter = None
        if elementwise_affine:
            self.weights = NeatParameter(normalized_shape, False, device, dtype)

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        tensor = tensor / torch.sqrt(torch.mean(tensor**2, dim=self.dim, keepdim=True) + self.eps)

        if self.elementwise_affine:
            tensor = tensor * self.expand(self.weights[keys], tensor, keys=keys)

        return tensor


class GroupNorm(NeatModule):
    """Group normalization."""
    
    def __init__(self, groups: int, channels: int, eps=1e-9, affine=True,
                 bias=True, device=torch.device('cpu'), dtype=torch.float32):
        super(GroupNorm, self).__init__(groups=groups, channels=channels, eps=eps, affine=affine, bias=affine and bias)
        if channels % groups != 0:
            raise ValueError("num_channels must be divisible by num_groups.")
        self.groups = groups
        self.channels = channels
        self.eps    = eps
        self.affine = affine
        self.bias   = bias

        self.weights: NeatParameter = None
        self.biases: NeatParameter  = None
        if self.affine:
            self.weights = NeatParameter(channels, False, device, dtype)
            if bias:
                self.biases = NeatParameter(channels, False, device, dtype)

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, permute: Iterable[int] = None):
        if permute is not None:
            assert len(permute) == tensor.ndim
            tensor = tensor.permute(permute)

        # Reshape the tensor to (N, G, C//G, *) for group normalization
        (genomes, batch_size, channels), pixels = tensor.shape[:3], tensor.shape[3:]
        if channels != self.channels:
            raise ValueError(f"Channels num do not match; Got '{channels}', expected '{self.channels}'")
        if self.weights is not None and (genomes > self.weights.shape[0] or genomes <= 0):
            raise ValueError(f"Genomes num do not match; Got '{genomes}', expected '{self.weights.shape[0]}'")
        tensor = tensor.view(genomes, batch_size, self.groups, channels//self.groups, *pixels)

        # Calculate mean and variance over the group and spatial dimensions
        dims = list(range(3, 4+len(pixels)))
        mean = tensor.mean(dim=dims, keepdim=True)
        var = tensor.var(dim=dims, keepdim=True, unbiased=False)

        # Normalize the tensor
        tensor = (tensor - mean) / torch.sqrt(var + self.eps)

        # Reshape back to (N, C, *)
        tensor = tensor.view(genomes, batch_size, channels, *pixels)

        # Apply affine transformation
        if self.affine:
            weights = self.expand(self.weights[keys], tensor, offset=1, keys=keys)
            tensor = tensor * weights
            if self.bias:
                tensor = tensor + self.expand(self.biases[keys], tensor, offset=1, keys=keys)

        if permute is not None:
            tensor = tensor.permute(permute)
        return tensor


class BatchNorm(NeatModule):
    """Batch normalization."""
    
    def __init__(self, groups: int, channels: int, eps=1e-8, affine=True,
                 device=torch.device('cpu'), dtype=torch.float32):
        super(BatchNorm, self).__init__(num_groups=groups, num_channels=channels, eps=eps, affine=affine)
        if channels % groups != 0:
            raise ValueError("num_channels must be divisible by num_groups.")
        self.num_groups = groups
        self.num_channels = channels
        self.dim = [1, 2]
        self.eps = eps
        self.affine = affine

        self.weights: NeatParameter = None
        self.biases: NeatParameter = None
        if affine:
            self.weights = NeatParameter(channels, False, device, dtype)
            self.biases = NeatParameter(channels, False, device, dtype)

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        # assuming shape (genomes, groups, channels, *extra)
        mean     = tensor.mean(dim=self.dim, keepdim=True)
        variance = tensor.var(dim=self.dim, unbiased=False, keepdim=True)
        tensor   = (tensor - mean) / torch.sqrt(variance + self.eps)

        if self.affine:
            tensor = tensor * self.expand(self.weights[keys], tensor, 1, keys=keys)
            tensor = tensor + self.expand(self.biases[keys], tensor, 1, keys=keys)
        return tensor


class _BatchNormND(BatchNorm):
    """Shared base for the convolution-flavored BatchNorm1d/2d/3d.

    Expects tensors shaped (genomes, batch_size, channels, *spatial_dims), matching the
    Convolution layers' output. Normalizes per-channel over the batch and all spatial dims,
    keeping genomes (dim 0) and channels (dim 2) untouched.
    """

    def __init__(self, channels: int, eps=1e-8, affine=True, ndim=1,
                 device=torch.device('cpu'), dtype=torch.float32):
        super(_BatchNormND, self).__init__(groups=1, channels=channels, eps=eps, affine=affine,
                                           device=device, dtype=dtype)
        self.ndim = ndim
        self.dim = [1] + list(range(3, 3 + ndim))


class BatchNorm1d(_BatchNormND):
    """Batch normalization over (genomes, batch_size, channels, length) inputs."""

    def __init__(self, channels: int, eps=1e-8, affine=True,
                 device=torch.device('cpu'), dtype=torch.float32):
        super(BatchNorm1d, self).__init__(channels, eps, affine, ndim=1, device=device, dtype=dtype)


class BatchNorm2d(_BatchNormND):
    """Batch normalization over (genomes, batch_size, channels, height, width) inputs."""

    def __init__(self, channels: int, eps=1e-8, affine=True,
                 device=torch.device('cpu'), dtype=torch.float32):
        super(BatchNorm2d, self).__init__(channels, eps, affine, ndim=2, device=device, dtype=dtype)


class BatchNorm3d(_BatchNormND):
    """Batch normalization over (genomes, batch_size, channels, depth, height, width) inputs."""

    def __init__(self, channels: int, eps=1e-8, affine=True,
                 device=torch.device('cpu'), dtype=torch.float32):
        super(BatchNorm3d, self).__init__(channels, eps, affine, ndim=3, device=device, dtype=dtype)

# TODO: AdaptiveBatchNorm / running statistics (current BatchNorm variants are training-stats-only)