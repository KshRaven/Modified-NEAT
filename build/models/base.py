
from build.nn.base import NeatModule, NeatParameter

from torch import Tensor
from typing import Union, Iterable

import torch
import torch.nn as nn
import numbers
import inspect


def has_parameter(method, param_name: str):
    """Check if a method has a specific parameter."""
    sig = inspect.signature(method)
    return param_name in sig.parameters


class Sequential(NeatModule):
    def __init__(self, *modules: NeatModule):
        super(Sequential, self).__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, fetch=False):
        for module in self.modules_list:
            if isinstance(module, NeatModule):
                if has_parameter(module.forward, 'fetch'):
                    tensor = module(tensor, keys=keys, fetch=fetch)
                else:
                    tensor = module(tensor, keys=keys)
            else:
                tensor = module(tensor)

        return tensor

    def __repr__(self):
        params = ""
        for i, module in enumerate(self.modules_list):
            params += f"\t({i}): {module}"
            if i < len(self.modules_list)-1:
                params += "\n"
        return f"{self.__class__.__name__}[NeatModule](\n{params}\n)"


class Linear(NeatModule):
    def __init__(self, inputs: int, outputs: int, bias=True,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        super(Linear, self).__init__(inputs=inputs, outputs=outputs, bias=bias)
        self.weights = NeatParameter((inputs, outputs), False, device, dtype)
        self.biases  = NeatParameter(outputs, False, device, dtype) if bias else None

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, verbose: int = None):
        # print(f"lin_mod={self.weights.data.shape, tensor.shape}")
        w = self.expand(self.weights[keys], tensor, keys=keys)
        # print(w.shape, tensor.shape)
        tensor = torch.matmul(tensor, w)
        if self.biases is not None:
            b = self.expand(self.biases[keys], tensor, keys=keys)
            tensor = tensor + b
        else:
            b = None

        if not verbose:
            return tensor
        else:
            return tensor, (w, b)


class LayerNorm(NeatModule):
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


class GroupNorm(NeatModule):
    def __init__(self, groups: int, channels: int, eps=1e-8, affine=True,
                 device=torch.device('cpu'), dtype=torch.float32):
        super(GroupNorm, self).__init__(num_groups=groups, num_channels=channels, eps=eps, affine=affine)
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

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, permute: Iterable[int] = None):
        if permute is not None:
            assert len(permute) == tensor.ndim
            tensor = tensor.permute(permute)

        # assuming shape (genomes, groups, channels, *extra)
        mean     = tensor.mean(dim=self.dim, keepdim=True)
        variance = tensor.var(dim=self.dim, unbiased=False, keepdim=True)
        tensor   = (tensor - mean) / torch.sqrt(variance + self.eps)

        if self.affine:
            tensor = tensor * self.expand(self.weights[keys], tensor, 1, keys=keys)
            tensor = tensor + self.expand(self.biases[keys], tensor, 1, keys=keys)

        if permute is not None:
            tensor = tensor.permute(permute)
        return tensor


class RMSNorm(NeatModule):
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

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, fetch=False):
        tensor = tensor / torch.sqrt(torch.mean(tensor**2, dim=self.dim, keepdim=True) + self.eps)

        if self.elementwise_affine:
            tensor = tensor * self.expand(self.weights[keys], tensor, keys=keys)

        return tensor
