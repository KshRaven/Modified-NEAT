from torch import Tensor
from typing import Union, Iterable

import torch
import torch.nn as nn


class Identity(nn.Module):
    """Identity layer - returns input unchanged."""
    
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        return tensor


class Transpose(nn.Module):
    """Transpose two dimensions of a tensor."""
    
    def __init__(self, dim0=-1, dim1=-2):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, tensor: Tensor):
        return torch.transpose(tensor, self.dim0, self.dim1)

    def extra_repr(self) -> str:
        return f"dim0={self.dim0}, dim1={self.dim1}"


class Permute(nn.Module):
    """Transpose two dimensions of a tensor."""
    
    def __init__(self, dims: tuple[int, ...]):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, tensor: Tensor):
        return torch.permute(tensor, self.dims)

    def extra_repr(self) -> str:
        return f"dims={self.dims}"


class Ignore(nn.Module):
    """Stochastic ignore layer - randomly zeros output during training."""
    
    def __init__(self, rate: float = 0):
        super(Ignore, self).__init__()
        assert 0 <= rate < 1
        self.rate = rate

    def forward(self, tensor: Tensor):
        if self.training and torch.rand(1).item() < self.rate:
            tensor = tensor * 0
        return tensor

    def extra_repr(self) -> str:
        return f"rate={self.rate}"
