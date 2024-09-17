
from torch import Tensor
from typing import Union
from torch.nn import ReLU

import torch
import torch.nn as nn


class Sigmoid(nn.Module):
    def __init__(self, multiplier: Union[float, None] = 5, limit=True):
        super(Sigmoid, self).__init__()
        self.multiplier = multiplier
        self.limit = limit

    def forward(self, tensor: Tensor):
        if self.multiplier is not None:
            tensor = self.multiplier * tensor
        if self.limit:
            tensor = torch.clamp(tensor, -60, 60)
        return 1.0 / (1.0 + torch.exp(-tensor))


class Tanh(nn.Module):
    def __init__(self, multiplier: Union[float, None] = 2.5, limit=True):
        super(Tanh, self).__init__()
        self.multiplier = multiplier
        self.limit = limit

    def forward(self, tensor: Tensor):
        if self.multiplier is not None:
            tensor = self.multiplier * tensor
        if self.limit:
            tensor = torch.clamp(tensor, -60, 60)
        return torch.tanh(tensor)


class Sin(nn.Module):
    def __init__(self, multiplier: Union[float, None] = 5, limit=True):
        super(Sin, self).__init__()
        self.multiplier = multiplier
        self.limit = limit

    def forward(self, tensor: Tensor):
        if self.multiplier is not None:
            tensor = self.multiplier * tensor
        if self.limit:
            tensor = torch.clamp(tensor, -60, 60)
        return torch.sin(tensor)


class Gauss(nn.Module):
    def __init__(self, multiplier: Union[float, None] = 5, limit=True):
        super(Gauss, self).__init__()
        self.multiplier = multiplier
        self.limit = limit

    def forward(self, tensor: Tensor):
        tensor = tensor ** 2
        if self.multiplier is not None:
            tensor = self.multiplier * tensor
        if self.limit:
            tensor = torch.clamp(tensor, -60, 60)
        return torch.exp(-tensor)


class SoftPlus(nn.Module):
    def __init__(self, multiplier: Union[float, None] = 5, limit=True):
        super(SoftPlus, self).__init__()
        self.multiplier = multiplier
        self.limit = limit

    def forward(self, tensor: Tensor):
        if self.multiplier is not None:
            tensor = self.multiplier * tensor
        if self.limit:
            tensor = torch.clamp(tensor, -60, 60)
        return 0.2 + torch.log(1 + torch.tanh(tensor))


class Identity(nn.Module):
    def __init__(self, multiplier: Union[float, None] = None, limit=False):
        super(Identity, self).__init__()
        self.multiplier = multiplier
        self.limit = limit

    def forward(self, tensor: Tensor):
        if self.multiplier is not None:
            tensor = self.multiplier * tensor
        if self.limit:
            tensor = torch.clamp(tensor, -60, 60)
        return tensor
