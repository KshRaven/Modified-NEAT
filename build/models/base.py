
from build.nn.base import NeatModule

from torch.nn import Module
from typing import Union

import torch
import numbers


class Linear(NeatModule):
    def __init__(self, inputs: int, outputs: int, bias=False, device: torch.device = 'cpu',
                 dtype: torch.dtype = torch.float32, activation: Module = None, enable_masking=True):
        super(Linear, self).__init__(inputs, outputs, bias, device, dtype, activation, enable_masking)


class LayerNorm(Module):
    def __init__(self, normalized_shape: Union[int, list[int]], eps=1e-8, elementwise_affine=True,
                 bias: bool = True, device=torch.device('cpu'), dtype=torch.float32, enable_masking=True):
        super().__init__()
        if isinstance(normalized_shape, (numbers.Integral, numbers.Real)):
            normalized_shape = (int(normalized_shape),)
        else:
            raise NotImplemented(f"Currently only implemented for 1-D Norms")
        self.normalized_shape = tuple(normalized_shape)
        self.dim = tuple([-(i+1) for i in range(len(self.normalized_shape))])
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.bias = bias

        self.norm: NeatModule = None
        if self.elementwise_affine:
            self.norm = NeatModule(1, normalized_shape[0], bias, device, dtype, None, enable_masking)

    def forward(self, tensor):
        mean        = torch.mean(tensor, dim=self.dim, keepdim=True)
        variance    = torch.var(tensor, dim=self.dim, keepdim=True, unbiased=False)
        tensor      = (tensor - mean) / torch.sqrt(variance + self.eps)

        if self.elementwise_affine:
            tensor = tensor * self.norm.weights[0][..., 0, :]
            if self.bias:
                tensor = tensor + self.norm.biases[0][..., 0, :]

        return tensor

    def __repr__(self) -> str:
        module_type = self.__class__.__name__
        module_type = f"{module_type}(NeatModule)"
        return f'{module_type}(g={self.norm.genomes_num}, shape={self.normalized_shape}, bias={self.bias})'
