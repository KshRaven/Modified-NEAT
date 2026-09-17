from ...base import NeatModule, NeatParameter, addindent
from ....util.fancy_text import CM, Fore

import torch
import torch.nn as nn
import numpy as np
import inspect
import math

from torch import Tensor
from typing import Union, Iterable


def model_size(model: nn.Module):
    return np.sum([param.numel() * param.element_size() for param in model.parameters()]) / (1024 ** 2)


def model_params(model: nn.Module):
    return np.sum([param.numel() for param in model.parameters()])


COLOURS = {lbl: clr for lbl, clr in vars(Fore).items() if
           not any(fltr in lbl for fltr in ['BLACK', 'WHITE']) and 'LIGHT' in lbl}


class color_fetch:
    def __init__(self):
        self.current_idx = 0
        self.colors = list(COLOURS.values())
        self.color_num = len(self.colors)

    def __call__(self):
        self.current_idx += 1
        return self.colors[self.current_idx % self.color_num]


get = color_fetch()


def get_tensor_info(tensor: Tensor, label: str = None, verbose: int = None, color: Fore = None):
    tensor = tensor.detach().clone().cpu().type(torch.float32)
    label = f"{CM(label, get() if not color else color)} => {tensor.shape}" if label is not None else ""
    stats = f"\n\tmean={tensor.mean()}, std={tensor.std()}, max={tensor.max()}, min={tensor.min()}\n" if verbose and verbose >= 3 else ""
    details = CM(f"\n{tensor}\n", color if color else Fore.LIGHTWHITE_EX) if verbose and verbose >= 4 else ""
    return f"{label}{stats}{details}"


def calc_padding(kernel_size: int, stride=1, dilation=1):
    """Calculate padding for 'same' convolution."""
    return math.ceil(((kernel_size - 1) * dilation - (stride - 1)) / 2)


def display(label: str, var, check=None):
    """Format label and variable for repr display."""
    if not (var if check is None else check):
        return ''
    else:
        return f", {label}={var}"


# TODO: Fix implementation or location to avoid circular imports
# def get_conv(inputs: Union[Iterable[int], int]):
#     """Get the appropriate Conv class based on input dimensions."""
#     if isinstance(inputs, (int, float)):
#         inputs = [1 for _ in range(int(inputs))]
#     if len(inputs) == 1:
#         from .convolution import Conv1d
#         return Conv1d
#     elif len(inputs) == 2:
#         from .convolution import Conv2d
#         return Conv2d
#     elif len(inputs) == 3:
#         from .convolution import Conv3d
#         return Conv3d
#     else:
#         raise ValueError(f"Unsupported num of image dimension '{len(inputs)}'")


def has_parameter(method, param_name: str):
    """Check if a method has a specific parameter."""
    sig = inspect.signature(method)
    return param_name in sig.parameters
