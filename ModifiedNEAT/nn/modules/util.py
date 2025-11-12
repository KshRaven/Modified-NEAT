
from ...util.fancy_text import CM, Fore

import torch
import torch.nn as nn
import numpy as np

from torch import Tensor


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
