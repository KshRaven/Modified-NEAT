from ...base import NeatModule, addindent
from ....util.fancy_text import CM, Fore

from torch import Tensor
from typing import Union, Iterable

import torch.nn as nn
import inspect


class Sequential(NeatModule):
    """Container for stacking modules in sequence."""
    
    def __init__(self, *modules: NeatModule | nn.Module, **kwargs):
        super(Sequential, self).__init__()
        self.modules_list = nn.ModuleList(modules)
        self.kwargs = kwargs

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, verbose: int = None):
        for m_idx, module in enumerate(self.modules_list):
            try:
                # print(f"Module {m_idx} '{module.__class__.__name__}': input = {tensor.shape}")
                if isinstance(module, NeatModule):
                    sig = inspect.signature(module.forward)
                    if "verbose" in sig.parameters and not isinstance(module, (nn.Linear,)):
                        tensor = module(tensor, keys=keys, verbose=verbose)
                    else:
                        tensor = module(tensor, keys=keys)
                else:
                    tensor = module(tensor)
                # print(f"Module {m_idx} '{module.__class__.__name__}': output = {tensor.shape}")
            except Exception as e:
                print(tensor)
                print(CM(f"Failed on module '{m_idx}' =>\n{module}\n"
                         f"\twith tensor shape {tensor.shape}", Fore.LIGHTRED_EX))
                raise e

        return tensor

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for index, module in enumerate(self.modules_list):
            mod_str = repr(module)
            mod_str = addindent(mod_str, 2)
            child_lines.append("(" + str(index) + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "[NeatModule]("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str
