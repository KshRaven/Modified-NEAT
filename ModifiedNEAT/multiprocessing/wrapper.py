
from torch.nn import Module
from torch import Tensor
from itertools import count

import torch


class ModelWrapper(object):
    __indexer = count(0)
    _verbosity = 0

    def __init__(self, model: Module, input_shape: tuple[int, ...], output_shape: tuple[int, ...],
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float64):
        self.model = model
        self.model_index = next(self.__indexer)
        self.inputs: Tensor = None # torch.zeros(*input_shape, device=device, dtype=dtype)
        self.outputs: Tensor = None # torch.zeros(*output_shape, device=device, dtype=dtype)

    def set(self, inputs: Tensor):
        if self._verbosity:
            print(f"Setting tensors to Module{self.model_index}: {self.model.__class__.__name__}")
        self.inputs = inputs
        self.outputs = None

    def get(self):
        if self._verbosity:
            print(f"Getting tensors from Module{self.model_index}: {self.model.__class__.__name__}")
        if self.outputs is None:
            raise RuntimeError(f"Inputs have not been calculated")
        return self.outputs

    def __call__(self, *args, **kw_args):
        if self._verbosity:
            print(f"Calculating output for Module{self.model_index}: {self.model.__class__.__name__}")
        tensor = self.model(self.inputs)
        self.outputs = tensor
        return tensor


class verbose(object):
    def __init__(self, verbosity=1):
        self.verbosity = verbosity
        ModelWrapper._verbosity = self.verbosity

    def __enter__(self):
        ModelWrapper._verbosity = self.verbosity

    def __exit__(self, exc_type, exc_val, exc_tb):
        ModelWrapper._verbosity = 0
