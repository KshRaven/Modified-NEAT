
from torch import Tensor
from typing import Union
from numpy import ndarray

import numpy as np
import torch.nn.functional as F


class ReplayBuffer(object):
    def __init__(self, buffers: list[str] = None):
        if buffers is None:
            buffers = []
        self.data: dict[str, list[Tensor]] = {b: list() for b in buffers}

    @property
    def size(self):
        return len(list(self.data.values())[0])

    def add_buffers(self, *buffers: str):
        if len(buffers) == 1 and isinstance(buffers[0], (list, tuple)):
            buffers = buffers[0]
        for buffer in buffers:
            self.data[buffer] = list()
        self.reset()

    def update(self, validate=False, add_dims: list[int] = None, **data: Tensor):
        if validate:
            if any(b not in self.data for b in data.keys()):
                raise ValueError(f"Update all buffers at once")
        for name, buffer in data.items():
            if name not in self.data:
                raise ValueError(f"buffer name '{name}' has not been added.")
            if isinstance(buffer, Tensor):
                buffer = buffer.detach().cpu()
            if add_dims:
                for dim in add_dims:
                    if isinstance(buffer, Tensor):
                        buffer = buffer.unsqueeze(dim)
                    elif isinstance(buffer, ndarray):
                        buffer = np.expand_dims(buffer, dim)
                    elif isinstance(buffer, (list, int, float)) and dim == 0:
                        buffer = [buffer]
            self.data[name].append(buffer)

    def reset(self):
        for b in self.data.keys():
            self.data[b] = list()

    def crop(self, max_length: int):
        for b, buffer in self.data.items():
            if len(buffer) > max_length:
                self.data[b] = buffer[-max_length:]

    def rollout(self, sequence_length: int = None, buffers: Union[str, list[str]] = None, as_list=False):
        if any(len(b) == 0 for b in self.data):
            raise ValueError(f"No buffers found")

        def get_max_shape(tensor_list: list[Tensor]):
            max_shape: list[int] = list(tensor_list[0].shape)
            for tensor in tensor_list:
                for i in range(len(max_shape)):
                    max_shape[i] = max(max_shape[i], tensor.shape[i])
            return max_shape

        # Step 2: Compute padding and pad the tensor with the max values along each dimension
        def pad_tensor(tensor, max_shape: list[int], max_values):
            # Calculate the padding for each dimension
            pad = []
            for dim, (tensor_size, max_size) in enumerate(zip(tensor.shape, max_shape)):
                pad.append(max_size - tensor_size)
                pad.append(0)  # No padding before the tensor
            pad = pad[::-1]  # Reverse the list as torch.pad expects padding for last dim first

            # Pad the tensor with the max value from that dimension
            return F.pad(tensor, pad, mode='constant', value=max_values)

        res = {}
        for label, data in self.data.items():
            if buffers is None or (buffers is not None and label in buffers):
                if sequence_length is not None:
                    data = data[-min(len(data), sequence_length):]
                    if sequence_length == 1:
                        data = data[0]
                res[label] = data
        if as_list:
            res = list(res.values())
        return res

    def deque(self, to_del: list[int]):
        for name, buffer in self.data.items():
            self.data[name] = [item for idx, item in enumerate(buffer) if idx not in to_del]
        pass

