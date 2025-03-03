
from torch import Tensor
from typing import Union, Any
from numpy import ndarray

import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, buffers: list[str] = None):
        if buffers is None:
            buffers = []
        self.buffer_names: list[str] = buffers
        self.data: dict[int, dict[str, list[Any]]] = {}
        self.mapping: dict[int, int] = {}
        self.reverse: dict[int, int] = {}

    def max_size(self):
        maximum = np.max([np.mean([len(buffer) for buffer in buffers.values()]) for buffers in self.data.values()])
        return maximum

    def min_size(self):
        maximum = np.min([np.mean([len(buffer) for buffer in buffers.values()]) for buffers in self.data.values()])
        return maximum

    def episodes(self):
        listing = []
        if 'ep_map' in self.buffer_names:
            for key_data in self.data.values():
                for ep_idx in key_data['ep_map']:
                    if ep_idx not in listing:
                        listing.append(ep_idx)
        return listing

    def buffer_sizes(self, normalize=False):
        lengths: dict[int, float] = {key: np.mean([len(buffer) for buffer in buffers.values()]) for key, buffers in self.data.items()}
        if normalize:
            maximum, minimum = np.max(list(lengths.values())), np.min(list(lengths.values()))
            if maximum > minimum:
                lengths = {key: (value - minimum) / (maximum - minimum) for key, value in lengths.items()}
            else:
                lengths = {key: 1.0 for key in lengths.keys()}
        return lengths

    def _init_buffers(self, key: int):
        self.data[key] = {b: list() for b in self.buffer_names if self.data.get(key) is None or b not in self.data[key]}

    def _reset_buffers(self, key: int):
        self.data[key] = {b: list() for b in self.buffer_names}

    def _crop_buffers(self, key: int, max_length: int):
        for name, buffer in self.data[key].items():
            if len(buffer) > max_length:
                self.data[key][name] = buffer[-max_length:]

    def _deque_buffers(self, key: int, to_del: list[int]):
        for name, buffer in self.data[key].items():
            self.data[key][name] = [item for idx, item in enumerate(buffer) if idx not in to_del]

    def update_mapping(self, mapping: dict[int, int]):
        """
        Map of every genome key to their index
        :param mapping:
        :return:
        """
        new_keys, new_indices = [], []
        # Add keys
        for key, index in mapping.items():
            # Update mapping
            self.mapping[key] = index
            self.reverse[index] = key
            # Initialize buffers if already existing
            if key not in self.data:
                self._init_buffers(key)
            new_keys.append(key)
            new_indices.append(index)
        # Delete keys that are not in the new mapping
        for key in list(self.mapping.keys()):
            if key not in new_keys:
                del self.mapping[key]
                del self.data[key]
        # Delete indices that are not in the new mapping
        for index in list(self.reverse.keys()):
            if index not in new_indices:
                del self.reverse[index]
        self.sort()

    def add_buffers(self, *buffers: str):
        if len(buffers) == 1 and isinstance(buffers[0], (list, tuple)):
            buffers = buffers[0]
        for buffer in buffers:
            if buffer not in self.buffer_names:
                self.buffer_names.append(buffer)
                for key in self.data.keys():
                    self.data[key][buffer] = list()
        self.reset()

    def update(self, add_dims: Union[int, list[int]] = None, validation: int = True, **inputs):
        if validation:
            buffer_names = inputs.keys()
            for name in buffer_names:
                if name not in self.buffer_names:
                    raise ValueError(f"buffer name '{name}' has not been added.")
            if validation == 2 and len(buffer_names) != len(self.buffer_names):
                raise ValueError(f"Update all buffers at once")

        def expand_dims(var, dims: Union[int, list[int]]):
            if not isinstance(dims, (list, tuple)):
                dims = [dims]
            for dim in dims:
                if isinstance(var, Tensor):
                    var = var.unsqueeze(dim)
                elif isinstance(var, ndarray):
                    var = np.expand_dims(var, dim)
                # elif isinstance(var, (list, int, float)) and dim == 0:
                #     var = [var]
            return var

        def check(var, label):
            try:
                error = False
                if not isinstance(var, (ndarray, Tensor, list)):
                    raise ValueError(f"Variable '{label}' is not a Tensor, Array or List; type = '{type(var)}'")
                if isinstance(var, (ndarray, Tensor)) and var.shape[0] != len(self.data):
                    error = True
                elif isinstance(var, list) and len(var) != len(self.data):
                    error = True
                if error:
                    out = var.shape if isinstance(var, (ndarray, Tensor)) else len(var)
                    raise ValueError(f"Shape of data '{label}' cannot be broadcast onto buffer; shape = {out}")
            except ValueError as e:
                raise e
            except IndexError as e:
                print(f"\nvar '{label}' shape = {var.shape if isinstance(var, (ndarray, Tensor)) else len(var)}\n")
                raise e
            except Exception as e:
                raise e

        def get_value(key: int, var, ctn: bool):
            index = self.mapping[key]
            if ctn:
                value = var[index]
            elif isinstance(var, Tensor):
                value = torch.select(var, dim=0, index=index)
            elif isinstance(var, list):
                value = var[index]
            else:
                raise ValueError(f"Cannot get the value from a variable of dtype '{type(var)}'")
            return value

        for name, data in inputs.items():
            if isinstance(data, Tensor):
                data = data.detach().cpu()
            if isinstance(data, (int, float, bool)):
                data = [data for _ in range(len(self.data))]
            check(data, name)
            if isinstance(data, ndarray):
                data = torch.tensor(data, device='cpu')
                to_numpy = True
            else:
                to_numpy = False
            if add_dims:
                expand_dims(data, add_dims)
            for k in self.data.keys():
                self.data[k][name].append(get_value(k, data, to_numpy))

    def reset(self, keys: list[int] = None):
        for key in self.data.keys():
            if keys is None or (keys is not None and key in keys):
                self._reset_buffers(key)

    def crop(self, max_length: int, keys: list[int] = None):
        for key in self.data.keys():
            if keys is None or (keys is not None and key in keys):
                self._crop_buffers(key, max_length)

    def deque(self, to_del: dict[int, list[int]], keys: list[int] = None):
        for key in self.data.keys():
            if keys is None or (keys is not None and key in keys):
                self._deque_buffers(key, to_del[key])

    def sort(self):
        self.data = dict(sorted(self.data.items(), key=lambda item: len(item[1])))
        self.mapping = dict(sorted(self.mapping.items(), key=lambda item: item[1]))
        self.reverse = dict(sorted(self.reverse.items(), key=lambda item: item[0]))

    def rollout(self, buffers: Union[str, list[str]] = None, sequence_length: int = None, keys: list[int] = None,
                as_list=False, stack=False):
        if len(self.mapping) == 0:
            raise ValueError(f"No mapping set")
        self.sort()
        if len(list(self.data.values())[0]) == 0:
            raise ValueError(f"No data in buffers")

        if isinstance(buffers, str):
            buffers = [buffers]
        elif buffers is None:
            buffers = self.buffer_names
        if keys is None:
            keys = list(self.mapping.keys())

        res = {}
        for label in buffers:
            group = {}
            for key in keys:
                # Get key's buffer
                buffer = self.data[key][label]
                # Limit length
                if sequence_length is not None:
                    buffer = buffer[-min(len(buffer), sequence_length):]
                    # if sequence_length == 1:
                    #     buffer = buffer[0]
                # Stack buffer
                if stack:
                    if isinstance(buffer[0], ndarray):
                        buffer = np.stack(buffer, axis=0)
                    elif isinstance(buffer[0], Tensor):
                        buffer = torch.stack(buffer, dim=0)
                    elif isinstance(buffer[0], (int, float, bool)):
                        pass
                group[key] = buffer
            res[label] = group
        if as_list:
            res = list(res.values())
        return res
