
from torch import Tensor
from typing import Union


class RollbackBuffer(object):
    def __init__(self, buffers: list[str] = None):
        if buffers is None:
            buffers = {}
        self.data: dict[str, list[Tensor]] = {b: list() for b in buffers}

    def add_buffers(self, *buffers: str):
        if len(buffers) == 1 and isinstance(buffers[0], (list, tuple)):
            buffers = buffers[0]
        for buffer in buffers:
            self.data[buffer] = list()
        self.reset()

    def update(self, validate=False, **buffers: Tensor):
        if validate:
            if any(b not in self.data for b in buffers.keys()):
                raise ValueError(f"Update all buffers at once")
        for b, buffer in buffers.items():
            self.data[b].append(buffer.detach().cpu())

    def reset(self):
        for b in self.data.keys():
            self.data[b] = list()

    def get(self, sequence_length: int = None, buffers: Union[str, list[str]] = None):
        res: Union[list[Tensor], dict[str, list[Tensor]]] = {}
        for label, data in self.data.items():
            if buffers is None or (buffers is not None and label in buffers):
                if sequence_length is not None:
                    data = data[-min(len(data), sequence_length)]
                res[label] = data
        if len(res) == 1:
            res = list(res.values())[0]
        if buffers is not None and len(res) == 0:
            raise ValueError(f"No buffers found")
        return res
