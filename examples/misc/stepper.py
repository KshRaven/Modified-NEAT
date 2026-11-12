"""BinaryStepper: log-scaled output resolution for gradient-free (NEAT) policies."""
from __future__ import annotations

import torch
from torch import Tensor

import ModifiedNEAT.nn as mn

__all__ = ["BinaryStepper", "bits_for"]


def bits_for(steps: int) -> int:
    """Raw outputs needed per continuous output for `steps` intervals: bit_length(steps), i.e. log2 growth.
    1 -> 1, 3 -> 2, 7 -> 3, 1023 -> 10, 10000 -> 14 (a linear scheme would need `steps` outputs)."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    return int(steps).bit_length()


class BinaryStepper(mn.NeatModule):
    """
    Turns saturating (tanh) binary extremes into stepped continuous values.

    Input  : (..., outputs * bits)  raw values
    Output : (..., outputs)         values in [0, 1]  (or [-1, 1] when bipolar)

    Every output owns `bits` raw values. Each one goes through tanh and is mapped to a bit b in [0, 1]
    (-1 -> 0, +1 -> 1). Bits are weighted by binary place value (first = most significant) and normalised
    by the highest reachable state, 2^bits - 1:

        value = sum_i(b_i * 2^(bits-1-i)) / (2^bits - 1)

    steps=3 -> bits=2:  (-1,-1)->0.00  (-1,+1)->0.33  (+1,-1)->0.67  (+1,+1)->1.00
    steps=1 -> bits=1:  value == (tanh(x)+1)/2; bipolar -> exactly tanh(x), i.e. the plain tanh policy.

    Extremes are what NEAT likes to output (no back-propagation), so resolution comes from MORE extreme outputs,
    not finer ones: 2^bits states for `bits` raw values. Values between the extremes stay smooth.
    """

    def __init__(self, outputs: int, steps: int = 1, bipolar: bool = False, bits: int | None = None):
        if bits is None:
            bits = bits_for(steps)
        if bits < 1:
            raise ValueError(f"bits must be >= 1, got {bits}")
        super().__init__(outputs=outputs, steps=steps, bipolar=bipolar, bits=bits)
        self.outputs = outputs
        self.steps = steps
        self.bipolar = bipolar
        self.bits = bits
        self.max_state = float(2 ** bits - 1)

    def forward(self, tensor: Tensor, keys=None, **kwargs) -> Tensor:
        assert tensor.shape[-1] == self.outputs * self.bits, (
            f"Expected last dim {self.outputs} * {self.bits} = {self.outputs * self.bits}, got {tuple(tensor.shape)}"
        )
        bits = (torch.tanh(tensor) + 1.0) * 0.5                          # (..., outputs * bits) in [0, 1]
        bits = bits.unflatten(-1, (self.outputs, self.bits))             # (..., outputs, bits)
        weights = 2.0 ** torch.arange(self.bits - 1, -1, -1, device=tensor.device, dtype=tensor.dtype)
        value = (bits * weights).sum(-1) / self.max_state                # (..., outputs) in [0, 1]
        return value * 2.0 - 1.0 if self.bipolar else value