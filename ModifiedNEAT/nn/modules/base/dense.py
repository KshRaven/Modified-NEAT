from ...base import NeatModule, NeatParameter

from torch import Tensor
from typing import Union, Iterable

import torch


class Linear(NeatModule):
    """Fully connected linear layer."""
    
    def __init__(self, inputs: int, outputs: int, bias=True,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        super(Linear, self).__init__(inputs=inputs, outputs=outputs, bias=bias)
        self.weights = NeatParameter((inputs, outputs), False, device, dtype)
        self.biases  = NeatParameter(outputs, False, device, dtype) if bias else None

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, verbose: int = None):
        verbose = False
        try:
            # print(f"lin_mod={self.weights.data.shape, tensor.shape}")
            w = self.expand(self.weights[keys], tensor, keys=keys)
            # print(w.shape, tensor.shape)
            tensor = torch.matmul(tensor, w)
            if self.biases is not None:
                b = self.expand(self.biases[keys], tensor, keys=keys)
                tensor = tensor + b
            else:
                b = None

            if not verbose:
                return tensor
            else:
                return tensor, (w, b)
        except Exception as e:
            print(f"Input shape = {tensor.shape}")
            print(f"Weights shape = {self.weights.shape}")
            print(f"Biases shape = {self.biases.shape if self.biases is not None else None}")
            raise e


class Polynomial(NeatModule):
    """Polynomial layer with learnable coefficients."""
    
    def __init__(self, inputs: int, outputs: int, coefficients: int = 2, bias=True,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        super(Polynomial, self).__init__(inputs=inputs, outputs=outputs, coefficients=coefficients, bias=bias)
        self.coefficients = coefficients
        self.has_bias = bias
        assert coefficients >= 1
        self.exponents = torch.arange(coefficients, device=device, dtype=dtype) + 1
        self.weights = NeatParameter((coefficients, inputs, outputs), False, device, dtype)
        self.biases  = NeatParameter(outputs, False, device, dtype) if bias else None

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, verbose: int = None):
        tensor = torch.transpose(tensor.unsqueeze(-1) ** self.exponents, -2, -1).unsqueeze(-2)
        w = self.expand(self.weights[keys], tensor, keys=keys, offset=1)
        try:
            # print(f"lin_mod={self.weights.data.shape, tensor.shape}")
            # print(w.shape, tensor.shape)
            tensor = torch.sum(torch.matmul(tensor, w).squeeze(-2), dim=-2)
            if self.has_bias:
                b = self.expand(self.biases[keys], tensor, keys=keys)
                tensor = tensor + b
            else:
                b = None

            if not verbose:
                return tensor
            else:
                return tensor, (w, b)
        except Exception as e:
            print(f"Input shape = {tensor.shape}")
            print(f"Weights shape = {self.weights.shape} -> {w.shape}")
            print(f"Biases shape = {self.biases.shape if self.biases is not None else None}")
            raise e


class Embedding(NeatModule):
    def __init__(
        self, vocab_size: int, dim_size: int,  padding_idx: int | None = None, 
        device: torch.device = 'cpu', dtype: torch.dtype = torch.float32,
    ):
        super(Embedding, self).__init__(
            vocab_size=vocab_size, dim_size=dim_size, padding_idx=padding_idx
        )

        self.vocab_size = vocab_size
        self.dim_size = dim_size
        self.padding_idx = padding_idx

        # Learnable embedding matrix
        self.weights = NeatParameter((vocab_size, dim_size), False, device, dtype)
        # shape (genomes, vocab_size, dim_size)

        # Optional padding handling
        if padding_idx is not None:
            with torch.no_grad():
                self.weights.data[padding_idx].fill_(0)

    def forward(self, tensor: Tensor | int, keys: int | list[int] = None):
        """
        input_ids: LongTensor of shape (genomes, ...)
        returns: embeddings of shape (genomes, ..., dim_size)
        """
        resolved = self.weights.get_list(keys)  # list[int] of valid slot indices
        genomes = torch.tensor(
            resolved, device=self.weights.device, dtype=torch.long
        )

        if isinstance(tensor, (int, float)):
            tensor = torch.full((genomes.shape[0],), int(tensor),
                device=self.weights.device, dtype=torch.long)
        elif tensor.ndim == 0:
            tensor = tensor.unsqueeze(0).to(self.weights.device)

        if genomes.numel() == 0:
            raise ValueError(
                f"No valid genome slots resolved for keys={keys!r}; "
                f"weights tensor has {self.weights.data.shape[0]} genomes."
            )

        # Expand genome indices to match tensor shape
        while genomes.ndim < tensor.ndim:
            genomes = genomes.unsqueeze(-1)

        genomes = genomes.expand_as(tensor)

        # Advanced indexing
        return self.weights.data[genomes, tensor]
