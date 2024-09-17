
from build.nn.genome import Genome, Network, INT, NETWORK
from build.util.datetime import eta, clock
from build.util import storage

from torch import Tensor
from itertools import count
from numba import njit, prange, types, typeof
from numba.typed import List, Dict
from numpy import ndarray
from typing import Union

import torch
import torch.nn as nn

LAYER_DEF = types.Tuple([INT, INT])


class NeatModule(nn.Module):
    __indexer = count(0)

    def __init__(self, inputs: int, outputs: int, bias=False, device: torch.device = 'cpu',
                 dtype: torch.dtype = torch.float32, activation: nn.Module = None, enable_masking=True):
        super(NeatModule, self).__init__()

        # BUILD
        self.module_index = next(self.__indexer)
        self.networks: dict[int, Network] = Dict.empty(INT, NETWORK)
        self.weights: list[nn.Parameter] = None
        self.biases: list[nn.Parameter] = None
        self.activation = activation

        # ATTRIBUTES
        self.genomes_num  = 0

        # STATE
        self.features_in  = inputs
        self.features_out = outputs
        self.enable_bias  = bias
        self.device       = device
        self.dtype        = dtype
        self.mask         = enable_masking

    def parameters(self, recurse: bool = True):
        return [p for p in self.weights] + ([p for p in self.biases] if self.biases is not None else [])

    def setup(self, genomes: dict[int, Genome], verbose: int = None):
        # Get weight structures
        self.genomes_num  = len(genomes)
        ts = clock.perf_counter()
        self.networks = self._create_networks(genomes, self.module_index, self.features_in, self.features_out)
        if verbose and verbose >= 2:
            print(f"for a {self.features_in} input and {self.features_out} output module")
            print(f"created neat-module-{self.module_index} networks in {(clock.perf_counter()-ts):.2f}s")
        self.update(verbose)
        self.eval()

    @staticmethod
    @njit(nogil=True)
    def _create_networks(genomes: dict[int, Genome], index: int, inputs: int, outputs: int):
        networks: dict[int, Network] = Dict()
        mapping = List(genomes.keys())
        for idx in prange(len(genomes)):
            genome = genomes[mapping[idx]]
            # Create network
            network = genome.add_network(index, inputs, outputs, None)
            networks[genome.key] = network
        return networks

    def update(self, verbose: int = None):
        ts = clock.perf_counter()
        build = self._get_build(self.networks)
        if verbose:
            print(f"got neat-module-{self.module_index} build in {(clock.perf_counter()-ts):.2f}s")
        # Set weights and biases
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones((self.genomes_num, i, o), device=self.device, dtype=self.dtype))
            for i, o in build.values()
        ])
        if self.enable_bias:
            self.biases = nn.ParameterList([
                nn.Parameter(torch.ones((self.genomes_num, 1, o), device=self.device, dtype=self.dtype))
                for _, o in build.values()
            ])

        # Fill weights and biases
        self.fill(verbose)

    @staticmethod
    @njit(nogil=True)
    def _get_build(networks: dict[int, Network]):
        build: dict[int, tuple[int, int]] = Dict.empty(INT, LAYER_DEF)
        mapping = List(networks.keys())
        for idx in prange(len(networks)):
            network = networks[mapping[idx]]
            # Get the max build value
            for layer_idx in prange(len(network.build)):
                i, o = network.build[layer_idx]
                m = build.get(layer_idx)
                if m is None:
                    build[layer_idx] = (i, o)
                else:
                    mi, mo = m
                    build[layer_idx] = (max(mi, i), max(mo, o))
        return build

    def fill(self, verbose: int = None):
        with torch.no_grad():
            ts = clock.perf_counter()
            w, b = self._get_values(self.networks, [w.cpu().numpy() for w in self.weights],
                                    [b.cpu().numpy() for b in self.biases] if self.enable_bias else None)
            if verbose:
                print(f"got neat-module-{self.module_index} values in {(clock.perf_counter()-ts):.2f}s")
            for weight, source in zip(self.weights, w):
                weight[:] = torch.tensor(source)
            if self.enable_bias:
                for bias, source in zip(self.biases, b):
                    bias[:] = torch.tensor(source)

    @staticmethod
    @njit(nogil=True)
    def _get_values(networks: dict[int, Network], weights: list[ndarray], biases: Union[list[ndarray], None]):
        mapping: list[int] = List(networks.keys())
        for idx in prange(len(networks)):
            network = networks[mapping[idx]]
            # Set the weights
            w_ = network.weights
            for i in prange(len(w_)):
                w = w_[i]
                y, x = w.shape
                weights[i][idx, :y, :x] = w
            # Set the biases
            if biases is not None:
                b_ = network.biases
                for i in prange(len(b_)):
                    b = b_[i]
                    y = b.size
                    biases[i][idx, :y] = b
        return weights, biases

    def initialize(self):
        with torch.no_grad():
            nn.init.xavier_normal_(self.weights, 1)
            if self.biases is not None:
                nn.init.xavier_normal_(self.biases, 1)

    def forward(self, inputs: Tensor, mask: Tensor = None):
        genomes = self.genomes_num if mask is None else torch.sum(mask).item()
        tensor, squeeze = handle_input_dims(inputs, genomes, True)
        if self.mask:
            tensor, mask = handle_input_mask(tensor, mask, -3)

        # Execute forward pass
        for idx in range(len(self.weights)):
            w = self.weights[idx]
            if mask is not None:
                w = w[mask]
            # print(tensor.shape, w.shape, mask.shape if mask is not None else None)
            tensor = torch.matmul(tensor, w)
            if self.biases is not None:
                b = self.biases[idx]
                if mask is not None:
                    b = b[mask]
                tensor += b
        # Apply activation
        if self.activation is not None:
            tensor = self.activation(tensor)

        if self.mask:
            tensor = handle_output_mask(tensor, mask, -3)
        outputs = handle_output_dims(tensor, squeeze, genomes)

        return outputs

    def __repr__(self) -> str:
        module_type = self.__class__.__name__
        if module_type != 'NeatModule':
            module_type = f"{module_type}(NeatModule)"
        return f'{module_type}(g={self.genomes_num}, in={self.features_in}, ' \
               f'out={self.features_out}, bias={self.enable_bias}, actv={self.activation})'


def handle_input_mask(tensor: Tensor, mask: Union[Tensor, None], genome_dim_idx: int):
    if genome_dim_idx < 0:
        genome_dim_idx = tensor.ndim + genome_dim_idx
    if mask is not None:
        if mask.ndim != 1 and tensor.shape[genome_dim_idx] != mask.numel():
            raise ValueError(f"Wrong mask shape; Expected ({tensor.shape[genome_dim_idx]},) got {mask.shape}.")
        mask_ = mask.clone()
        for dim in reversed(tensor.shape[:genome_dim_idx]):
            mask_ = mask_.unsqueeze(0).expand(dim, *mask_.shape)
        tensor = tensor[mask_].view(*tensor.shape[:genome_dim_idx], -1, *tensor.shape[genome_dim_idx+1:])
        # print(tensor.shape, mask.shape)
    return tensor, mask


def handle_output_mask(tensor: Tensor, mask: Union[Tensor, None], genome_dim_idx: int):
    if genome_dim_idx < 0:
        genome_dim_idx = tensor.ndim + genome_dim_idx
    if mask is not None:
        total = mask.numel()
        filling = torch.sum(mask).item()
        padding = total - filling
        if padding > 0:
            fill = torch.zeros(*tensor.shape[:genome_dim_idx], total, *tensor.shape[genome_dim_idx+1:],
                               device=tensor.device, dtype=tensor.dtype)
            for dim in reversed(tensor.shape[:genome_dim_idx]):
                mask = mask.unsqueeze(0).expand(dim, *mask.shape)
            fill[mask] = tensor.contiguous().view(-1, tensor.shape[-1])
            tensor = fill
    return tensor


def handle_input_dims(tensor: Tensor, genomes: int = None, is_module=False):
    """
    Ensures the tensor is of shape (*extra_dims, genomes, batch_size, features)
    :param tensor: Input tensor
    :param mask: Mask tensor to reduce computations
    :param genomes: Necessary when using mask
    :return:
    """
    squeeze = False
    # On 1-D tensor
    if tensor.ndim == 1:
        # Assume no genome dimension so add it, along with batch_dim. Remove batch dim later
        inputs = tensor.shape[0]
        tensor = tensor.unsqueeze(0).unsqueeze(1).expand(genomes, 1, inputs)
        squeeze = True
    # On 2-D tensor
    elif tensor.ndim == 2:
        # Assuming 3rd dim is naturally the genomes_im, then add it if it doesn't exist
        if tensor.shape[0] != genomes:
            extra_dim, inputs = tensor.shape
            tensor = tensor.unsqueeze(1).expand(extra_dim, genomes, inputs)
        # If it's there add the batch_dim and remove it later
        tensor = tensor.unsqueeze(0)
        squeeze = True
    # On tensor D greater than or equal to 3
    if tensor.ndim > 2:
        if is_module:
            # Assuming 3rd dim is naturally the genomes_dim but input has it on 2nd dim
            if tensor.shape[-2] == genomes:
                tensor = tensor.transpose(-2, -3)
            # If no other case math raise error
            elif tensor.shape[-3] != genomes:
                raise ValueError(f"Wrong input shape; Expected "
                                 f"(*{genomes}, *batch_size, features), got {tensor.shape}.")
        elif tensor.ndim == 3 and tensor.shape[-2] == genomes:
            tensor = tensor.unsqueeze(0)
            squeeze = True
        else:
            if tensor.shape[-2] != genomes:
                raise ValueError(f"Dim '-2' should be equal to {genomes} but got {tensor.shape}")
    return tensor, squeeze


def handle_output_dims(tensor: Tensor, squeeze: bool, genomes: int):
    # Return shape to (*extra_dims, genomes, features)
    if tensor.shape[-3] == genomes:
        tensor = tensor.transpose(-2, -3)
    # Squeeze when no batch_dim is required
    if squeeze:
        tensor = tensor.squeeze(0)
    return tensor


def get_modules(*models: nn.Module) -> list[NeatModule]:
    neat_modules = []
    for model in models:
        for module in model.modules():
            if isinstance(module, NeatModule) and module not in neat_modules:
                neat_modules.append(module)
    neat_modules = sorted(neat_modules, key=lambda m: m.features_in * m.features_out, reverse=True)
    return neat_modules


def bind_modules(modules: list[NeatModule], genomes: Union[dict[int, Genome], list[Genome]], verbose: int = None):
    if isinstance(genomes, dict):
        # raise ValueError(f"genomes must be a Numba dict, not a normal dict")
        genomes = Dict([(k, v) for k, v in genomes.items()])
    elif isinstance(genomes, (List, list)):
        # raise ValueError(f"genomes must be a Numba dict, not a normal dict")
        genomes = Dict([(g.key, g) for g in genomes])
    ts = clock.perf_counter()
    ud = 0
    ut = len(modules)
    for module in modules:
        ud += 1
        module.setup(genomes, verbose)
        if verbose:
            eta(ts, ud, ut, f"binding modules")
    if verbose:
        print(f"\rbound modules in {(clock.perf_counter() - ts):.2f}s")


class Model(nn.Module):
    def get_mean(self, latent: Tensor) -> Tensor:
        raise NotImplementedError(f"No 'get_mean' method")

    def get_std(self, latent: Tensor) -> Tensor:
        raise NotImplementedError(f"No 'get_std' method")

    def get_action(self, state: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(f"No 'get_action' method")

    def evaluate_action(self, state: Tensor, action: Tensor) -> [Tensor, Union[Tensor, None]]:
        raise NotImplementedError(f"No 'evaluate_action' method")

    def get_policy(self, state: Tensor, **options) -> Tensor:
        raise NotImplementedError(f"No 'get_policy' method")

    def get_value(self, state: Tensor) -> Tensor:
        raise NotImplementedError(f"No 'get_value' method")

#     def save(self, symbol: str, timeframe: Timeframe, file_no: int = None, replace: bool = False) -> None:
#         model_type = self.__class__.__name__
#         cons_name = f'{symbol}-{timeframe.name}-{model_type}'
#         storage.save(self, 'model', 'trading_models', file_no=file_no, replace=replace,
#                      subdirectory=cons_name, items_name=f'{model_type} Model')
#
#     def load(self, symbol: str, timeframe: Timeframe, file_no: int = None):
#         model_type = self.__class__.__name__
#         cons_name = f'{symbol}-{timeframe.name}-{model_type}'
#         model = storage.load('model', 'trading_models', file_no=file_no,
#                              subdirectory=cons_name, items_name=f'{model_type} Model')
#         if model is not None:
#             for attr, val in vars(model).items():
#                 setattr(self, attr, val)
#
#
# def save(
#         model: Model, symbol: str, timeframe: Timeframe, file_no: int = None, replace: bool = False
# ) -> None:
#     if model is None:
#         raise ValueError("Model cannot be None")
#
#     model_type = type(model).__name__
#     cons_name = f'{symbol}-{timeframe.name}-{model_type}'
#     storage.save(model, 'model', 'trading_models', file_no=file_no, replace=replace,
#                  subdirectory=cons_name, items_name='Model')
#
#
# def load(
#         model_class: str, symbol: str, timeframe: Timeframe, file_no: int = None
# ) -> tuple[Union[Model, None]]:
#     cons_name = f'{symbol}-{timeframe.name}-{model_class}'
#     model = storage.load('model', 'trading_models', file_no=file_no, subdirectory=cons_name, items_name='Model')
#     return model
