
from build.nn.genome import Genome, INT

from torch import Tensor
from itertools import count
from numba import njit, prange, types
from numpy import ndarray as CPUArray
from typing import Union, Iterable, Any

import torch
import torch.nn as nn

LAYER_DEF = types.Tuple([INT, INT])


class NeatParameter(nn.Module):
    __indexer = count(0)

    def __init__(self, shape: Union[int, Iterable[int]], requires_grad=False,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        super(NeatParameter, self).__init__()
        if isinstance(shape, (int, float)):
            shape = [shape]
        self.param_index = next(self.__indexer)
        self.original_shape: tuple[int, ...] = tuple(shape)

        # BUILD
        self.data = nn.Parameter(torch.randn(1, *shape, device=device, dtype=dtype), requires_grad=requires_grad)
        self.mapping: dict[int, int] = None

        # ATTRIBUTES
        self.genome_num: int = None

        self.cd: CPUArray = None
        self.md: CPUArray = None

    def __len__(self):
        return self.genome_num

    @property
    def shape(self):
        return self.data.shape

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self.data.dtype

    def _update_data(self, tensor: Tensor):
        remove_rg = self.data.requires_grad
        if remove_rg:
            self.data.requires_grad_(False)
        self.data[:] = tensor
        if remove_rg:
            self.data.requires_grad_(True)

    def update(self, genomes: dict[int, Genome], params: Union[Tensor, CPUArray] = None, verify=False):
        params = params[self.param_index] if params is not None else None
        if isinstance(params, CPUArray):
            params = torch.tensor(params, self.device, self.dtype)

        self.genome_num = len(genomes)

        if self.mapping is None:
            if params is None:
                self.data = nn.Parameter(self.data.expand(self.genome_num, *self.original_shape).clone(), self.data.requires_grad)
                # size = self.data[0].numel()
                # self.data[:] = (torch.arange(size).view(self.data.shape[1:]) / size)
            else:
                p_shape = torch.tensor(params.shape)
                d_shape = torch.tensor(self.original_shape)
                if not torch.all(d_shape[1:] == p_shape[1:]):
                    raise ValueError(f"Cannot initialize params of shape {p_shape.numpy()} "
                                     f"to data of shape {d_shape.numpy()}.")
                self.data = nn.Parameter(params.clone().to(self.device, self.dtype), self.data.requires_grad)
        else:
            if params is not None:
                # Verify pre-existing genomes
                new_indices = [index for index, genome in enumerate(genomes.values()) if genome.key in self.mapping]
                old_indices = [self.mapping[genome.key] for genome in genomes.values() if genome.key in self.mapping]
                if verify and len(old_indices) > 0:
                    if not torch.all(self.data[old_indices] == params[new_indices]):
                        raise ValueError(f"Some values from new params are not in old params after update.")

                # Set population
                self.data = nn.Parameter(params.clone().to(self.device, self.dtype), self.data.requires_grad)

        # TODO: Rearrange genomes according to fitness and gid

        self.mapping = {genome.key: index for index, genome in enumerate(genomes.values())}

        return self.mapping, self.genome_num

    def get(self, keys: Union[int, Iterable[int]] = None):
        if keys is None:
            return self.data
        elif isinstance(keys, (int, float)):
            keys = [keys]
        return self.data[[self.mapping[i] for i in keys]]

    def __getitem__(self, keys: Union[int, Iterable[int]] = None):
        return self.get(keys)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return self.__str__()

    def requires_gradients(self, mode=False):
        self.data.requires_grad_(mode)

    def __eq__(self, other):
        if isinstance(other, NeatParameter):
            return self.param_index == other.param_index
        return False

    def __hash__(self):
        return hash(self.param_index)


class NeatModule(nn.Module):
    __indexer = count(0)

    def __init__(self, **params):
        super(NeatModule, self).__init__()

        self.module_index = next(self.__indexer)
        self.genome_num: int         = None
        self.mapping: dict[int, int] = None
        self.params: dict[str, Any]  = params
        self.updated = False

    def neat_parameters(self):
        params: list[NeatParameter] = []

        def get(obj):
            if isinstance(obj, NeatParameter):
                params.append(obj)
            elif isinstance(obj, (NeatModule, Model)):
                params.extend(obj.neat_parameters())

        for n, item in vars(self).items():
            get(item)
            if isinstance(item, (tuple, list, dict)) and len(item) > 0:
                # print(f"module index {self.module_index}: {self.__class__.__name__} => {n}")
                if isinstance(item, dict):
                    item = list(item.values())
                # if any([isinstance(x, (nn.Module, nn.ModuleList, nn.ModuleDict, NeatParameter, NeatModule))
                #         for x in item]):
                for sub_item in item:
                    if isinstance(sub_item, nn.ModuleList):
                        for x in sub_item:
                            get(x)
                    elif isinstance(sub_item, nn.ModuleDict):
                        for x in sub_item.values:
                            get(x)
                    else:
                        get(sub_item)
        return list(set(params))

    def neat_modules(self, extensive=True):
        modules: list[NeatModule] = []

        def get(obj):
            if isinstance(obj, (NeatModule, Model)):
                modules.append(obj)
                if extensive:
                    modules.extend(obj.neat_modules(extensive))

        for n, item in vars(self).items():
            get(item)
            if isinstance(item, (tuple, list, dict)) and len(item) > 0:
                if isinstance(item, dict):
                    item = list(item.values())
                for sub_item in item:
                    if isinstance(sub_item, nn.ModuleList):
                        for x in sub_item:
                            get(x)
                    elif isinstance(sub_item, nn.ModuleDict):
                        for x in sub_item.values():
                            get(x)
                    else:
                        get(sub_item)
        return list(set(modules))

    def update_limit(self):
        pass

    def update(self, genomes: dict[int, Genome], params: Union[Tensor, CPUArray] = None, verify=False):
        # if not self.updated:
        self.mapping = {genome.key: index for index, genome in enumerate(genomes.values())}
        self.genome_num = len(self.mapping)

        for var in (self.neat_modules()+self.neat_parameters()):
            res = var.update(genomes, params, verify)
            if res is not None:
                self.updated = True
                if verify:
                    for key, index in var.mapping.items():
                        if key not in self.mapping or self.mapping[key] != index:
                            raise ValueError(f"Error in module mapping during update")
        # print(f"Module {self.module_index}: {self.genome_num} {list(self.mapping.items())[:2]}")

        return self.mapping, self.genome_num

    @staticmethod
    @njit(nogil=True)
    def crop(shape: tuple[int, ...]):
        res = [[0 for _ in range(0)] for _ in prange(len(shape))]
        for i in prange(len(shape)):
            res[i] = list(range(shape[i]))
        return res

    def fetch(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        if isinstance(keys, (int, float)):
            keys = [keys]
        elif keys is None:
            return tensor
        try:
            return tensor[[self.mapping[key] for key in keys]]
        except IndexError as e:
            print(self.module_index)
            print(self)
            print(tensor)
            print(tensor.shape)
            print(self.mapping)
            raise e
        # else:
        #     raise ValueError(f"Not all keys included in global tensor")

    def expand(self, tensor: Union[Tensor, None], target: Tensor, offset: int = None, keys=None):
        if tensor is not None:
            # print(tensor.shape)
            # tensor = self.fetch(tensor, keys)
            if not(tensor.shape[0] == 1 or tensor.shape[0] == target.shape[0]):
                raise ValueError(f"Tensors' key dims do not match; tensor={tensor.shape}, target={target.shape}.")
            extra   = target.ndim - tensor.ndim
            pre     = extra if offset is None else offset
            post    = 0 if offset is None else max(0, extra - offset)
            return tensor.view(tensor.shape[0], *[1 for _ in range(pre)], *tensor.shape[1:], *[1 for _ in range(post)])
        else:
            return None

    def __repr__(self):
        if len(self.params) == 0:
            return super().__repr__()
        else:
            params = ""
            for i, (param, value) in enumerate(self.params.items()):
                params += f"{param}={value}"
                if i < len(self.params)-1:
                    params += ", "
            return f"{self.__class__.__name__}[NeatModule]({params})"

    def requires_gradients(self, mode=False):
        for item in self.parameters():
            item.requires_grad_(mode)

    def __eq__(self, other):
        if isinstance(other, NeatModule):
            return self.module_index == other.module_index
        return False

    def __hash__(self):
        return hash(self.module_index)


class Model(NeatModule):
    def __init__(self):
        super().__init__()
        self.distribution: str = 'normal'

    def dist(self, m: Tensor, s: Tensor):
        if self.distribution == 'discrete':
            m = m.unsqueeze(-2)
            distribution = torch.distributions.Categorical(torch.softmax(m, -1))
        elif self.distribution == 'normal':
            distribution = torch.distributions.Normal(m, s)
        elif self.distribution == 'mult_var_normal':
            cov = torch.diag_embed(s**2)
            distribution = torch.distributions.MultivariateNormal(m, cov)
        else:
            raise NotImplementedError(f"Unsupported distribution")
        return distribution

    def get_mean(self, latent: Tensor, keys: Union[int, Iterable[int]] = None) -> Tensor:
        raise NotImplementedError(f"No 'get_mean' method")

    def get_std(self, latent: Tensor, keys: Union[int, Iterable[int]] = None) -> Tensor:
        raise NotImplementedError(f"No 'get_std' method")

    def get_action(self, state: Tensor, keys: Union[int, Iterable[int]] = None) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(f"No 'get_action' method")

    def evaluate_action(self, state: Tensor, action: Tensor, keys: Union[int, Iterable[int]] = None) -> [Tensor, Union[Tensor, None]]:
        raise NotImplementedError(f"No 'evaluate_action' method")

    def get_policy(self, state: Tensor, keys: Union[int, Iterable[int]] = None, **options) -> Tensor:
        raise NotImplementedError(f"No 'get_policy' method")

    def get_value(self, state: Tensor, keys: Union[int, Iterable[int]] = None) -> Tensor:
        raise NotImplementedError(f"No 'get_value' method")

    def __repr__(self):
        if len(self.params) == 0:
            return super(nn.Module, self).__repr__()
        else:
            params = ""
            for i, (param, value) in enumerate(self.params.items()):
                params += f"{param}={value}"
                if i < len(self.params)-1:
                    params += ", "
            return f"{self.__class__.__name__}[NeatModule]({params})"

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
