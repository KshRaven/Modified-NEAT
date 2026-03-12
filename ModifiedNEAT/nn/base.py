
from ModifiedNEAT.nn.genome import Genome, INT
from ModifiedNEAT.config import Config

from torch import Tensor
from itertools import count
from numba import njit, prange, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numpy import ndarray as CPUArray
from cupy import ndarray as GPUArray
from typing import Union, Iterable, Any

import torch
import torch.nn as nn
import numpy as np
import cupy as cp


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class NeatParameter(nn.Module):
    __indexer = count(0)

    def __init__(self, shape: int | Iterable[int], requires_grad=False,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        super(NeatParameter, self).__init__()
        if isinstance(shape, (int, float)):
            shape = [shape]
        self.param_index: int = next(self.__indexer)
        self._module_index: int | None = None  # Used to specify the module that a parameter is implemented TODO: Ensure all modules support this for debugging reasons. Though not necessary :)
        self.original_shape: tuple[int, ...] = tuple(shape)

        # BUILD
        self.data = nn.Parameter(torch.randn(1, *shape, device=device, dtype=dtype), requires_grad=requires_grad)
        self.mapping: dict[int, int] = None

        # ATTRIBUTES
        self.genome_num: int = None

        self.cd: CPUArray = None
        self.md: CPUArray = None

    def get_attr_name(self, value):
        for name, val in self.__dict__.items():
            if val is value:
                return name
        return None

    @property
    def module_index(self):
        if self._module_index is None:
            raise ValueError(f"Module index has not been set. It is set every time .neat_parameters() is called.")
        else:
            return self._module_index

    def reset(self):
        # BUILD
        self.data = nn.Parameter(
            torch.randn(1, *self.original_shape, device=self.device, dtype=self.dtype),
            requires_grad=self.requires_grad)
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
    def ndim(self):
        return self.data.ndim

    @property
    def numel(self):
        return self.data.numel()

    @property
    def element_size(self):
        return self.data.element_size()

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def requires_grad(self):
        return self.data.requires_grad

    def _update_data(self, tensor: Tensor):
        remove_rg = self.data.requires_grad
        if remove_rg:
            self.data.requires_grad_(False)
        self.data[:] = tensor
        if remove_rg:
            self.data.requires_grad_(True)

    def update(self, genomes: dict[int, Genome], params: Union[Tensor, GPUArray] = None, verify=False):
        if params is None and self.mapping is not None and not all([key in self.mapping for key in genomes.keys()]):
            raise ValueError(f"Updating genomes of a module without parameters!")
        if isinstance(params, dict):
            params = params[self.param_index]
        if isinstance(params, DeviceNDArray):
            params = params.copy_to_host()
        if isinstance(params, CPUArray):
            params = torch.tensor(params, device=self.device, dtype=self.dtype)
        elif isinstance(params, GPUArray):
            params = torch.from_dlpack(params).to(self.device, self.dtype)

        self.genome_num = len(genomes)

        if self.mapping is None:
            if params is None:
                self.data = nn.Parameter(self.data.expand(self.genome_num, *self.original_shape).clone(), self.data.requires_grad)
                # size = self.data[0].numel()
                # self.data[:] = (torch.arange(size).view(self.data.shape[1:]) / size)
            else:
                d_shape = torch.tensor(self.original_shape)
                p_shape = torch.tensor(params.shape[-len(d_shape):])
                if not torch.all(d_shape == p_shape):
                    raise ValueError(f"Cannot initialize params of shape {p_shape.numpy()} "
                                     f"to data of shape {d_shape.numpy()}.")
                self.data = nn.Parameter(params.clone().to(self.device, self.dtype), self.data.requires_grad)
        else:
            if params is not None:
                # Verify pre-existing genomes
                new_indices = [index for index, genome in enumerate(genomes.values()) if genome.key in self.mapping]
                old_indices = [self.mapping[genome.key] for genome in genomes.values() if genome.key in self.mapping]
                try:
                    if verify and len(old_indices) > 0:
                        if not torch.all(self.data[old_indices] == (params[new_indices].to(self.data.device))):
                            # raise ValueError(f"Some values from new params are not in old params after update.")
                            pass
                except Exception as e:
                    print(new_indices)
                    print(old_indices)
                    print(list(genomes.keys()))
                    print(params.shape, self.data.shape)
                    raise e
                # Set population
                self.data = nn.Parameter(params.clone(), self.data.requires_grad)

        # TODO: Rearrange genomes according to fitness and gid

        self.mapping = {genome.key: index for index, genome in enumerate(genomes.values())}

        return self.mapping, self.genome_num

    def get(self, keys: int | Iterable[int] = None):
        if keys is None:
            return self.data
        elif isinstance(keys, (int, float)):
            keys = [keys]
        return self.data[[self.mapping[i] for i in keys]]

    def __getitem__(self, keys: int | Iterable[int] = None):
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
        self.genus: int = None

    def get_attr_name(self, value):
        for name, val in vars(self).items():
            if val is value:
                return name
        return None

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

        # Remove duplicate parameters and sort them by parameter index
        params = sorted(list(set(params)), key=lambda p: p.param_index)

        # TODO: For testing purpose update all neat parameters within the module with the module's index for debugging. Temporary!
        for param in params:
            if isinstance(param, NeatParameter) and param._module_index is None:
                param._module_index = self.module_index

        return params

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
        return sorted(list(set(modules)), key=lambda m: m.module_index)

    def update_limit(self):
        pass

    def update(self, genomes: dict[int, Genome], params: Union[Tensor, CPUArray] = None, verify=False):
        # if not self.updated:
        self.mapping = {genome.key: index for index, genome in enumerate(genomes.values())}
        self.genome_num = len(self.mapping)

        for var in (self.neat_parameters()+self.neat_modules()):
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

    def fetch(self, tensor: Tensor, keys: int | Iterable[int] = None):
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

    @staticmethod
    def expand(tensor: Union[Tensor, None], target: Tensor, offset: int = None, keys=None, padding: int = 0):
        if tensor is not None:
            # print(tensor.shape)
            # tensor = self.fetch(tensor, keys)
            if not(tensor.shape[0] == 1 or tensor.shape[0] == target.shape[0]):
                raise ValueError(f"Tensors' key dims do not match; tensor={tensor.shape}, target={target.shape}.")
            extra   = target.ndim - tensor.ndim + padding
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
    def __init__(self, distribution: str = 'normal'):
        super().__init__()
        self.distribution: str = distribution
    
    def set_ditribution(self, distribution: str):
        self.distribution = distribution

    def dist(self, mean: Tensor, std: Union[Tensor, None], latent: Tensor = None, verbose: int = None):
        mean, std = mean.float(), std.float() if std is not None else None
        def fill_std(std_dev: Tensor):
            if std_dev is not None:
                return std_dev
            else:
                return torch.full_like(mean, 1e-12)

        extra = {}
        if self.distribution == 'discrete':
            if std is not None:
                logits = mean + (torch.randn_like(std) * std)
            else:
                logits = mean
            distribution = torch.distributions.Categorical(logits=torch.softmax(logits, -1))
        elif self.distribution == 'normal':
            distribution = torch.distributions.Normal(mean, fill_std(std))
        elif self.distribution == 'mult_var_normal':
            std = fill_std(std)
            if latent is not None:
                try:
                    # Create and cache the correlation layer and its lower-triangular indices, if needed.
                    if not hasattr(self, 'corr'):
                        embedding, features = latent.shape[-1], std.shape[-1]
                        tril_params_num = (features * (features - 1)) // 2
                        self.corr = torch.nn.Linear(embedding, tril_params_num, device=std.device, dtype=std.dtype,
                                                    bias=self.bias_enabled)
                        self.corr_indices = torch.tril_indices(features, features, offset=-1, device=std.device)

                    # Compute the lower-triangular correlation parameters and reduce them to (-1, 1)
                    corr_params = torch.tanh(self.corr(latent))

                    # Determine the batch shape from the correlation output (could be one or more batch dims)
                    features_out = std.shape[-1]  # number of features (must equal 'features')

                    # Create a full identity matrix of shape (F, F) and expand it to the batch shape.
                    identity = torch.eye(features_out, device=corr_params.device, dtype=corr_params.dtype)
                    # Clone after expansion to ensure a writable (contiguous) tensor.
                    corr_matrix = identity.expand(*std.shape[:-1], features_out, features_out).clone()
                    epsilon = corr_matrix * 1e-3

                    # Fill the lower-triangular part (excluding the diagonal) with the computed correlations
                    # Enforce symmetry: copy the lower triangle to the upper triangle.
                    # We add the transpose and subtract the duplicate diagonal..
                    corr_matrix[..., self.corr_indices[0], self.corr_indices[1]] = corr_params
                    corr_matrix[..., self.corr_indices[1], self.corr_indices[0]] = corr_params

                    # Build the covariance matrix: std_diag * corr_matrix * std_diag.
                    assert torch.all(std >= 0)
                    std_diag = torch.diag_embed(std)
                    cov_matrix = std_diag @ corr_matrix @ std_diag

                    if verbose and verbose >= 2:
                        extra['corr_params'] = corr_params
                        extra['corr_matrix'] = corr_matrix
                        extra['std_diag'] = std_diag

                    cov_matrix = cov_matrix + epsilon
                except Exception:
                    cov_matrix = torch.diag_embed(std ** 2)
            else:
                cov_matrix = torch.diag_embed(std ** 2)
            try:
                distribution = torch.distributions.MultivariateNormal(mean, cov_matrix)
            except ValueError as e:
                if verbose is None or verbose < 2:
                    raise e
                else:
                    distribution = None
        else:
            raise NotImplementedError(f"Unsupported distribution")
        if verbose and verbose >= 2:
            return distribution, extra
        else:
            return distribution

    def get_mean(self, latent: Tensor, keys: int | Iterable[int] = None) -> Tensor:
        raise NotImplementedError(f"No 'get_mean' method")

    def get_std(self, latent: Tensor, keys: int | Iterable[int] = None) -> Tensor | None:
        raise NotImplementedError(f"No 'get_std' method")

    def get_mean_std(self, latent: Tensor, keys: int | Iterable[int] = None) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError(f"No 'get_mean_std' method")

    def get_action(self, state: Tensor, keys: int | Iterable[int] = None) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(f"No 'get_action' method")

    def evaluate_action(self, state: Tensor, action: Tensor, keys: int | Iterable[int] = None) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError(f"No 'evaluate_action' method")

    def get_policy(self, state: Tensor, keys: int | Iterable[int] = None, **options) -> Tensor:
        raise NotImplementedError(f"No 'get_policy' method")

    def get_value(self, state: Tensor, keys: int | Iterable[int] = None) -> Tensor:
        raise NotImplementedError(f"No 'get_value' method")
    
    def extra_repr(self) -> str:
        return f"distribution='{self.distribution}'"


def check_for_illegal_zeros(config: Config, array: Union[CPUArray, GPUArray], param: NeatParameter, modules: Union[NeatModule, list[NeatModule], dict[Any, NeatModule]]):
    array_has_zeros = (np.any(array == 0.0) if isinstance(array, CPUArray) else cp.any(array == 0.0).get()).item()
    if config.genome.weight_del_prob == 0.0 and array_has_zeros:
        # NOTE: By this point the module_index of parameters should be set.
        error = f"Illegal zero value has been encountered when parameter epsilon or deletion has been enabled!"
        # Attempt to find modules it belongs to
        if isinstance(modules, NeatModule):
            modules = [modules]
        if isinstance(modules, dict):
            modules = list(modules.values())
        for module in modules:
            for m in module.neat_modules():
                if param.module_index is not None and param.module_index == m.module_index:
                    pn = m.get_attr_name(param)
                    param_name = f"'{pn}' " if pn is not None else ""
                    error += (f"\n\tModule {m.__class__.__name__} (module_index={m.module_index}) "
                              f"with parameter {param_name}"
                              f"of shape {param.shape}")
        raise ValueError(error)
