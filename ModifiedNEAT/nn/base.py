
from ModifiedNEAT.nn.genome import Genome, INT
from ModifiedNEAT.config import Config
from ModifiedNEAT.util.storage import save, load

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
import traceback
import warnings


def addindent(text: str, spaces: int):
    s = text.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return text
    first = s.pop(0)
    s = [(spaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class NeatParameter(nn.Parameter):
    __indexer = count(0)

    def __new__(cls, 
                shape: int | Iterable[int],
                requires_grad: bool = False,
                device: torch.device = 'cpu',
                dtype: torch.dtype = torch.float32):
        
        if isinstance(shape, (int, float)):
            shape = [shape]
        
        # Tensor construction MUST happen here in __new__
        data = torch.randn(1, *shape, device=device, dtype=dtype)
        
        # Use _make_subclass to properly wrap data as this subclass type
        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        return instance

    def __init__(self,
                 shape: int | Iterable[int],
                 requires_grad: bool = False,
                 device: torch.device = 'cpu',
                 dtype: torch.dtype = torch.float32):
        
        # Tensor is already built — DO NOT call super().__init__(data=...) 
        super().__init__()  # object.__init__() with no args
        
        if isinstance(shape, (int, float)):
            shape = [shape]

        # Pure Python attribute setup only
        self.param_index: int = next(self.__indexer)
        self._module_index: int | None = None
        self.original_shape: tuple[int, ...] = tuple(shape)

        self.mapping: dict[int, int] = None
        self.genome_num: int = None
        
        # Debugging
        self.cd: CPUArray = None
        self.md: CPUArray = None
        
    @property
    def module_index(self):
        if self._module_index is None:
            raise ValueError(f"Module index has not been set. It is set every time Module.neat_parameters() is called.")
        else:
            return self._module_index

    def reset(self):
        self.data = torch.randn_like(self.data)
        self.requires_grad_(self.requires_grad)
        self.mapping = None
        self.genome_num = None
        self.cd = None
        self.md = None

    def _update_data(self, tensor: Tensor):
        remove_rg = self.requires_grad
        if remove_rg:
            self.requires_grad_(False)
        
        # Convert numpy to tensor if needed
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor).float()
        
        # Handle shape mismatches by broadcasting
        if tensor.shape != self.data.shape:
            # If tensor is single genome (shape [1, ...]) and self.data is multi-genome (shape [N, ...])
            # broadcast/replicate the tensor across all genomes
            if len(tensor.shape) > 0 and tensor.shape[0] == 1 and len(self.data.shape) > 0 and self.data.shape[0] > 1:
                # Replicate the single genome weights across all genomes
                replicated = tensor.expand(self.data.shape[0], *tensor.shape[1:]).clone()
                self.data[:] = replicated
            else:
                # Try direct assignment (will error if incompatible)
                self.data[:] = tensor
        else:
            self.data[:] = tensor
        
        if remove_rg:
            self.requires_grad_(True)

    def update(self, genomes: dict[int, Genome], params: Union[Tensor, GPUArray] = None, verify=False):
        if params is None and self.mapping is not None and not all([key in self.mapping for key in genomes.keys()]):
            raise ValueError(f"Updating genomes of a module without parameters!")
        if isinstance(params, dict):
            params = params[self.param_index] # -> Any Array or Tensor
        if isinstance(params, DeviceNDArray):
            params = params.copy_to_host() # -> NumpyArray
        if isinstance(params, Tensor): # Handle Tensors first to avoid copying data twice
            params = params.clone().to(self.device, self.dtype).requires_grad_(self.requires_grad)
        if isinstance(params, CPUArray):
            params = torch.tensor(params, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        elif isinstance(params, GPUArray):
            params = torch.from_dlpack(params).to(self.device, self.dtype).requires_grad_(self.requires_grad)

        self.genome_num = len(genomes)

        if self.mapping is None:
            # Expand the tensor on first Population initialization
            if params is None:
                self.data = self.data.expand(self.genome_num, *self.original_shape)
                # size = self.data[0].numel()
                # self.data[:] = (torch.arange(size).view(self.data.shape[1:]) / size)
            # Load data from state dictionary
            else:
                d_shape = torch.tensor(self.original_shape)
                p_shape = torch.tensor(params.shape[-len(d_shape):])
                if not torch.all(d_shape == p_shape):
                    raise ValueError(f"Cannot initialize params of shape {p_shape.numpy()} "
                                     f"to data of shape {d_shape.numpy()}.")
                self.data = params.clone().to(self.device, self.dtype).requires_grad_(self.requires_grad)
        else:
            # Replace parameters with updated Population's parameters (ie. New genomes)
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
                self.data = params

        # TODO: Rearrange genomes according to fitness and gid

        self.mapping = {genome.key: index for index, genome in enumerate(genomes.values())}

        return self.mapping, self.genome_num

    def get_list(self, keys: int | Iterable[int] = None):
        if isinstance(keys, (int, float)):
            keys = [int(keys)]
        elif keys is None or len(keys) == 0:
            return list(range(self.data.shape[0]))
        try:
            return [self.mapping[i] for i in keys]
        except KeyError as e:
            raise KeyError(f"One or more genome keys passed that are not in Population")

    def get(self, keys: int | Iterable[int] = None):
        if isinstance(keys, (int, float)):
            keys = [int(keys)]
        elif keys is None or len(keys) == 0:
            return self.data
        try:
            return self.data[[self.mapping[i] for i in keys]]
        except KeyError as e:
            raise KeyError(f"One or more genome keys passed that are not in Population; {e}")

    # TODO: Might raise issues with default get_item
    def __getitem__(self, keys: int | Iterable[int] = None):
        return self.get(keys)

    def __repr__(self):
        return f"NeatParameter(g={self.genome_num}, s={self.original_shape})"

    # def __eq__(self, other): # TODO: Work of equalities for the sake of some default torch ops
    #     if isinstance(other, NeatParameter):
    #         return self.param_index == other.param_index
    #     return False

    # def __hash__(self):
    #     return hash(self.param_index)


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

    def neat_parameters(self, recurse=True):
        params: list[NeatParameter] = []

        def get(obj):
            if isinstance(obj, NeatParameter):
                params.append(obj)
            elif isinstance(obj, (NeatModule, Model)) and recurse:
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

    def neat_modules(self, recurse=True):
        modules: list[NeatModule] = []

        def get(obj):
            if isinstance(obj, (NeatModule, Model)):
                modules.append(obj)
                if recurse:
                    modules.extend(obj.neat_modules(recurse))

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

    def get_attr_name(self, value):
        for name, val in vars(self).items():
            if val is value:
                return name
        return None

    def _get_parameter_mapping(self):
        # Build state dict - extract all neat parameters by name
        state_dict: dict[str, CPUArray] = dict()
        param_dict: dict[str, NeatParameter] = dict()
        param_name_map: dict[int, str] = dict()
        
        # Get all registered parameters with their names
        for name, param in self.named_parameters(recurse=True):
            if isinstance(param, NeatParameter):
                param_name_map[id(param)] = name
        
        # Get any NeatParameters that weren't registered
        for param in self.neat_parameters(recurse=True):
            if isinstance(param, NeatParameter) and id(param) not in param_name_map:
                name = self.get_attr_name(param)
                if name is None:
                    # TODO: The use of param_index might raise retrieval issues or just general indexing issues
                    name = f'param_{param.param_index}' 
                param_name_map[id(param)] = name
        
        # Save all parameters using their names
        for param in self.neat_parameters(recurse=True):
            if isinstance(param, NeatParameter):
                name = param_name_map.get(id(param), f'param_{param.param_index}')
                
                param_dict[name] = param
                # Store as numpy array for compatibility
                if isinstance(param, torch.Tensor):
                    param = param.detach().cpu().numpy()
                if isinstance(param, GPUArray):
                    param = param.get()
                state_dict[name] = param
        
        def sort_key(item: tuple[str, Any]):
            name = item[0]
            return param_dict[name].param_index
        
        state_dict = dict(sorted(state_dict.items(), key=sort_key))
        param_dict = dict(sorted(param_dict.items(), key=sort_key))
        return state_dict, param_dict

    def _get_params_metadata(self):
        """Build metadata for all NeatParameters in this module.
        
        Returns:
            Dict mapping parameter names to metadata dicts containing:
            - 'shape': original shape tuple
            - 'genomes_total': number of genomes
            - 'param_index': unique parameter index
            - 'requires_grad': whether parameter requires gradients
            - 'dtype': torch dtype string
            - 'device': device string
        """
        metadata: dict[str, dict[str, Any]] = dict()
        seen_ids: set[int] = set()
        
        # Get direct registered Parameters
        for name, param in self.named_parameters(recurse=False):
            if isinstance(param, NeatParameter):
                metadata[name] = {
                    'registered': True,
                    'shape': param.original_shape,
                    'genome_total': param.genome_num,
                    'param_index': param.param_index,
                    'requires_grad': param.requires_grad,
                    'dtype': str(param.dtype),
                    'device': str(param.device),
                }
                seen_ids.add(id(param))
        
        # Get any other unregistered NeatParameters
        for param in self.neat_parameters(recurse=False):
            if isinstance(param, NeatParameter) and id(param) not in seen_ids:
                # Try to find its name
                name = self.get_attr_name(param)
                if name is None:
                    name = f'param_{param.param_index}'
                
                if name not in metadata:
                    metadata[name] = {
                        'registered': False,
                        'shape': param.original_shape,
                        'genome_total': param.genome_num,
                        'param_index': param.param_index,
                        'requires_grad': param.requires_grad,
                        'dtype': str(param.dtype),
                        'device': str(param.device),
                    }
                seen_ids.add(id(param))
        
        return metadata
    
    def _verify_params_metadata(self, metadata: dict[str, dict[str, Any]], strict: bool = False):
        if len(metadata) == 0:
            return True
        
        registered = [(n, d) for n, d in metadata.items() if d.get('registered', False)]
        unregistered = [(n, d) for n, d in metadata.items() if not d.get('registered', True)]
        registered_ids: set[int] = set()
        
        # Check registered Parameters
        count = 0
        for idx, (name, param) in enumerate(self.named_parameters(recurse=False)):
            if isinstance(param, NeatParameter):
                count += 1
                
                if strict and name not in metadata:
                    warnings.warn(
                        f"A NeatParameter's attribute name does not exist or has been modified",
                        category=RuntimeWarning,
                    )
                    return False
                
                param_data = metadata[name] if name in metadata else registered[idx]
                new_shape = param_data.get('shape')
                if new_shape is None or new_shape != param.original_shape:
                    warnings.warn(
                        f"Mismatch in NeatParameter metadata being set; "
                        f"Attempting to replace param of shape {param.original_shape} with {new_shape}",
                        category=RuntimeWarning,
                    )
                    return False
                
                registered_ids.add(id(param))
        if count != len(registered):
            old_params = [p.original_shape for _, p in self.named_parameters(recurse=False) if isinstance(p, NeatParameter)]
            new_params = [d.get('original_shape') for (_, d) in registered]
            warnings.warn(
                f"Mismatch in NeatParameter metadata being set. \nExpected registered parameters ({len(new_params)}): {new_params} "
                f"\nBut module currently has ({len(old_params)}): {old_params} \n",
                category=RuntimeWarning,
            )
            return False 
        
        # Check any other unregistered NeatParameters
        count = 0
        for idx, param in enumerate(self.neat_parameters(recurse=False)):
            if isinstance(param, NeatParameter) and id(param) not in registered_ids:
                count += 1
                name = self.get_attr_name(param)
                if name is None:
                    name = f'param_{param.param_index}'
                    
                if strict and name not in metadata:
                    warnings.warn(
                        f"A NeatParameter's attribute name does not exist or has been modified",
                        category=RuntimeWarning,
                    )
                    return False
                
                param_data = metadata[name] if name in metadata else unregistered[idx]
                new_shape = param_data.get('original_shape')
                if new_shape is None or new_shape != param.original_shape:
                    warnings.warn(
                        f"Mismatch in NeatParameter metadata being set; "
                        f"Attempting to replace param of shape {param.original_shape} with {new_shape}",
                        category=RuntimeWarning,
                    )
                    return False
        if count != len(unregistered):
            old_params = [p.original_shape for _, p in self.named_parameters(recurse=False) if isinstance(p, NeatParameter)]
            new_params = [d.get('original_shape') for (_, d) in unregistered]
            warnings.warn(
                f"Mismatch in NeatParameter metadata being set. \nExpected unregistered parameters ({len(new_params)}): {new_params} "
                f"\nBut module currently has ({len(old_params)}): {old_params} \n",
                category=RuntimeWarning,
            )
            return False
        
        return True

    def _get_all_params_with_module_path(self, module_path: str = "root") -> dict[str, str]:
        """Recursively collect all parameters with their owning module path.
        
        Args:
            module_path: current module's path in the hierarchy (default "root")
            
        Returns:
            Dict mapping parameter name -> owning module path
            Example: {
                "lat_proj.weight": "root.lat_proj",
                "pol_proj.weight": "root.pol_proj",
                "nested.child.param": "root.nested.child"
            }
        """
        param_module_map: dict[str, str] = {}
        
        # Get direct parameters of this module
        seen_ids: set[int] = set()
        for name, param in self.named_parameters(recurse=False):
            if isinstance(param, NeatParameter):
                param_module_map[name] = module_path
                seen_ids.add(id(param))
        
        # Add unregistered NeatParameters from this module
        for param in self.neat_parameters(recurse=False):
            if isinstance(param, NeatParameter) and id(param) not in seen_ids:
                name = self.get_attr_name(param)
                if name is None:
                    name = f'param_{param.param_index}'
                param_module_map[name] = module_path
                seen_ids.add(id(param))
        
        # Recursively process child modules
        for child_name, module in self.named_children():
            if isinstance(module, NeatModule):
                child_path = f"{module_path}.{child_name}"
                child_params = module._get_all_params_with_module_path(child_path)
                param_module_map.update(child_params)
        
        return param_module_map

    def _get_modules_metadata(self, recurse: bool = True, index: int = 0) -> dict[str, Any]:
        """Build tree structure of nested NeatModules with parameter tracking.
        
        Returns:
            Dict with:
            - 'class_name': name of this module's class
            - 'module_index': unique module index
            - 'child_name': name of this module (in parent context)
            - 'children': list of child module metadata dicts
            - 'params': list of parameter names owned by this module (direct only)
        """
        return _get_module_metadata(self)

    def _verify_modules_metadata(self, metadata: dict[str, Any], strict: bool = False, recurse: bool = True):
        if metadata is None or len(metadata) == 0:
            return True
        
        name = self.__class__.__name__
        if strict and name != metadata['class_name']:
            warnings.warn(
                f"A NeatModule's class name does not match that in the metadata",
                category=RuntimeWarning,
            )
            return False
    
        children = metadata['children']
        if recurse and len(children) > 0:
            count = 0
            for name, module in self.named_children():
                if isinstance(module, NeatModule):
                    count += 1
                    if strict and name not in children:
                        warnings.warn(
                            f"A NeatModule '{self.__class__.__name__} is missing a child module '{name}' "
                            f"or its attribute name has been modified",
                            category=RuntimeWarning,
                        )
                        return False
            if count != len(children):
                warnings.warn(
                    f"Mismatch in NeatModule metadata being set. "
                    f"Expected {len(children)} child modules but current module requires {count} ",
                    category=RuntimeWarning,
                )
                return False
        
        return metadata

    def neat_dict(self):
        """Build serialization dict with state, architecture, and metadata.
        
        Returns a comprehensive dict suitable for pickling that includes:
        - 'version': format version number
        - 'state_dict': tensor/parameter values (like nn.Module.state_dict())
        - 'architecture': module structure and parameter metadata
        - 'metadata': genomes_total,  mapping, and other runtime state
        - 'param_module_map': mapping of parameter name -> owning module path
        
        Format:
        {
            'version': 1,
            'state_dict': {param_name: tensor_or_array},
            'architecture': {
                'neat_params': {name: metadata_dict},
                'neat_modules': module_tree_dict,
            },
            'param_module_map': {param_name: module_path},
            'metadata': {
                'genomes_total': int,
                'mapping': {genome_key: index},
                'params': self.params,  # Constructor params
            }
        }
        """
        # Build state dict - extract all neat parameters by name
        state_dict, _ = self._get_parameter_mapping()
        
        # Build architecture info
        architecture = { # TODO: Convert to metadata if param metadata is depracated
            'neat_params': self._get_params_metadata(), # TODO: Might be useless with new module_metadata fetch method
            'neat_modules': self._get_modules_metadata(recurse=True),
        }
        
        # Build parameter-to-module mapping for frontend filtering
        param_module_map = self._get_all_params_with_module_path(module_path="root") # TODO: Might be useless with new module_metadata fetch method
        
        # Build population
        population = {
            'genomes_total': self.genome_num,
            'mapping': self.mapping,
            'params': self.params, # TODO: Implement in PseudoModule or depracate
            'extra_repr': self.extra_repr(), # TODO: Implement in PseudoModule or depracate
            'genus': self.genus,
        }
        
        return {
            'version': 1,
            'state_dict': state_dict,
            'architecture': architecture,
            'param_module_map': param_module_map,
            'population': population,
        }

    def load_neat_dict(
        self,
        state: dict[str, Any],
        strict: bool = True,
        verbose: bool | int = False
    ):
        """Load state dict and restore module with flexible matching.
        
        Args:
            serialized_dict: Output from state_dict_with_metadata()
            module_instance: Existing module to populate (required if strict=True)
            strict: If True, require module_instance and validate parameter names/counts.
                   If False, load by position/shape only and auto-instantiate if needed.
            debug: If True, print debug information
        
        Returns:
            Populated NeatModule instance
        
        Raises:
            ValueError: If strict=True and requirements not met, or if version mismatch
        """
        version = state.get('version', 1)
        if version != 1:
            raise ValueError(f"Unsupported serialization format version: {version}")
        
        state_dict: dict[str, CPUArray] = state.get('state_dict', {})
        architecture = state.get('architecture', {})
        population = state.get('population', {})
        param_meta = architecture.get('neat_params', {})
        module_meta = architecture.get('neat_modules', {})
        
        self._verify_params_metadata(param_meta, strict) # TODO: Might want to chagne the RuntimeWarning to RuntimeError
        self._verify_modules_metadata(module_meta, strict, recurse=True)
        
        # Set the weights
        with torch.no_grad():
            old_state_dict, old_param_dict = self._get_parameter_mapping()
            counts_match = len(old_state_dict) == len(state_dict)
            if not counts_match:
                raise RuntimeError(
                    f"Current number of parameters does not match that in the state_dict; "
                    f"{len(old_state_dict)} != {len(state_dict)}"
                )
            shapes_match = all([nt.shape[1:] == ot.shape[1:] for (nt, ot) in zip(state_dict.values(), old_state_dict.values())])
            if not shapes_match:
                raise RuntimeError(
                    f"Current order and shape of parameters does not match of the state_dict"
                )
            for (nn, new_array), (on, param) in zip(state_dict.items(), old_param_dict.items()):
                if strict and nn != on:
                    raise RuntimeError(
                        f"Mismatch in parameter names or order of parameters in loading of NeatModule; {nn} != {on}"
                    )
                param.data = torch.tensor(
                    new_array, 
                    device=param.device,
                    dtype=param.dtype,
                    requires_grad=param.requires_grad
                )
        
        
        # # Strict mode: require module_instance
        # if strict:
        #     if module_instance is None:
        #         raise ValueError(
        #             "strict=True requires module_instance parameter. "
        #             "Pass an initialized NeatModule instance to populate."
        #         )
        #     module = module_instance
            
        #     # Build a map of parameter names to parameters
        #     param_by_name = {}
        #     param_ids_mapped = set()
            
        #     for name, param in module.named_parameters(recurse=True):
        #         if isinstance(param, NeatParameter):
        #             param_by_name[name] = param
        #             param_ids_mapped.add(id(param))
            
        #     # For any neat_parameters not in named_parameters, try to find by get_attr_name
        #     for param in module.neat_parameters(recurse=True):
        #         if isinstance(param, NeatParameter) and id(param) not in param_ids_mapped:
        #             param_name = module.get_attr_name(param)
        #             if param_name:
        #                 param_by_name[param_name] = param
        #                 param_ids_mapped.add(id(param))
            
        #     # Validate and restore parameters by name
        #     param_meta = architecture.get('neat_params', {})
        #     # if isinstance(cls, NeatModule):
        #     print(param_meta)
        #     self._verify_params_metadata(param_meta, strict)
        #     print(f"\nCompleted parameter verification\n")
        #     for name, meta in param_meta.items():
        #         if name in param_by_name:
        #             param = param_by_name[name]
                    
        #             # Validate shape matches
        #             if param.original_shape != tuple(meta['shape']):
        #                 raise ValueError(
        #                     f"Parameter '{name}' shape mismatch: "
        #                     f"expected {param.original_shape}, got {tuple(meta['shape'])}"
        #                 )
                    
        #             # Restore tensor data by parameter name
        #             if name in state_dict:
        #                 value = state_dict[name]
        #                 if isinstance(value, np.ndarray):
        #                     value = torch.from_numpy(value).float()
        #                 param._update_data(value)
        #             elif debug:
        #                 print(f"Warning: Parameter '{name}' not found in state_dict")
        #         elif debug:
        #             print(f"Warning: Parameter '{name}' not found in module")
        # else:
        #     # Non-strict mode: reconstruct module from architecture if needed
        #     if module_instance is None:
        #         # Try to instantiate from class_name
        #         class_name = architecture.get('neat_modules', {}).get('class_name', 'NeatModule')
        #         constructor_params = population.get('params', {})
                
        #         try:
        #             if class_name == 'NeatModule':
        #                 module = self(**constructor_params)
        #             else:
        #                 # Try to find and import the class
        #                 module = self(**constructor_params)
        #                 if debug:
        #                     print(f"Warning: Non-strict load created {self.__name__} (expected {class_name})")
        #         except Exception as e:
        #             if debug:
        #                 print(f"Warning: Could not instantiate {class_name}, using {self.__name__}: {e}")
        #             module = self(**constructor_params)
        #     else:
        #         module = module_instance
            
        #     # Load by position/shape (ignore names) - match state_dict values to parameters by order
        #     param_list = module.neat_parameters(recurse=True)
        #     param_list = [p for p in param_list if isinstance(p, NeatParameter)]
            
        #     # Get state_dict values and metadata in the same order
        #     param_meta = architecture.get('neat_params', {})
        #     state_values = []
        #     param_shapes = []
        #     for name in param_meta.keys():
        #         if name in state_dict:
        #             state_values.append(state_dict[name])
        #             param_shapes.append(tuple(param_meta[name]['shape']))
            
        #     # If module has no parameters, create them from architecture
        #     if len(param_list) == 0 and len(state_values) > 0:
        #         if debug:
        #             print(f"Creating {len(state_values)} parameters on module from architecture")
                
        #         for i, (state_value, shape) in enumerate(zip(state_values, param_shapes)):
        #             # Create a NeatParameter with the correct shape
        #             param = NeatParameter(shape)
        #             # Restore the data
        #             if isinstance(state_value, np.ndarray):
        #                 state_value = torch.from_numpy(state_value).float()
        #             param._update_data(state_value)
                    
        #             # Attach to module with auto-generated name
        #             param_name = f'param_{i}'
        #             setattr(module, param_name, param)
        #     else:
        #         # Load values by position into existing parameters
        #         for i, param in enumerate(param_list):
        #             if i < len(state_values):
        #                 value = state_values[i]
        #                 if isinstance(value, np.ndarray):
        #                     value = torch.from_numpy(value).float()
        #                 param._update_data(value)
        
        # Restore metadata
        self.genome_num = population.get('genomes_total')
        self.mapping    = population.get('mapping')
        
        if verbose:
            print(f"Loaded {self.__class__.__name__} with genome_num={self.genome_num}")
        
        return self

    def save(self, filename: str = None, directory: str = None, file_no: int = None, 
             replace: bool = False, debug: bool = True) -> tuple[bool, int]:
        """Save NeatModule state and architecture to .module.pkl file.
        
        Instead of pickling the full object, saves state_dict, architecture metadata,
        and parameter information. This enables reliable loading without class resolution issues.
        
        Args:
            filename: Base filename (without .module.pkl extension)
            directory: Directory to save to
            file_no: Specific file number (auto-finds latest if None)
            replace: If True, replace existing file instead of creating new numbered version
            debug: If True, print debug messages
        
        Returns:
            Tuple of (success: bool, file_no: int)
        """
        try:
            serialized = self.neat_dict()
            return save(
                items=serialized,
                items_name=f"NeatModule-{self.__class__.__name__}",
                extension='.module.pkl',
                filename=filename,
                directory=directory,
                file_no=file_no,
                replace=replace,
                debug=debug
            )
        except Exception as e:
            if debug:
                print(f"Error saving NeatModule: {e}\n{traceback.format_exc()}")
            return False, file_no

    def load(self, filename: str = None, directory: str = None, file_no: int = None, 
             strict: bool = True, debug: bool = True) -> 'NeatModule':
        """Load a NeatModule from a .module.pkl file.
        
        Loads state_dict and architecture metadata, reconstructing the module with
        all parameters restored. Supports both strict and flexible loading modes.
        
        Args:
            filename: Base filename (without .module.pkl extension)
            directory: Directory to load from
            file_no: Specific file number to load (auto-finds latest if None)
            strict: If True, require exact module instance; if False, reconstruct from architecture
            debug: If True, print debug messages
        
        Returns:
            Loaded NeatModule object with all parameters restored
        
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If format is invalid or strict requirements not met
        """
        try:
            # Load the serialized dict
            data = load(
                items_name=f"NeatModule-{self.__class__.__name__}",
                extension='.module.pkl',
                filename=filename,
                directory=directory,
                file_no=file_no,
                debug=debug
            )
            
            if data is None:
                raise FileNotFoundError(
                    f"Could not load file: {filename} from {directory}"
                )
            
            # Extract the serialized dict if wrapped
            if isinstance(data, dict) and 'items' in data:
                serialized_dict = data['items']
            else:
                serialized_dict = data
                
            return self.load_neat_dict(
                serialized_dict,
                strict=strict,
                verbose=debug
            )
        except Exception as e:
            if debug:
                print(f"Error loading NeatModule: {e}\n{traceback.format_exc()}")
            raise 

    def __eq__(self, other):
        # if isinstance(other, NeatModule):
        #     return self.module_index == other.module_index
        return super().__eq__(other)

    def __hash__(self):
        # # TODO: Find a better way to hash this class
        # return hash(self.module_index)
        return super().__hash__()


def _get_module_metadata(module: NeatModule | nn.Module, index: int = 0):
    """Build tree structure of nested NeatModules with parameter tracking.
    
    Returns:
        Dict with:
        - 'class_name': name of this module's class
        - 'module_index': unique module index
        - 'child_name': name of this module (in parent context)
        - 'children': list of child module metadata dicts
        - 'params': list of parameter names owned by this module (direct only)
    """
    is_base = index == 0
    is_neat_module = isinstance(module, NeatModule)

    metadata = {
        'class_name': module.__class__.__name__,
        'module_index': index,
        'children': [],
        'params': [],  # Direct parameters of this module
        'neat_module': is_neat_module,
    }
    
    # Collect direct parameters (not from children)
    seen_ids: set[int] = set()
    for name, param in module.named_parameters(recurse=False):
        if isinstance(param, NeatParameter):
            metadata['params'].append(name)
            seen_ids.add(id(param))
    
    # Add any unregistered NeatParameters from this module
    all_parameters = module.neat_parameters if is_neat_module else module.parameters
    for param in all_parameters(recurse=False):
        if isinstance(param, NeatParameter) and id(param) not in seen_ids:
            name = module.get_attr_name(param)
            if name is None:
                name = f'param_{param.param_index}'
            metadata['params'].append(name)
            seen_ids.add(id(param))
    
    # Get child modules
    for name, module in module.named_children():
        if isinstance(module, nn.Module):
            index += 1
        else:
            raise ValueError(f"Unsupported module type '{type(module)}'")
        child_metadata, index = _get_module_metadata(module, index=index)
        child_metadata['child_name'] = name
        metadata['children'].append(child_metadata)
    
    if is_base:
        return metadata
    else:
        return metadata, index

class PseudoModule(nn.Module): # | NeatModule
    """
Recreates a NeatModule and all its child modules from a NeatModule's metadata.\n
Should be able to recreate all ModifiedNEAT.nn.NeatModule and torch.nn.Module that
are directly or nested within the current module.
    """
 
    # Cache the two variants so we don't recreate them on every instantiation
    _neat_cls: type = None
    _plain_cls: type = None
    
    def __new__(cls, metadata, state_dict, population):
        is_neat = metadata['neat_module']
        if is_neat:
            if PseudoModule._neat_cls is None:
                # PseudoModule first so its __init__ wins in the MRO over NeatModule.__init__;
                # NeatModule second to make isinstance(x, NeatModule) return True.
                PseudoModule._neat_cls = type('PseudoModule', (PseudoModule, NeatModule), {})
            return object.__new__(PseudoModule._neat_cls)
        # plain nn.Module path — PseudoModule itself already inherits nn.Module
        return object.__new__(PseudoModule)
    
    # TODO: All variable names and types are subject to change
    def __init__(self, metadata: dict[str, Any], state_dict: dict[str, ], population: dict[str, Any]):
        # target_cls = NeatModule if metadata['neat_module'] else nn.Module
        nn.Module.__init__(self)
        
        # metadata essentials: {class_name, *population_info, *neat_parameters, ...,
        # *children: {class_name, *population_info (would prob be a redundancy), ..., *children}}
        
        # Class info
        self.params: dict[str, Any] = {}
        self.name: str = metadata['class_name'] # Don't know if you can change a __class__.__name__
        self._attr_name: str | None = metadata.get('child_name', None)
        self._param_ext: str | None = metadata.get('param_ext', None)
        self.module_index: int = metadata['module_index']
        self.is_neat_module: bool = metadata['neat_module']

        self.device: torch.device = torch.device('cpu')
        self.dtype: torch.dtype = torch.float32
                
        # Population info
        self.genome_num: int | None = population['genomes_total']
        self.mapping: dict[int, int] = population['mapping']
        self.genus: int | None = population['genus']
        
        # NeatParameters and NeatModules/torch.nn.Modules
        self._pseudo_children: dict[str, 'PseudoModule'] = {}
        self.build(metadata, state_dict, population, self._param_ext)
        
    @property
    def class_name(self): # Alternative to using __class__.__name__
        return self.name
    
    def build(
        self, metadata: dict, states: dict[str, CPUArray], 
        population: dict[str, Any], param_ext: str | None = None
    ):
        """
        metadata: {
            class_name: str,
            module_index: int,
            params: list[str],
            children: list[{
                child_name: str,
                *child_metadata
            }]
        }
        """
        # If the module has any parameters
        for param_name in metadata.get('params', []):
            _param_name = param_name if param_ext is None else f"{param_ext}{param_name}"
            # print(f"Getting '{param_name}' for {self.class_name} with '{_param_name}'")
            
            param_data = states[_param_name]
            param = NeatParameter(param_data.shape[1:], False, self.device, self.dtype)
            param.data = torch.tensor(param_data, device=self.device, dtype=self.dtype, requires_grad=False)
            
            setattr(self, param_name, param)
            
        # If module has any child modules
        for child in metadata.get('children', []):
            child_name = child['child_name']
            ext = f"{param_ext if param_ext is not None else ''}{child_name}."
            child['param_ext'] = ext
            
            module = PseudoModule(child, states, population)
            # print(f"Child '{child['class_name']}': NeatModule={isinstance(module, NeatModule)}, "
            #       f"TorchModule={isinstance(module, nn.Module)}")
            
            setattr(self, child_name, module)
            self._pseudo_children[child_name] = module
            
    def extra_repr(self) -> str:
        return ''

    def __repr__(self) -> str:
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
            
        child_lines = []
        for key, module in self._pseudo_children.items():
            mod_str = repr(module)
            mod_str = addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = f"{self.class_name}{'[NeatModule]' if self.is_neat_module else ''}" + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


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
            # if std is None:
            logits = mean
            # else:
            #     logits = mean + (torch.randn_like(std) * std)
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

    def _get_mean(self, latent: Tensor, keys: int | Iterable[int] = None) -> Tensor:
        raise NotImplementedError(f"No '_get_mean' method")

    def _get_std(self, latent: Tensor, keys: int | Iterable[int] = None) -> Tensor | None:
        raise NotImplementedError(f"No '_get_std' method")

    def _get_mean_std(self, latent: Tensor, keys: int | Iterable[int] = None) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError(f"No 'get_mean_std' method")

    # def get_mean(self, state: Tensor, keys: int | Iterable[int] = None) -> Tensor:
    #     raise NotImplementedError(f"No 'get_mean' method")

    # def get_std(self, state: Tensor, keys: int | Iterable[int] = None) -> Tensor | None:
    #     raise NotImplementedError(f"No 'get_std' method")

    # def get_mean_std(self, state: Tensor, keys: int | Iterable[int] = None) -> tuple[Tensor, Tensor | None]:
    #     raise NotImplementedError(f"No 'get_mean_std' method")

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
