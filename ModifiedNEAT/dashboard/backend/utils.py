"""Utility functions for module and tensor processing."""

from ModifiedNEAT.nn.base import NeatModule, NeatParameter, Model, PseudoModule
from ModifiedNEAT.nn.modules.base import *
from ModifiedNEAT.nn.genome import Genome
from ModifiedNEAT.population import Population
from ModifiedNEAT.util.storage import *
from torch import Tensor
from torch.nn import Parameter, Module as TorchModule
from enum import Enum
from numpy import ndarray as CPUArray
from typing import Optional, Union, Tuple, Any

import torch
import torch.nn as nn
import numpy as np
import logging

from .models import ModuleInfo, ModuleNode, ModuleEdge

logger = logging.getLogger(__name__.split(".")[-1]+".py")

Module = NeatModule | TorchModule | PseudoModule

class ModelType(Enum):
    TORCH = 0
    NEAT  = 1
    PSEUDO = 3


def get_module_type(module: Module) -> str:
    """Get the class name of a module."""
    return module.__class__.__name__


def is_neat_module(module: Module) -> bool:
    """Check if module is a NeatModule."""
    # return module.__class__.__name__ == 'NeatModule'
    return isinstance(module, NeatModule)


def count_parameters(module: Module) -> int:
    """Count total parameters in a module."""
    count = sum(p.numel() for p in module.parameters())
    return count


def has_weights(module: Module) -> bool:
    """Check if module has trainable weights or buffers."""
    for _ in module.parameters(recurse=False): return True
    # Use neat_paramters since it might not find the .data attribute that is Parameter
    if is_neat_module(module):
        for _ in module.neat_parameters(recurse=False): return True
    for _ in module.buffers(recurse=False): return True
    return False


# ---------------------------------------------------------------------------
# Module Reconstruction using PseudoModule from base.py
# ---------------------------------------------------------------------------

def reconstruct_module_from_metadata(
    neat_dict: dict[str, Any],
    module_name: str = "root"
) -> PseudoModule:
    """
    Reconstruct a module structure from neat_dict() output using PseudoModule.
    
    Args:
        neat_dict: Dictionary from NeatModule.neat_dict() containing:
            - 'state_dict': {param_name: numpy_array}
            - 'architecture': {'neat_params': {...}, 'neat_modules': {...}}
            - 'param_module_map': {param_name: module_path}  (optional)
            - 'population': {'genomes_total': int, 'mapping': dict, 'genus': int}
        module_name: Name for the root module (default: "root")
    
    Returns:
        PseudoModule instance that mimics the original module structure
    """
    logger.debug(f"reconstruct_module_from_metadata: Starting (module_name={module_name})")
    
    state_dict = neat_dict.get('state_dict', {})
    architecture = neat_dict.get('architecture', {})
    neat_modules_meta = architecture.get('neat_modules', {})
    population = neat_dict.get('population', {})
    
    logger.debug(f"  state_dict keys: {len(state_dict)} items")
    logger.debug(f"  architecture keys: {list(architecture.keys())}")
    logger.debug(f"  neat_modules_meta type: {type(neat_modules_meta).__name__}")
    if isinstance(neat_modules_meta, dict):
        logger.debug(f"  neat_modules_meta keys: {list(neat_modules_meta.keys())}")
    
    # Create PseudoModule using the new implementation from base.py
    # PseudoModule handles reconstruction of parameters and child modules
    try:
        pseudo_module = PseudoModule(neat_modules_meta, state_dict, population)
        logger.debug(f"reconstruct_module_from_metadata: Successfully created PseudoModule")
    except Exception as e:
        logger.error(f"reconstruct_module_from_metadata: Failed to create PseudoModule: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise
    
    return pseudo_module


def extract_module_tree(module: Module, prefix: str = "") -> list[ModuleInfo]:
    """Extract hierarchical module tree."""
    result = []
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        child_info = ModuleInfo(
            name=full_name,
            type=get_module_type(child),
            has_weights=has_weights(child),
            children=extract_module_tree(child, full_name),
        )
        result.append(child_info)
    return result


def build_graph(
    module: Module,
    prefix: Optional[str] = None,
    parent_id: Optional[str] = None,
    nodes: Optional[list[ModuleNode]] = None,
    edges: Optional[list[ModuleEdge]] = None,
    edge_counter: Optional[list[int]] = None,
    total_params: Optional[list[int]] = None,
    visited_ids: Optional[set] = None,
    state_dict: Optional[dict[str, Tensor | CPUArray]] = None,
) -> Tuple[list[ModuleNode], list[ModuleEdge], int]:
    """Build a hierarchical graph representation of the module structure.
    
    Works with both actual torch modules and PseudoModule instances reconstructed from metadata.
    Creates nested ModuleNode structure where child modules are stored in the parent.
    
    Args:
        module: The module to build the graph from
        prefix: Full module path prefix (e.g., 'root.child.grandchild')
        parent_id: Parent node ID
        state_dict: Optional state_dict for parameter information
        ... other args for recursion
    """
    if not prefix:
        prefix = "root"
    if nodes is None:
        nodes = []
    if edges is None:
        edges = []
    if edge_counter is None:
        edge_counter = [0]
    if total_params is None:
        total_params = [0]
    if visited_ids is None:
        visited_ids = set()

    module_id = prefix
    
    # Skip if we've already processed this node (handles shared modules)
    if module_id in visited_ids:
        if parent_id is not None:
            edges.append(
                ModuleEdge(id=f"edge_{edge_counter[0]}", source=parent_id, target=module_id)
            )
            edge_counter[0] += 1
        return nodes, edges, total_params[0]
    
    visited_ids.add(module_id)
    
    # Handle both real modules and PseudoModule instances
    if isinstance(module, PseudoModule):
        param_count = count_parameters(module)
        module_type = module.class_name
        has_module_weights = has_weights(module)
        is_neat = module.is_neat_module
        children_list = list(module.named_children())
    else:
        param_count = count_parameters(module)
        module_type = get_module_type(module)
        has_module_weights = has_weights(module)
        is_neat = is_neat_module(module)
        children_list = list(module.named_children())
    
    total_params[0] += param_count
    is_nested = len(children_list) > 0
    
    # Extract direct parameters (not from children)
    parameters = []
    if has_module_weights:
        for param_name, param_obj in module.named_parameters(recurse=False):
            extended_name = f"{prefix}.{param_name}"
            
            # Get shape and dtype from state_dict if available, otherwise from param_obj
            if state_dict and extended_name in state_dict:
                tensor_info = state_dict[extended_name]
                if hasattr(tensor_info, 'shape'):
                    shape = list(tensor_info.shape)
                    dtype = str(tensor_info.dtype)
                else:
                    shape = list(tensor_info.shape) if hasattr(tensor_info, 'shape') else []
                    dtype = 'unknown'
            else:
                # Fallback to param_obj
                if hasattr(param_obj, 'shape'):
                    shape = list(param_obj.shape)
                    dtype = str(param_obj.dtype)
                else:
                    shape = []
                    dtype = 'unknown'
            
            parameters.append({
                'name': param_name,
                'extended_name': extended_name,
                'shape': shape,
                'dtype': dtype,
            })
    
    # Create the node
    node = ModuleNode(
        id=module_id,
        name=prefix.split(".")[-1] if prefix else "root",
        type=module_type,
        has_weights=has_module_weights,
        param_count=param_count,
        is_neat_module=is_neat,
        is_nested=is_nested,
        parent_id=parent_id,
        parameters=parameters if parameters else None,
        full_module_path=prefix,
    )

    # Add edge from parent if exists
    if parent_id is not None:
        edges.append(
            ModuleEdge(id=f"edge_{edge_counter[0]}", source=parent_id, target=module_id)
        )
        edge_counter[0] += 1

    # Process children
    for name, child in children_list:
        child_id = f"{module_id}.{name}" if module_id != "root" else name
        build_graph(
            child, child_id, module_id, nodes, edges, edge_counter, total_params, visited_ids, state_dict
        )
        child_nodes = [n for n in nodes if n.id == child_id]
        if child_nodes:
            node.children.append(child_nodes[0])

    nodes.append(node)
    return nodes, edges, total_params[0]


def build_graph_from_metadata(
    neat_dict: dict[str, Any],
    module_name: str = "root",
    module_prefix: str = ""
) -> Tuple[list[ModuleNode], list[ModuleEdge], int]:
    """Build module graph directly from neat_dict metadata.
    
    This avoids unpickling and reconstructs the graph purely from metadata.
    
    Args:
        neat_dict: Dictionary from NeatModule.neat_dict()
        module_name: Name for root module (default: "root")
        module_prefix: Prefix to add to all node IDs for multi-module cases
    
    Returns:
        Tuple of (nodes, edges, total_params)
    """
    logger.debug(f"build_graph_from_metadata: Starting (prefix={module_prefix})")
    logger.debug(f"  neat_dict type: {type(neat_dict).__name__}")
    if isinstance(neat_dict, dict):
        logger.debug(f"  neat_dict keys: {list(neat_dict.keys())}")
    
    # Reconstruct module structure from metadata
    try:
        metadata_module = reconstruct_module_from_metadata(neat_dict, module_name)
        logger.debug(f"build_graph_from_metadata: PseudoModule reconstructed")
    except Exception as e:
        logger.error(f"build_graph_from_metadata: Failed to reconstruct: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise
    
    # Build graph from reconstructed module with the specified prefix
    try:
        state_dict = neat_dict.get('state_dict', {})
        nodes, edges, total_params = build_graph(metadata_module, prefix=module_prefix, state_dict=state_dict)
        logger.debug(f"build_graph_from_metadata: Graph built: {len(nodes)} nodes, {len(edges)} edges, {total_params} params")
    except Exception as e:
        logger.error(f"build_graph_from_metadata: Failed to build graph: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise
    
    return nodes, edges, total_params


def parse_slice_spec(slice_spec: str, shape: list[int]) -> Tuple:
    """Parse a slice specification string like '[:100, 10:20, 0]' into slice objects."""
    if not slice_spec:
        return tuple(slice(None) for _ in shape)

    # Remove brackets if present
    slice_spec = slice_spec.strip().strip("[]()")
    parts = [p.strip() for p in slice_spec.split(",")]

    slices = []
    for i, part in enumerate(parts):
        if i >= len(shape):
            break

        if part == ":" or part == "":
            slices.append(slice(None))
        elif ":" in part:
            # Range like "0:100" or ":100" or "10:"
            range_parts = part.split(":")
            if len(range_parts) == 2:
                start = int(range_parts[0]) if range_parts[0] else None
                end = int(range_parts[1]) if range_parts[1] else None
                slices.append(slice(start, end))
            else:
                raise ValueError(f"Invalid slice specification: {part}")
        else:
            # Single index
            try:
                slices.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid slice specification: {part}")

    # Pad with full slices if needed
    while len(slices) < len(shape):
        slices.append(slice(None))

    return tuple(slices)


def numpy_to_torch(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    return torch.from_numpy(arr).float()


def get_tensor_slice(
    tensor: Union[torch.Tensor, np.ndarray],
    slice_spec: Optional[str], 
    max_display: Optional[int] = None
) -> Tuple[np.ndarray, bool]:
    """Get a slice of the tensor and return as numpy array.
    
    Works with both torch tensors and numpy arrays.
    """
    # Convert numpy to torch if needed for slicing
    if isinstance(tensor, np.ndarray):
        tensor = numpy_to_torch(tensor)
    
    shape = list(tensor.shape)

    # Build slice tuple
    slices = parse_slice_spec(slice_spec, shape) if slice_spec else tuple(slice(None) for _ in shape)

    # Apply slice
    try:
        result = tensor[slices]
    except Exception as e:
        raise ValueError(f"Invalid slice: {str(e)}")

    # Convert to numpy
    arr = result.detach().cpu().numpy()

    # If more than 2 dims, collapse to 2D
    while arr.ndim > 2:
        arr = arr[0]

    # If 0D (scalar), keep as is
    if arr.ndim == 0:
        return arr, False

    # If 1D, reshape to 2D for display
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    reached_limit = False
    if max_display is not None:
        if arr.shape[0] > max_display:
            arr = arr[:max_display]
            reached_limit = True
        if arr.shape[1] > max_display:
            arr = arr[:, :max_display]
            reached_limit = True

    return arr, reached_limit


def compute_tensor_stats(tensor: Union[torch.Tensor, np.ndarray]) -> dict:
    """Compute statistics for a tensor or numpy array."""
    if isinstance(tensor, np.ndarray):
        tensor = numpy_to_torch(tensor)
    
    flat = tensor.detach().flatten()
    has_nan = bool(torch.isnan(flat).any().item())
    has_inf = bool(torch.isinf(flat).any().item())

    if has_nan or has_inf:
        # Filter out NaN and Inf for statistics
        valid = flat[torch.isfinite(flat)]
        if valid.numel() == 0:
            return {
                "has_nan": has_nan,
                "has_inf": has_inf,
                "min_val": None,
                "max_val": None,
                "mean_val": None,
                "std_val": None,
                "sparsity": None,
            }
        flat = valid

    return {
        "has_nan": has_nan,
        "has_inf": has_inf,
        "min_val": float(flat.min().item()),
        "max_val": float(flat.max().item()),
        "mean_val": float(flat.mean().item()),
        "std_val": float(flat.std().item()) if flat.numel() > 1 else 0.0,
        "sparsity": float((flat == 0).sum().item() / flat.numel() * 100),
    }
