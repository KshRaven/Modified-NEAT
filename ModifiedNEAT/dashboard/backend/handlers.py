"""FastAPI route handlers for the dashboard backend."""

import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import HTTPException

import torch
import torch.nn as nn
import numpy as np

from ModifiedNEAT.nn.base import NeatModule, PseudoModule
from ModifiedNEAT.population import Population

from .utils import (
    build_graph_from_metadata,
    build_graph,
    reconstruct_module_from_metadata,
    compute_tensor_stats,
    get_tensor_slice,
    numpy_to_torch,
)
from .models import (
    SliceRequest, FileInfo, TensorResponse, TensorData,
    ModuleGraph, ModuleNode, ModuleEdge,
)

# Set up logging
logger = logging.getLogger(__name__.split(".")[-1]+".py")


# ---------------------------------------------------------------------------
# sys.path: ensure backend dir is on path for pickle deserialization
# ---------------------------------------------------------------------------
_backend_dir = Path(__file__).parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open_pkl(filepath: Path):
    """Load a pickle file with robust handling of multiple formats.
    
    Supports:
    - Full module objects (NeatModule, torch.nn.Module)
    - Metadata dictionaries (neat_dict output)
    - Storage wrapper format from util.storage.save() (with 'items' and 'time' keys)
    - Legacy wrapped formats with nested structures
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        The loaded object (unwrapped if needed)
        
    Raises:
        HTTPException: On pickle load errors
    """
    logger.debug(f"_open_pkl: Loading file: {filepath}")
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        logger.debug(f"_open_pkl: Successfully loaded, type={type(data).__name__}")
    except Exception as e:
        logger.error(f"_open_pkl: Failed to load: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load pickle file: {e}"
        )
    
    # Handle storage wrapper format from util.storage.save()
    # The storage.save() function wraps items in: {'items': items, 'time': timestamp}
    if isinstance(data, dict):
        # Check for storage wrapper with 'items' key
        if "items" in data and "time" in data:
            # This is definitely a storage wrapper
            logger.debug(f" _open_pkl: Unwrapping storage wrapper")
            data = data["items"]
        elif "items" in data and len(data) == 1:
            # Might be a storage wrapper with only 'items' (edge case)
            logger.debug(f" _open_pkl: Unwrapping edge-case wrapper (only 'items')")
            data = data["items"]
        
        # Handle nested wrappers (unlikely but possible in legacy files)
        while isinstance(data, dict) and "items" in data and "time" in data and isinstance(data.get("items"), dict):
            logger.debug(f" _open_pkl: Unwrapping nested wrapper")
            data = data["items"]
    
    if isinstance(data, dict):
        logger.debug(f"_open_pkl: Final data is dict with keys: {list(data.keys())[:5]}")
    
    return data


def _resolve_path(base_dir: Path, filename: str) -> Path:
    """Resolve filename relative to base_dir; raise 404/403 on failure."""
    filepath = base_dir / filename
    if not filepath.exists():
        matches = list(base_dir.rglob(filename))
        if matches:
            filepath = matches[0]
        else:
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    try:
        filepath = filepath.resolve()
        filepath.relative_to(base_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid file path")
    return filepath


def _detect_object_type(obj) -> str:
    """
    Detect the type of loaded object.
    
    Returns:
        'module' for NeatModule or torch.nn.Module objects
        'metadata' for neat_dict() output dictionaries
        'population_dict' for Population.save_dict() output
        'unknown' for unsupported types
    """
    # Check for Population.save_dict() format
    if isinstance(obj, dict) and "module_state_dicts" in obj:
        logger.debug(f"_detect_object_type: Detected Population.save_dict() format")
        return "population_dict"
    
    # Check for neat_dict() format
    if isinstance(obj, dict) and "state_dict" in obj and "architecture" in obj:
        logger.debug(f"_detect_object_type: Detected neat_dict() metadata format")
        return "metadata"
    
    # Check for full module objects
    if isinstance(obj, (NeatModule, nn.Module)) and not isinstance(obj, PseudoModule):
        logger.debug(f"_detect_object_type: Detected full module object: {type(obj).__name__}")
        return "module"
    
    # Check for Population object
    if isinstance(obj, Population):
        logger.debug(f"_detect_object_type: Detected Population object")
        return "population"
    
    obj_type = type(obj).__name__
    logger.debug(f"_detect_object_type: Unknown type: {obj_type}")
    return "unknown"


def _extract_module_data(obj) -> tuple:
    """
    Extract module data from loaded object with robust format detection.
    
    Handles:
    - Full module objects (NeatModule, torch.nn.Module)
    - Population.save_dict() with 'module_state_dicts'
    - Direct neat_dict() dictionary
    - Single module wrapped in dict
    - Dict of modules (by genus/id)
    - Dict of neat_dicts
    
    Returns:
        (modules_dict, module_names, object_type)
        where modules_dict = {module_id: module_object_or_metadata}
        
    Raises:
        HTTPException: On unsupported format
    """
    object_type = _detect_object_type(obj)
    
    # Case 1: Full module object
    if object_type == "module":
        return {"0": obj}, ["Module"], "module"
    
    # Case 2: Population.save_dict() output
    if object_type == "population_dict":
        logger.debug(f" _extract_module_data: Extracting Population.save_dict() format")
        module_dicts = obj.get("module_state_dicts", {})
        if not module_dicts:
            logger.error(f" _extract_module_data: Population save_dict has no module_state_dicts")
            raise HTTPException(
                status_code=400,
                detail="Population save_dict has no module_state_dicts"
            )
        # Sort by genus ID for consistent ordering
        sorted_keys = sorted(module_dicts.keys())
        logger.debug(f" _extract_module_data: Found {len(sorted_keys)} module(s) with IDs: {sorted_keys}")
        names = [f"Module (Genus {g})" for g in sorted_keys]
        # Return with keys as strings for consistency
        result = {str(k): module_dicts[k] for k in sorted_keys}
        logger.debug(f" _extract_module_data: Returning metadata dicts for {len(result)} module(s)")
        for k, v in result.items():
            if isinstance(v, dict):
                logger.debug(f"   Module {k}: keys={list(v.keys())}")
        return result, names, "metadata"
    
    # Case 3: Direct neat_dict() output
    if object_type == "metadata":
        return {"0": obj}, ["Module"], "metadata"
    
    # Case 4: Population object - explicitly reject (must use save_dict)
    if object_type == "population":
        raise HTTPException(
            status_code=400,
            detail="Cannot load Population object directly; use save_dict() format"
        )
    
    # Case 5: Dict-wrapped module(s)
    if isinstance(obj, dict) and len(obj) > 0:
        first_val = next(iter(obj.values()))
        
        # Case 5a: Dict of modules (by genus or ID)
        if isinstance(first_val, (NeatModule, nn.Module)) and not isinstance(first_val, PseudoModule):
            names = [f"Module {k}" for k in sorted(obj.keys())]
            return {str(k): obj[k] for k in sorted(obj.keys())}, names, "module"
        
        # Case 5b: Dict of neat_dicts (by genus or ID)
        if isinstance(first_val, dict) and "state_dict" in first_val and "architecture" in first_val:
            names = [f"Module {k}" for k in sorted(obj.keys())]
            return {str(k): obj[k] for k in sorted(obj.keys())}, names, "metadata"
        
        # Case 5c: Try to treat entire dict as a single neat_dict
        if "state_dict" in obj or "architecture" in obj:
            return {"0": obj}, ["Module"], "metadata"
    
    # Unsupported format
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported object format. Expected NeatModule, torch.nn.Module, "
               f"Population.save_dict(), NeatModule.neat_dict(), or dict of modules/metadata. "
               f"Got: {type(obj).__name__} with keys: {list(obj.keys())[:5] if isinstance(obj, dict) else 'N/A'}"
    )


# ---------------------------------------------------------------------------
# ModuleHandler
# ---------------------------------------------------------------------------

class ModuleHandler:

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def list_pkl_files(self) -> dict:
        """List all .module.pkl and .neat.pkl files recursively."""
        files      = []
        seen_paths = set()

        for pattern in ["*.module.pkl", "*.neat.pkl"]:
            for path in self.base_dir.rglob(pattern):
                if not path.is_file():
                    continue
                abs_path = path.resolve()
                if abs_path in seen_paths:
                    continue
                seen_paths.add(abs_path)

                stat      = path.stat()
                rel_path  = path.relative_to(self.base_dir)
                directory = str(rel_path.parent) if str(rel_path.parent) != "." else ""

                files.append({
                    "name":      path.name,
                    "path":      str(rel_path),
                    "size":      stat.st_size,
                    "modified":  datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "directory": directory if directory else None,
                })

        return {
            "files":      sorted(files, key=lambda x: x["name"]),
            "source_dir": str(self.base_dir),
        }

    def load_module(self, filename: str):
        """Load module graph from saved file.
        
        Works with:
        - Full module objects (NeatModule, torch.nn.Module)
        - Population.save_dict() format (multiple genera)
        - NeatModule.neat_dict() format (metadata)
        - Legacy util.storage.save() wrapped format
        """
        filepath = _resolve_path(self.base_dir, filename)

        try:
            obj = _open_pkl(filepath)
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            raise HTTPException(
                status_code=500,
                detail=f"Error loading pickle file: {e}\n{traceback.format_exc()}",
            )

        # Extract module data (handles both full objects and metadata)
        try:
            modules_dict, module_names, object_type = _extract_module_data(obj)
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            raise HTTPException(
                status_code=500,
                detail=f"Error extracting module data: {e}\n{traceback.format_exc()}"
            )

        all_nodes, all_edges, total_params = [], [], 0
        module_graphs = []

        # Build graph for each module
        for module_id, module_data in modules_dict.items():
            try:
                logger.debug(f"load_module: Processing module {module_id} (type={object_type})")
                if isinstance(module_data, dict) and object_type == "metadata":
                    logger.debug(f"  Module {module_id} metadata keys: {list(module_data.keys())}")
                
                # Generate a module-specific prefix for node IDs to ensure uniqueness
                module_prefix = f"mod_{module_id}" if module_id != "0" else ""
                
                if object_type == "module":
                    # Full module object - build graph directly with module prefix
                    logger.debug(f"load_module: Building graph from module object")
                    # Try to get state_dict for parameter information
                    state_dict = None
                    if hasattr(module_data, 'state_dict') and callable(module_data.state_dict):
                        try:
                            state_dict = module_data.state_dict()
                        except Exception:
                            pass
                    nodes, edges, params = build_graph(
                        module_data,
                        prefix=module_prefix,
                        state_dict=state_dict
                    )
                else:
                    # Metadata - reconstruct and build graph with module prefix
                    logger.debug(f"load_module: Building graph from metadata (module_id={module_id})")
                    nodes, edges, params = build_graph_from_metadata(
                        module_data,
                        module_name="root",
                        module_prefix=module_prefix
                    )
                
                logger.debug(f"load_module: Module {module_id} graph built: {len(nodes)} nodes, {len(edges)} edges, {params} params")
                
                all_nodes.extend(nodes)
                all_edges.extend(edges)
                total_params += params
                
                module_graphs.append({
                    "module_id":    str(module_id),
                    "nodes":        [n.model_dump() for n in nodes],
                    "edges":        [e.model_dump() for e in edges],
                    "total_params": params,
                })
            except Exception as e:
                import traceback
                err_msg = f"Error building graph for module {module_id}: {e}\n{traceback.format_exc()}"
                print(err_msg)
                raise HTTPException(
                    status_code=500,
                    detail=err_msg
                )

        if len(modules_dict) == 1:
            return ModuleGraph(nodes=all_nodes, edges=all_edges, total_params=total_params)

        return ModuleGraph(
            nodes=all_nodes, edges=all_edges, total_params=total_params,
            modules=module_graphs, module_names=module_names,
        )


# ---------------------------------------------------------------------------
# TensorHandler
# ---------------------------------------------------------------------------

class TensorHandler:
    """Handle tensor loading and slicing from full modules and metadata."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def _load_and_navigate(self, filename: str, module_path: str, model_id: Optional[str] = None) -> tuple:
        """Load file and navigate to a module by path.
        
        Args:
            filename: Pickle filename
            module_path: Dot-separated path to module
            model_id: Specific model/genus ID to use (optional, defaults to first module)
        
        Returns:
            (module, state_dict_or_none, module_id, object_type, module_path_prefix)
        """
        filepath = _resolve_path(self.base_dir, filename)
        logger.debug(f" _load_and_navigate: filename={filename}, module_path={module_path}, model_id={model_id}")

        try:
            obj = _open_pkl(filepath)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading file: {e}")

        # Extract module data (handles both full objects and metadata)
        try:
            modules_dict, _, object_type = _extract_module_data(obj)
        except HTTPException:
            raise

        parts = module_path.split(".")
        logger.debug(f" _load_and_navigate: module_path parts={parts}, modules_dict keys={list(modules_dict.keys())}, object_type={object_type}")
        
        # Handle module path - model_id takes priority if specified
        if len(modules_dict) == 1:
            selected_module_id = next(iter(modules_dict.keys()))
            selected_module = modules_dict[selected_module_id]
            nav_parts = parts
            logger.debug(f" _load_and_navigate: Single module, ID={selected_module_id}, nav_parts={nav_parts}")
        else:
            # Multi-module file - use model_id if provided, otherwise try to extract from path
            if model_id is not None:
                # Use explicitly provided model_id
                selected_module = modules_dict.get(str(model_id))
                if selected_module is None and model_id.isdigit():
                    selected_module = modules_dict.get(int(model_id))
                
                if selected_module is None:
                    available_ids = list(modules_dict.keys())
                    logger.error(f" _load_and_navigate: Model '{model_id}' not found. Available: {available_ids}")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model '{model_id}' not found. Available models: {available_ids}"
                    )
                
                selected_module_id = str(model_id)
                nav_parts = parts
                logger.debug(f" _load_and_navigate: Using explicit model_id={model_id}, nav_parts={nav_parts}")
            else:
                # Try to extract module ID from first path part (e.g., "mod_1" -> "1")
                if len(parts) < 1:
                    raise HTTPException(
                        status_code=400,
                        detail="Module path too short for multi-module file and no model_id provided"
                    )
                
                prefix_part = parts[0]
                module_id = None
                
                # Try to extract module ID from prefix (e.g., "mod_1" -> "1")
                if prefix_part.startswith("mod_"):
                    module_id = prefix_part[4:]  # Remove "mod_" prefix
                else:
                    module_id = prefix_part
                
                logger.debug(f" _load_and_navigate: Multi-module, prefix_part={prefix_part}, extracted module_id={module_id}")
                
                # Try to find the module with this ID (try as string first, then as int)
                selected_module = modules_dict.get(module_id)
                if selected_module is None and module_id.isdigit():
                    selected_module = modules_dict.get(int(module_id))
                    if selected_module is None:
                        # Also try the string representation of the int
                        selected_module = modules_dict.get(str(int(module_id)))
                
                if selected_module is None:
                    available_ids = list(modules_dict.keys())
                    logger.error(f" _load_and_navigate: Module '{module_id}' not found. Available: {available_ids}")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Module '{module_id}' not found. Available modules: {available_ids}"
                    )
                
                selected_module_id = module_id
                nav_parts = parts[1:]
                logger.debug(f" _load_and_navigate: Found module {selected_module_id}, nav_parts={nav_parts}")

        # Get or create the module to navigate
        if object_type == "module":
            # Full module object - use directly
            target_module = selected_module
            state_dict = None  # Will get from module directly
            module_path_prefix = ""  # Track the path prefix for state_dict lookups
            logger.debug(f" _load_and_navigate: Using full module object")
        else:
            # Metadata - reconstruct pseudo module
            target_module = reconstruct_module_from_metadata(selected_module)
            state_dict = selected_module.get('state_dict', {})
            # Start with empty prefix for root module
            module_path_prefix = ""
            logger.debug(f" _load_and_navigate: Reconstructed from metadata, state_dict has {len(state_dict)} keys")

        # Navigate to target module using the path
        logger.debug(f" _load_and_navigate: Starting navigation through {len(nav_parts)} parts: {nav_parts}")
        for i, part in enumerate(nav_parts):
            if not part:  # Skip empty parts
                logger.debug(f" _load_and_navigate: Skipping empty part at index {i}")
                continue
            logger.debug(f" _load_and_navigate: Navigating to '{part}', current module: {type(target_module).__name__}")
            child = None
            try:
                children = list(target_module.named_children())
                logger.debug(f" _load_and_navigate:   Available children: {[name for name, _ in children]}")
            except Exception as e:
                logger.error(f" _load_and_navigate:   Error listing children: {e}")
                children = []
            
            for name, module in children:
                if name == part:
                    child = module
                    break
            if child is None:
                logger.error(f" _load_and_navigate: Sub-module '{part}' not found in path '{module_path}'")
                raise HTTPException(
                    status_code=404,
                    detail=f"Sub-module '{part}' not found in path '{module_path}'"
                )
            target_module = child
            # Update the path prefix for state_dict lookups
            module_path_prefix = f"{module_path_prefix}{part}." if module_path_prefix else f"{part}."
            logger.debug(f" _load_and_navigate: Navigated to '{part}', new prefix={module_path_prefix}")

        logger.debug(f" _load_and_navigate: Navigation complete, returning module={type(target_module).__name__}, prefix={module_path_prefix}")
        return target_module, state_dict, selected_module_id, object_type, module_path_prefix

    def get_tensor(self, request):
        """Get a specific tensor with optional slicing.
        
        Args:
            request: SliceRequest with filename, module_path, model_id, etc.
            
        Returns:
            TensorResponse with tensor data or error message
        """
        try:
            module, state_dict, _, object_type, *module_path_prefix_list = self._load_and_navigate(
                request.filename, request.module_path, request.model_id
            )
            module_path_prefix = module_path_prefix_list[0] if module_path_prefix_list else ""

            # Get tensors from the module
            tensors_raw = []
            try:
                if object_type == "module":
                    # Full module object - get parameters directly
                    for name, param in module.named_parameters(recurse=False):
                        try:
                            tensors_raw.append((name, param.data))
                        except Exception as e:
                            print(f"Error accessing parameter {name}: {e}")
                else:
                    # Metadata - get from state_dict and match with module params
                    state_dict = state_dict or {}
                    
                    # For metadata modules, get direct parameter names
                    try:
                        direct_param_names = set(name for name, _ in module.named_parameters(recurse=False))
                    except Exception as e:
                        print(f"Error getting parameter names: {e}")
                        direct_param_names = set()
                    
                    # Look up parameters in state_dict using the module_path_prefix
                    for param_name in direct_param_names:
                        # Construct the full state_dict key by prepending the module path
                        state_dict_key = f"{module_path_prefix}{param_name}"
                        if state_dict_key in state_dict:
                            try:
                                tensor = numpy_to_torch(state_dict[state_dict_key])
                                tensors_raw.append((param_name, tensor))
                            except Exception as e:
                                print(f"Error converting tensor {state_dict_key}: {e}")
                    
                    # If no parameters found with prefix, try without prefix as fallback
                    if not tensors_raw and state_dict:
                        for param_name in direct_param_names:
                            if param_name in state_dict:
                                try:
                                    tensor = numpy_to_torch(state_dict[param_name])
                                    tensors_raw.append((param_name, tensor))
                                except Exception as e:
                                    print(f"Error converting tensor {param_name}: {e}")
            except Exception as e:
                import traceback
                print(f"Error listing parameters for {request.module_path}: {e}")
                print(traceback.format_exc())
                return TensorResponse(
                    error=f"Failed to list tensors in {request.module_path}: {e}"
                )

            if not tensors_raw:
                return TensorResponse(
                    error=f"No tensors in {request.module_path} (object_type={object_type}, prefix={module_path_prefix})"
                )

            idx = min(request.tensor_idx, len(tensors_raw) - 1)
            name, tensor = tensors_raw[idx]

            # Compute stats for all tensors
            tensors_meta = []
            for tname, t in tensors_raw:
                try:
                    s = compute_tensor_stats(t)
                    tensors_meta.append({
                        "name": tname,
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "numel": int(t.numel()),
                        **s,
                    })
                except Exception as e:
                    tensors_meta.append({
                        "name": tname,
                        "shape": list(t.shape) if hasattr(t, "shape") else [],
                        "dtype": str(t.dtype) if hasattr(t, "dtype") else "unknown",
                        "numel": int(t.numel()) if hasattr(t, "numel") else 0,
                        "error": str(e),
                    })

            # Handle genome dimension if present
            has_genome_dim = False
            genome_keys = None
            actual = tensor

            if request.genome_key is not None and tensor.ndim > 1:
                has_genome_dim = True
                try:
                    actual = tensor[request.genome_key]
                except (IndexError, RuntimeError):
                    pass
            elif tensor.ndim >= 2 and tensor.shape[0] > 1:
                has_genome_dim = True
                genome_keys = list(range(min(tensor.shape[0], 100)))  # Limit list size
                actual = tensor[0]

            # Slice and get display values
            arr, _ = get_tensor_slice(actual, request.slice_spec, request.max_display)
            stats = compute_tensor_stats(actual)

            common = dict(
                tensor_idx=idx,
                tensor_name=name,
                has_genome_dim=has_genome_dim,
                genome_keys=genome_keys,
                dtype=str(actual.dtype),
                **stats,
            )

            if arr.ndim == 0:
                return TensorResponse(
                    tensors=tensors_meta,
                    selected=TensorData(
                        shape=[],
                        is_scalar=True,
                        scalar_value=float(arr),
                        **common
                    ),
                )

            values = arr.tolist()
            if arr.ndim == 1:
                values = [[v] for v in values]

            return TensorResponse(
                tensors=tensors_meta,
                selected=TensorData(
                    shape=list(actual.shape),
                    values=values,
                    is_scalar=False,
                    **common
                ),
            )

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            err_msg = f"Error in get_tensor: {e}\n{traceback.format_exc()}"
            print(err_msg)
            raise HTTPException(status_code=500, detail=err_msg)

    def list_tensors(self, filename: str, module_path: str, model_id: Optional[str] = None) -> dict:
        """List all available tensors for a module.
        
        Args:
            filename: Pickle file name
            module_path: Dot-separated path to module
            model_id: Specific model/genus ID (optional, defaults to first if not provided)
            
        Returns:
            Dict with 'tensors' list containing tensor metadata
        """
        logger.debug(f" list_tensors: filename={filename}, module_path={module_path}, model_id={model_id}")
        try:
            module, state_dict, _, object_type, *module_path_prefix_list = self._load_and_navigate(filename, module_path, model_id)
            module_path_prefix = module_path_prefix_list[0] if module_path_prefix_list else ""
            
            logger.debug(f" list_tensors: Successfully loaded module, object_type={object_type}, prefix={module_path_prefix}")
            
            tensors = []
            
            # Get tensors from the module
            try:
                if object_type == "module":
                    # Full module object - get parameters directly
                    logger.debug(f" list_tensors: Getting parameters from module object")
                    for name, param in module.named_parameters(recurse=False):
                        try:
                            tensor = param.data
                            s = compute_tensor_stats(tensor)
                            tensors.append({
                                "name": name,
                                "shape": list(tensor.shape),
                                "dtype": str(tensor.dtype),
                                "numel": int(tensor.numel()),
                                **s,
                            })
                        except Exception as e:
                            logger.error(f" list_tensors: Error processing parameter {name}: {e}")
                            tensors.append({
                                "name": name,
                                "shape": [],
                                "dtype": "unknown",
                                "numel": 0,
                                "error": str(e),
                            })
                    logger.debug(f" list_tensors: Found {len(tensors)} parameters in module object")
                else:
                    # Metadata - get from state_dict
                    logger.debug(f" list_tensors: Getting parameters from metadata")
                    state_dict = state_dict or {}
                    try:
                        direct_param_names = set(name for name, _ in module.named_parameters(recurse=False))
                        logger.debug(f" list_tensors: Direct param names: {direct_param_names}")
                    except Exception as e:
                        logger.error(f" list_tensors: Error getting parameter names from metadata: {e}")
                        direct_param_names = set()
                    
                    # Look up parameters in state_dict using the module_path_prefix
                    logger.debug(f" list_tensors: Looking up {len(direct_param_names)} params with prefix '{module_path_prefix}' in state_dict ({len(state_dict)} keys)")
                    for param_name in direct_param_names:
                        # Construct the full state_dict key by prepending the module path
                        state_dict_key = f"{module_path_prefix}{param_name}"
                        if state_dict_key in state_dict:
                            try:
                                tensor = numpy_to_torch(state_dict[state_dict_key])
                                s = compute_tensor_stats(tensor)
                                tensors.append({
                                    "name": param_name,
                                    "shape": list(tensor.shape),
                                    "dtype": str(tensor.dtype),
                                    "numel": int(tensor.numel()),
                                    **s,
                                })
                            except Exception as e:
                                logger.error(f" list_tensors: Error processing tensor {state_dict_key}: {e}")
                                tensors.append({
                                    "name": param_name,
                                    "shape": list(state_dict[state_dict_key].shape) if hasattr(state_dict[state_dict_key], "shape") else [],
                                    "dtype": str(state_dict[state_dict_key].dtype) if hasattr(state_dict[state_dict_key], "dtype") else "unknown",
                                    "numel": int(np.prod(state_dict[state_dict_key].shape)) if hasattr(state_dict[state_dict_key], "shape") else 0,
                                    "error": str(e),
                                })
                        else:
                            logger.debug(f" list_tensors: Key not found: {state_dict_key}")
                    
                    # If no parameters found with prefix, try without prefix as fallback
                    if not tensors and state_dict:
                        logger.debug(f" list_tensors: No params found with prefix, trying without prefix")
                        for param_name in direct_param_names:
                            if param_name in state_dict:
                                try:
                                    tensor = numpy_to_torch(state_dict[param_name])
                                    s = compute_tensor_stats(tensor)
                                    tensors.append({
                                        "name": param_name,
                                        "shape": list(tensor.shape),
                                        "dtype": str(tensor.dtype),
                                        "numel": int(tensor.numel()),
                                        **s,
                                    })
                                except Exception as e:
                                    logger.error(f" list_tensors: Error processing tensor {param_name}: {e}")
                                    tensors.append({
                                        "name": param_name,
                                        "shape": list(state_dict[param_name].shape) if hasattr(state_dict[param_name], "shape") else [],
                                        "dtype": str(state_dict[param_name].dtype) if hasattr(state_dict[param_name], "dtype") else "unknown",
                                        "numel": int(np.prod(state_dict[param_name].shape)) if hasattr(state_dict[param_name], "shape") else 0,
                                        "error": str(e),
                                    })
                    logger.debug(f" list_tensors: Found {len(tensors)} parameters total")
            except Exception as e:
                import traceback
                logger.error(f" list_tensors: Error iterating tensors: {e}")
                print(traceback.format_exc())
                return {
                    "tensors": tensors,
                    "error": f"Error listing tensors: {e}"
                }

            if not tensors:
                logger.debug(f" list_tensors: No tensors found in {module_path}")
                return {
                    "tensors": tensors,
                    "message": f"No tensors in {module_path}"
                }
            logger.debug(f" list_tensors: Returning {len(tensors)} tensors")
            return {"tensors": tensors}

        except HTTPException as e:
            logger.error(f" list_tensors: HTTPException: {e.detail}")
            raise
        except Exception as e:
            import traceback
            msg = f"Error listing tensors for {module_path}: {e}\n{traceback.format_exc()}"
            logger.error(f" list_tensors: {msg}")
            raise HTTPException(status_code=500, detail=msg)