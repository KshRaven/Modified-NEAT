"""Pydantic models for the dashboard API."""

from ModifiedNEAT.nn.base import NeatModule, NeatParameter, Model
from ModifiedNEAT.nn.modules.base import *
from ModifiedNEAT.nn.genome import Genome
from ModifiedNEAT.population import Population
from ModifiedNEAT.util.storage import *
from typing import Optional
from pydantic import BaseModel


class SliceRequest(BaseModel):
    """Request model for tensor slicing."""
    filename: str
    module_path: str
    tensor_idx: int = 0
    slice_spec: Optional[str] = None
    max_display: Optional[int] = None
    genome_key: Optional[int] = None  # Specific genome key to index if first dim is genomes
    model_id: Optional[str] = None  # Model/genus ID when multiple models are loaded


class ModuleInfo(BaseModel):
    """Hierarchical module information."""
    name: str
    type: str
    has_weights: bool
    children: list["ModuleInfo"] = []


ModuleInfo.model_rebuild()


class TensorData(BaseModel):
    """Tensor data and statistics."""
    shape: list[int]
    values: Optional[list[list[float]]] = None
    dtype: str
    tensor_idx: int = 0
    tensor_name: Optional[str] = None
    is_scalar: bool = False
    scalar_value: Optional[float] = None
    has_nan: bool = False
    has_inf: bool = False
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    std_val: Optional[float] = None
    sparsity: Optional[float] = None  # percentage of zeros
    has_genome_dim: bool = False  # Whether first dimension is genome indices
    genome_keys: Optional[list[int]] = None  # Available genome keys if applicable
    genome_key_mapping: Optional[dict] = None  # Mapping from genome key to index


class ModuleNode(BaseModel):
    """Node in the module graph."""
    id: str
    name: str
    type: str
    has_weights: bool
    param_count: Optional[int] = None
    is_neat_module: bool = False  # Whether this is a NeatModule
    is_nested: bool = False  # Whether this module contains child modules
    children: list["ModuleNode"] = []  # Child modules for hierarchical display
    position: Optional[dict] = None  # Position for nested modules (x, y)
    parent_id: Optional[str] = None  # Parent module ID
    parameters: Optional[list[dict]] = None  # List of parameter info: [{'name': 'weights', 'extended_name': '...path.to.weights', 'shape': [...], 'dtype': 'float32'}, ...]
    full_module_path: Optional[str] = None  # Full path to this module from root (e.g., 'root.lat_proj.modules_list.0')


ModuleNode.model_rebuild()


class ModuleEdge(BaseModel):
    """Edge in the module graph."""
    id: str
    source: str
    target: str


class ModuleGraph(BaseModel):
    """Complete module graph structure."""
    nodes: list[ModuleNode]
    edges: list[ModuleEdge]
    total_params: int = 0
    modules: Optional[list[dict]] = None  # Multiple modules for Population objects
    module_names: Optional[list[str]] = None  # Names of modules if multiple


class TensorResponse(BaseModel):
    """Response model for tensor queries."""
    tensors: list[dict] = []
    selected: Optional[TensorData] = None
    error: Optional[str] = None


class FileInfo(BaseModel):
    """Information about a pickle file."""
    name: str
    path: str
    size: int
    modified: str
    total_params: Optional[int] = None
    directory: Optional[str] = None  # Directory path for display
