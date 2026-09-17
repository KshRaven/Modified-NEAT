export interface PklFile {
  name: string;
  path: string;
  size: number;
  modified: string;
  total_params?: number;
  directory?: string;  // Directory path for display
}

export interface ModuleNode {
  id: string;
  name: string;
  type: string;
  has_weights: boolean;
  param_count?: number;
  is_neat_module?: boolean;
  is_nested?: boolean;
  children?: ModuleNode[];
  position?: { x: number; y: number };
  parent_id?: string | null;
}

export interface ModuleEdge {
  id: string;
  source: string;
  target: string;
}

export interface ModuleGraph {
  nodes: ModuleNode[];
  edges: ModuleEdge[];
  total_params: number;
  modules?: Array<{
    module_id: string;
    nodes: ModuleNode[];
    edges: ModuleEdge[];
    total_params: number;
  }>;
  module_names?: string[];
}

export interface TensorInfo {
  name: string;
  shape: number[];
  dtype: string;
  numel: number;
  has_nan: boolean;
  has_inf: boolean;
  min_val: number | null;
  max_val: number | null;
  mean_val: number | null;
  std_val: number | null;
  sparsity: number | null;
}

export interface TensorData {
  shape: number[];
  values: number[][] | null;
  dtype: string;
  tensor_idx: number;
  tensor_name?: string;
  is_scalar: boolean;
  scalar_value: number | null;
  has_nan: boolean;
  has_inf: boolean;
  min_val: number | null;
  max_val: number | null;
  mean_val: number | null;
  std_val: number | null;
  sparsity: number | null;
  has_genome_dim?: boolean;
  genome_keys?: number[] | null;
}

export interface TensorResponse {
  tensors?: TensorInfo[];
  selected?: TensorData;
  error?: string;
}

export interface SliceRequest {
  filename: string;
  module_path: string;
  tensor_idx: number;
  slice_spec?: string;
  max_display?: number;
  genome_key?: number;
  model_id?: string;  // Model/genus ID for multi-model files
}

export interface AppInfo {
  source_dir: string;
  has_files: boolean;
}

export interface FilesResponse {
  files: PklFile[];
  source_dir: string;
}
