import type { ModuleGraph, TensorData, SliceRequest, TensorInfo, AppInfo, FilesResponse, TensorResponse } from '../types';

// Dynamically determine API base URL
const getAPIBase = (): string => {
  if (typeof window === 'undefined') {
    return 'http://localhost:8000';
  }
  // Use current location protocol and hostname, keep API on same server
  return `${window.location.protocol}//${window.location.host}`;
};

const API_BASE = getAPIBase();

// Add cache busting to all requests
const fetchWithCache = async (url: string, options?: RequestInit) => {
  // Add timestamp to prevent caching
  const separator = url.includes('?') ? '&' : '?';
  const cacheBustUrl = `${url}${separator}_t=${Date.now()}`;
  return fetch(cacheBustUrl, {
    ...options,
    headers: {
      ...options?.headers,
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Pragma': 'no-cache',
      'Expires': '0',
    }
  });
};

export const api = {
  async getInfo(): Promise<AppInfo> {
    const response = await fetchWithCache(`${API_BASE}/api/info`);
    if (!response.ok) throw new Error('Failed to fetch info');
    return response.json();
  },

  async getFiles(): Promise<FilesResponse> {
    const response = await fetchWithCache(`${API_BASE}/api/files`);
    if (!response.ok) throw new Error('Failed to fetch files');
    const data = await response.json();
    // Add source_dir if not present
    return {
      ...data,
      source_dir: data.source_dir || '',
      has_demo: data.has_demo !== undefined ? data.has_demo : true,
    };
  },

  async refreshFiles(): Promise<FilesResponse> {
    const response = await fetchWithCache(`${API_BASE}/api/refresh`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to refresh files');
    const data = await response.json();
    return {
      ...data,
      source_dir: data.source_dir || '',
      has_demo: data.has_demo !== undefined ? data.has_demo : true,
    };
  },

  async loadModule(filename: string): Promise<ModuleGraph> {
    const response = await fetchWithCache(`${API_BASE}/api/load?filename=${encodeURIComponent(filename)}`, {
      method: 'POST',
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to load module');
    }
    return response.json();
  },

  async getTensor(request: SliceRequest): Promise<TensorData | null> {
    const response = await fetch(`${API_BASE}/api/tensor`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch tensor');
    }
    const data: TensorResponse = await response.json();
    return data.selected || null;
  },

  async listTensors(filename: string, modulePath: string, modelId?: string): Promise<TensorInfo[]> {
    const params = new URLSearchParams({
      filename: filename,
      module_path: modulePath,
    });
    if (modelId !== undefined && modelId !== null) {
      params.append('model_id', modelId);
    }
    const response = await fetchWithCache(`${API_BASE}/api/tensor/list?${params}`);
    if (!response.ok) throw new Error('Failed to list tensors');
    const data = await response.json();
    return data.tensors;
  },
};
