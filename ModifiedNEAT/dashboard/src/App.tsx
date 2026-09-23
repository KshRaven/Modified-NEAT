import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { Header } from './components/Header';
import { FileList } from './components/FileList';
import { ModuleDiagram } from './components/ModuleDiagram';
import { TensorViewer } from './components/TensorViewer';
import { SettingsPanel } from './components/SettingsPanel';
import { StatusBar } from './components/StatusBar';
import { ThemeProvider } from './contexts/ThemeContext';
import { SettingsProvider, useSettings } from './contexts/SettingsContext';
import { api } from './services/api';
import type { PklFile, ModuleGraph, TensorData, TensorInfo } from './types';
import { Loader2, AlertCircle, Search, ZoomIn, ZoomOut, Maximize2, ChevronRight, ChevronLeft } from 'lucide-react';

const AppContent: React.FC = () => {
  const { settings } = useSettings();
  const [files, setFiles] = useState<PklFile[]>([]);
  const [sourceDir, setSourceDir] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [moduleGraph, setModuleGraph] = useState<ModuleGraph | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);  // Track selected model/genus
  const [tensorPanelWidth, setTensorPanelWidth] = useState(520); // 384px = w-96 equivalent
  const [isResizingTensorPanel, setIsResizingTensorPanel] = useState(false);
  const [isTensorPanelExpanded, setIsTensorPanelExpanded] = useState(false);
  const [tensorList, setTensorList] = useState<TensorInfo[]>([]);
  const [selectedTensorData, setSelectedTensorData] = useState<TensorData | null>(null);
  const [viewingTensor, setViewingTensor] = useState(false);
  const [loading, setLoading] = useState(false);
  const [fileLoading, setFileLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [zoomLevel, setZoomLevel] = useState(1);
  const [activeModuleTab, setActiveModuleTab] = useState(0);  // Active module tab index
  
  // Diagram control callbacks
  const expandAllRef = useRef<(() => void) | null>(null);
  const collapseAllRef = useRef<(() => void) | null>(null);
  const fitViewRef = useRef<(() => void) | null>(null);

  // Auto refresh
  const autoRefreshRef = useRef<number | null>(null);

  const handleRefreshTensor = useCallback(async () => {
    if (selectedNode && selectedFile) {
      try {
        const tensors = await api.listTensors(selectedFile, selectedNode, selectedModelId || undefined);
        setTensorList(tensors);
      } catch (err) {
        console.error('Failed to refresh tensors:', err);
      }
    }
  }, [selectedNode, selectedFile, selectedModelId]);

  useEffect(() => {
    if (settings.autoRefresh && selectedFile) {
      autoRefreshRef.current = window.setInterval(() => {
        if (selectedNode && viewingTensor) {
          handleRefreshTensor();
        }
      }, settings.refreshInterval * 1000);
    }

    return () => {
      if (autoRefreshRef.current) {
        clearInterval(autoRefreshRef.current);
      }
    };
  }, [settings.autoRefresh, settings.refreshInterval, selectedFile, selectedNode, viewingTensor, handleRefreshTensor]);

  const fetchFiles = useCallback(async () => {
    setFileLoading(true);
    setError(null);
    try {
      const result = await api.getFiles();
      setFiles(result.files);
      setSourceDir(result.source_dir || '');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch files');
    } finally {
      setFileLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  const handleSelectFile = useCallback(async (filename: string) => {
    setSelectedFile(filename);
    setSelectedNode(null);
    setSelectedModelId(null);  // Reset selected model when file changes
    setTensorList([]);
    setSelectedTensorData(null);
    setViewingTensor(false);
    setActiveModuleTab(0);  // Reset to first module tab
    setLoading(true);
    setGraphError(null);
    try {
      const result = await api.loadModule(filename);
      setModuleGraph(result);
      // Set default model ID to first one if multiple models exist
      if (result.modules && result.modules.length > 0) {
        setSelectedModelId(result.modules[0].module_id);
      }
    } catch (err) {
      setGraphError(err instanceof Error ? err.message : 'Failed to load module');
      setModuleGraph(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleNodeClick = useCallback(async (nodeId: string) => {
    if (nodeId === selectedNode) {
      setSelectedNode(null);
      setTensorList([]);
      setSelectedTensorData(null);
      setViewingTensor(false);
      return;
    }

    setSelectedNode(nodeId);

    if (!selectedFile) return;

    // Strip module prefix (e.g., "mod_1.lat_proj" → "lat_proj") when navigating
    // since we're now passing model_id explicitly
    let navigationPath = nodeId;
    if (selectedModelId && nodeId.startsWith(`mod_${selectedModelId}.`)) {
      navigationPath = nodeId.slice(`mod_${selectedModelId}.`.length);
    }

    const node = moduleGraph?.nodes.find(n => n.id === nodeId);
    if (node?.has_weights) {
      try {
        const tensors = await api.listTensors(selectedFile, navigationPath, selectedModelId || undefined);
        setTensorList(tensors);
        const tensorData = await api.getTensor({
          filename: selectedFile,
          module_path: navigationPath,
          tensor_idx: 0,
          max_display: settings.maxDisplay,
          model_id: selectedModelId || undefined,
        });
        if (!tensorData) {
          setTensorList([]);
          setSelectedTensorData(null);
          setViewingTensor(false);
        } else {
          setSelectedTensorData(tensorData);
          setViewingTensor(true);
        }
      } catch (err) {
        console.error('Failed to fetch tensor:', err);
        setTensorList([]);
        setSelectedTensorData(null);
        setViewingTensor(false);
      }
    } else {
      setTensorList([]);
      setSelectedTensorData(null);
      setViewingTensor(false);
    }
  }, [selectedFile, selectedNode, selectedModelId, moduleGraph, settings.maxDisplay]);

  const handleDiagramPaneClick = useCallback(() => {
    setSelectedNode(null);
    setTensorList([]);
    setSelectedTensorData(null);
    setViewingTensor(false);
  }, []);

  const handleCloseTensorViewer = useCallback(() => {
    setViewingTensor(false);
  }, []);

  const handleSettingsClick = useCallback(() => {
    setSettingsOpen(true);
  }, []);

  const handleCloseSettings = useCallback(() => {
    setSettingsOpen(false);
  }, []);

  const handleRefreshFiles = useCallback(async () => {
    setIsRefreshing(true);
    await fetchFiles();
    setIsRefreshing(false);
  }, [fetchFiles]);

  // Filter files based on search
  const filteredFiles = useMemo(() => {
    if (!searchQuery.trim()) return files;
    const query = searchQuery.toLowerCase();
    return files.filter(f => f.name.toLowerCase().includes(query));
  }, [files, searchQuery]);

  // Zoom controls
  const handleZoomIn = useCallback(() => {
    setZoomLevel(prev => Math.min(prev + 0.1, 2));
  }, []);

  const handleZoomOut = useCallback(() => {
    setZoomLevel(prev => Math.max(prev - 0.1, 0.3));
  }, []);

  const handleZoomReset = useCallback(() => {
    setZoomLevel(1);
  }, []);

  const handleExpandAll = useCallback(() => {
    if (expandAllRef.current) expandAllRef.current();
  }, []);

  const handleCollapseAll = useCallback(() => {
    if (collapseAllRef.current) collapseAllRef.current();
  }, []);

  const handleFitView = useCallback(() => {
    if (fitViewRef.current) fitViewRef.current();
  }, []);

  // Tensor panel resize handlers
  const handleStartResizeTensorPanel = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizingTensorPanel(true);
  }, []);

  const handleToggleTensorPanelExpand = useCallback(() => {
    if (isTensorPanelExpanded) {
      setIsTensorPanelExpanded(false);
      setTensorPanelWidth(520); // Reset to default
    } else {
      setIsTensorPanelExpanded(true);
      setTensorPanelWidth(Math.min(window.innerWidth * 0.4, 800)); // Expand to 40% or max 800px
    }
  }, [isTensorPanelExpanded]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizingTensorPanel) return;
      
      const mainElement = document.querySelector('main');
      if (!mainElement) return;
      
      const rect = mainElement.getBoundingClientRect();
      const newWidth = Math.max(250, Math.min(window.innerWidth * 0.5, rect.right - e.clientX));
      setTensorPanelWidth(newWidth);
    };

    const handleMouseUp = () => {
      setIsResizingTensorPanel(false);
    };

    if (isResizingTensorPanel) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isResizingTensorPanel]);

  const graphKey = useMemo(() => `graph-${moduleGraph ? 'loaded' : 'empty'}`, [moduleGraph]);

  // Compute current module graph based on active tab
  const currentModuleGraph = useMemo(() => {
    if (!moduleGraph) return null;
    
    // If there are multiple modules, return the one for the active tab
    if (moduleGraph.modules && moduleGraph.modules.length > 0 && activeModuleTab < moduleGraph.modules.length) {
      const activeModule = moduleGraph.modules[activeModuleTab];
      return {
        nodes: activeModule.nodes,
        edges: activeModule.edges,
        total_params: activeModule.total_params,
      };
    }
    
    // Otherwise return the main graph (single module case)
    return {
      nodes: moduleGraph.nodes,
      edges: moduleGraph.edges,
      total_params: moduleGraph.total_params,
    };
  }, [moduleGraph, activeModuleTab]);

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-gray-100 via-gray-50 to-gray-200 dark:from-gray-950 dark:via-gray-900 dark:to-gray-950 transition-colors duration-500">
      <Header
        onSettingsClick={handleSettingsClick}
        onRefreshClick={handleRefreshFiles}
        isRefreshing={isRefreshing}
      />

      <div className="flex-1 flex overflow-hidden">
        <aside className="w-72 border-r border-gray-200 dark:border-white/5 bg-white/30 dark:bg-black/20 backdrop-blur-xl transition-all duration-300 flex flex-col">
          {/* Search */}
          <div className="p-3 border-b border-white/5">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Filter files..."
                className="w-full pl-9 pr-3 py-1.5 text-xs bg-gray-800/50 border border-white/10 rounded-lg text-gray-200 placeholder-gray-500 focus:outline-none focus:border-cyan-400/50"
              />
            </div>
          </div>

          <FileList
            files={filteredFiles}
            loading={fileLoading}
            selectedFile={selectedFile}
            onSelectFile={handleSelectFile}
            error={error}
          />
        </aside>

        <main className="flex-1 flex flex-col overflow-hidden">
          {loading && selectedFile ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="flex flex-col items-center gap-3">
                <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Loading module structure...
                </p>
              </div>
            </div>
          ) : graphError ? (
            <div className="flex-1 flex items-center justify-center p-8">
              <div className="flex flex-col items-center gap-3 text-center max-w-md">
                <AlertCircle className="w-12 h-12 text-red-400" />
                <h3 className="text-lg font-semibold text-red-400">Failed to Load Module</h3>
                <p className="text-sm text-gray-400">{graphError}</p>
              </div>
            </div>
          ) : moduleGraph ? (
            <div key={graphKey} className="flex-1 flex flex-col overflow-hidden">
              {/* Module tabs for multiple modules */}
              {moduleGraph.modules && moduleGraph.modules.length > 1 && (
                <div className="flex items-center gap-1 px-4 py-2 border-b border-white/5 bg-gray-800/20 overflow-x-auto custom-scrollbar">
                  {moduleGraph.module_names?.map((name, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setActiveModuleTab(idx);
                        // Set the model ID when switching tabs
                        if (moduleGraph.modules?.[idx]) {
                          setSelectedModelId(moduleGraph.modules[idx].module_id);
                        }
                        setSelectedNode(null);
                        setTensorList([]);
                        setSelectedTensorData(null);
                        setViewingTensor(false);
                      }}
                      className={`px-3 py-1.5 rounded text-xs font-medium whitespace-nowrap transition-all ${
                        activeModuleTab === idx
                          ? 'bg-cyan-400/30 text-cyan-300 border border-cyan-400/50'
                          : 'bg-gray-700/30 text-gray-400 border border-gray-600/30 hover:bg-gray-700/50 hover:text-gray-300'
                      }`}
                    >
                      {name}
                    </button>
                  ))}
                </div>
              )}

              {/* Toolbar */}
              {currentModuleGraph && (
                <div className="flex items-center justify-between px-4 py-2 border-b border-white/5 bg-gray-800/20">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-400">Total params:</span>
                    <span className="text-xs font-mono text-cyan-300">
                      {currentModuleGraph.total_params?.toLocaleString() || 'N/A'}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={handleExpandAll}
                      className="px-2 py-1 rounded text-xs font-medium bg-cyan-600 hover:bg-cyan-500 text-white transition-colors"
                      title="Expand all modules"
                    >
                      Expand
                    </button>
                    <button
                      onClick={handleCollapseAll}
                      className="px-2 py-1 rounded text-xs font-medium bg-gray-700 hover:bg-gray-600 text-white transition-colors"
                      title="Collapse all modules"
                    >
                      Collapse
                    </button>
                    <button
                      onClick={handleFitView}
                      className="px-2 py-1 rounded text-xs font-medium bg-gray-700 hover:bg-gray-600 text-white transition-colors"
                      title="Fit view"
                    >
                      Fit
                    </button>
                    <div className="w-px h-4 bg-white/10 mx-1"></div>
                    <button
                      onClick={handleZoomOut}
                      className="p-1.5 rounded hover:bg-white/10 transition-colors"
                      title="Zoom out"
                    >
                      <ZoomOut className="w-3.5 h-3.5 text-gray-400" />
                    </button>
                    <span className="text-xs text-gray-400 w-12 text-center">{Math.round(zoomLevel * 100)}%</span>
                    <button
                      onClick={handleZoomIn}
                      className="p-1.5 rounded hover:bg-white/10 transition-colors"
                      title="Zoom in"
                    >
                      <ZoomIn className="w-3.5 h-3.5 text-gray-400" />
                    </button>
                    <button
                      onClick={handleZoomReset}
                      className="p-1.5 rounded hover:bg-white/10 transition-colors"
                      title="Reset zoom"
                    >
                      <Maximize2 className="w-3.5 h-3.5 text-gray-400" />
                    </button>
                  </div>
                </div>
              )}

              {/* Main content area - horizontal layout */}
              <div className="flex-1 flex overflow-hidden">
                {/* Left panel - Module Diagram */}
                <div className="flex-1 overflow-hidden">
                  <ModuleDiagram
                    nodes={currentModuleGraph?.nodes || []}
                    edges={currentModuleGraph?.edges || []}
                    selectedNode={selectedNode}
                    onNodeClick={handleNodeClick}
                    onPaneClick={handleDiagramPaneClick}
                    onExpandAllChange={(fn) => { expandAllRef.current = fn; }}
                    onCollapseAllChange={(fn) => { collapseAllRef.current = fn; }}
                    onFitViewChange={(fn) => { fitViewRef.current = fn; }}
                    onViewTensor={handleNodeClick}
                  />
                </div>

                {/* Resizer between graph and tensor panel */}
                {viewingTensor && tensorList.length > 0 && selectedTensorData && (
                  <div
                    className="group relative w-1 bg-gray-700 hover:bg-cyan-500 cursor-col-resize transition-colors"
                    onMouseDown={handleStartResizeTensorPanel}
                  >
                    <div className="absolute left-1/2 -translate-x-1/2 top-1/2 -translate-y-1/2 hidden group-hover:flex items-center justify-center h-12 w-8 bg-cyan-500/20 rounded">
                      <div className="flex gap-0.5">
                        <div className="w-0.5 h-4 bg-cyan-400 rounded-full" />
                        <div className="w-0.5 h-4 bg-cyan-400 rounded-full" />
                        <div className="w-0.5 h-4 bg-cyan-400 rounded-full" />
                      </div>
                    </div>
                  </div>
                )}

                {/* Right panel - Tensor Viewer */}
                {viewingTensor && tensorList.length > 0 && selectedTensorData && (
                  <div 
                    className="border-l border-gray-200 dark:border-white/5 bg-white/30 dark:bg-black/20 backdrop-blur-xl transition-all duration-300 overflow-hidden flex flex-col relative"
                    style={{ width: `${tensorPanelWidth}px` }}
                  >
                    {/* Expand/Collapse Button */}
                    <button
                      onClick={handleToggleTensorPanelExpand}
                      className="absolute -left-8 top-16 z-10 p-2 rounded-lg bg-gray-800/80 hover:bg-gray-700 border border-white/10 hover:border-cyan-400/50 transition-all"
                      title={isTensorPanelExpanded ? 'Collapse tensor panel' : 'Expand tensor panel'}
                    >
                      {isTensorPanelExpanded ? (
                        <ChevronRight className="w-4 h-4 text-cyan-400" />
                      ) : (
                        <ChevronLeft className="w-4 h-4 text-gray-400 hover:text-cyan-400" />
                      )}
                    </button>

                    <TensorViewer
                      tensors={tensorList}
                      selectedTensorData={selectedTensorData}
                      moduleName={selectedNode || ''}
                      filename={selectedFile || ''}
                      modelId={selectedModelId || undefined}
                      onClose={handleCloseTensorViewer}
                      onRefresh={handleRefreshTensor}
                    />
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center p-8">
              <div className="flex flex-col items-center gap-4 text-center max-w-md">
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-blue-500/20 flex items-center justify-center border border-cyan-400/20">
                  <svg
                    className="w-10 h-10 text-cyan-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
                    />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">
                    PyTorch Module Visualizer
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Select a .pkl file from the left panel to visualize the module hierarchy.
                    Click on nodes with weights to view tensor data.
                  </p>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      <StatusBar
        sourceDir={sourceDir}
        fileCount={files.length}
        totalParams={moduleGraph?.total_params || null}
        selectedModule={selectedNode}
      />

      <SettingsPanel isOpen={settingsOpen} onClose={handleCloseSettings} />
    </div>
  );
};

const App: React.FC = () => {
  return (
    <ThemeProvider>
      <SettingsProvider>
        <AppContent />
      </SettingsProvider>
    </ThemeProvider>
  );
};

export default App;