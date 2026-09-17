import React, { useMemo, useState, useEffect, useCallback } from 'react';
import { AgGridReact } from 'ag-grid-react';
import { AllCommunityModule, ModuleRegistry, themeQuartz, colorSchemeDark } from 'ag-grid-community';
import type { ColDef } from 'ag-grid-community';
import { X, Box, AlertCircle, RotateCcw, Download, BarChart2, TrendingUp, Percent, Key } from 'lucide-react';
import type { TensorData, TensorInfo, SliceRequest } from '../types';
import { useTheme } from '../contexts/ThemeContext';
import { useSettings } from '../contexts/SettingsContext';
import { api } from '../services/api';

ModuleRegistry.registerModules([AllCommunityModule]);

interface TensorViewerProps {
  tensors: TensorInfo[];
  selectedTensorData: TensorData | null;
  moduleName: string;
  filename: string;
  modelId?: string;
  onClose: () => void;
  onRefresh: () => void;
}

const darkTheme = themeQuartz.withPart(colorSchemeDark).withParams({
  backgroundColor: 'rgba(0, 0, 0, 0.4)',
  foregroundColor: '#e5e7eb',
  headerBackgroundColor: 'rgba(0, 0, 0, 0.6)',
  headerTextColor: '#9ca3af',
  oddRowBackgroundColor: 'rgba(255, 255, 255, 0.02)',
  accentColor: '#22d3ee',
});

const lightTheme = themeQuartz.withParams({
  backgroundColor: 'rgba(255, 255, 255, 0.9)',
  foregroundColor: '#1f2937',
  headerBackgroundColor: 'rgba(243, 244, 246, 1)',
  headerTextColor: '#6b7280',
  oddRowBackgroundColor: 'rgba(0, 0, 0, 0.02)',
  accentColor: '#0891b2',
});

// Advanced slice parser supporting Python tensor syntax
/**
 * Full Python-style slice parser.
 * Supports:
 *   - Single indices: 0, -1
 *   - Slice notation: 1:10, ::2, ::-1, 0:100:-1
 *   - List of indices: [0, 5, 99]
 *   - Full tensor syntax including genome (first) dim
 * e.g. "0:100:-1, 1:4, ::-1, 0, :4"  or  "[0, 99], :, :4"
 */
const parseSlice = (input: string): { valid: boolean; spec?: string; error?: string } => {
  const trimmed = input.trim();
  if (!trimmed) {
    return { valid: true, spec: undefined };
  }

  try {
    // Strip outer [...] wrapper if present (but NOT inner lists)
    let content = trimmed;
    if (/^\[.*\]$/.test(content)) {
      // Only strip if the outer brackets are not a list-of-indices for a single dim
      // Check by finding matching closing bracket
      let depth = 0;
      let isOuterWrapper = false;
      for (let i = 0; i < content.length; i++) {
        if (content[i] === '[') depth++;
        if (content[i] === ']') {
          depth--;
          if (depth === 0 && i === content.length - 1) {
            // Outer brackets span the whole string - strip them
            // But only if the inside has commas at depth 0 (i.e. multiple dims)
            const inner = content.slice(1, -1);
            let d2 = 0;
            let hasTopComma = false;
            for (const ch of inner) {
              if (ch === '[') d2++;
              else if (ch === ']') d2--;
              else if (ch === ',' && d2 === 0) { hasTopComma = true; break; }
            }
            if (hasTopComma) {
              content = inner;
              isOuterWrapper = true;
            }
            break;
          }
        }
      }
    }

    // Split on commas at depth 0
    const parts: string[] = [];
    let depth = 0;
    let cur = '';
    for (const ch of content) {
      if (ch === '[') depth++;
      else if (ch === ']') depth--;
      if (ch === ',' && depth === 0) {
        parts.push(cur.trim());
        cur = '';
      } else {
        cur += ch;
      }
    }
    if (cur.trim()) parts.push(cur.trim());

    for (const part of parts) {
      if (part === '' || part === ':') continue;

      // Single integer (positive or negative)
      if (/^-?\d+$/.test(part)) continue;

      // Slice notation: optional_int : optional_int (: optional_int)?
      // covers :, 1:, :5, 1:5, ::2, 1::2, :5:2, 1:5:2, ::-1, etc.
      if (/^-?\d*:-?\d*(?::-?\d*)?$/.test(part)) continue;

      // List of integers: [1, 2, 3] or [-1, 0, 5]
      if (/^\[-?\d+(?:\s*,\s*-?\d+)*\]$/.test(part)) continue;

      return { valid: false, error: `Invalid slice part: "${part}"` };
    }

    return { valid: true, spec: `[${content}]` };
  } catch (e) {
    return { valid: false, error: 'Parse error' };
  }
};

/**
 * Parse genome key input: a single integer key or a list [k1, k2, ...]
 * Returns array of numeric keys or null on error.
 */
const parseGenomeKeys = (input: string): { keys: number[] | null; error?: string } => {
  const trimmed = input.trim();
  if (!trimmed) return { keys: null };

  // Single key
  if (/^-?\d+$/.test(trimmed)) {
    return { keys: [parseInt(trimmed, 10)] };
  }

  // List of keys
  if (/^\[-?\d+(?:\s*,\s*-?\d+)*\]$/.test(trimmed)) {
    const inner = trimmed.slice(1, -1);
    const keys = inner.split(',').map(k => parseInt(k.trim(), 10));
    return { keys };
  }

  return { keys: null, error: 'Expected a genome key like 12345 or a list [12345, 67890]' };
};

const formatNumber = (num: number | null | undefined): string => {
  if (num === null || num === undefined) return 'N/A';
  if (Math.abs(num) < 0.0001 && num !== 0) {
    return num.toExponential(3);
  }
  return num.toFixed(4);
};

export const TensorViewer: React.FC<TensorViewerProps> = ({
  tensors,
  selectedTensorData,
  moduleName,
  filename,
  modelId,
  onClose,
  onRefresh,
}) => {
  const { theme } = useTheme();
  const { settings } = useSettings();
  const [selectedTensorIdx, setSelectedTensorIdx] = useState(0);
  const [sliceInput, setSliceInput] = useState('');
  const [sliceError, setSliceError] = useState<string | null>(null);
  const [tableData, setTableData] = useState<TensorData | null>(selectedTensorData);
  const [showAllDims, setShowAllDims] = useState(false);
  const [loading, setLoading] = useState(false);
  const [selectedGenomeKey, setSelectedGenomeKey] = useState<number | undefined>(undefined);
  const [useGenomeKeyMode, setUseGenomeKeyMode] = useState(false);
  const [genomeKeyInput, setGenomeKeyInput] = useState('');
  const [genomeKeyError, setGenomeKeyError] = useState<string | null>(null);

  // Reset when tensor selection changes
  useEffect(() => {
    setSelectedTensorIdx(0);
    setSliceInput('');
    setSliceError(null);
    setSelectedGenomeKey(undefined);
    setGenomeKeyInput('');
    setGenomeKeyError(null);
  }, [moduleName]);

  // Genome keys sorted by index (ascending index = ascending position in list)
  const sortedGenomeEntries = useMemo(() => {
    if (!tableData?.has_genome_dim || !tableData?.genome_keys || !tableData?.genome_key_map) return [];
    // genome_key_map: { [key: number]: index }
    const map = tableData.genome_key_map as Record<number, number>;
    return Object.entries(map)
      .map(([key, idx]) => ({ key: parseInt(key), idx }))
      .sort((a, b) => a.idx - b.idx);
  }, [tableData]);

  useEffect(() => {
    const fetchData = async () => {
      if (!filename || tensors.length === 0) {
        setTableData(selectedTensorData);
        return;
      }

      // Validate slice
      const parseResult = parseSlice(sliceInput);
      if (!parseResult.valid) {
        setSliceError(parseResult.error || 'Invalid slice syntax');
        return;
      }
      setSliceError(null);

      // Validate genome keys if in key mode
      let resolvedGenomeKeys: number[] | undefined;
      if (useGenomeKeyMode && genomeKeyInput.trim()) {
        const { keys, error } = parseGenomeKeys(genomeKeyInput);
        if (error || !keys) {
          setGenomeKeyError(error || 'Invalid genome key input');
          return;
        }
        // Validate keys exist in map
        if (tableData?.genome_key_map) {
          const map = tableData.genome_key_map as Record<number, number>;
          const notFound = keys.filter(k => !(k in map));
          if (notFound.length > 0) {
            setGenomeKeyError(`Genome key(s) not found: ${notFound.join(', ')}`);
            return;
          }
          resolvedGenomeKeys = keys;
        }
      }
      setGenomeKeyError(null);

      setLoading(true);
      try {
        const request: SliceRequest = {
          filename,
          module_path: moduleName,
          tensor_idx: selectedTensorIdx,
          slice_spec: parseResult.spec,
          max_display: showAllDims ? undefined : settings.maxDisplay,
          genome_key: selectedGenomeKey,
          genome_keys: resolvedGenomeKeys,
          use_genome_key_mode: useGenomeKeyMode,
          model_id: modelId,
        };
        const data = await api.getTensor(request);
        setTableData(data);
      } catch (err) {
        setSliceError(err instanceof Error ? err.message : 'Failed to fetch tensor');
        setTableData(null);
      } finally {
        setLoading(false);
      }
    };

    const timeout = setTimeout(fetchData, 300);
    return () => clearTimeout(timeout);
  }, [sliceInput, selectedTensorIdx, showAllDims, filename, moduleName, modelId, selectedTensorData, tensors, settings.maxDisplay, selectedGenomeKey, useGenomeKeyMode, genomeKeyInput]);

  const heatmapColor = useCallback(
    (value: number, min: number, max: number): string => {
      const hasNaNInf = tableData?.has_nan || tableData?.has_inf;
      if (hasNaNInf || min === max) {
        return 'transparent';
      }

      const normalized = (value - min) / (max - min);

      // Get RGB values from hex colors
      const hexToRgb = (hex: string) => {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result
          ? {
              r: parseInt(result[1], 16),
              g: parseInt(result[2], 16),
              b: parseInt(result[3], 16),
            }
          : { r: 0, g: 0, b: 0 };
      };

      const low = hexToRgb(settings.heatmapLowColor);
      const mid = hexToRgb(settings.heatmapMidColor);
      const high = hexToRgb(settings.heatmapHighColor);

      let r: number, g: number, b: number;

      if (normalized < 0.5) {
        const t = normalized * 2;
        r = Math.round(low.r + (mid.r - low.r) * t);
        g = Math.round(low.g + (mid.g - low.g) * t);
        b = Math.round(low.b + (mid.b - low.b) * t);
      } else {
        const t = (normalized - 0.5) * 2;
        r = Math.round(mid.r + (high.r - mid.r) * t);
        g = Math.round(mid.g + (high.g - mid.g) * t);
        b = Math.round(mid.b + (high.b - mid.b) * t);
      }

      return `rgba(${r}, ${g}, ${b}, 0.4)`;
    },
    [tableData, settings]
  );

  const exportCSV = useCallback(() => {
    if (!tableData?.values) return;

    const csvRows = tableData.values.map(row => row.join(','));
    const csvContent = csvRows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${moduleName.replace(/\./g, '_')}_tensor.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [tableData, moduleName]);

  const rowData = useMemo(() => {
    if (!tableData?.values) return [];

    if (tableData.is_scalar) {
      return [{ id: 0, col_0: tableData.scalar_value }];
    }

    return tableData.values.map((row, idx) => ({
      id: idx,
      ...row.reduce((acc: Record<string, number>, val, colIdx) => {
        acc[`col_${colIdx}`] = val;
        return acc;
      }, {}),
    }));
  }, [tableData]);

  const columnDefs = useMemo(() => {
    const defs: ColDef[] = [];
    if (!tableData) return defs;
    
    if (tableData.is_scalar) {
      return [
        {
          field: 'col_0',
          headerName: 'Value',
          width: 120,
          type: 'numericColumn',
          valueFormatter: (params: { value: number }) => {
            if (typeof params.value === 'number') {
              if (isNaN(params.value)) return 'NaN';
              if (!isFinite(params.value)) return 'Inf';
              return params.value.toFixed(6);
            }
            return '';
          },
          cellClass: 'cell-center',
        },
      ];
    }

    if (!tableData.values || tableData.values.length === 0) return defs;
    const numCols = tableData.values[0]?.length || 0;
    const showHeatmap = settings.showHeatmap && !tableData.has_nan && !tableData.has_inf;
    const min = tableData.min_val ?? 0;
    const max = tableData.max_val ?? 0;

    // Row index column (left axis)
    const colDefs: ColDef[] = [
      {
        field: 'id',
        // headerName: '',
        width: 44,
        pinned: 'left',
        cellClass: 'cell-center',
        headerClass: 'text-xs',
        cellStyle: { color: '#6b7280', fontSize: '10px', textAlign: 'right', paddingRight: '4px' },
      },
    ];

    colDefs.push(...Array.from({ length: numCols }, (_, i) => ({
      field: `col_${i}`,
      headerName: `${i}`,
      width: 62,
      type: 'numericColumn',
      headerClass: 'text-xs',
      valueFormatter: (params: { value: number }) => {
        if (typeof params.value === 'number') {
          if (isNaN(params.value)) return 'NaN';
          if (!isFinite(params.value)) return 'Inf';
          return params.value.toFixed(3);
        }
        return '';
      },
      cellClass: 'cell-center',
      cellStyle: showHeatmap
        ? (params: any) => {
            const value = params.value;
            if (typeof value !== 'number' || isNaN(value) || !isFinite(value)) {
              return { backgroundColor: 'rgba(255, 0, 0, 0.2)', fontSize: '10px' };
            }
            return { backgroundColor: heatmapColor(value, min, max), fontSize: '10px' };
          }
        : { fontSize: '10px' },
    } as ColDef)));

    return colDefs;
  }, [tableData, settings.showHeatmap, heatmapColor]);

  if (tensors.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500 dark:text-gray-400">
        <Box className="w-6 h-6 mr-2 opacity-50" />
        <span>No tensors in this module</span>
      </div>
    );
  }

  const currentTensor = tensors[selectedTensorIdx];
  const shapeDisplay = currentTensor ? `[${currentTensor.shape.join(', ')}]` : '';
  // const hasMoreDims = currentTensor && currentTensor.shape.length > 2; // TODO: was made irrelevant by recent edits. Fix for column and row truncation

  return (
    <div className="h-full flex flex-col p-2.5 w-full text-xs">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <Box className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
          <h3 className="text-xs font-semibold text-gray-800 dark:text-white truncate">
            {moduleName.split('.').pop() || moduleName}
          </h3>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={exportCSV}
            disabled={!tableData?.values}
            className="p-1 rounded-lg hover:bg-white/10 transition-colors group disabled:opacity-30"
            aria-label="Export to CSV"
          >
            <Download className="w-3 h-3 text-gray-400 group-hover:text-cyan-400" />
          </button>
          <button
            onClick={onRefresh}
            className="p-1 rounded-lg hover:bg-white/10 transition-colors group"
            aria-label="Refresh tensor"
          >
            <RotateCcw className="w-3 h-3 text-gray-400 group-hover:text-cyan-400" />
          </button>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-white/10 transition-colors group"
            aria-label="Close tensor viewer"
          >
            <X className="w-3 h-3 text-gray-400 group-hover:text-white" />
          </button>
        </div>
      </div>

      {/* Tensor selector tabs */}
      <div className="flex gap-0.5 mb-1.5 flex-wrap items-center border-b border-white/10 pb-1 overflow-x-auto">
        {tensors.map((tensor, idx) => (
          <button
            key={idx}
            onClick={() => setSelectedTensorIdx(idx)}
            className={`px-2 py-1 rounded-t text-xs font-medium transition-all duration-200 whitespace-nowrap
              ${selectedTensorIdx === idx
                ? 'bg-cyan-500/20 text-cyan-300 border-b-2 border-cyan-400'
                : 'text-gray-400 border-b-2 border-transparent hover:text-gray-200 hover:bg-white/5'
              }`}
            title={`${tensor.name}: Shape ${JSON.stringify(tensor.shape)}`}
          >
            <span className="font-semibold text-xs">{tensor.name}</span>
            <span className="text-gray-500 text-xs ml-0.5">[{tensor.shape.join(', ')}]</span>
          </button>
        ))}
      </div>

      {/* Genome key selector — sorted by index ascending, shows key labels */}
      {tableData?.has_genome_dim && sortedGenomeEntries.length > 0 && (
        <div className="flex items-center gap-2 mb-1.5 px-2 py-1 rounded-lg bg-amber-500/10 border border-amber-400/20">
          <span className="text-xs text-amber-400">Genome:</span>
          <select
            value={selectedGenomeKey ?? sortedGenomeEntries[0]?.key}
            onChange={(e) => setSelectedGenomeKey(parseInt(e.target.value))}
            className="px-2 py-0.5 text-xs bg-gray-800/50 border border-amber-400/30 rounded text-amber-300 focus:outline-none focus:border-amber-400/50"
          >
            {sortedGenomeEntries.map(({ key, idx }) => (
              <option key={key} value={key}>
                key {key} (idx {idx})
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Stats display */}
      {tableData && (
        <div className="grid grid-cols-4 gap-1 mb-1.5">
          <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-gray-800/30 border border-white/5">
            <TrendingUp className="w-2 h-2 text-cyan-400 flex-shrink-0" />
            <span className="text-xs text-gray-400">Min:</span>
            <span className="text-xs font-mono text-cyan-300">{formatNumber(tableData.min_val)}</span>
          </div>
          <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-gray-800/30 border border-white/5">
            <TrendingUp className="w-2 h-2 text-cyan-400 rotate-180 flex-shrink-0" />
            <span className="text-xs text-gray-400">Max:</span>
            <span className="text-xs font-mono text-cyan-300">{formatNumber(tableData.max_val)}</span>
          </div>
          <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-gray-800/30 border border-white/5">
            <BarChart2 className="w-2 h-2 text-cyan-400 flex-shrink-0" />
            <span className="text-xs text-gray-400">Mean:</span>
            <span className="text-xs font-mono text-cyan-300">{formatNumber(tableData.mean_val)}</span>
          </div>
          <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-gray-800/30 border border-white/5">
            <Percent className="w-2 h-2 text-cyan-400 flex-shrink-0" />
            <span className="text-xs text-gray-400">Sparse:</span>
            <span className="text-xs font-mono text-cyan-300">
              {tableData.sparsity !== null ? `${tableData.sparsity.toFixed(1)}%` : 'N/A'}
            </span>
          </div>
        </div>
      )}

      {/* Shape info */}
      <div className="flex items-center gap-2 mb-1.5 p-1 rounded-lg bg-gray-800/50 border border-white/5 text-xs">
        <span className="text-gray-400">Shape:</span>
        <span className="font-mono text-cyan-300">{shapeDisplay}</span>
        <span className="text-gray-500">({currentTensor?.dtype})</span>
        {currentTensor?.numel && (
          <span className="text-gray-500 ml-auto">
            {currentTensor.numel.toLocaleString()} elements
          </span>
        )}
      </div>

      {/* TODO: Section moved from above scalar display. Ensure it does not work */}
      {/* Warnings */}
      {(tableData?.has_nan || tableData?.has_inf) && (
        <div className="mb-1.5 px-2 py-1 rounded-lg bg-yellow-500/10 border border-yellow-400/20 flex items-center gap-2">
          <AlertCircle className="w-3 h-3 text-yellow-400 flex-shrink-0" />
          <span className="text-xs text-yellow-300">
            Contains {tableData?.has_nan ? 'NaN' : ''}{tableData?.has_nan && tableData?.has_inf ? ' and ' : ''}{tableData?.has_inf ? 'Inf' : ''} values.
          </span>
        </div>
      )}

      {/* Grid - Takes remaining space */}
      <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-white/10">
        {loading ? (
          <div className="h-full flex items-center justify-center">
            <div className="animate-spin w-4 h-4 border-2 border-cyan-400 border-t-transparent rounded-full" />
          </div>
        ) : sliceError ? (
          <div className="h-full flex items-center justify-center text-gray-500 text-xs">
            Enter valid slice to display tensor
          </div>
        ) : tableData?.values || tableData?.is_scalar ? (
          <div className="h-full ag-theme-quartz" style={{ height: '100%' }}>
            <AgGridReact
              rowData={rowData}
              columnDefs={columnDefs}
              theme={theme === 'dark' ? darkTheme : lightTheme}
              defaultColDef={{
                sortable: false,
                filter: false,
                resizable: true,
              }}
              animateRows={false}
              headerHeight={24}
              rowHeight={22}
              enableCellTextSelection
            />
          </div>
        ) : tableData?.values === null && !tableData?.is_scalar ? (
          <div className="h-full flex items-center justify-center text-gray-500 text-xs">
            Tensor too large or slice returned empty
          </div>
        ) : null}
      </div>

      {/* Slice input — at the bottom */}
      <div className="mt-2 pt-2 border-t border-white/10">
        <div className="flex items-center gap-1.5 mb-1">
          <span className="text-xs text-gray-400 flex-shrink-0">Slice:</span>
          <input
            type="text"
            value={sliceInput}
            onChange={(e) => setSliceInput(e.target.value)}
            placeholder="e.g., 0:100:-1, :, :4  or  [0,99], :, ::-1"
            className="flex-1 min-w-0 px-2 py-1 text-xs bg-gray-800/50 border border-white/10 rounded text-gray-200 placeholder-gray-500 focus:outline-none focus:border-cyan-400/50 font-mono"
          />
          {/* Genome key mode toggle */}
          <button
            onClick={() => setUseGenomeKeyMode(!useGenomeKeyMode)}
            className={`flex-shrink-0 flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors ${
              useGenomeKeyMode
                ? 'bg-amber-500/20 border border-amber-400/50 text-amber-300'
                : 'bg-gray-800/50 border border-white/10 text-gray-400 hover:border-white/20'
            }`}
            title="Genome key indexing mode: pass genome keys instead of indices for first dim"
          >
            <Key className="w-3 h-3" />
          </button>
        </div>

        {/* Genome key input (visible when key mode enabled) */}
        {useGenomeKeyMode && (
          <input
            type="text"
            value={genomeKeyInput}
            onChange={(e) => setGenomeKeyInput(e.target.value)}
            placeholder="Genome key(s) for dim 0, e.g. 12345 or [12345, 67890]"
            className="w-full px-2 py-1 text-xs bg-gray-800/50 border border-amber-400/30 rounded text-amber-300 placeholder-gray-500 focus:outline-none focus:border-amber-400/50 font-mono mb-1"
          />
        )}

        {/* Error display */}
        {sliceError && (
          <div className="mt-1 px-2 py-1 rounded-lg bg-red-500/10 border border-red-400/20 flex items-center gap-1.5">
            <AlertCircle className="w-3 h-3 text-red-400 flex-shrink-0" />
            <span className="text-xs text-red-300">{sliceError}</span>
          </div>
        )}

        {genomeKeyError && (
          <div className="mt-1 px-2 py-1 rounded-lg bg-red-500/10 border border-red-400/20 flex items-center gap-1.5">
            <AlertCircle className="w-3 h-3 text-red-400 flex-shrink-0" />
            <span className="text-xs text-red-300">{genomeKeyError}</span>
          </div>
        )}
      </div>

      {/*// TODO: Ensures scalar display when a slice input return a single scalar value*/}
      {/* Scalar display */}
      {tableData?.is_scalar && (
          <div className="mb-3 p-4 rounded-lg bg-cyan-500/10 border border-cyan-400/20 text-center">
          <span className="text-2xl font-mono text-cyan-300">
            {tableData.scalar_value?.toFixed(6)}
          </span>
          </div>
      )}

      {/* TODO: Old implementation that truncates long tensors displayed, at given dimension */}
      {/* Slice input */}
      {/* {hasMoreDims && (
        <div className="mb-2">
          <div className="flex items-center gap-2 mb-1">
            <input
              type="text"
              value={sliceInput}
              onChange={(e) => setSliceInput(e.target.value)}
              placeholder="e.g., [:100, 10:20, 0] or [0, :100, :64]"
              className="flex-1 px-3 py-1.5 text-xs bg-gray-800/50 border border-white/10 rounded-lg text-gray-200 placeholder-gray-500 focus:outline-none focus:border-cyan-400/50 font-mono"
            />
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500">
              Showing last 2 dims. Use slice notation to select different indices.
            </span>
            <button
              onClick={() => setShowAllDims(!showAllDims)}
              className="flex items-center gap-1 px-2 py-1 text-xs text-cyan-400 hover:text-cyan-300 transition-colors"
            >
              {showAllDims ? (
                <>
                  <ChevronUp className="w-3 h-3" />
                  Limit display
                </>
              ) : (
                <>
                  <ChevronDown className="w-3 h-3" />
                  Show all
                </>
              )}
            </button>
          </div>
        </div>
      )} */}
    </div>
  );
};
