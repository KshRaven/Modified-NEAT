import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  applyNodeChanges,
  NodeResizer,
  addEdge,
  type Connection,
  type Node,
  type Edge,
  MarkerType,
  BackgroundVariant,
  Handle,
  Position,
  useReactFlow,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Box, Layers, ChevronDown, ChevronUp, Search, Maximize2, ChevronsDown, ChevronsUp, Eye, EyeOff } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import type { ModuleNode as ModuleNodeType } from '../types';

// TODO: Fix Dark mode not getting enabled in built files error
// TODO: Fix Minimap not getting shrunk accordingly on its position and instead translating off module diagram panel along with color scheme error

// ─── Types ──────────────────────────────────────────────────────────────────

interface ModuleDiagramProps {
  nodes: ModuleNodeType[];
  edges: { id: string; source: string; target: string }[];
  selectedNode: string | null;
  onNodeClick: (nodeId: string) => void;
  onPaneClick?: () => void;
  onExpandAllChange?: (callback: () => void) => void;
  onCollapseAllChange?: (callback: () => void) => void;
  onFitViewChange?: (callback: () => void) => void;
  onFitSelectedChange?: (callback: () => void) => void;
  /** Called when the user clicks "View Tensors" inside an expanded node */
  onViewTensor?: (nodeId: string) => void;
}

interface ExpandedState { [nodeId: string]: boolean }

// ─── Constants ───────────────────────────────────────────────────────────────

const NODE_W        = 180;   // collapsed node width
const NODE_H        = 64;    // collapsed node height
const CHILD_PAD     = 20;    // padding around children inside a bubble
const CHILD_GAP_X   = 24;    // horizontal gap between children
const CHILD_GAP_Y   = 80;    // vertical gap between child rows
const SIBLING_GAP   = 60;    // horizontal gap between sibling subtrees

// ─── Bubble-size calculation ──────────────────────────────────────────────────
/**
 * Recursively computes the total rendered width & height a node will occupy
 * when expanded (including all its expanded descendants).
 */
function subtreeSize(
  nodeId: string,
  childMap: Map<string, string[]>,
  expanded: ExpandedState,
): { w: number; h: number } {
  const isExpanded = expanded[nodeId] !== false;
  const children   = childMap.get(nodeId) ?? [];

  if (!isExpanded || children.length === 0) {
    return { w: NODE_W, h: NODE_H };
  }

  // Compute every child's subtree size
  const childSizes = children.map(c => subtreeSize(c, childMap, expanded));

  // Total inner width = sum of child widths + gaps
  const innerW = childSizes.reduce((s, { w }) => s + w, 0)
               + (children.length - 1) * CHILD_GAP_X;

  // Total inner height = max child height + room for the header row
  const innerH = Math.max(...childSizes.map(({ h }) => h)) + CHILD_GAP_Y;

  const bubbleW = Math.max(NODE_W, innerW + CHILD_PAD * 2);
  const bubbleH = NODE_H + innerH + CHILD_PAD * 2;

  return { w: bubbleW, h: bubbleH };
}

/**
 * Computes absolute {x, y} positions for every node in the tree.
 * Parent nodes are rendered as "group" nodes in ReactFlow; children are
 * positioned *relative to the parent's origin*.
 */
function computePositions(
  nodeId: string,
  absX: number,
  absY: number,
  childMap: Map<string, string[]>,
  expanded: ExpandedState,
  positions: Map<string, { x: number; y: number }>,
  parentId: Map<string, string | null>,
  sizes: Map<string, { w: number; h: number }>,
) {
  const sz = subtreeSize(nodeId, childMap, expanded);
  sizes.set(nodeId, sz);
  positions.set(nodeId, { x: absX, y: absY });

  const isExpanded = expanded[nodeId] !== false;
  const children   = childMap.get(nodeId) ?? [];

  if (isExpanded && children.length > 0) {
    const childSizes = children.map(c => subtreeSize(c, childMap, expanded));
    const totalChildW = childSizes.reduce((s, { w }) => s + w, 0)
                      + (children.length - 1) * CHILD_GAP_X;
    let cx = absX + (sz.w - totalChildW) / 2;
    const cy = absY + NODE_H + CHILD_PAD;

    children.forEach((childId, i) => {
      parentId.set(childId, nodeId);
      computePositions(childId, cx, cy, childMap, expanded, positions, parentId, sizes);
      cx += childSizes[i].w + CHILD_GAP_X;
    });
  }
}

/**
 * Build level-based positions for top-level nodes (roots) and their visible
 * subtrees. Top-level nodes without a parent are laid out side-by-side.
 */
function buildLayout(
  allNodes: ModuleNodeType[],
  allEdges: { id: string; source: string; target: string }[],
  expanded: ExpandedState,
): {
  positions: Map<string, { x: number; y: number }>;
  sizes:     Map<string, { w: number; h: number }>;
  parentId:  Map<string, string | null>;
  childMap:  Map<string, string[]>;
} {
  const childMap  = new Map<string, string[]>();
  const parentMap = new Map<string, string>();

  allNodes.forEach(n => childMap.set(n.id, []));
  allEdges.forEach(e => {
    const ch = childMap.get(e.source) ?? [];
    ch.push(e.target);
    childMap.set(e.source, ch);
    parentMap.set(e.target, e.source);
  });

  // Find true roots (no incoming edges)
  const roots = allNodes.filter(n => !parentMap.has(n.id)).map(n => n.id);

  const positions = new Map<string, { x: number; y: number }>();
  const sizes     = new Map<string, { w: number; h: number }>();
  const parentId  = new Map<string, string | null>();
  roots.forEach(r => parentId.set(r, null));

  // Compute sizes first so we can space roots
  const rootSizes = roots.map(r => subtreeSize(r, childMap, expanded));
  const totalRootW = rootSizes.reduce((s, { w }) => s + w, 0)
                   + (roots.length - 1) * SIBLING_GAP;

  let rx = -totalRootW / 2;
  roots.forEach((r, i) => {
    computePositions(r, rx, 0, childMap, expanded, positions, parentId, sizes);
    rx += rootSizes[i].w + SIBLING_GAP;
  });

  return { positions, sizes, parentId, childMap };
}

// ─── Custom node renderers ────────────────────────────────────────────────────

interface BubbleNodeData extends Record<string, unknown> {
  label: string;
  type: string;
  has_weights: boolean;
  selected: boolean;
  is_neat_module: boolean;
  is_expanded: boolean;
  has_children: boolean;
  child_count: number;
  is_container: boolean;   // true → render as translucent bubble (group)
  width: number;
  height: number;
  onToggle: () => void;
  onSelect: () => void;
  onExpandAll?: () => void;
  onCollapseAll?: () => void;
  onFit?: () => void;
  onViewTensors?: () => void;
  onResize?: (width: number, height: number) => void;
}

const BubbleNode: React.FC<{ id: string; data: BubbleNodeData }> = ({ id, data }) => {
  const { theme } = useTheme();
  const {
    label, type, has_weights, selected, is_neat_module,
    has_children, child_count,
    is_container, width, height,
    onToggle, onSelect, onExpandAll, onCollapseAll, onFit, onViewTensors, onResize,
  } = data;
  const toggleClickTimer = useRef<number | null>(null);
  const handleToggleClick = useCallback(() => {
    if (toggleClickTimer.current !== null) window.clearTimeout(toggleClickTimer.current);
    toggleClickTimer.current = window.setTimeout(() => {
      toggleClickTimer.current = null;
      onToggle();
    }, 220);
  }, [onToggle]);
  const handleDoubleClick = useCallback(() => {
    if (toggleClickTimer.current !== null) {
      window.clearTimeout(toggleClickTimer.current);
      toggleClickTimer.current = null;
    }
    onSelect();
  }, [onSelect]);

  const actionBar = selected && onExpandAll && onCollapseAll && onFit ? (
    <div
      className="absolute right-2 top-2 z-20 flex items-center gap-0.5 rounded-md border border-cyan-400/30 bg-gray-950/90 p-0.5 shadow-lg backdrop-blur-md"
      onClick={(event) => event.stopPropagation()}
      onDoubleClick={(event) => event.stopPropagation()}
    >
      <button
        type="button"
        onClick={onExpandAll}
        className="rounded p-1 text-cyan-300 transition-colors hover:bg-cyan-400/20"
        title="Expand all inner modules"
        aria-label="Expand all inner modules"
      >
        <ChevronsDown className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        onClick={onCollapseAll}
        className="rounded p-1 text-gray-300 transition-colors hover:bg-white/10"
        title="Collapse all inner modules"
        aria-label="Collapse all inner modules"
      >
        <ChevronsUp className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        onClick={onFit}
        className="rounded p-1 text-gray-300 transition-colors hover:bg-white/10"
        title="Fit selected module"
        aria-label="Fit selected module"
      >
        <Maximize2 className="h-3.5 w-3.5" />
      </button>
    </div>
  ) : null;

  // ── Container (expanded bubble) ───────────────────────────────────────────
  if (is_container) {
    return (
      <div
        style={{ width: '100%', height: '100%', position: 'relative' }}
        onClick={onSelect}
        className={`rounded-2xl border-2 transition-colors duration-300
          ${selected
            ? 'border-cyan-400/70 bg-cyan-500/5 shadow-xl shadow-cyan-500/20'
            : has_weights
              ? theme === 'dark'
                ? 'border-cyan-400/30 bg-gray-900/40 hover:border-cyan-400/50'
                : 'border-cyan-700/40 bg-white/65 hover:border-cyan-700/60'
              : theme === 'dark'
                ? 'border-white/10 bg-gray-800/30 hover:border-white/20'
                : 'border-gray-300/80 bg-white/55 hover:border-gray-400'
          }`}
      >
        {actionBar}
        <NodeResizer
          nodeId={id}
          minWidth={NODE_W}
          minHeight={NODE_H + CHILD_PAD * 2 + NODE_H}
          isVisible={selected}
          onResize={(_, params) => onResize?.(params.width, params.height)}
          lineClassName="!border-cyan-400/60"
          handleClassName="!w-2.5 !h-2.5 !bg-cyan-400 !border-gray-950"
        />
        {/* Top handle */}
        <Handle
          type="target"
          position={Position.Top}
          className="!w-3 !h-3 !bg-cyan-400 !border-none"
          style={{ top: 0 }}
        />

        {/* Header bar */}
        <div
          className={`flex items-center justify-between px-3 py-2 cursor-pointer rounded-t-2xl
            ${selected ? 'bg-cyan-500/20' : theme === 'dark' ? 'bg-black/20' : 'bg-white/45'}`}
          onClick={handleToggleClick}
          onDoubleClick={handleDoubleClick}
          style={{ height: NODE_H }}
        >
          <div className="flex items-center gap-2 min-w-0">
            {has_weights
              ? <Box className="w-4 h-4 text-cyan-400 shrink-0" />
              : <Layers className={`w-4 h-4 shrink-0 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`} />}
            <div className="min-w-0">
              <div className={`text-xs font-medium truncate ${selected ? 'text-cyan-300' : theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                {type}
              </div>
              <div className={`text-sm font-semibold truncate ${selected ? 'text-white' : theme === 'dark' ? 'text-gray-200' : 'text-gray-800'}`}>
                {label || 'root'}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0 ml-2">
            {/* View tensors button */}
            {has_weights && onViewTensors && (
              <button
                onClick={(e) => { e.stopPropagation(); onViewTensors(); }}
                className="flex items-center gap-1 px-2 py-1 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/40 border border-cyan-400/30 hover:border-cyan-400/60 text-cyan-300 text-xs font-medium transition-all"
                title="View tensors in Tensor Viewer"
              >
                <Search className="w-3 h-3" />
                <span>Tensors</span>
              </button>
            )}

            {/* Collapse toggle */}
            <button
              onClick={(e) => { e.stopPropagation(); onToggle(); }}
              className="flex items-center gap-1 text-cyan-400 hover:text-cyan-300 transition-colors"
            >
              <ChevronUp className="w-3.5 h-3.5" />
              <span className="text-xs">{child_count}</span>
            </button>
          </div>
        </div>

        {/* NeatModule indicator */}
        {is_neat_module && (
          <div className="absolute -top-1 -left-1 w-2.5 h-2.5 rounded-full bg-amber-400 shadow-lg shadow-amber-400/50" title="NeatModule" />
        )}
        {has_weights && (
          <div className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-lg shadow-cyan-400/50" />
        )}

        {/* Bottom handle */}
        <Handle
          type="source"
          position={Position.Bottom}
          className="!w-3 !h-3 !bg-cyan-400 !border-none"
          style={{ bottom: 0 }}
        />
      </div>
    );
  }

  // ── Collapsed leaf node ───────────────────────────────────────────────────
  return (
    <div
      style={{ width: NODE_W, minHeight: NODE_H, position: 'relative' }}
      className={`relative px-4 py-3 rounded-xl transition-all duration-200 cursor-pointer
        backdrop-blur-xl border
        ${selected
          ? 'bg-gradient-to-br from-cyan-500/30 to-blue-500/20 border-cyan-400/50 shadow-xl shadow-cyan-500/20 scale-105'
          : has_weights
            ? 'bg-gradient-to-br from-gray-800/80 to-gray-900/80 border-cyan-400/30 hover:border-cyan-400/50 hover:shadow-lg hover:shadow-cyan-500/10'
            : 'bg-gradient-to-br from-gray-700/60 to-gray-800/60 border-white/10 hover:border-white/30'
        } ${has_children ? 'border-2' : ''}`}
      onClick={handleToggleClick}
      onDoubleClick={handleDoubleClick}
    >
      {actionBar}
      <Handle
        type="target"
        position={Position.Top}
        className="!w-3 !h-3 !bg-cyan-400 !border-none"
      />

      <div className="flex items-center justify-between gap-2 mb-0.5">
        <div className="flex items-center gap-2 min-w-0">
          {has_weights
            ? <Box className="w-4 h-4 text-cyan-400 shrink-0" />
            : <Layers className={`w-4 h-4 shrink-0 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`} />}
          <span className={`text-xs font-medium truncate ${selected ? 'text-cyan-300' : theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
            {type}
          </span>
        </div>
        {has_children && (
          <div className="flex items-center gap-1 shrink-0">
            <ChevronDown className="w-3 h-3 text-cyan-400" />
            <span className="text-xs text-cyan-400">{child_count}</span>
          </div>
        )}
      </div>

      <div className={`font-semibold text-sm truncate ${selected ? 'text-white' : theme === 'dark' ? 'text-gray-200' : 'text-gray-800'}`}>
        {label || 'root'}
      </div>

      {/* View tensors bar — shown on selected leaf with weights */}
      {has_weights && onViewTensors && selected && (
        <button
          onClick={(e) => { e.stopPropagation(); onViewTensors(); }}
          className="mt-2 w-full flex items-center justify-center gap-1.5 py-1 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/35 border border-cyan-400/30 hover:border-cyan-400/60 text-cyan-300 text-xs font-medium transition-all"
          title="Open in Tensor Viewer"
        >
          <Search className="w-3 h-3" />
          View Tensors
        </button>
      )}

      {is_neat_module && (
        <div className="absolute -top-1 -left-1 w-2.5 h-2.5 rounded-full bg-amber-400 shadow-lg shadow-amber-400/50" title="NeatModule" />
      )}
      {has_weights && (
        <div className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-lg shadow-cyan-400/50" />
      )}

      <Handle
        type="source"
        position={Position.Bottom}
        className="!w-3 !h-3 !bg-cyan-400 !border-none"
      />
    </div>
  );
};

const nodeTypes = { bubble: BubbleNode };

// ─── Layout → ReactFlow elements ──────────────────────────────────────────────

function buildReactFlowElements(
  allNodes: ModuleNodeType[],
  allEdges: { id: string; source: string; target: string }[],
  expanded: ExpandedState,
  selectedNode: string | null,
  onToggle: (id: string) => void,
  onSelect: (id: string) => void,
  onExpandAll: (id: string) => void,
  onCollapseAll: (id: string) => void,
  onFit: (id: string) => void,
  onViewTensor: ((id: string) => void) | undefined,
): { rfNodes: Node[]; rfEdges: Edge[] } {
  const { positions, sizes, parentId, childMap } = buildLayout(allNodes, allEdges, expanded);

  const nodeMap = new Map<string, ModuleNodeType>();
  allNodes.forEach(n => nodeMap.set(n.id, n));

  // Nodes that are visible (have a computed position)
  const visible = new Set(positions.keys());

  const rfNodes: Node<BubbleNodeData>[] = [];

  positions.forEach((pos, id) => {
    const meta = nodeMap.get(id);
    if (!meta) return;

    const sz        = sizes.get(id)!;
    const isExp     = expanded[id] !== false;
    const children  = childMap.get(id) ?? [];
    const isCont    = isExp && children.length > 0;

    const parent = parentId.get(id);
    const parentPosition = parent ? positions.get(parent) : undefined;
    rfNodes.push({
      id,
      type: 'bubble',
      position: parentPosition
        ? { x: pos.x - parentPosition.x, y: pos.y - parentPosition.y }
        : pos,
      ...(parent ? { parentId: parent, extent: 'parent' as const } : {}),
      zIndex: isCont ? 0 : 1,
      style: {
        width:  isCont ? sz.w : NODE_W,
        height: isCont ? sz.h : undefined,
        // Elevated z-index for selected
        zIndex: id === selectedNode ? 10 : isCont ? 0 : 1,
      },
      data: {
        label:         meta.name,
        type:          meta.type,
        has_weights:   meta.has_weights,
        selected:      id === selectedNode,
        is_neat_module:meta.is_neat_module,
        is_expanded:   isExp,
        has_children:  children.length > 0,
        child_count:   children.length,
        is_container:  isCont,
        width:         sz.w,
        height:        sz.h,
        onToggle:      () => onToggle(id),
        onSelect:      () => onSelect(id),
        onExpandAll:   () => onExpandAll(id),
        onCollapseAll: () => onCollapseAll(id),
        onFit:         () => onFit(id),
        onViewTensors: meta.has_weights ? () => onViewTensor?.(id) : undefined,
      } as BubbleNodeData,
    });
  });

  // Edges — show only between visible nodes that are NOT a child-of-child
  // hidden inside a still-collapsed ancestor. When both endpoints are visible
  // and neither is "inside" an expanded container, show the edge normally.
  // Edges leading INTO a container node are dimmed (the interior is shown).
  const rfEdges: Edge[] = allEdges
    .filter(e => visible.has(e.source) && visible.has(e.target))
    .map(e => {
      const targetChildren = childMap.get(e.source) ?? [];
      const isInterior = (expanded[e.source] !== false) && targetChildren.includes(e.target);

      return {
        id:     e.id,
        source: e.source,
        target: e.target,
        type:   'smoothstep',
        animated: !isInterior,
        style: {
          stroke:    isInterior ? '#22d3ee40' : '#22d3ee',
          strokeWidth: isInterior ? 1 : 2,
          opacity:   isInterior ? 0.2 : 0.7,
          strokeDasharray: isInterior ? '4 4' : undefined,
        },
        markerEnd: isInterior ? undefined : {
          type:  MarkerType.ArrowClosed,
          color: '#22d3ee',
        },
      } as Edge;
    });

  return { rfNodes, rfEdges };
}

// ─── Diagram component ────────────────────────────────────────────────────────

export const ModuleDiagram: React.FC<ModuleDiagramProps> = ({
  nodes,
  edges,
  selectedNode,
  onNodeClick,
  onPaneClick,
  onExpandAllChange,
  onCollapseAllChange,
  onFitViewChange,
  onFitSelectedChange,
  onViewTensor,
}) => {
  // Initialise all nodes collapsed except the root
  const [expanded, setExpanded] = useState<ExpandedState>(() => {
    const s: ExpandedState = {};
    nodes.forEach(n => { s[n.id] = n.id === 'root'; });
    return s;
  });
  const [layoutResetVersion, setLayoutResetVersion] = useState(0);
  const appliedLayoutResetVersion = useRef(0);
  const fitSelectedRef = useRef<(() => void) | null>(null);

  // Reset expansion when the node list changes (new file loaded)
  const prevNodesKey = useRef('');
  useEffect(() => {
    const key = nodes.map(n => n.id).join(',');
    if (key !== prevNodesKey.current) {
      prevNodesKey.current = key;
      const s: ExpandedState = {};
      nodes.forEach(n => { s[n.id] = n.id === 'root'; });
      setExpanded(s);
    }
  }, [nodes]);

  const handleToggle = useCallback((nodeId: string) => {
    onNodeClick(nodeId);
    const meta = nodes.find(n => n.id === nodeId);
    const children = edges.filter(e => e.source === nodeId);
    if (meta && (meta.is_nested || children.length > 0)) {
      setExpanded(prev => ({ ...prev, [nodeId]: !prev[nodeId] }));
    }
  }, [onNodeClick, nodes, edges]);

  const handleSelect = useCallback((nodeId: string) => {
    onNodeClick(nodeId);
  }, [onNodeClick]);

  const handleViewTensor = useCallback((nodeId: string) => {
    onViewTensor?.(nodeId);
  }, [onViewTensor]);

  const handleFitSelected = useCallback(() => {
    fitSelectedRef.current?.();
  }, []);

  const expandAll = useCallback(() => {
    const s: ExpandedState = {};
    nodes.forEach(n => { s[n.id] = true; });
    setExpanded(s);
  }, [nodes]);

  const collapseAll = useCallback(() => {
    setLayoutResetVersion(version => version + 1);
    const s: ExpandedState = {};
    nodes.forEach(n => { s[n.id] = n.id === 'root'; });
    setExpanded(s);
  }, [nodes]);

  const getDescendants = useCallback((nodeId: string) => {
    const descendants = new Set<string>();
    const pending = [nodeId];
    while (pending.length > 0) {
      const current = pending.pop()!;
      edges.filter(edge => edge.source === current).forEach(edge => {
        if (!descendants.has(edge.target)) {
          descendants.add(edge.target);
          pending.push(edge.target);
        }
      });
    }
    return descendants;
  }, [edges]);

  const expandSelected = useCallback(() => {
    if (!selectedNode) return;
    const descendants = getDescendants(selectedNode);
    setExpanded(previous => {
      const next = { ...previous, [selectedNode]: true };
      descendants.forEach(id => { next[id] = true; });
      return next;
    });
  }, [getDescendants, selectedNode]);

  const collapseSelected = useCallback(() => {
    if (!selectedNode) return;
    const descendants = getDescendants(selectedNode);
    setExpanded(previous => {
      const next = { ...previous, [selectedNode]: false };
      descendants.forEach(id => { next[id] = false; });
      return next;
    });
  }, [getDescendants, selectedNode]);

  // Expose callbacks to parent (toolbar buttons)
  useEffect(() => {
    if (onExpandAllChange)  onExpandAllChange(expandAll);
    if (onCollapseAllChange) onCollapseAllChange(collapseAll);
  }, [expandAll, collapseAll, onExpandAllChange, onCollapseAllChange]);

  const { rfNodes, rfEdges } = useMemo(
    () => buildReactFlowElements(nodes, edges, expanded, selectedNode, handleToggle, handleSelect, expandSelected, collapseSelected, handleFitSelected, handleViewTensor),
    [nodes, edges, expanded, selectedNode, handleToggle, handleSelect, expandSelected, collapseSelected, handleFitSelected, handleViewTensor],
  );

  const [reactNodes, setReactNodes] = useNodesState(rfNodes);
  const [reactEdges, setReactEdges, onEdgesChange] = useEdgesState(rfEdges);

  const onNodesChange = useCallback((changes: import('@xyflow/react').NodeChange[]) => {
    setReactNodes(currentNodes => {
      const nextNodes = applyNodeChanges(changes, currentNodes);
      const byId = new Map(nextNodes.map(node => [node.id, node]));
      const dimension = (node: Node) => ({
        width: node.measured?.width ?? node.width ?? Number(node.style?.width ?? NODE_W),
        height: node.measured?.height ?? node.height ?? Number(node.style?.height ?? NODE_H),
      });

      for (let pass = 0; pass < nextNodes.length; pass += 1) {
        let changed = false;
        const childrenByParent = new Map<string, Node[]>();
        nextNodes.forEach(node => {
          if (!node.parentId) return;
          const siblings = childrenByParent.get(node.parentId) ?? [];
          siblings.push(node);
          childrenByParent.set(node.parentId, siblings);
        });

        childrenByParent.forEach((siblings, parentId) => {
          const parent = byId.get(parentId);
          if (!parent) return;
          const parentSize = dimension(parent);
          const ordered = siblings.slice().sort((left, right) => left.position.x - right.position.x);
          ordered.forEach((node, index) => {
            if (index === 0) return;
            const previous = ordered[index - 1];
            const previousSize = dimension(previous);
            const nodeSize = dimension(node);
            const overlapsVertically = node.position.y < previous.position.y + previousSize.height
              && node.position.y + nodeSize.height > previous.position.y;
            const minimumX = previous.position.x + previousSize.width + CHILD_GAP_X;
            if (overlapsVertically && node.position.x < minimumX) {
              node.position = { ...node.position, x: minimumX };
              changed = true;
            }
          });

          const requiredWidth = Math.max(
            NODE_W,
            ...siblings.map(node => node.position.x + dimension(node).width + CHILD_PAD),
          );
          const requiredHeight = Math.max(
            NODE_H + CHILD_PAD,
            ...siblings.map(node => node.position.y + dimension(node).height + CHILD_PAD),
          );
          const styledParent = parent as Node & { style?: Record<string, unknown> };
          const parentStyle = { ...styledParent.style };
          if (requiredWidth > parentSize.width) {
            parentStyle.width = requiredWidth;
            parent.width = requiredWidth;
            changed = true;
          }
          if (requiredHeight > parentSize.height) {
            parentStyle.height = requiredHeight;
            parent.height = requiredHeight;
            changed = true;
          }
          styledParent.style = parentStyle;
        });

        nextNodes.slice().reverse().forEach(node => {
          const parent = node.parentId ? byId.get(node.parentId) : undefined;
          if (!parent) return;

          const parentSize = dimension(parent);
          const nodeSize = dimension(node);
          const x = Math.max(CHILD_PAD, Math.min(node.position.x, Math.max(CHILD_PAD, parentSize.width - nodeSize.width - CHILD_PAD)));
          const y = Math.max(NODE_H, Math.min(node.position.y, Math.max(NODE_H, parentSize.height - nodeSize.height - CHILD_PAD)));
          if (x !== node.position.x || y !== node.position.y) {
            node.position = { x, y };
            changed = true;
          }

          const requiredWidth = x + nodeSize.width + CHILD_PAD;
          const requiredHeight = y + nodeSize.height + CHILD_PAD;
          const styledParent = parent as Node & { style?: Record<string, unknown> };
          const parentStyle = { ...styledParent.style };
          if (requiredWidth > parentSize.width) {
            parentStyle.width = requiredWidth;
            parent.width = requiredWidth;
            changed = true;
          }
          if (requiredHeight > parentSize.height) {
            parentStyle.height = requiredHeight;
            parent.height = requiredHeight;
            changed = true;
          }
          styledParent.style = parentStyle;
        });
        if (!changed) break;
      }

      return nextNodes;
    });
  }, [setReactNodes]);

  // Keep ReactFlow state in sync with computed layout
  useEffect(() => {
    setReactNodes(currentNodes => {
      const currentById = new Map(currentNodes.map(node => [node.id, node]));
      const shouldResetLayout = layoutResetVersion !== appliedLayoutResetVersion.current;
      if (shouldResetLayout) appliedLayoutResetVersion.current = layoutResetVersion;
      const reconciled = rfNodes.map(computed => {
        const current = currentById.get(computed.id);
        if (!current || shouldResetLayout) return computed;

        const currentStyle = current.style ?? {};
        const computedStyle = computed.style ?? {};
        const computedWidth = Number(computedStyle.width ?? NODE_W);
        const computedHeight = Number(computedStyle.height ?? NODE_H);
        const currentWidth = current.width ?? Number(currentStyle.width ?? NODE_W);
        const currentHeight = current.height ?? Number(currentStyle.height ?? NODE_H);
        const width = Math.max(currentWidth, computedWidth);
        const height = Math.max(currentHeight, computedHeight);
        return {
          ...computed,
          position: current.position,
          width,
          height,
          measured: current.measured,
          style: {
            ...computedStyle,
            width,
            height: computedStyle.height !== undefined || currentStyle.height !== undefined ? height : undefined,
          },
        };
      });

      const byId = new Map(reconciled.map(node => [node.id, node]));
      reconciled.filter(node => !currentById.has(node.id)).forEach(node => {
        if (!node.parentId) return;
        const parent = byId.get(node.parentId);
        if (!parent) return;

        const siblings = reconciled.filter(sibling => sibling.parentId === node.parentId && sibling.id !== node.id);
        const nodeWidth = node.width ?? Number(node.style?.width ?? NODE_W);
        const nodeHeight = node.height ?? Number(node.style?.height ?? NODE_H);
        let x = node.position.x;
        let y = node.position.y;
        let overlaps = true;
        while (overlaps) {
          overlaps = siblings.some(sibling => {
            const siblingWidth = sibling.width ?? Number(sibling.style?.width ?? NODE_W);
            const siblingHeight = sibling.height ?? Number(sibling.style?.height ?? NODE_H);
            return x < sibling.position.x + siblingWidth
              && x + nodeWidth > sibling.position.x
              && y < sibling.position.y + siblingHeight
              && y + nodeHeight > sibling.position.y;
          });
          if (overlaps) x += nodeWidth + CHILD_GAP_X;
        }
        node.position = { x: Math.max(CHILD_PAD, x), y: Math.max(NODE_H, y) };

        const requiredWidth = node.position.x + nodeWidth + CHILD_PAD;
        const requiredHeight = node.position.y + nodeHeight + CHILD_PAD;
        const parentWidth = parent.width ?? Number(parent.style?.width ?? NODE_W);
        const parentHeight = parent.height ?? Number(parent.style?.height ?? NODE_H);
        if (requiredWidth > parentWidth) {
          parent.width = requiredWidth;
          parent.style = { ...parent.style, width: requiredWidth };
        }
        if (requiredHeight > parentHeight) {
          parent.height = requiredHeight;
          parent.style = { ...parent.style, height: requiredHeight };
        }
      });

      // A deep expansion can increase a child several levels below the root.
      // Revisit the hierarchy until every ancestor contains its direct children.
      for (let pass = 0; pass < reconciled.length; pass += 1) {
        let changed = false;
        reconciled.slice().reverse().forEach(node => {
          if (!node.parentId) return;
          const parent = byId.get(node.parentId);
          if (!parent) return;

          const nodeWidth = node.width ?? Number(node.style?.width ?? NODE_W);
          const nodeHeight = node.height ?? Number(node.style?.height ?? NODE_H);
          const requiredWidth = node.position.x + nodeWidth + CHILD_PAD;
          const requiredHeight = node.position.y + nodeHeight + CHILD_PAD;
          const parentWidth = parent.width ?? Number(parent.style?.width ?? NODE_W);
          const parentHeight = parent.height ?? Number(parent.style?.height ?? NODE_H);

          if (requiredWidth > parentWidth) {
            parent.width = requiredWidth;
            parent.style = { ...parent.style, width: requiredWidth };
            changed = true;
          }
          if (requiredHeight > parentHeight) {
            parent.height = requiredHeight;
            parent.style = { ...parent.style, height: requiredHeight };
            changed = true;
          }
        });
        if (!changed) break;
      }

      return reconciled;
    });
    setReactEdges(rfEdges);
  }, [rfNodes, rfEdges, layoutResetVersion, setReactNodes, setReactEdges]);

  useEffect(() => {
    setReactNodes(currentNodes => currentNodes.map(node => {
      const computed = rfNodes.find(candidate => candidate.id === node.id);
      if (!computed) return node;
      return {
        ...node,
        zIndex: computed.zIndex,
        data: computed.data,
      };
    }));
  }, [rfNodes, setReactNodes]);

  const onConnect = useCallback(
    (params: Connection) => setReactEdges(eds => addEdge(params, eds)),
    [setReactEdges],
  );

  return (
    <DiagramCanvas
      nodes={reactNodes}
      edges={reactEdges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onFitViewChange={onFitViewChange}
      onFitSelectedChange={(callback) => { fitSelectedRef.current = callback; }}
      selectedNode={selectedNode}
      onPaneClick={onPaneClick}
    />
  );
};

// ─── Inner canvas (needs ReactFlow context for fitView hook) ──────────────────

interface DiagramCanvasProps {
  nodes: Node[];
  edges: Edge[];
  onNodesChange: (changes: import('@xyflow/react').NodeChange[]) => void;
  onEdgesChange: (changes: import('@xyflow/react').EdgeChange[]) => void;
  onConnect: (p: Connection) => void;
  onFitViewChange?: (cb: () => void) => void;
  onFitSelectedChange?: (cb: () => void) => void;
  selectedNode: string | null;
  onPaneClick?: () => void;
}

const DiagramCanvas: React.FC<DiagramCanvasProps> = ({
  nodes, edges, onNodesChange, onEdgesChange, onConnect, onFitViewChange,
  onFitSelectedChange,
  selectedNode,
  onPaneClick,
}) => {
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onPaneClick={onPaneClick}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={{ padding: 0.15 }}
      minZoom={0.05}
      maxZoom={2.5}
      // Disable drag so expansion feels snappy (users zoom/pan with mouse)
      nodesDraggable={true}
      nodesConnectable={false}
      elementsSelectable={true}
      proOptions={{ hideAttribution: true }}
    >
      <CanvasControls
        onFitViewChange={onFitViewChange}
        onFitSelectedChange={onFitSelectedChange}
        selectedNode={selectedNode}
      />
    </ReactFlow>
  );
};

// ─── Controls panel (uses hook so must be inside ReactFlow context) ───────────

const CanvasControls: React.FC<{
  onFitViewChange?: (cb: () => void) => void;
  onFitSelectedChange?: (cb: () => void) => void;
  selectedNode: string | null;
}> = ({ onFitViewChange, onFitSelectedChange, selectedNode }) => {
  const { fitView } = useReactFlow();
  const { theme } = useTheme();
  const [isMiniMapVisible, setIsMiniMapVisible] = useState(true);
  const [miniMapSize, setMiniMapSize] = useState(180);
  const [isResizingMiniMap, setIsResizingMiniMap] = useState(false);

  const handleStartMiniMapResize = useCallback((event: React.PointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setIsResizingMiniMap(true);
  }, []);

  useEffect(() => {
    if (!isResizingMiniMap) return;

    const handlePointerMove = (event: PointerEvent) => {
      setMiniMapSize((currentSize) => Math.max(120, Math.min(320, currentSize + event.movementX + event.movementY)));
    };
    const handlePointerUp = () => setIsResizingMiniMap(false);

    document.addEventListener('pointermove', handlePointerMove);
    document.addEventListener('pointerup', handlePointerUp);
    return () => {
      document.removeEventListener('pointermove', handlePointerMove);
      document.removeEventListener('pointerup', handlePointerUp);
    };
  }, [isResizingMiniMap]);

  useEffect(() => {
    if (onFitViewChange) onFitViewChange(() => fitView({ padding: 0.15, duration: 400 }));
    if (onFitSelectedChange) {
      onFitSelectedChange(() => {
        if (selectedNode) fitView({ nodes: [selectedNode], padding: 0.25, duration: 400 });
      });
    }
  }, [fitView, onFitViewChange, onFitSelectedChange, selectedNode]);

  return (
    <>
      <Background
        variant={BackgroundVariant.Dots}
        gap={16}
        size={1}
        color="rgba(255,255,255,0.08)"
      />
      <Controls className={theme === 'dark'
        ? '!border-white/10 !bg-gray-800/80 !shadow-xl'
        : '!border-gray-300 !bg-white/90 !shadow-xl'} />
      {isMiniMapVisible ? (
        <div
          className="absolute bottom-3 right-3 z-10 rounded-lg shadow-xl"
          style={{ width: miniMapSize, height: miniMapSize }}
        >
          <MiniMap
            nodeColor={(n) => {
              const d = n.data as BubbleNodeData;
              if (d?.selected)     return '#22d3ee';
              if (d?.has_weights)  return '#0e7490';
              return '#374151';
            }}
            className={theme === 'dark'
              ? '!static !h-full !w-full !border-white/10 !bg-gray-900/80 transition-colors duration-300'
              : '!static !h-full !w-full !border-gray-300 !bg-white/90 transition-colors duration-300'}
            maskColor={theme === 'dark' ? 'rgba(0,0,0,0.5)' : 'rgba(255,255,255,0.48)'}
          />
          <button
            type="button"
            onClick={() => setIsMiniMapVisible(false)}
            className={`absolute right-1 top-1 rounded p-1 transition-colors hover:bg-cyan-400/20 hover:text-cyan-300 ${theme === 'dark' ? 'bg-gray-950/80 text-gray-300' : 'bg-white/90 text-gray-600'}`}
            title="Hide minimap"
            aria-label="Hide minimap"
          >
            <EyeOff className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onPointerDown={handleStartMiniMapResize}
            className="absolute bottom-0 right-0 h-4 w-4 cursor-se-resize rounded-tl bg-cyan-400/80"
            title="Resize minimap"
            aria-label="Resize minimap"
          />
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setIsMiniMapVisible(true)}
          className={`absolute bottom-3 right-3 z-10 rounded border p-2 shadow-xl transition-colors hover:border-cyan-400/40 hover:text-cyan-300 ${theme === 'dark' ? 'border-white/10 bg-gray-900/90 text-gray-300' : 'border-gray-300 bg-white/90 text-gray-600'}`}
          title="Show minimap"
          aria-label="Show minimap"
        >
          <Eye className="h-4 w-4" />
        </button>
      )}
    </>
  );
};

export default ModuleDiagram;