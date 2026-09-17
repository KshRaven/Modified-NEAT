import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
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
import { Box, Layers, ChevronDown, ChevronUp, Search } from 'lucide-react';
import type { ModuleNode as ModuleNodeType } from '../types';

// ─── Types ──────────────────────────────────────────────────────────────────

interface ModuleDiagramProps {
  nodes: ModuleNodeType[];
  edges: { id: string; source: string; target: string }[];
  selectedNode: string | null;
  onNodeClick: (nodeId: string) => void;
  onExpandAllChange?: (callback: () => void) => void;
  onCollapseAllChange?: (callback: () => void) => void;
  onFitViewChange?: (callback: () => void) => void;
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

interface BubbleNodeData {
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
  onViewTensors?: () => void;
}

const BubbleNode: React.FC<{ id: string; data: BubbleNodeData }> = ({ data }) => {
  const {
    label, type, has_weights, selected, is_neat_module,
    has_children, child_count,
    is_container, width, height,
    onToggle, onViewTensors,
  } = data;

  // ── Container (expanded bubble) ───────────────────────────────────────────
  if (is_container) {
    return (
      <div
        style={{ width, height, position: 'relative' }}
        className={`rounded-2xl border-2 transition-all duration-300
          ${selected
            ? 'border-cyan-400/70 bg-cyan-500/5 shadow-xl shadow-cyan-500/20'
            : has_weights
              ? 'border-cyan-400/30 bg-gray-900/40 hover:border-cyan-400/50'
              : 'border-white/10 bg-gray-800/30 hover:border-white/20'
          }`}
      >
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
            ${selected ? 'bg-cyan-500/20' : 'bg-black/20'}`}
          onClick={onToggle}
          style={{ height: NODE_H }}
        >
          <div className="flex items-center gap-2 min-w-0">
            {has_weights
              ? <Box className="w-4 h-4 text-cyan-400 shrink-0" />
              : <Layers className="w-4 h-4 text-gray-400 shrink-0" />}
            <div className="min-w-0">
              <div className={`text-xs font-medium truncate ${selected ? 'text-cyan-300' : 'text-gray-400'}`}>
                {type}
              </div>
              <div className={`text-sm font-semibold truncate ${selected ? 'text-white' : 'text-gray-200'}`}>
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
      onClick={onToggle}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="!w-3 !h-3 !bg-cyan-400 !border-none"
      />

      <div className="flex items-center justify-between gap-2 mb-0.5">
        <div className="flex items-center gap-2 min-w-0">
          {has_weights
            ? <Box className="w-4 h-4 text-cyan-400 shrink-0" />
            : <Layers className="w-4 h-4 text-gray-400 shrink-0" />}
          <span className={`text-xs font-medium truncate ${selected ? 'text-cyan-300' : 'text-gray-400'}`}>
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

      <div className={`font-semibold text-sm truncate ${selected ? 'text-white' : 'text-gray-200'}`}>
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
  onViewTensor: ((id: string) => void) | undefined,
): { rfNodes: Node[]; rfEdges: Edge[] } {
  const { positions, sizes, parentId, childMap } = buildLayout(allNodes, allEdges, expanded);

  const nodeMap = new Map<string, ModuleNodeType>();
  allNodes.forEach(n => nodeMap.set(n.id, n));

  // Nodes that are visible (have a computed position)
  const visible = new Set(positions.keys());

  const rfNodes: Node[] = [];

  positions.forEach((pos, id) => {
    const meta = nodeMap.get(id);
    if (!meta) return;

    const sz        = sizes.get(id)!;
    const isExp     = expanded[id] !== false;
    const children  = childMap.get(id) ?? [];
    const isCont    = isExp && children.length > 0;

    // Position is absolute; ReactFlow parent-relative positioning is done by
    // setting `parentId` + `extent: 'parent'` — but that requires parent to
    // be sized first, and ReactFlow's group nodes require explicit `style` with
    // width/height. We handle containment visually via bubble borders and
    // absolute co-ordinates instead.

    rfNodes.push({
      id,
      type: 'bubble',
      position: pos,
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
  onExpandAllChange,
  onCollapseAllChange,
  onFitViewChange,
  onViewTensor,
}) => {
  // Initialise all nodes collapsed except the root
  const [expanded, setExpanded] = useState<ExpandedState>(() => {
    const s: ExpandedState = {};
    nodes.forEach(n => { s[n.id] = n.id === 'root'; });
    return s;
  });

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

  const handleViewTensor = useCallback((nodeId: string) => {
    onViewTensor?.(nodeId);
  }, [onViewTensor]);

  const expandAll = useCallback(() => {
    const s: ExpandedState = {};
    nodes.forEach(n => { s[n.id] = true; });
    setExpanded(s);
  }, [nodes]);

  const collapseAll = useCallback(() => {
    const s: ExpandedState = {};
    nodes.forEach(n => { s[n.id] = n.id === 'root'; });
    setExpanded(s);
  }, [nodes]);

  // Expose callbacks to parent (toolbar buttons)
  useEffect(() => {
    if (onExpandAllChange)  onExpandAllChange(expandAll);
    if (onCollapseAllChange) onCollapseAllChange(collapseAll);
  }, [expandAll, collapseAll, onExpandAllChange, onCollapseAllChange]);

  const { rfNodes, rfEdges } = useMemo(
    () => buildReactFlowElements(nodes, edges, expanded, selectedNode, handleToggle, handleViewTensor),
    [nodes, edges, expanded, selectedNode, handleToggle, handleViewTensor],
  );

  const [reactNodes, setReactNodes, onNodesChange] = useNodesState(rfNodes);
  const [reactEdges, setReactEdges, onEdgesChange] = useEdgesState(rfEdges);

  // Keep ReactFlow state in sync with computed layout
  useEffect(() => {
    setReactNodes(rfNodes);
    setReactEdges(rfEdges);
  }, [rfNodes, rfEdges, setReactNodes, setReactEdges]);

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
}

const DiagramCanvas: React.FC<DiagramCanvasProps> = ({
  nodes, edges, onNodesChange, onEdgesChange, onConnect, onFitViewChange,
}) => {
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={{ padding: 0.15 }}
      minZoom={0.05}
      maxZoom={2.5}
      // Disable drag so expansion feels snappy (users zoom/pan with mouse)
      nodesDraggable={true}
      nodesConnectable={false}
      elementsSelectable={false}
      proOptions={{ hideAttribution: true }}
    >
      <CanvasControls onFitViewChange={onFitViewChange} />
    </ReactFlow>
  );
};

// ─── Controls panel (uses hook so must be inside ReactFlow context) ───────────

const CanvasControls: React.FC<{ onFitViewChange?: (cb: () => void) => void }> = ({
  onFitViewChange,
}) => {
  const { fitView } = useReactFlow();

  useEffect(() => {
    if (onFitViewChange) onFitViewChange(() => fitView({ padding: 0.15, duration: 400 }));
  }, [fitView, onFitViewChange]);

  return (
    <>
      <Background
        variant={BackgroundVariant.Dots}
        gap={16}
        size={1}
        color="rgba(255,255,255,0.08)"
      />
      <Controls className="!border-white/10 !bg-gray-800/80 !shadow-xl" />
      <MiniMap
        nodeColor={(n) => {
          const d = n.data as BubbleNodeData;
          if (d?.selected)     return '#22d3ee';
          if (d?.has_weights)  return '#0e7490';
          return '#374151';
        }}
        className="!border-white/10 !bg-gray-900/80"
        maskColor="rgba(0,0,0,0.5)"
      />
    </>
  );
};

export default ModuleDiagram;