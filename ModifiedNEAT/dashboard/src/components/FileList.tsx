import React, { useState, useMemo } from 'react';
import { FileCode, Loader2, CaseSensitive, Quote, Regex, ArrowUp, ArrowDown, List, GitBranch } from 'lucide-react';
import type { PklFile } from '../types';

interface FileListProps {
  files: PklFile[];
  loading: boolean;
  selectedFile: string | null;
  onSelectFile: (filename: string) => void;
  error: string | null;
}

const formatSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const truncateText = (text: string, maxLength: number = 30): string => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3) + '...';
};

type SortType = 'alphabetical' | 'created' | 'modified';
type SortOrder = 'asc' | 'desc';
type DisplayType = 'full' | 'tree';

interface DirectoryNode {
  name: string;
  path: string;
  children: DirectoryNode[];
  file?: PklFile;
}

const buildTreeStructure = (files: PklFile[], rootDir: string): DirectoryNode => {
  const root: DirectoryNode = { name: rootDir || 'root', path: rootDir || '', children: [] };
  
  for (const file of files) {
    const pathParts = file.directory ? file.directory.split('/').filter(p => p) : [];
    let current = root;
    
    for (const part of pathParts) {
      let child = current.children.find(c => c.name === part);
      if (!child) {
        child = { name: part, path: current.path ? `${current.path}/${part}` : part, children: [] };
        current.children.push(child);
      }
      current = child;
    }
    
    current.children.push({ name: file.name, path: file.path, file, children: [] });
  }
  
  return root;
};

export const FileList: React.FC<FileListProps> = ({
  files,
  loading,
  selectedFile,
  onSelectFile,
  error,
}) => {
  // const [hoveredFile, setHoveredFile] = useState<string | null>(null); // TODO: Removed in a previous edit but might have functionality around
  const [searchQuery, setSearchQuery] = useState('');
  const [caseMatch, setCaseMatch] = useState(false);
  const [exactWord, setExactWord] = useState(false);
  const [useRegex, setUseRegex] = useState(false);
  const [sortType, setSortType] = useState<SortType>('modified');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  const [displayType, setDisplayType] = useState<DisplayType>('tree');

  const filteredAndSortedFiles = useMemo(() => {
    let result = [...files];
    
    // Filter by search query
    if (searchQuery.trim()) {
      result = result.filter(file => {
        const fileName = caseMatch ? file.name : file.name.toLowerCase();
        const dirName = caseMatch && file.directory ? file.directory : file.directory?.toLowerCase() || '';
        const query = caseMatch ? searchQuery : searchQuery.toLowerCase();
        const fullPath = `${dirName}/${fileName}`;
        
        try {
          if (useRegex) {
            const regex = new RegExp(query, caseMatch ? '' : 'i');
            return regex.test(fullPath) || regex.test(fileName);
          } else if (exactWord) {
            const wordRegex = new RegExp(`\\b${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, caseMatch ? '' : 'i');
            return wordRegex.test(fullPath) || wordRegex.test(fileName);
          } else {
            return fullPath.includes(query) || fileName.includes(query);
          }
        } catch {
          return false;
        }
      });
    }
    
    // Sort
    result.sort((a, b) => {
      let comparison = 0;
      
      if (sortType === 'alphabetical') {
        comparison = a.name.localeCompare(b.name);
      } else if (sortType === 'created') {
        comparison = new Date(a.modified).getTime() - new Date(b.modified).getTime();
      } else if (sortType === 'modified') {
        comparison = new Date(a.modified).getTime() - new Date(b.modified).getTime();
      }
      
      return sortOrder === 'asc' ? comparison : -comparison;
    });
    
    return result;
  }, [files, searchQuery, caseMatch, exactWord, useRegex, sortType, sortOrder]);

  const treeStructure = useMemo(() => {
    return buildTreeStructure(filteredAndSortedFiles, '');
  }, [filteredAndSortedFiles]);

  const expandedDirs = useState<Set<string>>(new Set());
  const [expandedDirSet, setExpandedDirSet] = expandedDirs;

  const toggleDir = (path: string) => {
    const newSet = new Set(expandedDirSet);
    if (newSet.has(path)) {
      newSet.delete(path);
    } else {
      newSet.add(path);
    }
    setExpandedDirSet(newSet);
  };

  const TreeNode: React.FC<{ node: DirectoryNode; depth: number }> = ({ node, depth }) => {
    const isDir = node.children.length > 0 && !node.file;
    const isExpanded = expandedDirSet.has(node.path);
    
    if (node.file) {
      return (
        <li key={node.path}>
          <button
            onClick={() => onSelectFile(node.path)}
            // onMouseEnter={() => setHoveredFile(node.path)}
            // onMouseLeave={() => setHoveredFile(null)}
            className={`w-full text-left px-2 py-1.5 rounded-lg transition-all duration-200 group flex items-center gap-1.5 text-xs ${
              selectedFile === node.path
                ? 'bg-gradient-to-r from-cyan-500/20 to-blue-500/20 border border-cyan-400/30'
                : 'hover:bg-white/5 border border-transparent'
            }`}
            title={node.name}
            style={{ marginLeft: `${depth * 12}px` }}
          >
            <FileCode className={`w-2.5 h-2.5 flex-shrink-0 ${
              selectedFile === node.path ? 'text-cyan-400' : 'text-gray-500'
            }`} />
            <span className={`truncate text-xs ${
              selectedFile === node.path ? 'text-cyan-300' : 'text-gray-300'
            }`}>
              {truncateText(node.name, 22)}
            </span>
            {node.file?.size && (
              <span className="text-gray-500 ml-auto flex-shrink-0 text-xs">
                {formatSize(node.file.size)}
              </span>
            )}
          </button>
        </li>
      );
    }
    
    if (!isDir) return null;
    
    const fileCount = node.children.filter(c => c.file).length;
    
    return (
      <li key={node.path}>
        <button
          onClick={() => toggleDir(node.path)}
          className="w-full text-left px-2 py-1 rounded hover:bg-white/5 transition-colors group flex items-center gap-1.5" // text-xs"
          style={{ marginLeft: `${depth * 12}px` }}
        >
          <span className="text-gray-500 w-3 flex items-center justify-center text-xs font-mono">
            {isExpanded ? '−' : '+'}
          </span>
          <span className="text-gray-400 text-xs font-medium">{node.name}</span>
          {fileCount > 0 && (
            <span className="text-gray-600 text-xs ml-1">({fileCount})</span>
          )}
        </button>
        {isExpanded && (
          <ul className="space-y-0">
            {node.children.map(child => (
              <TreeNode
                  key={child.path}
                  node={child}
                  depth={depth + 1}
              />
            ))}
          </ul>
        )}
      </li>
    );
  };

  return (
    <div className="h-full flex flex-col">
      {/* Search and Controls Section */}
      <div className="p-3 border-b border-white/10 dark:border-white/5 space-y-2">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-600 dark:text-gray-400">
          Model Files
        </h2>

        {/* Search bar with filter toggles on the right */}
        <div className="flex items-center gap-1.5">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search files and directories..."
            className="flex-1 min-w-0 px-2 py-1.5 text-xs bg-gray-800/50 border border-white/10 rounded-lg text-gray-200 placeholder-gray-500 focus:outline-none focus:border-cyan-400/50"
          />
          <button
            onClick={() => setCaseMatch(!caseMatch)}
            className={`flex-shrink-0 flex items-center px-1.5 py-1.5 rounded text-xs transition-colors ${
              caseMatch
                ? 'bg-cyan-500/20 border border-cyan-400/50 text-cyan-300'
                : 'bg-gray-800/50 border border-white/10 text-gray-400 hover:border-white/20'
            }`}
            title="Match case"
          >
            <CaseSensitive className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setExactWord(!exactWord)}
            className={`flex-shrink-0 flex items-center px-1.5 py-1.5 rounded text-xs transition-colors ${
              exactWord
                ? 'bg-cyan-500/20 border border-cyan-400/50 text-cyan-300'
                : 'bg-gray-800/50 border border-white/10 text-gray-400 hover:border-white/20'
            }`}
            title="Match exact word (bounded by space, _, -, /)"
          >
            <Quote className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setUseRegex(!useRegex)}
            className={`flex-shrink-0 flex items-center px-1.5 py-1.5 rounded text-xs transition-colors ${
              useRegex
                ? 'bg-cyan-500/20 border border-cyan-400/50 text-cyan-300'
                : 'bg-gray-800/50 border border-white/10 text-gray-400 hover:border-white/20'
            }`}
            title="Use regex pattern"
          >
            <Regex className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* File count + sort options + display type */}
        <div className="flex items-center justify-between gap-1.5 flex-wrap">
          <p className="text-xs text-gray-500 shrink-0">
            {filteredAndSortedFiles.length} file{filteredAndSortedFiles.length !== 1 ? 's' : ''} found
          </p>
          <div className="flex items-center gap-1 flex-wrap justify-end">
            {/* Sort Buttons */}
            <div className="flex gap-1">
              <select
                value={sortType}
                onChange={(e) => setSortType(e.target.value as SortType)}
                className="px-2 py-1.5 text-xs bg-gray-800/50 border border-white/10 rounded text-gray-300 focus:outline-none focus:border-cyan-400/50"
              >
                <option value="alphabetical">A-Z</option>
                <option value="created">Created</option>
                <option value="modified">Modified</option>
              </select>
              <button
                onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                className="px-2 py-1.5 rounded bg-gray-800/50 border border-white/10 text-gray-400 hover:border-cyan-400/50 transition-colors"
                title={sortOrder === 'asc' ? 'Ascending' : 'Descending'}
              >
                {sortOrder === 'asc' ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />}
              </button>
            </div>

            {/* Display Type Buttons */}
            <div className="flex gap-1">
              <button
                onClick={() => setDisplayType('full')}
                className={`px-2 py-1.5 rounded text-xs transition-colors ${
                  displayType === 'full'
                    ? 'bg-cyan-500/20 border border-cyan-400/50 text-cyan-300'
                    : 'bg-gray-800/50 border border-white/10 text-gray-400 hover:border-white/20'
                }`}
                title="Full list display"
              >
                <List className="w-3 h-3" />
              </button>
              <button
                onClick={() => setDisplayType('tree')}
                className={`px-2 py-1.5 rounded text-xs transition-colors ${
                  displayType === 'tree'
                    ? 'bg-cyan-500/20 border border-cyan-400/50 text-cyan-300'
                    : 'bg-gray-800/50 border border-white/10 text-gray-400 hover:border-white/20'
                }`}
                title="Tree view display"
              >
                <GitBranch className="w-3 h-3" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-2 custom-scrollbar">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-5 h-5 animate-spin text-cyan-400" />
          </div>
        ) : error ? (
          <div className="px-3 py-2 text-sm text-red-400 bg-red-400/10 rounded-lg">
            {error}
          </div>
        ) : filteredAndSortedFiles.length === 0 ? (
          <div className="px-3 py-8 text-center">
            <FileCode className="w-8 h-8 mx-auto mb-2 text-gray-400 dark:text-gray-600" />
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {searchQuery ? 'No files match your search' : 'No .module.pkl or .neat.pkl files found'}
            </p>
          </div>
        ) : displayType === 'full' ? (
          <ul className="space-y-1">
            {filteredAndSortedFiles.map((file) => (
              <li key={file.path}>
                <button
                  onClick={() => onSelectFile(file.path)}
                  // onMouseEnter={() => setHoveredFile(file.path)}
                  // onMouseLeave={() => setHoveredFile(null)}
                  className={`w-full text-left px-3 py-2 rounded-lg transition-all duration-200
                    group flex items-center gap-3
                    ${selectedFile === file.path
                      ? 'bg-gradient-to-r from-cyan-500/20 to-blue-500/20 border border-cyan-400/30 shadow-lg shadow-cyan-500/10'
                      : 'hover:bg-white/5 dark:hover:bg-white/5 border border-transparent'
                    }`}
                  title={file.name}
                >
                  <div className={`p-1.5 rounded-lg transition-colors flex-shrink-0 ${
                    selectedFile === file.name
                      ? 'bg-cyan-400/20'
                      : 'bg-gray-100 dark:bg-gray-800 group-hover:bg-cyan-400/10'
                  }`}>
                    <FileCode className={`w-3.5 h-3.5 ${
                      selectedFile === file.name
                        ? 'text-cyan-400'
                        : 'text-gray-500 dark:text-gray-400 group-hover:text-cyan-400'
                    }`} />
                  </div>
                  <div className="flex-1 min-w-0">
                      <p
                        className={`text-xs font-medium truncate transition-colors ${
                        selectedFile === file.name
                          ? 'text-cyan-300'
                          : 'text-gray-800 dark:text-gray-200 group-hover:text-cyan-400'
                      }`}
                      title={file.name}
                      >
                        {truncateText(file.name)}
                      </p>
                      {file.directory && (
                        <p
                          className={`text-xs transition-colors ${
                          selectedFile === file.name
                            ? 'text-cyan-200/60'
                            : 'text-gray-500 dark:text-gray-500'
                        }`}
                        title={file.directory}
                        >
                          {truncateText(file.directory, 25)}
                        </p>
                      )}
                      <p className="text-xs text-gray-500 dark:text-gray-500">
                        {formatSize(file.size)}
                      </p>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <ul className="space-y-0">
            {treeStructure.children.map(node => (
              <TreeNode key={node.path} node={node} depth={0} />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};
