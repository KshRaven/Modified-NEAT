import React from 'react';
import { FolderOpen, Layers, Hash, Activity } from 'lucide-react';

interface StatusBarProps {
  sourceDir: string;
  fileCount: number;
  totalParams: number | null;
  selectedModule: string | null;
}

const formatNumber = (num: number): string => {
  if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
  if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
  if (num >= 1e3) return `${(num / 1e3).toFixed(1)}K`;
  return num.toString();
};

export const StatusBar: React.FC<StatusBarProps> = ({
  sourceDir,
  fileCount,
  totalParams,
  selectedModule,
}) => {
  return (
    <footer className="h-8 px-4 flex items-center justify-between text-xs border-t backdrop-blur-xl bg-white/20 dark:bg-black/30 border-gray-200 dark:border-white/5 transition-colors duration-300">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1.5 text-gray-600 dark:text-gray-400">
          <FolderOpen className="w-3.5 h-3.5" />
          <span className="font-mono truncate max-w-xs" title={sourceDir}>
            {sourceDir || 'No directory'}
          </span>
        </div>

        <div className="flex items-center gap-1.5 text-gray-600 dark:text-gray-400">
          <Layers className="w-3.5 h-3.5" />
          <span>{fileCount} file{fileCount !== 1 ? 's' : ''}</span>
        </div>

        {totalParams !== null && totalParams > 0 && (
          <div className="flex items-center gap-1.5 text-cyan-500">
            <Hash className="w-3.5 h-3.5" />
            <span>{formatNumber(totalParams)} params</span>
          </div>
        )}
      </div>

      <div className="flex items-center gap-4">
        {selectedModule && (
          <div className="flex items-center gap-1.5 text-cyan-500">
            <Activity className="w-3.5 h-3.5" />
            <span className="max-w-48 truncate">{selectedModule.split('.').pop()}</span>
          </div>
        )}
      </div>
    </footer>
  );
};
