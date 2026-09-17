import React from 'react';
import { Sun, Moon, Network, Settings, RefreshCw } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

interface HeaderProps {
  onSettingsClick: () => void;
  onRefreshClick: () => void;
  isRefreshing?: boolean;
}

export const Header: React.FC<HeaderProps> = ({ onSettingsClick, onRefreshClick, isRefreshing }) => {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="h-14 px-6 flex items-center justify-between border-b backdrop-blur-xl bg-white/30 dark:bg-black/20 border-gray-200 dark:border-white/5 transition-colors duration-300">
      <div className="flex items-center gap-3">
        <div className="p-1.5 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-500/20 border border-cyan-400/20">
          <Network className="w-5 h-5 text-cyan-400" />
        </div>
        <div>
          <h1 className="text-base font-semibold tracking-tight text-gray-900 dark:text-white">
            PyTorch Module Visualizer
          </h1>
          <p className="text-xs text-gray-500 dark:text-gray-400 hidden sm:block">
            Architecture explorer for neural networks
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onRefreshClick}
          disabled={isRefreshing}
          className="p-2 rounded-lg hover:bg-white/10 dark:hover:bg-white/5 transition-colors group disabled:opacity-50"
          aria-label="Refresh files"
        >
          <RefreshCw
            className={`w-4 h-4 text-gray-500 dark:text-gray-400 group-hover:text-cyan-400 transition-colors ${
              isRefreshing ? 'animate-spin' : ''
            }`}
          />
        </button>

        <button
          onClick={onSettingsClick}
          className="p-2 rounded-lg hover:bg-white/10 dark:hover:bg-white/5 transition-colors group"
          aria-label="Settings"
        >
          <Settings className="w-4 h-4 text-gray-500 dark:text-gray-400 group-hover:text-cyan-400 transition-colors" />
        </button>

        <button
          onClick={toggleTheme}
          className="relative w-14 h-8 rounded-full transition-all duration-300
            bg-gradient-to-r from-gray-200 to-gray-300 dark:from-gray-700 dark:to-gray-600
            shadow-inner hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-offset-2
            dark:focus:ring-offset-gray-900"
          aria-label="Toggle theme"
        >
          <div
            className={`absolute top-1 w-6 h-6 rounded-full transition-all duration-500 transform
              bg-gradient-to-br from-yellow-300 to-orange-400 dark:from-indigo-400 dark:to-purple-500
              shadow-md flex items-center justify-center
              ${theme === 'dark' ? 'left-7 rotate-0' : 'left-1 rotate-180'}`}
          >
            {theme === 'dark' ? (
              <Moon className="w-3.5 h-3.5 text-white" />
            ) : (
              <Sun className="w-3.5 h-3.5 text-white" />
            )}
          </div>
        </button>
      </div>
    </header>
  );
};
