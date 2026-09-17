import React from 'react';
import { X, RefreshCw, Palette, RotateCcw } from 'lucide-react';
import { useSettings } from '../contexts/SettingsContext';

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const SettingsPanel: React.FC<SettingsPanelProps> = ({ isOpen, onClose }) => {
  const { settings, updateSettings, resetSettings } = useSettings();

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-end p-4">
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-80 max-h-[80vh] overflow-y-auto rounded-xl bg-gradient-to-br from-gray-900/95 to-gray-800/95 backdrop-blur-xl border border-white/10 shadow-2xl">
        <div className="sticky top-0 flex items-center justify-between p-4 border-b border-white/10 bg-gray-900/50 backdrop-blur">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-200">Settings</h2>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-white/10 transition-colors"
          >
            <X className="w-4 h-4 text-gray-400" />
          </button>
        </div>

        <div className="p-4 space-y-6">
          {/* Auto Refresh */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-gray-300 flex items-center gap-2">
                <RefreshCw className="w-4 h-4 text-cyan-400" />
                Auto Refresh
              </label>
              <button
                onClick={() => updateSettings({ autoRefresh: !settings.autoRefresh })}
                className={`w-10 h-5 rounded-full transition-all duration-300 ${
                  settings.autoRefresh
                    ? 'bg-cyan-500'
                    : 'bg-gray-700'
                }`}
              >
                <div
                  className={`w-4 h-4 rounded-full bg-white transition-transform duration-300 ${
                    settings.autoRefresh ? 'translate-x-5' : 'translate-x-0.5'
                  }`}
                />
              </button>
            </div>
            {settings.autoRefresh && (
              <div className="flex items-center gap-2 pl-6">
                <span className="text-xs text-gray-400">Interval:</span>
                <input
                  type="number"
                  value={settings.refreshInterval}
                  onChange={(e) => {
                    const val = parseInt(e.target.value, 10);
                    if (val >= 1 && val <= 300) {
                      updateSettings({ refreshInterval: val });
                    }
                  }}
                  min={1}
                  max={300}
                  className="w-16 px-2 py-1 text-xs bg-gray-800 border border-white/10 rounded text-gray-200 focus:outline-none focus:border-cyan-400"
                />
                <span className="text-xs text-gray-400">seconds</span>
              </div>
            )}
          </div>

          {/* Max Display */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">Max Display per Dimension</label>
            <input
              type="number"
              value={settings.maxDisplay}
              onChange={(e) => {
                const val = parseInt(e.target.value, 10);
                if (val >= 10 && val <= 1000) {
                  updateSettings({ maxDisplay: val });
                }
              }}
              min={10}
              max={1000}
              className="w-full px-3 py-2 text-sm bg-gray-800 border border-white/10 rounded-lg text-gray-200 focus:outline-none focus:border-cyan-400"
            />
            <p className="text-xs text-gray-500">Number of indices to show per dimension before scroll</p>
          </div>

          {/* Heatmap Settings */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-gray-300 flex items-center gap-2">
                <Palette className="w-4 h-4 text-cyan-400" />
                Show Heatmap
              </label>
              <button
                onClick={() => updateSettings({ showHeatmap: !settings.showHeatmap })}
                className={`w-10 h-5 rounded-full transition-all duration-300 ${
                  settings.showHeatmap ? 'bg-cyan-500' : 'bg-gray-700'
                }`}
              >
                <div
                  className={`w-4 h-4 rounded-full bg-white transition-transform duration-300 ${
                    settings.showHeatmap ? 'translate-x-5' : 'translate-x-0.5'
                  }`}
                />
              </button>
            </div>

            {settings.showHeatmap && (
              <div className="space-y-2 pl-6">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-400 w-12">Low:</span>
                  <input
                    type="color"
                    value={settings.heatmapLowColor}
                    onChange={(e) => updateSettings({ heatmapLowColor: e.target.value })}
                    className="w-8 h-6 rounded cursor-pointer"
                  />
                  <span className="text-xs text-gray-500 font-mono">{settings.heatmapLowColor}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-400 w-12">Mid:</span>
                  <input
                    type="color"
                    value={settings.heatmapMidColor}
                    onChange={(e) => updateSettings({ heatmapMidColor: e.target.value })}
                    className="w-8 h-6 rounded cursor-pointer"
                  />
                  <span className="text-xs text-gray-500 font-mono">{settings.heatmapMidColor}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-400 w-12">High:</span>
                  <input
                    type="color"
                    value={settings.heatmapHighColor}
                    onChange={(e) => updateSettings({ heatmapHighColor: e.target.value })}
                    className="w-8 h-6 rounded cursor-pointer"
                  />
                  <span className="text-xs text-gray-500 font-mono">{settings.heatmapHighColor}</span>
                </div>
              </div>
            )}
          </div>

          {/* Reset */}
          <button
            onClick={resetSettings}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-gray-800 border border-white/10 text-gray-300 hover:bg-gray-700 hover:border-cyan-400/30 transition-all duration-200"
          >
            <RotateCcw className="w-4 h-4" />
            <span className="text-sm">Reset to Defaults</span>
          </button>
        </div>
      </div>
    </div>
  );
};
