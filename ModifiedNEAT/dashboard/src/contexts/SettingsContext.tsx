import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface Settings {
  autoRefresh: boolean;
  refreshInterval: number;
  maxDisplay: number;
  showHeatmap: boolean;
  heatmapLowColor: string;
  heatmapMidColor: string;
  heatmapHighColor: string;
}

interface SettingsContextType {
  settings: Settings;
  updateSettings: (updates: Partial<Settings>) => void;
  resetSettings: () => void;
}

const defaultSettings: Settings = {
  autoRefresh: false,
  refreshInterval: 10,
  maxDisplay: 100,
  showHeatmap: true,
  heatmapLowColor: '#1e3a8a',
  heatmapMidColor: '#1e1e1e',
  heatmapHighColor: '#22d3ee',
};

const STORAGE_KEY = 'pytorch-visualizer-settings';

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export const SettingsProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [settings, setSettings] = useState<Settings>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        return { ...defaultSettings, ...JSON.parse(stored) };
      }
    } catch {
      // Ignore parse errors
    }
    return defaultSettings;
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  const updateSettings = (updates: Partial<Settings>) => {
    setSettings((prev) => ({ ...prev, ...updates }));
  };

  const resetSettings = () => {
    setSettings(defaultSettings);
  };

  return (
    <SettingsContext.Provider value={{ settings, updateSettings, resetSettings }}>
      {children}
    </SettingsContext.Provider>
  );
};

export const useSettings = (): SettingsContextType => {
  const context = useContext(SettingsContext);
  if (!context) {
    throw new Error('useSettings must be used within a SettingsProvider');
  }
  return context;
};
