/**
 * ComplexityToggle - Controls visibility of Math and Code sections
 *
 * This component allows readers to customize their viewing experience
 * by showing/hiding mathematical derivations and code implementations.
 *
 * It works by toggling CSS classes on the document body, which then
 * hide/show the corresponding content layers via CSS rules.
 */

import React, { useState, useEffect } from 'react';

// Storage key
const STORAGE_KEY = 'rlbook-complexity-preferences';

interface ComplexityPreferences {
  showMath: boolean;
  showCode: boolean;
}

const defaultPreferences: ComplexityPreferences = {
  showMath: true,
  showCode: true,
};

/**
 * Toggle UI component that controls content visibility via CSS
 */
export function ComplexityToggle() {
  const [preferences, setPreferences] = useState<ComplexityPreferences>(defaultPreferences);
  const [isHydrated, setIsHydrated] = useState(false);

  // Load preferences from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        setPreferences(parsed);
      }
    } catch (e) {
      console.warn('Failed to load complexity preferences:', e);
    }
    setIsHydrated(true);
  }, []);

  // Apply CSS classes to document body and save preferences
  useEffect(() => {
    if (!isHydrated) return;

    // Toggle CSS classes on body
    if (preferences.showMath) {
      document.body.classList.remove('hide-math');
    } else {
      document.body.classList.add('hide-math');
    }

    if (preferences.showCode) {
      document.body.classList.remove('hide-code');
    } else {
      document.body.classList.add('hide-code');
    }

    // Save to localStorage
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
    } catch (e) {
      console.warn('Failed to save complexity preferences:', e);
    }
  }, [preferences, isHydrated]);

  const toggleMath = () => {
    setPreferences(prev => ({ ...prev, showMath: !prev.showMath }));
  };

  const toggleCode = () => {
    setPreferences(prev => ({ ...prev, showCode: !prev.showCode }));
  };

  return (
    <div className="complexity-toggle p-4 bg-gray-100 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Content Visibility
        </span>
        <span className="text-xs text-gray-500 dark:text-gray-400">
          Toggle sections on/off
        </span>
      </div>

      <div className="flex items-center gap-3">
        <button
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all ${
            preferences.showMath
              ? 'bg-purple-600 dark:bg-purple-700 text-white shadow-sm'
              : 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400 line-through'
          }`}
          onClick={toggleMath}
          aria-pressed={preferences.showMath}
          title={preferences.showMath ? 'Click to hide math sections' : 'Click to show math sections'}
        >
          <span className="text-base">∑</span>
          <span>Math</span>
          <span className={`ml-1 text-xs ${preferences.showMath ? 'opacity-75' : 'opacity-50'}`}>
            {preferences.showMath ? 'ON' : 'OFF'}
          </span>
        </button>

        <button
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all ${
            preferences.showCode
              ? 'bg-emerald-600 dark:bg-emerald-700 text-white shadow-sm'
              : 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400 line-through'
          }`}
          onClick={toggleCode}
          aria-pressed={preferences.showCode}
          title={preferences.showCode ? 'Click to hide code sections' : 'Click to show code sections'}
        >
          <span className="text-base">&lt;/&gt;</span>
          <span>Code</span>
          <span className={`ml-1 text-xs ${preferences.showCode ? 'opacity-75' : 'opacity-50'}`}>
            {preferences.showCode ? 'ON' : 'OFF'}
          </span>
        </button>
      </div>
    </div>
  );
}

// Keep the provider and hook for backward compatibility, but they're no longer needed
export function ComplexityProvider({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

export function useComplexity() {
  return {
    showMath: true,
    showCode: true,
    setShowMath: () => {},
    setShowCode: () => {},
    toggleMath: () => {},
    toggleCode: () => {},
  };
}

export default ComplexityToggle;
