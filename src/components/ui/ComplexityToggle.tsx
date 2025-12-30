/**
 * ComplexityToggle - Controls visibility of Math and Code sections
 *
 * This component allows readers to customize their viewing experience
 * by showing/hiding mathematical derivations and code implementations.
 *
 * Usage:
 *   <ComplexityToggle />
 *
 * The toggle state is persisted in localStorage and shared via React Context.
 */

import React, { createContext, useContext, useState, useEffect } from 'react';

// Types
interface ComplexityPreferences {
  showMath: boolean;
  showCode: boolean;
}

interface ComplexityContextType extends ComplexityPreferences {
  setShowMath: (show: boolean) => void;
  setShowCode: (show: boolean) => void;
  toggleMath: () => void;
  toggleCode: () => void;
}

// Default preferences
const defaultPreferences: ComplexityPreferences = {
  showMath: true,
  showCode: true,
};

// Context
const ComplexityContext = createContext<ComplexityContextType | null>(null);

// Storage key
const STORAGE_KEY = 'rlbook-complexity-preferences';

/**
 * Provider component that manages complexity state
 */
export function ComplexityProvider({ children }: { children: React.ReactNode }) {
  const [preferences, setPreferences] = useState<ComplexityPreferences>(defaultPreferences);

  // Load preferences from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        setPreferences(JSON.parse(stored));
      }
    } catch (e) {
      console.warn('Failed to load complexity preferences:', e);
    }
  }, []);

  // Save preferences to localStorage when they change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
    } catch (e) {
      console.warn('Failed to save complexity preferences:', e);
    }
  }, [preferences]);

  const setShowMath = (show: boolean) => {
    setPreferences(prev => ({ ...prev, showMath: show }));
  };

  const setShowCode = (show: boolean) => {
    setPreferences(prev => ({ ...prev, showCode: show }));
  };

  const toggleMath = () => {
    setPreferences(prev => ({ ...prev, showMath: !prev.showMath }));
  };

  const toggleCode = () => {
    setPreferences(prev => ({ ...prev, showCode: !prev.showCode }));
  };

  return (
    <ComplexityContext.Provider
      value={{
        ...preferences,
        setShowMath,
        setShowCode,
        toggleMath,
        toggleCode,
      }}
    >
      {children}
    </ComplexityContext.Provider>
  );
}

/**
 * Hook to access complexity preferences
 * Returns default values if used outside provider (for SSR compatibility)
 */
export function useComplexity(): ComplexityContextType {
  const context = useContext(ComplexityContext);
  if (!context) {
    // Return default values for SSR - content will be visible by default
    return {
      showMath: true,
      showCode: true,
      setShowMath: () => {},
      setShowCode: () => {},
      toggleMath: () => {},
      toggleCode: () => {},
    };
  }
  return context;
}

/**
 * Toggle UI component
 */
export function ComplexityToggle() {
  const { showMath, showCode, toggleMath, toggleCode } = useComplexity();

  return (
    <div className="complexity-toggle">
      <span className="toggle-label">Show:</span>

      <button
        className={`toggle-button ${showMath ? 'active' : ''}`}
        onClick={toggleMath}
        aria-pressed={showMath}
        title="Toggle mathematical content"
      >
        <span className="toggle-icon">∑</span>
        <span className="toggle-text">Math</span>
      </button>

      <button
        className={`toggle-button ${showCode ? 'active' : ''}`}
        onClick={toggleCode}
        aria-pressed={showCode}
        title="Toggle code examples"
      >
        <span className="toggle-icon">&lt;/&gt;</span>
        <span className="toggle-text">Code</span>
      </button>
    </div>
  );
}

export default ComplexityToggle;
