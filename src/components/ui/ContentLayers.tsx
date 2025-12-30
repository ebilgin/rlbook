/**
 * Content Layer Components - Intuition, Mathematical, Implementation
 *
 * These components wrap content at different complexity levels.
 * Mathematical and Implementation sections are collapsible using native HTML
 * <details>/<summary> elements for no-JS compatibility.
 *
 * Usage:
 *   <Intuition>
 *     Always visible content explaining the concept in plain terms.
 *   </Intuition>
 *
 *   <Mathematical>
 *     Formal definitions, equations, and derivations.
 *   </Mathematical>
 *
 *   <Implementation>
 *     Code examples and practical implementation details.
 *   </Implementation>
 */

import React from 'react';

interface LayerProps {
  children: React.ReactNode;
  title?: string;
  defaultOpen?: boolean;
}

/**
 * Intuition layer - always visible
 * Core concepts explained in plain English with visualizations
 */
export function Intuition({ children, title }: LayerProps) {
  return (
    <div className="content-layer layer-intuition">
      {title && <h3 className="layer-title">{title}</h3>}
      <div className="layer-content">{children}</div>
    </div>
  );
}

/**
 * Mathematical layer - collapsible
 * Formal definitions, equations, derivations
 * Uses native <details>/<summary> for no-JS accordion behavior
 */
export function Mathematical({ children, title = 'Mathematical Details', defaultOpen = true }: LayerProps) {
  return (
    <details className="content-layer layer-mathematical" open={defaultOpen}>
      <summary className="layer-header">
        <span className="layer-icon">∑</span>
        <span className="layer-title">{title}</span>
        <span className="layer-toggle-icon" aria-hidden="true"></span>
      </summary>
      <div className="layer-content">{children}</div>
    </details>
  );
}

/**
 * Implementation layer - collapsible
 * Code examples, practical details, debugging tips
 * Uses native <details>/<summary> for no-JS accordion behavior
 */
export function Implementation({ children, title = 'Implementation', defaultOpen = true }: LayerProps) {
  return (
    <details className="content-layer layer-implementation" open={defaultOpen}>
      <summary className="layer-header">
        <span className="layer-icon">&lt;/&gt;</span>
        <span className="layer-title">{title}</span>
        <span className="layer-toggle-icon" aria-hidden="true"></span>
      </summary>
      <div className="layer-content">{children}</div>
    </details>
  );
}

/**
 * DeepDive - optional advanced content
 * For curious readers who want to go beyond the main material
 * Collapsed by default to not overwhelm readers
 */
export function DeepDive({ children, title = 'Deep Dive', defaultOpen = false }: LayerProps) {
  return (
    <details className="content-layer layer-deepdive" open={defaultOpen}>
      <summary className="layer-header">
        <span className="layer-icon">🔍</span>
        <span className="layer-title">{title}</span>
        <span className="layer-toggle-icon" aria-hidden="true"></span>
      </summary>
      <div className="layer-content">{children}</div>
    </details>
  );
}

export default { Intuition, Mathematical, Implementation, DeepDive };
