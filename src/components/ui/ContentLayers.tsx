/**
 * Content Layer Components - Intuition, Mathematical, Implementation
 *
 * These components wrap content at different complexity levels.
 * They work with ComplexityToggle via CSS classes that get toggled
 * on the document body.
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
}

/**
 * Intuition layer - always visible
 * Core concepts explained in plain English with visualizations
 */
export function Intuition({ children, title }: LayerProps) {
  return (
    <div className="content-layer layer-intuition not-prose my-6 rounded-lg overflow-hidden">
      {title && (
        <div className="layer-header px-4 py-2 bg-blue-600 dark:bg-blue-700">
          <h4 className="layer-title font-semibold text-white">{title}</h4>
        </div>
      )}
      <div className="layer-content p-4 bg-blue-50 dark:bg-gray-800 border-l-4 border-blue-500 prose prose-blue dark:prose-invert max-w-none">{children}</div>
    </div>
  );
}

/**
 * Mathematical layer - controlled by ComplexityToggle via CSS
 * Formal definitions, equations, derivations
 */
export function Mathematical({ children, title = 'Mathematical Details' }: LayerProps) {
  return (
    <div className="content-layer layer-mathematical not-prose my-6 rounded-lg overflow-hidden" data-layer="math">
      <div className="layer-header flex items-center gap-2 px-4 py-2 bg-purple-600 dark:bg-purple-700">
        <span className="layer-icon text-white text-lg">∑</span>
        <span className="layer-title font-semibold text-white">{title}</span>
      </div>
      <div className="layer-content p-4 bg-purple-50 dark:bg-gray-800 border-l-4 border-purple-500 prose prose-purple dark:prose-invert max-w-none">{children}</div>
    </div>
  );
}

/**
 * Implementation layer - controlled by ComplexityToggle via CSS
 * Code examples, practical details, debugging tips
 */
export function Implementation({ children, title = 'Implementation' }: LayerProps) {
  return (
    <div className="content-layer layer-implementation not-prose my-6 rounded-lg overflow-hidden" data-layer="code">
      <div className="layer-header flex items-center gap-2 px-4 py-2 bg-emerald-600 dark:bg-emerald-700">
        <span className="layer-icon text-white text-lg">&lt;/&gt;</span>
        <span className="layer-title font-semibold text-white">{title}</span>
      </div>
      <div className="layer-content p-4 bg-emerald-50 dark:bg-gray-800 border-l-4 border-emerald-500 prose prose-emerald dark:prose-invert max-w-none">{children}</div>
    </div>
  );
}

/**
 * DeepDive - optional advanced content
 * For curious readers who want to go beyond the main material
 * Uses native details/summary for individual toggling
 */
export function DeepDive({ children, title = 'Deep Dive' }: LayerProps) {
  return (
    <details className="content-layer layer-deepdive not-prose my-6 rounded-lg overflow-hidden group">
      <summary className="layer-header flex items-center gap-2 px-4 py-2 bg-amber-600 dark:bg-amber-700 cursor-pointer list-none">
        <span className="layer-icon text-white text-lg">🔍</span>
        <span className="layer-title font-semibold text-white">{title}</span>
        <span className="ml-auto text-white group-open:rotate-90 transition-transform">▶</span>
      </summary>
      <div className="layer-content p-4 bg-amber-50 dark:bg-gray-800 border-l-4 border-amber-500 prose prose-amber dark:prose-invert max-w-none">{children}</div>
    </details>
  );
}

export default { Intuition, Mathematical, Implementation, DeepDive };
