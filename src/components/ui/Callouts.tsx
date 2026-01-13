/**
 * Callout Components - Note, Warning, Tip, etc.
 *
 * Styled callout boxes for important information.
 *
 * Usage:
 *   <Note>Important context that doesn't fit the main flow.</Note>
 *   <Warning>A common mistake or pitfall to avoid.</Warning>
 *   <Tip>A practical suggestion or shortcut.</Tip>
 */

import React from "react";

interface CalloutProps {
  children: React.ReactNode;
  title?: string;
}

/**
 * Note - Important context or additional information
 */
export function Note({ children, title = "Note" }: CalloutProps) {
  return (
    <div className="callout callout-note" role="note">
      <div className="callout-header">
        <span className="callout-icon">ℹ️</span>
        <span className="callout-title">{title}</span>
      </div>
      <div className="callout-content">{children}</div>
    </div>
  );
}

/**
 * Warning - Common mistakes or pitfalls
 */
export function Warning({ children, title = "Warning" }: CalloutProps) {
  return (
    <div className="callout callout-warning" role="alert">
      <div className="callout-header">
        <span className="callout-icon">⚠️</span>
        <span className="callout-title">{title}</span>
      </div>
      <div className="callout-content">{children}</div>
    </div>
  );
}

/**
 * Tip - Practical suggestions or shortcuts
 */
export function Tip({ children, title = "Tip" }: CalloutProps) {
  return (
    <div className="callout callout-tip">
      <div className="callout-header">
        <span className="callout-icon">💡</span>
        <span className="callout-title">{title}</span>
      </div>
      <div className="callout-content">{children}</div>
    </div>
  );
}

/**
 * Example - Worked example or demonstration
 */
export function Example({ children, title = "Example" }: CalloutProps) {
  return (
    <div className="callout callout-example">
      <div className="callout-header">
        <span className="callout-icon">📌</span>
        <span className="callout-title">{title}</span>
      </div>
      <div className="callout-content">{children}</div>
    </div>
  );
}

export function Question({ children, title = "Question" }: CalloutProps) {
  return (
    <div className="callout callout-question">
      <div className="callout-header">
        <span className="callout-icon">❓</span>
        <span className="callout-title">{title}</span>
      </div>
      <div className="callout-content">{children}</div>
    </div>
  );
}

/**
 * Definition - Formal definition of a term
 */
export function Definition({ children, title }: CalloutProps) {
  return (
    <div className="callout callout-definition">
      <div className="callout-header">
        <span className="callout-icon">📖</span>
        <span className="callout-title">{title || "Definition"}</span>
      </div>
      <div className="callout-content">{children}</div>
    </div>
  );
}

/**
 * Theorem - Mathematical theorem or proposition
 */
export function Theorem({ children, title }: CalloutProps) {
  return (
    <div className="callout callout-theorem">
      <div className="callout-header">
        <span className="callout-icon">📐</span>
        <span className="callout-title">{title || "Theorem"}</span>
      </div>
      <div className="callout-content">{children}</div>
    </div>
  );
}

/**
 * Proof - Mathematical proof (collapsible)
 */
export function Proof({ children, title = "Proof" }: CalloutProps) {
  const [isExpanded, setIsExpanded] = React.useState(false);

  return (
    <div
      className={`callout callout-proof ${
        isExpanded ? "expanded" : "collapsed"
      }`}
    >
      <button
        className="callout-header"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span className="callout-icon">✏️</span>
        <span className="callout-title">{title}</span>
        <span className="callout-toggle">{isExpanded ? "−" : "+"}</span>
      </button>
      {isExpanded && (
        <div className="callout-content">
          {children}
          <div className="proof-end">∎</div>
        </div>
      )}
    </div>
  );
}

export default {
  Note,
  Warning,
  Tip,
  Example,
  Question,
  Definition,
  Theorem,
  Proof,
};
