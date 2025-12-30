/**
 * Chapter Navigation Components
 *
 * Components for navigating between chapters and sections.
 * All navigation uses slugs instead of chapter numbers for stability.
 *
 * Usage:
 *   <ChapterObjectives>
 *     - Objective 1
 *     - Objective 2
 *   </ChapterObjectives>
 *
 *   <NextChapter slug="exploration-exploitation" title="Exploration vs Exploitation" />
 *
 *   <CrossRef slug="intro-to-td" section="td-error" />
 */

import React from 'react';

interface ChapterObjectivesProps {
  children: React.ReactNode;
}

/**
 * Learning objectives displayed at the start of each chapter
 */
export function ChapterObjectives({ children }: ChapterObjectivesProps) {
  return (
    <div className="chapter-objectives">
      <h2 className="objectives-title">What You'll Learn</h2>
      <div className="objectives-list">{children}</div>
    </div>
  );
}

interface NextChapterProps {
  slug: string;
  title: string;
}

/**
 * Link to the next chapter
 */
export function NextChapter({ slug, title }: NextChapterProps) {
  const href = `/chapters/${slug}`;

  return (
    <a href={href} className="next-chapter">
      <span className="next-label">Next Chapter</span>
      <span className="next-title">{title}</span>
      <span className="next-arrow">→</span>
    </a>
  );
}

interface CrossRefProps {
  slug: string;
  section?: string;
  children?: React.ReactNode;
}

/**
 * Cross-reference to another chapter or section
 */
export function CrossRef({ slug, section, children }: CrossRefProps) {
  const href = section ? `/chapters/${slug}#${section}` : `/chapters/${slug}`;

  return (
    <a href={href} className="cross-ref">
      {children || slug}
    </a>
  );
}

interface KeyTakeawaysProps {
  children: React.ReactNode;
}

/**
 * Key takeaways summary at end of chapter
 */
export function KeyTakeaways({ children }: KeyTakeawaysProps) {
  return (
    <div className="key-takeaways">
      <h2 className="takeaways-title">Key Takeaways</h2>
      <div className="takeaways-list">{children}</div>
    </div>
  );
}

interface SectionSummaryProps {
  section: string;
  chapters: string[]; // Array of slugs
}

/**
 * Summary shown at end of a multi-chapter section
 */
export function SectionSummary({ section, chapters }: SectionSummaryProps) {
  return (
    <div className="section-summary">
      <h2 className="summary-title">Section Complete: {section}</h2>
      <p className="summary-text">
        You've completed the {section} section.
      </p>
      <div className="summary-chapters">
        {chapters.map(slug => (
          <a key={slug} href={`/chapters/${slug}`} className="summary-chapter-link">
            {slug}
          </a>
        ))}
      </div>
    </div>
  );
}

interface NextSectionProps {
  title: string;
  slug: string;
}

/**
 * Link to the next section
 */
export function NextSection({ title, slug }: NextSectionProps) {
  return (
    <a href={`/chapters/${slug}`} className="next-section">
      <span className="next-label">Next Section</span>
      <span className="next-title">{title}</span>
      <span className="next-arrow">→</span>
    </a>
  );
}

interface ColabLinkProps {
  notebook: string;
  children: React.ReactNode;
}

/**
 * Link to Google Colab notebook
 */
export function ColabLink({ notebook, children }: ColabLinkProps) {
  const colabUrl = `https://colab.research.google.com/github/ebilgin/rlbook/blob/main/notebooks/${notebook}`;

  return (
    <a href={colabUrl} target="_blank" rel="noopener noreferrer" className="colab-link">
      <span className="colab-icon">📓</span>
      <span className="colab-text">{children}</span>
      <span className="external-icon">↗</span>
    </a>
  );
}

export default {
  ChapterObjectives,
  NextChapter,
  CrossRef,
  KeyTakeaways,
  SectionSummary,
  NextSection,
  ColabLink,
};
