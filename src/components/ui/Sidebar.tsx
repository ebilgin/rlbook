/**
 * Chapter Sidebar Navigation
 *
 * Displays hierarchical chapter navigation grouped by section.
 * Supports three levels: Section > Chapter > Subsection
 * Also includes global complexity controls for Math/Code visibility.
 */

import React, { useState, useEffect } from 'react';
import type { SubsectionData } from '../../lib/chapters';

// Storage key for complexity preferences
const STORAGE_KEY = 'rlbook-complexity-preferences';

interface ComplexityPreferences {
  showMath: boolean;
  showCode: boolean;
}

const defaultPreferences: ComplexityPreferences = {
  showMath: true,
  showCode: true,
};

interface ChapterData {
  slug: string;
  title: string;
  section: string;
  subsections?: SubsectionData[];
}

interface SidebarProps {
  chapters: ChapterData[];
  currentSlug?: string;
  currentSubsectionSlug?: string;
  prevChapter?: { slug: string; title: string };
  nextChapter?: { slug: string; title: string };
  subsections?: SubsectionData[];
}

export function Sidebar({
  chapters,
  currentSlug,
  currentSubsectionSlug,
  prevChapter,
  nextChapter,
  subsections = [],
}: SidebarProps) {
  // Group chapters by section
  const sections = new Map<string, ChapterData[]>();
  for (const chapter of chapters) {
    const existing = sections.get(chapter.section) || [];
    existing.push(chapter);
    sections.set(chapter.section, existing);
  }

  // Track which sections are expanded (default: current section is open)
  const currentSection = chapters.find(c => c.slug === currentSlug)?.section;
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(currentSection ? [currentSection] : [])
  );

  // Track which chapters are expanded (default: current chapter if it has subsections)
  const [expandedChapters, setExpandedChapters] = useState<Set<string>>(
    new Set(currentSlug && subsections.length > 0 ? [currentSlug] : [])
  );

  // Complexity preferences state
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

  const toggleSection = (section: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev);
      if (next.has(section)) {
        next.delete(section);
      } else {
        next.add(section);
      }
      return next;
    });
  };

  const toggleChapter = (chapterSlug: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setExpandedChapters(prev => {
      const next = new Set(prev);
      if (next.has(chapterSlug)) {
        next.delete(chapterSlug);
      } else {
        next.add(chapterSlug);
      }
      return next;
    });
  };

  return (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <a href="/" className="block px-4 py-4 text-xl font-bold text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 border-b border-gray-200 dark:border-gray-700">
        rlbook.ai
      </a>

      {/* Content visibility toggles */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
          Show Content
        </div>
        <div className="flex gap-2">
          <button
            onClick={toggleMath}
            className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded text-xs font-medium transition-all ${
              preferences.showMath
                ? 'bg-purple-600 dark:bg-purple-700 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-400 dark:text-gray-500'
            }`}
            title={preferences.showMath ? 'Hide math sections' : 'Show math sections'}
          >
            <span>∑</span>
            <span>Math</span>
          </button>
          <button
            onClick={toggleCode}
            className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded text-xs font-medium transition-all ${
              preferences.showCode
                ? 'bg-emerald-600 dark:bg-emerald-700 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-400 dark:text-gray-500'
            }`}
            title={preferences.showCode ? 'Hide code sections' : 'Show code sections'}
          >
            <span>&lt;/&gt;</span>
            <span>Code</span>
          </button>
        </div>
      </div>

      {/* Chapter navigation */}
      <nav className="flex-1 overflow-y-auto py-4">
        {Array.from(sections.entries()).map(([section, sectionChapters]) => {
          const isExpanded = expandedSections.has(section);
          const hasCurrentChapter = sectionChapters.some(c => c.slug === currentSlug);

          return (
            <div key={section} className="mb-2">
              {/* Section header */}
              <button
                onClick={() => toggleSection(section)}
                className={`w-full flex items-center justify-between px-4 py-2 text-sm font-medium text-left transition-colors ${
                  hasCurrentChapter
                    ? 'text-primary-700 dark:text-primary-300 bg-primary-50 dark:bg-primary-900/30'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800'
                }`}
              >
                <span>{section}</span>
                <svg
                  className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>

              {/* Chapter list */}
              {isExpanded && (
                <ul className="mt-1 space-y-1">
                  {sectionChapters.map(chapter => {
                    const isCurrent = chapter.slug === currentSlug;
                    const hasSubsections = chapter.subsections && chapter.subsections.length > 0;
                    const isChapterExpanded = expandedChapters.has(chapter.slug);
                    // Get subsections for this chapter (either from props or chapter data)
                    const chapterSubsections = isCurrent ? subsections : (chapter.subsections || []);

                    return (
                      <li key={chapter.slug}>
                        <div className="flex items-center">
                          <a
                            href={`/chapters/${chapter.slug}`}
                            className={`flex-1 block px-4 py-2 pl-8 text-sm transition-colors ${
                              isCurrent && !currentSubsectionSlug
                                ? 'text-primary-700 dark:text-primary-300 bg-primary-100 dark:bg-primary-900/50 font-medium border-l-2 border-primary-600 dark:border-primary-400'
                                : isCurrent
                                  ? 'text-primary-700 dark:text-primary-300 bg-primary-50 dark:bg-primary-900/30 font-medium'
                                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-50 dark:hover:bg-gray-800'
                            }`}
                          >
                            {chapter.title}
                          </a>
                          {hasSubsections && (
                            <button
                              onClick={(e) => toggleChapter(chapter.slug, e)}
                              className="p-2 text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300"
                              aria-label={isChapterExpanded ? 'Collapse' : 'Expand'}
                            >
                              <svg
                                className={`w-3 h-3 transition-transform ${isChapterExpanded ? 'rotate-90' : ''}`}
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                              >
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                              </svg>
                            </button>
                          )}
                        </div>

                        {/* Subsection list */}
                        {hasSubsections && isChapterExpanded && (
                          <ul className="ml-4 mt-1 space-y-0.5 border-l border-gray-200 dark:border-gray-700">
                            {chapterSubsections.map((sub, i) => {
                              const isCurrentSub = isCurrent && currentSubsectionSlug === sub.slug;
                              return (
                                <li key={sub.slug}>
                                  <a
                                    href={`/chapters/${chapter.slug}/${sub.slug}`}
                                    className={`block px-4 py-1.5 pl-6 text-xs transition-colors ${
                                      isCurrentSub
                                        ? 'text-primary-700 dark:text-primary-300 bg-primary-100 dark:bg-primary-900/50 font-medium'
                                        : 'text-gray-500 dark:text-gray-500 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-50 dark:hover:bg-gray-800'
                                    }`}
                                  >
                                    <span className="text-gray-400 dark:text-gray-600 mr-1.5">{i + 1}.</span>
                                    {sub.title}
                                  </a>
                                </li>
                              );
                            })}
                          </ul>
                        )}
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          );
        })}
      </nav>

      {/* Prev/Next navigation */}
      {(prevChapter || nextChapter) && (
        <div className="border-t border-gray-200 dark:border-gray-700 p-4 space-y-2">
          {prevChapter && (
            <a
              href={`/chapters/${prevChapter.slug}`}
              className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              <span className="truncate">{prevChapter.title}</span>
            </a>
          )}
          {nextChapter && (
            <a
              href={`/chapters/${nextChapter.slug}`}
              className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
            >
              <span className="truncate">{nextChapter.title}</span>
              <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </a>
          )}
        </div>
      )}

      {/* All chapters link */}
      <div className="border-t border-gray-200 dark:border-gray-700 p-4">
        <a
          href="/chapters"
          className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
          </svg>
          <span>All Chapters</span>
        </a>
      </div>
    </div>
  );
}

export default Sidebar;
