/**
 * Chapter data definitions
 *
 * This is the single source of truth for chapter metadata.
 *
 * Structure:
 * - Level 1: Section (e.g., "Foundations", "Q-Learning Foundations")
 * - Level 2: Chapter (e.g., "Introduction to RL", "Multi-Armed Bandits")
 * - Level 3: Subsection/Topic (e.g., "What is RL?", "The Agent-Environment Interface")
 *
 * Directory numbering uses increments of 10 (0010, 0020, etc.) to allow inserting.
 * Subsection numbering uses increments of 10 (01, 02, etc.) within each chapter.
 */

export type ChapterStatus = 'draft' | 'editor_reviewed' | 'community_reviewed' | 'verified';

export interface SubsectionData {
  slug: string;           // e.g., "what-is-rl"
  title: string;          // e.g., "What is Reinforcement Learning?"
  description: string;
  order: number;          // For sorting (10, 20, 30...)
}

export interface ChapterData {
  slug: string;           // e.g., "intro-to-rl"
  dirName: string;        // e.g., "0010-intro-to-rl"
  title: string;
  section: string;
  description: string;
  status: ChapterStatus;
  subsections?: SubsectionData[];
}

// Full URL path for a subsection: /chapters/{chapterSlug}/{subsectionSlug}
export interface NavigationItem {
  type: 'chapter' | 'subsection';
  chapterSlug: string;
  subsectionSlug?: string;
  title: string;
  section: string;
  fullPath: string;
}

export const chapters: Record<string, ChapterData> = {
  // Foundations Section
  'intro-to-rl': {
    slug: 'intro-to-rl',
    dirName: '0010-intro-to-rl',
    title: 'Introduction to Reinforcement Learning',
    section: 'Foundations',
    description: 'Understand the core concepts of RL: agents, environments, rewards, and the learning loop',
    status: 'draft',
    subsections: [
      {
        slug: 'what-is-rl',
        title: 'What is Reinforcement Learning?',
        description: 'The big picture: learning from interaction',
        order: 10,
      },
      {
        slug: 'agent-environment',
        title: 'The Agent-Environment Interface',
        description: 'States, actions, and the interaction loop',
        order: 20,
      },
      {
        slug: 'rewards-returns',
        title: 'Rewards and Returns',
        description: 'Defining goals through reward signals',
        order: 30,
      },
      {
        slug: 'policies-values',
        title: 'Policies and Value Functions',
        description: 'How agents represent knowledge',
        order: 40,
      },
      {
        slug: 'rl-landscape',
        title: 'The RL Landscape',
        description: 'Model-free vs model-based, value vs policy methods',
        order: 50,
      },
    ],
  },
  'multi-armed-bandits': {
    slug: 'multi-armed-bandits',
    dirName: '0020-multi-armed-bandits',
    title: 'Multi-Armed Bandits',
    section: 'Foundations',
    description: 'Master the exploration-exploitation tradeoff in the simplest RL setting',
    status: 'draft',
    subsections: [
      {
        slug: 'bandit-problem',
        title: 'The Bandit Problem',
        description: 'A simplified RL setting for understanding exploration',
        order: 10,
      },
      {
        slug: 'action-value-methods',
        title: 'Action-Value Methods',
        description: 'Estimating the value of actions',
        order: 20,
      },
      {
        slug: 'epsilon-greedy',
        title: 'Epsilon-Greedy Exploration',
        description: 'Balancing exploration and exploitation',
        order: 30,
      },
      {
        slug: 'ucb',
        title: 'Upper Confidence Bound',
        description: 'Optimism in the face of uncertainty',
        order: 40,
      },
      {
        slug: 'thompson-sampling',
        title: 'Thompson Sampling',
        description: 'Bayesian approach to exploration',
        order: 50,
      },
    ],
  },
  'contextual-bandits': {
    slug: 'contextual-bandits',
    dirName: '0030-contextual-bandits',
    title: 'Contextual Bandits',
    section: 'Foundations',
    description: 'Learn to make personalized decisions based on context features',
    status: 'draft',
  },
  // Q-Learning Foundations Section
  'intro-to-td': {
    slug: 'intro-to-td',
    dirName: '1010-intro-to-td',
    title: 'Introduction to TD Learning',
    section: 'Q-Learning Foundations',
    description: 'Learn how TD methods combine the best of Monte Carlo and Dynamic Programming',
    status: 'draft',
  },
  'q-learning-basics': {
    slug: 'q-learning-basics',
    dirName: '1020-q-learning-basics',
    title: 'Q-Learning Basics',
    section: 'Q-Learning Foundations',
    description: 'Master the foundational algorithm for learning optimal behavior',
    status: 'draft',
  },
  'exploration-exploitation': {
    slug: 'exploration-exploitation',
    dirName: '1030-exploration-exploitation',
    title: 'Exploration vs Exploitation',
    section: 'Q-Learning Foundations',
    description: 'Balance discovery with optimization using proven strategies',
    status: 'draft',
  },
  'deep-q-networks': {
    slug: 'deep-q-networks',
    dirName: '1040-deep-q-networks',
    title: 'Deep Q-Networks',
    section: 'Q-Learning Foundations',
    description: 'Scale Q-learning with neural networks, experience replay, and target networks',
    status: 'draft',
  },
  'q-learning-applications': {
    slug: 'q-learning-applications',
    dirName: '1050-q-learning-applications',
    title: 'Q-Learning Applications',
    section: 'Q-Learning Foundations',
    description: 'Apply Q-learning to real-world problems in games, robotics, and finance',
    status: 'draft',
  },
  'q-learning-frontiers': {
    slug: 'q-learning-frontiers',
    dirName: '1060-q-learning-frontiers',
    title: 'Q-Learning Frontiers',
    section: 'Q-Learning Foundations',
    description: 'Explore the limits of Q-learning and preview what comes next',
    status: 'draft',
  },
  // Policy Gradient Methods Section
  'intro-to-policy-gradients': {
    slug: 'intro-to-policy-gradients',
    dirName: '2010-intro-to-policy-gradients',
    title: 'Introduction to Policy-Based Methods',
    section: 'Policy Gradient Methods',
    description: 'Discover a fundamentally different approach: learning policies directly instead of value functions',
    status: 'draft',
  },
  'policy-gradient-theorem': {
    slug: 'policy-gradient-theorem',
    dirName: '2020-policy-gradient-theorem',
    title: 'The Policy Gradient Theorem and REINFORCE',
    section: 'Policy Gradient Methods',
    description: 'Master the fundamental theorem that enables learning policies through gradient ascent',
    status: 'draft',
  },
  'actor-critic-methods': {
    slug: 'actor-critic-methods',
    dirName: '2030-actor-critic-methods',
    title: 'Actor-Critic Methods',
    section: 'Policy Gradient Methods',
    description: 'Combine the best of policy gradients and value-based learning for stable, efficient training',
    status: 'draft',
  },
  'ppo-and-advanced-pg': {
    slug: 'ppo-and-advanced-pg',
    dirName: '2040-ppo-and-advanced-pg',
    title: 'PPO and Trust Region Methods',
    section: 'Policy Gradient Methods',
    description: 'Master the most popular deep RL algorithm and understand why it works',
    status: 'draft',
  },
  'policy-methods-applications': {
    slug: 'policy-methods-applications',
    dirName: '2050-policy-methods-applications',
    title: 'Policy Gradient Methods in Practice',
    section: 'Policy Gradient Methods',
    description: 'Apply policy gradient methods to real-world challenges in robotics, RLHF, and beyond',
    status: 'draft',
  },
};

/**
 * Get all chapter slugs
 */
export function getChapterSlugs(): string[] {
  return Object.keys(chapters);
}

/**
 * Get chapter data by slug
 */
export function getChapter(slug: string): ChapterData | undefined {
  return chapters[slug];
}

/**
 * Get all chapters as an array, sorted by directory name
 */
export function getChaptersArray(): ChapterData[] {
  return Object.values(chapters).sort((a, b) => a.dirName.localeCompare(b.dirName));
}

/**
 * Get chapters grouped by section, maintaining order
 */
export function getChaptersBySection(): Map<string, ChapterData[]> {
  const sorted = getChaptersArray();
  const sections = new Map<string, ChapterData[]>();

  for (const chapter of sorted) {
    const existing = sections.get(chapter.section) || [];
    existing.push(chapter);
    sections.set(chapter.section, existing);
  }

  return sections;
}

/**
 * Get previous and next chapters for navigation
 */
export function getAdjacentChapters(currentSlug: string): { prev?: ChapterData; next?: ChapterData } {
  const sorted = getChaptersArray();
  const currentIndex = sorted.findIndex(c => c.slug === currentSlug);

  if (currentIndex === -1) {
    return {};
  }

  return {
    prev: currentIndex > 0 ? sorted[currentIndex - 1] : undefined,
    next: currentIndex < sorted.length - 1 ? sorted[currentIndex + 1] : undefined,
  };
}

/**
 * Get a subsection by chapter and subsection slug
 */
export function getSubsection(chapterSlug: string, subsectionSlug: string): SubsectionData | undefined {
  const chapter = chapters[chapterSlug];
  if (!chapter?.subsections) return undefined;
  return chapter.subsections.find(s => s.slug === subsectionSlug);
}

/**
 * Get sorted subsections for a chapter
 */
export function getSubsectionsForChapter(chapterSlug: string): SubsectionData[] {
  const chapter = chapters[chapterSlug];
  if (!chapter?.subsections) return [];
  return [...chapter.subsections].sort((a, b) => a.order - b.order);
}

/**
 * Get adjacent subsections for navigation within a chapter
 */
export function getAdjacentSubsections(
  chapterSlug: string,
  subsectionSlug: string
): { prev?: SubsectionData; next?: SubsectionData; prevChapter?: ChapterData; nextChapter?: ChapterData } {
  const subsections = getSubsectionsForChapter(chapterSlug);
  const currentIndex = subsections.findIndex(s => s.slug === subsectionSlug);

  if (currentIndex === -1) {
    return {};
  }

  const result: ReturnType<typeof getAdjacentSubsections> = {};

  if (currentIndex > 0) {
    result.prev = subsections[currentIndex - 1];
  } else {
    // First subsection - link to previous chapter
    const { prev } = getAdjacentChapters(chapterSlug);
    result.prevChapter = prev;
  }

  if (currentIndex < subsections.length - 1) {
    result.next = subsections[currentIndex + 1];
  } else {
    // Last subsection - link to next chapter
    const { next } = getAdjacentChapters(chapterSlug);
    result.nextChapter = next;
  }

  return result;
}

/**
 * Build the full navigation path for a subsection
 */
export function getSubsectionPath(chapterSlug: string, subsectionSlug: string): string {
  return `/chapters/${chapterSlug}/${subsectionSlug}`;
}

/**
 * Get all navigation items (chapters + subsections) as a flat list
 * Useful for generating all static paths
 */
export function getAllNavigationItems(): NavigationItem[] {
  const items: NavigationItem[] = [];
  const sorted = getChaptersArray();

  for (const chapter of sorted) {
    // Add chapter as navigation item
    items.push({
      type: 'chapter',
      chapterSlug: chapter.slug,
      title: chapter.title,
      section: chapter.section,
      fullPath: `/chapters/${chapter.slug}`,
    });

    // Add subsections if they exist
    if (chapter.subsections) {
      const sortedSubs = [...chapter.subsections].sort((a, b) => a.order - b.order);
      for (const sub of sortedSubs) {
        items.push({
          type: 'subsection',
          chapterSlug: chapter.slug,
          subsectionSlug: sub.slug,
          title: sub.title,
          section: chapter.section,
          fullPath: `/chapters/${chapter.slug}/${sub.slug}`,
        });
      }
    }
  }

  return items;
}
