/**
 * Chapter data definitions
 *
 * This is the single source of truth for chapter metadata.
 * Directory numbering uses increments of 10 (1010, 1020, etc.) to allow inserting chapters.
 * Navigation uses slugs instead of numbers for URL stability.
 */

export type ChapterStatus = 'draft' | 'editor_reviewed' | 'community_reviewed' | 'verified';

export interface ChapterData {
  slug: string;
  dirName: string;
  title: string;
  section: string;
  description: string;
  status: ChapterStatus;
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
  },
  'multi-armed-bandits': {
    slug: 'multi-armed-bandits',
    dirName: '0020-multi-armed-bandits',
    title: 'Multi-Armed Bandits',
    section: 'Foundations',
    description: 'Master the exploration-exploitation tradeoff in the simplest RL setting',
    status: 'draft',
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
