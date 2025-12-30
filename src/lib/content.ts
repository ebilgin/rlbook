/**
 * Generic content utilities for papers, applications, infrastructure, environments
 *
 * Scans content directories and extracts frontmatter for listing pages.
 */

export type ContentStatus = 'draft' | 'editor_reviewed' | 'community_reviewed' | 'verified';

export interface ContentItem {
  slug: string;
  dirName: string;
  title: string;
  description: string;
  section?: string;
  status: ContentStatus;
}

/**
 * Get all content items from a specific content type directory
 */
export function getContentItems(
  modules: Record<string, unknown>,
  contentType: string
): ContentItem[] {
  const items: ContentItem[] = [];

  for (const [path, module] of Object.entries(modules)) {
    const mod = module as { frontmatter?: Record<string, unknown> };
    if (!mod.frontmatter) continue;

    // Extract directory name from path like /content/papers/grpo/index.mdx
    const match = path.match(new RegExp(`/content/${contentType}/([^/]+)/index\\.mdx$`));
    if (!match) continue;

    const dirName = match[1];
    const fm = mod.frontmatter;

    items.push({
      slug: (fm.slug as string) || dirName,
      dirName,
      title: (fm.title as string) || dirName,
      description: (fm.description as string) || '',
      section: fm.section as string | undefined,
      status: (fm.status as ContentStatus) || 'draft',
    });
  }

  // Sort by directory name
  return items.sort((a, b) => a.dirName.localeCompare(b.dirName));
}

/**
 * Status display helpers
 */
export const statusIcons: Record<ContentStatus, string> = {
  draft: '📝',
  editor_reviewed: '✅',
  community_reviewed: '👥',
  verified: '🔒',
};

export const statusLabels: Record<ContentStatus, string> = {
  draft: 'AI Generated',
  editor_reviewed: 'Editor Reviewed',
  community_reviewed: 'Community Reviewed',
  verified: 'Verified',
};
