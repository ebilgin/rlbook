/**
 * Blog utilities for managing blog posts with author support
 */

export interface Author {
  name: string;
  slug: string;
  bio?: string;
  avatar?: string;
  twitter?: string;
  linkedin?: string;
  github?: string;
  website?: string;
}

export interface BlogPost {
  slug: string;
  dirName: string;
  title: string;
  description: string;
  author: Author;
  publishedDate: string;
  updatedDate?: string;
  tags?: string[];
}

// Known authors - add new authors here
export const authors: Record<string, Author> = {
  'enes-bilgin': {
    name: 'Enes Bilgin',
    slug: 'enes-bilgin',
    bio: 'Creator of rlbook.ai and author of "Mastering Reinforcement Learning with Python".',
    linkedin: 'https://www.linkedin.com/in/enesbilgin',
    github: 'https://github.com/ebilgin',
  },
};

/**
 * Get all blog posts from MDX modules
 */
export function getBlogPosts(modules: Record<string, unknown>): BlogPost[] {
  const posts: BlogPost[] = [];

  for (const [path, module] of Object.entries(modules)) {
    const mod = module as { frontmatter?: Record<string, unknown> };
    if (!mod.frontmatter) continue;

    // Extract directory name from path like /content/blog/post-slug/index.mdx
    const match = path.match(/\/content\/blog\/([^/]+)\/index\.mdx$/);
    if (!match) continue;

    const dirName = match[1];
    const fm = mod.frontmatter;

    // Get author info
    const authorSlug = (fm.author as string) || 'enes-bilgin';
    const author = authors[authorSlug] || {
      name: authorSlug,
      slug: authorSlug,
    };

    posts.push({
      slug: (fm.slug as string) || dirName,
      dirName,
      title: (fm.title as string) || dirName,
      description: (fm.description as string) || '',
      author,
      publishedDate: (fm.publishedDate as string) || new Date().toISOString(),
      updatedDate: fm.updatedDate as string | undefined,
      tags: (fm.tags as string[]) || [],
    });
  }

  // Sort by published date, newest first
  return posts.sort(
    (a, b) => new Date(b.publishedDate).getTime() - new Date(a.publishedDate).getTime()
  );
}

/**
 * Get posts by a specific author
 */
export function getPostsByAuthor(posts: BlogPost[], authorSlug: string): BlogPost[] {
  return posts.filter((post) => post.author.slug === authorSlug);
}

/**
 * Get all unique authors from posts
 */
export function getAuthorsFromPosts(posts: BlogPost[]): Author[] {
  const authorMap = new Map<string, Author>();
  for (const post of posts) {
    if (!authorMap.has(post.author.slug)) {
      authorMap.set(post.author.slug, post.author);
    }
  }
  return Array.from(authorMap.values());
}

/**
 * Format date for display
 * Handles date strings without time component by parsing as local date
 */
export function formatDate(dateString: string): string {
  // Parse as local date to avoid timezone shifting
  // If dateString is "2025-12-30", create date in local timezone
  const parts = dateString.split('-');
  const date = parts.length === 3
    ? new Date(parseInt(parts[0]), parseInt(parts[1]) - 1, parseInt(parts[2]))
    : new Date(dateString);

  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
}
