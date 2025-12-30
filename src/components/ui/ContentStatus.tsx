/**
 * ContentStatus - Displays the review status of content
 *
 * Shows readers whether content is AI-generated, editor-reviewed,
 * community-reviewed, or fully verified.
 *
 * Usage:
 *   <ContentStatus status="editor_reviewed" lastReviewed="2024-01-15" />
 */

import React from 'react';

type Status = 'draft' | 'editor_reviewed' | 'community_reviewed' | 'verified';

interface ContentStatusProps {
  status: Status;
  lastReviewed?: string | null;
  reviewedBy?: string | null;
}

const STATUS_CONFIG: Record<
  Status,
  {
    label: string;
    description: string;
    icon: string;
    className: string;
  }
> = {
  draft: {
    label: 'Draft',
    description: 'AI-generated content, pending review',
    icon: '📝',
    className: 'status-draft',
  },
  editor_reviewed: {
    label: 'Editor Reviewed',
    description: 'Reviewed and approved by the editor',
    icon: '✅',
    className: 'status-editor-reviewed',
  },
  community_reviewed: {
    label: 'Community Reviewed',
    description: 'Incorporates community feedback',
    icon: '👥',
    className: 'status-community-reviewed',
  },
  verified: {
    label: 'Verified',
    description: 'Code tested, demos verified working',
    icon: '🔒',
    className: 'status-verified',
  },
};

export function ContentStatus({ status, lastReviewed, reviewedBy }: ContentStatusProps) {
  const config = STATUS_CONFIG[status];

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <div className={`content-status ${config.className}`} title={config.description}>
      <span className="status-icon">{config.icon}</span>
      <span className="status-label">{config.label}</span>

      {lastReviewed && (
        <span className="status-date">Last reviewed: {formatDate(lastReviewed)}</span>
      )}

      {reviewedBy && <span className="status-reviewer">by {reviewedBy}</span>}
    </div>
  );
}

/**
 * Compact version for use in chapter lists
 */
export function ContentStatusBadge({ status }: { status: Status }) {
  const config = STATUS_CONFIG[status];

  return (
    <span className={`status-badge ${config.className}`} title={config.description}>
      {config.icon}
    </span>
  );
}

export default ContentStatus;
