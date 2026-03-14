import { memo } from "react";
import { formatDateTime } from "../utils/formatting";
import type { ProjectChapterRevision } from "../types";

export type DraftRevisionListProps = {
  draftRevisions: ProjectChapterRevision[];
  disabled: boolean;
  onRollbackDraftToVersion: (targetVersion: number) => Promise<void>;
};

export const DraftRevisionList = memo(function DraftRevisionList({
  draftRevisions,
  disabled,
  onRollbackDraftToVersion,
}: DraftRevisionListProps) {
  return (
    <details className="draft-history">
      <summary>版本历史（最近 {draftRevisions.length} 条）</summary>
      <div className="draft-revision-list">
        {draftRevisions.length === 0 ? <p className="empty">暂无版本历史</p> : null}
        {draftRevisions.map((revision) => (
          <article key={revision.id} className="draft-revision-card">
            <div className="msg-head">
              <span>
                v{revision.version} · {revision.source}
              </span>
              <small>{formatDateTime(revision.created_at)}</small>
            </div>
            {(revision.semantic_summary ?? []).length > 0 ? (
              <ul className="revision-semantic-list">
                {(revision.semantic_summary ?? []).map((line, idx) => (
                  <li key={`${revision.id}-semantic-${idx}`}>{line}</li>
                ))}
              </ul>
            ) : null}
            <pre>{revision.content.slice(0, 220) || "(空正文)"}</pre>
            <div className="action-ops">
              <button
                className="btn ghost tiny"
                onClick={() => void onRollbackDraftToVersion(revision.version)}
                disabled={disabled}
              >
                回滚到此版本
              </button>
            </div>
          </article>
        ))}
      </div>
    </details>
  );
});
