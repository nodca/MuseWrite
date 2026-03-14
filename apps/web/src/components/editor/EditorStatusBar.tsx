import { memo } from "react";
import type { DraftAutoSaveState } from "../../types";
import { formatDateTime } from "../../utils/formatting";

export type EditorStatusBarProps = {
  draftWordCount: number;
  draftVersion: number;
  draftUpdatedAt: string | null;
  autoSaveState: DraftAutoSaveState;
  autoSaveAt: string | null;
  draftFocusMode: boolean;
  typewriterModeEnabled: boolean;
  selectedDraftText: string;
  latestAssistantReply: string;
};

function renderAutoSaveText(state: DraftAutoSaveState, autoSaveAt: string | null): string {
  switch (state) {
    case "pending":
      return "自动保存：等待中";
    case "saving":
      return "自动保存：保存中";
    case "saved":
      return `已保存(${formatDateTime(autoSaveAt)})`;
    case "error":
      return "自动保存：失败";
    case "idle":
      return "自动保存：空闲";
    default:
      return "";
  }
}

export const EditorStatusBar = memo(function EditorStatusBar({
  draftWordCount,
  draftVersion,
  draftUpdatedAt,
  autoSaveState,
  autoSaveAt,
  draftFocusMode,
  typewriterModeEnabled,
  selectedDraftText,
  latestAssistantReply,
}: EditorStatusBarProps) {
  return (
    <div className="flex items-center justify-between px-4 py-1.5 text-xs text-text-secondary border-t border-border-default bg-surface-base/50">
      <div className="flex items-center gap-3">
        <span>{draftWordCount} 字</span>
        <span>v{draftVersion}</span>
        {draftUpdatedAt && <span>{formatDateTime(draftUpdatedAt)}</span>}
        <span>{renderAutoSaveText(autoSaveState, autoSaveAt)}</span>
        {draftFocusMode && <span>专注模式</span>}
        {typewriterModeEnabled && <span>打字机</span>}
      </div>
      <div className="flex items-center gap-3">
        {selectedDraftText.length > 0 && <span>已选 {selectedDraftText.length} 字</span>}
        {latestAssistantReply ? (
          <span>可写回正文</span>
        ) : (
          <span>暂无助手回复</span>
        )}
      </div>
    </div>
  );
});
