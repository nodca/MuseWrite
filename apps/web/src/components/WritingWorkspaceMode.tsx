import { memo } from "react";
import { ErrorBoundary } from "./ErrorBoundary";
import { DraftWorkspacePanel, type DraftWorkspacePanelProps } from "./DraftWorkspacePanel";

export type WritingWorkspaceModeProps = DraftWorkspacePanelProps;

export const WritingWorkspaceMode = memo(function WritingWorkspaceMode(props: WritingWorkspaceModeProps) {
  return (
    <ErrorBoundary
      fallbackTitle="写作区暂时不可用"
      fallbackDescription="编辑器出现异常，已进入降级模式。请刷新页面后继续写作。"
    >
      <DraftWorkspacePanel {...props} />
    </ErrorBoundary>
  );
});
