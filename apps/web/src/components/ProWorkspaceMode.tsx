import { memo, type ReactNode } from "react";
import type { ChatStreamTimingMetrics } from "../api/chatApi";
import {
  DebugSnapshotGrid,
  PromptWorkshopPanel,
  type DebugSnapshotGridProps,
  type PromptWorkshopPanelProps,
} from "../debugPanels";
import { StoryPlanningPanel, type StoryPlanningPanelProps } from "./StoryPlanningPanel";

type WorkspaceStatusBarProps = {
  sessionId: number | null;
  ghostAutoEnabled: boolean;
  referenceProjectIds: number[];
  retrievalDegraded: boolean;
  degradedReasons: string[];
  lastStreamMetrics: ChatStreamTimingMetrics | null;
};

const WorkspaceStatusBar = memo(function WorkspaceStatusBar({
  sessionId,
  ghostAutoEnabled,
  referenceProjectIds,
  retrievalDegraded,
  degradedReasons,
  lastStreamMetrics,
}: WorkspaceStatusBarProps) {
  const firstTokenLabel =
    lastStreamMetrics?.firstTokenMs === null || lastStreamMetrics?.firstTokenMs === undefined
      ? "--"
      : `${Math.round(lastStreamMetrics.firstTokenMs)}ms`;
  return (
    <section className="workspace-bar">
      <div className="status-chip">
        <span>当前模式</span>
        <strong>工作台模式</strong>
      </div>
      <div className="status-chip">
        <span>会话 ID</span>
        <strong>{sessionId ?? "未创建"}</strong>
      </div>
      <div className="status-chip">
        <span>Ghost 策略</span>
        <strong>{ghostAutoEnabled ? "自动触发" : "手动触发"}</strong>
      </div>
      <div className="status-chip">
        <span>引用项目</span>
        <strong>{referenceProjectIds.length ? referenceProjectIds.join(", ") : "无"}</strong>
      </div>
      <div className={`status-chip ${retrievalDegraded ? "warn" : ""}`}>
        <span>检索状态</span>
        <strong>{retrievalDegraded ? "已降级（不中断写作）" : "正常"}</strong>
        {degradedReasons.length > 0 ? <small>{degradedReasons.slice(0, 2).join(" / ")}</small> : null}
      </div>
      <div className="status-chip">
        <span>Stream 指标</span>
        {lastStreamMetrics ? (
          <strong>{`TTFB ${Math.round(lastStreamMetrics.ttfbMs)}ms / 首 token ${firstTokenLabel}`}</strong>
        ) : (
          <strong>未采样</strong>
        )}
      </div>
    </section>
  );
});

export type WorkbenchPanelVisibility = {
  actions: boolean;
  prompt: boolean;
  planning: boolean;
  snapshot: boolean;
};

type WorkbenchPanelKey = keyof WorkbenchPanelVisibility;

const WORKBENCH_PANEL_LABELS: Record<WorkbenchPanelKey, string> = {
  actions: "动作提议 + 图谱",
  prompt: "Prompt + 知识库",
  planning: "结构化大纲",
  snapshot: "检索快照",
};

type WorkbenchPanelBarProps = {
  visibility: WorkbenchPanelVisibility;
  onToggle: (panelKey: WorkbenchPanelKey) => void;
};

const WorkbenchPanelBar = memo(function WorkbenchPanelBar({ visibility, onToggle }: WorkbenchPanelBarProps) {
  const allHidden = !visibility.actions && !visibility.prompt && !visibility.planning && !visibility.snapshot;
  return (
    <section className="panel workbench-panel-bar">
      <div className="panel-title sub">
        <h3>工作台面板</h3>
        <small>默认仅展示核心面板，可按需展开</small>
      </div>
      <div className="workbench-panel-toggles">
        {(Object.keys(WORKBENCH_PANEL_LABELS) as WorkbenchPanelKey[]).map((panelKey) => (
          <label key={panelKey} className="workbench-panel-toggle">
            <input type="checkbox" checked={visibility[panelKey]} onChange={() => onToggle(panelKey)} />
            <span>{WORKBENCH_PANEL_LABELS[panelKey]}</span>
          </label>
        ))}
      </div>
      {allHidden ? <p className="workbench-panel-hint">当前未显示任何工作台面板，建议至少开启一个。</p> : null}
    </section>
  );
});

export type ProWorkspaceModeProps = {
  statusBar: WorkspaceStatusBarProps;
  workbenchPanelVisibility: WorkbenchPanelVisibility;
  onToggleWorkbenchPanel: (panelKey: WorkbenchPanelKey) => void;
  actionsPanelNode: ReactNode;
  promptPanelReady: boolean;
  promptPanelProps: PromptWorkshopPanelProps;
  planningPanelProps: StoryPlanningPanelProps;
  snapshotPanelReady: boolean;
  snapshotPanelProps: DebugSnapshotGridProps;
};

export const ProWorkspaceMode = memo(function ProWorkspaceMode({
  statusBar,
  workbenchPanelVisibility,
  onToggleWorkbenchPanel,
  actionsPanelNode,
  promptPanelReady,
  promptPanelProps,
  planningPanelProps,
  snapshotPanelReady,
  snapshotPanelProps,
}: ProWorkspaceModeProps) {
  return (
    <>
      <WorkspaceStatusBar {...statusBar} />

      <WorkbenchPanelBar visibility={workbenchPanelVisibility} onToggle={onToggleWorkbenchPanel} />

      {workbenchPanelVisibility.actions ? actionsPanelNode : null}

      {workbenchPanelVisibility.prompt && promptPanelReady ? <PromptWorkshopPanel {...promptPanelProps} /> : null}

      {workbenchPanelVisibility.planning ? <StoryPlanningPanel {...planningPanelProps} /> : null}

      {workbenchPanelVisibility.snapshot && snapshotPanelReady ? <DebugSnapshotGrid {...snapshotPanelProps} /> : null}
    </>
  );
});
