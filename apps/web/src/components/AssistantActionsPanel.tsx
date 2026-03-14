import { memo, useCallback, useEffect, useMemo, useState } from "react";
import {
  getActionBlastRadius,
  summarizeBlastRadius,
  buildBlastRadiusSummaryChips,
  normalizeGraphPreviewToken,
  buildGraphPreviewEdgeKey,
  resolveBlastRadiusTone,
} from "../lib/blastRadius";
import { summarizeAction } from "../lib/actionHelpers";
import { formatDateTime } from "../utils/formatting";
import { ActionCard } from "./ActionCard";
import { ActionLogsList } from "./ActionLogsList";
import { BlastRadiusDetailDisclosure } from "./BlastRadiusDetailDisclosure";
import { useTimelineGraphFlow } from "../hooks/useTimelineGraphFlow";
import type {
  ActionAuditLog,
  ChatAction,
  ChatStreamTraceEvent,
  ConsistencyAuditReport,
  GraphTimelineSnapshot,
} from "../types";

export type StreamLatencySample = {
  at: string;
  completeMs: number;
};

export type TokenUsageSample = {
  at: string;
  total: number;
};

export type RetrievalHitSample = {
  at: string;
  dsl: number;
  graph: number;
  rag: number;
};

// TODO(blast-radius): Apply 前总览确认区块（two-step apply）将在此处复用 blast radius 明细组件。
export type AssistantActionsPanelProps = {
  sortedActions: ChatAction[];
  pendingActionIds: number[];
  mutatingActionId: number | null;
  streamLatencySamples: StreamLatencySample[];
  tokenUsageSamples: TokenUsageSample[];
  retrievalHitSamples: RetrievalHitSample[];
  consistencyAudits: ConsistencyAuditReport[];
  consistencyAuditRunning: boolean;
  traceEvents: ChatStreamTraceEvent[];
  graphTimeline: GraphTimelineSnapshot | null;
  graphTimelineLoading: boolean;
  graphTimelineChapterIndex: number;
  maxChapterIndex: number;
  setGraphTimelineChapterIndex: (chapterIndex: number) => void;
  selectedActionId: number | null;
  actionLogs: ActionAuditLog[];
  onLoadLogs: (actionId: number) => Promise<void>;
  onMutateAction: (action: ChatAction, decision: "apply" | "reject" | "undo") => Promise<void>;
  onRunConsistencyAudit: () => Promise<void>;
};

export const AssistantActionsPanel = memo(function AssistantActionsPanel({
  sortedActions,
  pendingActionIds,
  mutatingActionId,
  streamLatencySamples,
  tokenUsageSamples,
  retrievalHitSamples,
  consistencyAudits,
  consistencyAuditRunning,
  traceEvents,
  graphTimeline,
  graphTimelineLoading,
  graphTimelineChapterIndex,
  maxChapterIndex,
  setGraphTimelineChapterIndex,
  selectedActionId,
  actionLogs,
  onLoadLogs,
  onMutateAction,
  onRunConsistencyAudit,
}: AssistantActionsPanelProps) {
  const pendingActionSet = useMemo(() => new Set(pendingActionIds), [pendingActionIds]);
  const latestAudits = useMemo(() => consistencyAudits.slice(0, 3), [consistencyAudits]);
  const [hoveredActionId, setHoveredActionId] = useState<number | null>(null);
  const hoveredAction = useMemo(
    () => sortedActions.find((item) => item.id === hoveredActionId) ?? null,
    [hoveredActionId, sortedActions]
  );
  const selectedAction = useMemo(
    () => sortedActions.find((item) => item.id === selectedActionId) ?? null,
    [selectedActionId, sortedActions]
  );
  const activeBlastAction = hoveredAction ?? selectedAction;
  const previewBlastRadius = useMemo(() => getActionBlastRadius(activeBlastAction), [activeBlastAction]);
  const previewBlastRadiusSummary = useMemo(() => summarizeBlastRadius(previewBlastRadius), [previewBlastRadius]);
  const previewBlastRadiusChips = useMemo(
    () => buildBlastRadiusSummaryChips(previewBlastRadius),
    [previewBlastRadius]
  );
  const previewActionLabel = useMemo(
    () => (activeBlastAction ? summarizeAction(activeBlastAction)[0] ?? `动作 #${activeBlastAction.id}` : ""),
    [activeBlastAction]
  );
  const previewNodeChangeMap = useMemo(() => {
    const next = new Map<string, string>();
    if (!previewBlastRadius) return next;
    previewBlastRadius.nodes.forEach((item) => {
      next.set(normalizeGraphPreviewToken(item.id || item.label), String(item.change || "touch"));
    });
    return next;
  }, [previewBlastRadius]);
  const previewEdgeChangeMap = useMemo(() => {
    const next = new Map<string, string>();
    if (!previewBlastRadius) return next;
    previewBlastRadius.edges.forEach((item) => {
      next.set(
        buildGraphPreviewEdgeKey(item.source, item.relation, item.target),
        String(item.change || "add")
      );
    });
    return next;
  }, [previewBlastRadius]);
  const hasPreviewOverlay = Boolean(
    previewBlastRadius && (previewBlastRadius.nodes.length > 0 || previewBlastRadius.edges.length > 0)
  );
  const hasBlastRadiusPanel = Boolean(hasPreviewOverlay || (previewBlastRadius?.notes?.length ?? 0) > 0);

  useEffect(() => {
    if (hoveredActionId === null) return;
    if (sortedActions.some((item) => item.id === hoveredActionId)) return;
    setHoveredActionId(null);
  }, [hoveredActionId, sortedActions]);

  const traceGroups = useMemo(() => {
    const source = traceEvents.slice(-24);
    const groups: Array<{
      key: string;
      scope: string;
      stage: string;
      status: string;
      items: ChatStreamTraceEvent[];
    }> = [];
    source.forEach((item) => {
      const scope = String(item.scope || "pipeline");
      const stage = String(item.stage || "step");
      const status = String(item.status || "info");
      const previous = groups[groups.length - 1];
      if (previous && previous.scope === scope && previous.stage === stage) {
        previous.items.push(item);
        previous.status = status;
        return;
      }
      groups.push({
        key: `${scope}:${stage}:${item.seq}`,
        scope,
        stage,
        status,
        items: [item],
      });
    });
    return groups;
  }, [traceEvents]);
  const {
    timelineEntityQuery,
    setTimelineEntityQuery,
    timelineRelationFilter,
    setTimelineRelationFilter,
    selectedTimelineNodeId,
    setSelectedTimelineNodeId,
    timelineRelationOptions,
    graphNodes,
    graphEdges,
    highlightedEdgeIdSet,
    highlightedNodeIdSet,
    selectedTimelineNodeLabel,
    graphLayout,
    sliderValue,
    timelineNodesTotal,
    timelineEdgesTotal,
    timelineNodesCount,
    timelineEdgesCount,
    hasTimelineFilter,
    graphNodeLimit,
    graphEdgeLimit,
    graphNodesCandidateCount,
    graphEdgesCandidateCount,
    graphNodesTruncated,
    graphEdgesTruncated,
  } = useTimelineGraphFlow({
    graphTimeline,
    graphTimelineChapterIndex,
    maxChapterIndex,
    previewBlastRadius,
  });
  const graphPreviewTruncated = graphNodesTruncated || graphEdgesTruncated;
  const resolveTraceChapterIndex = (item: ChatStreamTraceEvent): number | null => {
    const metaRecord = item.meta && typeof item.meta === "object" ? (item.meta as Record<string, unknown>) : null;
    const chapterRaw = metaRecord?.chapter_index ?? metaRecord?.current_chapter;
    if (typeof chapterRaw === "number" && Number.isFinite(chapterRaw) && chapterRaw > 0) {
      return Math.floor(chapterRaw);
    }
    if (typeof chapterRaw === "string") {
      const chapterValue = Number(chapterRaw);
      if (Number.isFinite(chapterValue) && chapterValue > 0) {
        return Math.floor(chapterValue);
      }
    }
    const text = String(item.message || "");
    const match = text.match(/第\s*([0-9]{1,6})\s*章/u);
    if (!match) return null;
    const chapterValue = Number(match[1]);
    if (!Number.isFinite(chapterValue) || chapterValue <= 0) return null;
    return Math.floor(chapterValue);
  };
  const clampChapterIndex = (value: number): number =>
    Math.max(1, Math.min(Math.max(1, maxChapterIndex), Math.floor(value)));

  const renderedEdges = useMemo(() => {
    return graphEdges.map((edge) => {
      const source = graphLayout.positions[edge.source];
      const target = graphLayout.positions[edge.target];
      if (!source || !target) return null;
      const previewChange = hasPreviewOverlay
        ? previewEdgeChangeMap.get(buildGraphPreviewEdgeKey(edge.source, edge.relation, edge.target))
        : null;
      const isPreviewGhostEdge = String(edge.id || "").startsWith("preview:");
      const edgeClassName = hasPreviewOverlay
        ? previewChange
          ? `timeline-edge preview-${resolveBlastRadiusTone(previewChange)}${
              isPreviewGhostEdge ? " is-preview-ghost" : ""
            }`
          : "timeline-edge is-dim"
        : selectedTimelineNodeId
          ? highlightedEdgeIdSet.has(edge.id)
            ? "timeline-edge is-linked"
            : "timeline-edge is-dim"
          : "timeline-edge";
      return (
        <line
          key={edge.id}
          className={edgeClassName}
          x1={source.x}
          y1={source.y}
          x2={target.x}
          y2={target.y}
        >
          <title>{`${edge.source} -[${edge.relation}]-> ${edge.target}`}</title>
        </line>
      );
    });
  }, [graphEdges, graphLayout.positions, hasPreviewOverlay, previewEdgeChangeMap, selectedTimelineNodeId, highlightedEdgeIdSet]);

  const renderedNodes = useMemo(() => {
    return graphNodes.map((node) => {
      const point = graphLayout.positions[node.id];
      if (!point) return null;
      const isSelected = selectedTimelineNodeId === node.id;
      const isNeighbor = highlightedNodeIdSet.has(node.id);
      const previewChange = hasPreviewOverlay
        ? previewNodeChangeMap.get(normalizeGraphPreviewToken(node.id || node.label))
        : null;
      const previewTone = previewChange ? resolveBlastRadiusTone(previewChange) : null;
      const nodeClassName = hasPreviewOverlay
        ? previewTone
          ? `timeline-node preview-${previewTone}${node.kind === "preview" ? " is-preview-ghost" : ""}`
          : "timeline-node is-dim"
        : selectedTimelineNodeId
          ? isSelected
            ? "timeline-node is-active"
            : isNeighbor
              ? "timeline-node is-neighbor"
              : "timeline-node is-dim"
          : "timeline-node";
      const toggleNodeHighlight = () => {
        setSelectedTimelineNodeId((prev) => (prev === node.id ? null : node.id));
      };
      return (
        <g
          key={node.id}
          className={nodeClassName}
          role="button"
          tabIndex={0}
          aria-label={`高亮节点 ${node.label}`}
          onClick={toggleNodeHighlight}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              toggleNodeHighlight();
            }
          }}
        >
          <circle cx={point.x} cy={point.y} r={10 + Math.min(4, Number(node.degree || 0))} />
          <text x={point.x} y={point.y + 21} textAnchor="middle">
            {node.label}
          </text>
        </g>
      );
    });
  }, [graphNodes, graphLayout.positions, selectedTimelineNodeId, highlightedNodeIdSet, hasPreviewOverlay, previewNodeChangeMap]);

  return (
    <section
      className="panel side-panel"
      onMouseLeave={() => setHoveredActionId(null)}
      onBlurCapture={(event) => {
        const nextTarget = event.relatedTarget;
        if (nextTarget instanceof Node && event.currentTarget.contains(nextTarget)) return;
        setHoveredActionId(null);
      }}
    >
      <div className="panel-title">
        <h2>动作提议</h2>
        <small>pending: {pendingActionIds.length}</small>
      </div>
      <div className="trace-list">
        <div className="trace-list-head">
          <strong>导演视角 · 推演轨迹</strong>
        </div>
        {traceGroups.length === 0 ? <p className="empty">等待 brainstorm 流式轨迹...</p> : null}
        {traceGroups.map((group, groupIndex) => (
          <details
            key={group.key}
            className={`trace-group status-${group.status}`}
            open={groupIndex === traceGroups.length - 1}
          >
            <summary className="trace-group-head">
              <span>{`${group.scope} · ${group.stage}`}</span>
              <small>{`${group.items.length} 条`}</small>
            </summary>
            <div className="trace-group-body">
              {group.items.map((item) => (
                (() => {
                  const traceChapter = resolveTraceChapterIndex(item);
                  const jumpChapter = traceChapter ? clampChapterIndex(traceChapter) : null;
                  return (
                    <article
                      key={`trace-${item.seq}-${item.stage}`}
                      className={`trace-item status-${String(item.status || "info")}`}
                    >
                      <div className="trace-item-head">
                        <span>{item.step && item.total ? `[${item.step}/${item.total}]` : `#${item.seq}`}</span>
                        <small>{item.scope}</small>
                      </div>
                      <p>{item.message}</p>
                      {jumpChapter ? (
                        <div className="trace-item-actions">
                          <button
                            type="button"
                            className="btn ghost tiny"
                            onClick={() => setGraphTimelineChapterIndex(jumpChapter)}
                            disabled={graphTimelineLoading}
                          >
                            {`跳到第 ${jumpChapter} 章`}
                          </button>
                        </div>
                      ) : null}
                    </article>
                  );
                })()
              ))}
            </div>
          </details>
        ))}
      </div>
      <div className="audit-list">
        <div className="audit-list-head">
          <strong>创作体检</strong>
          <button
            type="button"
            className="btn ghost tiny"
            onClick={() => void onRunConsistencyAudit()}
            disabled={consistencyAuditRunning || mutatingActionId !== null}
          >
            {consistencyAuditRunning ? "体检中..." : "立即体检"}
          </button>
        </div>
        {latestAudits.length === 0 ? <p className="empty">暂无体检报告</p> : null}
        {latestAudits.map((item) => {
          const summary = item.summary ?? {};
          const issueCount = Number(summary.issues ?? 0) || 0;
          const temporal = Number(summary.temporal_conflicts ?? 0) || 0;
          const foreshadow = Number(summary.foreshadow_overdue ?? 0) || 0;
          return (
            <article key={item.report_id} className="audit-item">
              <div className="audit-item-head">
                <span className={`audit-status ${item.status === "warning" ? "warning" : "ok"}`}>
                  {item.status === "warning" ? "告警" : "正常"}
                </span>
                <small>{formatDateTime(item.generated_at)}</small>
              </div>
              <p>{`问题 ${issueCount} · 时序冲突 ${temporal} · 伏笔超期 ${foreshadow}`}</p>
            </article>
          );
        })}
      </div>
      <div className="timeline-list">
        <div className="timeline-list-head">
          <strong>时序图谱</strong>
          <small>{`第 ${sliderValue} 章`}</small>
        </div>
        <label className="timeline-slider" htmlFor="timeline-chapter-slider">
          <span>{`章节滑块 1-${Math.max(1, maxChapterIndex)}`}</span>
          <input
            id="timeline-chapter-slider"
            type="range"
            min={1}
            max={Math.max(1, maxChapterIndex)}
            value={sliderValue}
            onChange={(event) => setGraphTimelineChapterIndex(Number(event.target.value || 1))}
            disabled={graphTimelineLoading}
          />
        </label>
        <div className="timeline-filter-row">
          <label className="timeline-filter-field" htmlFor="timeline-entity-filter">
            <span>实体检索</span>
            <input
              id="timeline-entity-filter"
              type="search"
              placeholder="人物/物品/地名"
              value={timelineEntityQuery}
              onChange={(event) => setTimelineEntityQuery(event.target.value)}
              disabled={graphTimelineLoading}
            />
          </label>
          <label className="timeline-filter-field" htmlFor="timeline-relation-filter">
            <span>关系类型</span>
            <select
              id="timeline-relation-filter"
              value={timelineRelationFilter}
              onChange={(event) => setTimelineRelationFilter(event.target.value)}
              disabled={graphTimelineLoading}
            >
              <option value="all">全部关系</option>
              {timelineRelationOptions.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
        </div>
        {graphTimelineLoading ? <p className="empty">图谱加载中...</p> : null}
        {!graphTimelineLoading && graphNodes.length === 0 ? (
          <p className="empty">{hasTimelineFilter ? "筛选后暂无可视化关系" : "该章节暂无可视化关系"}</p>
        ) : null}
        {!graphTimelineLoading && graphNodes.length > 0 ? (
          <svg
            className="timeline-graph"
            viewBox={`0 0 ${graphLayout.width} ${graphLayout.height}`}
            role="img"
            aria-label="章节时序图谱"
          >
            {renderedEdges}
            {renderedNodes}
          </svg>
        ) : null}
        {hasBlastRadiusPanel ? (
          <div className="timeline-preview-bar blast-radius-panel" aria-live="polite">
            <div className="blast-radius-head">
              <strong>动作爆炸半径</strong>
              <small>{activeBlastAction ? `#${activeBlastAction.id} · ${previewActionLabel}` : "待选择动作"}</small>
            </div>
            <div className="blast-radius-summary">
              {previewBlastRadiusChips.length > 0
                ? previewBlastRadiusChips.map((item) => (
                    <span key={item.key} className={`blast-radius-chip is-${item.tone}`}>
                      {item.label}
                    </span>
                  ))
                : <span className="blast-radius-chip is-update">仅锚点波及</span>}
            </div>
            <div className="blast-radius-legend">
              <span className="blast-radius-chip is-add">新增 / 投影</span>
              <span className="blast-radius-chip is-update">更新 / 波及</span>
              <span className="blast-radius-chip is-delete">删除 / 置换</span>
            </div>
            <div className="blast-radius-node-list">
              {(previewBlastRadius?.nodes ?? []).slice(0, 8).map((item) => (
                <span
                  key={`${item.id}-${item.label}`}
                  className={`blast-radius-node-pill is-${resolveBlastRadiusTone(item.change)}${item.in_current_graph ? "" : " is-ghost"}`}
                  title={`${item.role || "related"} · ${item.in_current_graph ? "已存在节点" : "投影节点"}`}
                >
                  {item.label}
                </span>
              ))}
            </div>
            {graphPreviewTruncated ? (
              <small className="blast-radius-truncation">
                {`图谱已截断显示：节点仅展示 ${Math.min(graphNodeLimit, graphNodesCandidateCount)}/${graphNodesCandidateCount}，边仅展示 ${Math.min(graphEdgeLimit, graphEdgesCandidateCount)}/${graphEdgesCandidateCount}。可用上方筛选缩小范围。`}
              </small>
            ) : null}
            {previewBlastRadiusSummary ? <small>{previewBlastRadiusSummary}</small> : null}
            {previewBlastRadius?.notes?.[0] ? <small>{previewBlastRadius.notes[0]}</small> : null}
            <BlastRadiusDetailDisclosure
              preview={previewBlastRadius}
              resetKey={activeBlastAction?.id ?? "none"}
              summaryLabel="查看明细"
            />
          </div>
        ) : selectedTimelineNodeId ? (
          <div className="timeline-focus-bar">
            <small>{`已高亮：${selectedTimelineNodeLabel}`}</small>
            <button
              type="button"
              className="btn ghost tiny"
              onClick={() => setSelectedTimelineNodeId(null)}
              disabled={graphTimelineLoading}
            >
              清除高亮
            </button>
          </div>
        ) : null}
        <small className="timeline-meta">{`当前 节点 ${timelineNodesCount}/${timelineNodesTotal} · 边 ${timelineEdgesCount}/${timelineEdgesTotal}`}</small>
      </div>
      <div className="action-list">
        {sortedActions.length === 0 ? <p className="empty">暂无动作记录</p> : null}
        {sortedActions.map((action) => (
          <ActionCard
            key={action.id}
            action={action}
            isPending={pendingActionSet.has(action.id)}
            isPreviewActive={hoveredActionId === action.id || (hoveredActionId === null && selectedActionId === action.id)}
            controlsDisabled={mutatingActionId !== null}
            onLoadLogs={onLoadLogs}
            onPreviewEnter={setHoveredActionId}
            onMutateAction={onMutateAction}
          />
        ))}
      </div>

      <ActionLogsList selectedActionId={selectedActionId} actionLogs={actionLogs} />
    </section>
  );
});