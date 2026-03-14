import { memo, useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import {
  resolveActionIntent,
  resolveActionIntentLabel,
  resolveActionStatusLabel,
  summarizeAction,
  actionRiskHints,
  isAliasDiffField,
  parseAliasTags,
  buildActionDiffRows,
  isEntityMergeActionType,
} from "../lib/actionHelpers";
import {
  getActionBlastRadius,
  summarizeBlastRadius,
  buildBlastRadiusSummaryChips,
  formatBlastRadiusMarkdown,
} from "../lib/blastRadius";
import { formatJson } from "../utils/formatting";
import { BlastRadiusDetailDisclosure } from "./BlastRadiusDetailDisclosure";
import type { ChatAction } from "../types";

export type ActionCardProps = {
  action: ChatAction;
  isPending: boolean;
  isPreviewActive: boolean;
  controlsDisabled: boolean;
  onLoadLogs: (actionId: number) => Promise<void>;
  onPreviewEnter: (actionId: number) => void;
  onMutateAction: (action: ChatAction, decision: "apply" | "reject" | "undo") => Promise<void>;
};

export const ActionCard = memo(function ActionCard({
  action,
  isPending,
  isPreviewActive,
  controlsDisabled,
  onLoadLogs,
  onPreviewEnter,
  onMutateAction,
}: ActionCardProps) {
  const actionIntent = useMemo(() => resolveActionIntent(action.action_type), [action.action_type]);
  const actionIntentLabel = useMemo(() => resolveActionIntentLabel(actionIntent), [actionIntent]);
  const actionStatusLabel = useMemo(() => resolveActionStatusLabel(action.status), [action.status]);
  const summaryLines = useMemo(() => summarizeAction(action), [action]);
  const riskHints = useMemo(() => actionRiskHints(action), [action]);
  const blastRadius = useMemo(() => getActionBlastRadius(action), [action]);
  const blastRadiusSummary = useMemo(() => summarizeBlastRadius(blastRadius), [blastRadius]);
  const canPreviewBlastRadius = action.status === "proposed" && blastRadius !== null;
  const blastRadiusChips = useMemo(() => buildBlastRadiusSummaryChips(blastRadius), [blastRadius]);
  const [applyConfirmOpen, setApplyConfirmOpen] = useState(false);
  const diffRows = useMemo(() => buildActionDiffRows(action), [action]);
  const [undoFlash, setUndoFlash] = useState(false);
  const previousStatusRef = useRef(action.status);
  const undoFlashTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (action.status === "proposed") return;
    setApplyConfirmOpen(false);
  }, [action.status]);

  useEffect(() => {
    const previousStatus = previousStatusRef.current;
    previousStatusRef.current = action.status;
    if (action.status !== "undone" || previousStatus === "undone") return;
    if (typeof window === "undefined") return;
    if (undoFlashTimerRef.current !== null) {
      window.clearTimeout(undoFlashTimerRef.current);
      undoFlashTimerRef.current = null;
    }
    setUndoFlash(true);
    undoFlashTimerRef.current = window.setTimeout(() => {
      setUndoFlash(false);
      undoFlashTimerRef.current = null;
    }, 520);
  }, [action.status]);

  useEffect(() => {
    return () => {
      if (undoFlashTimerRef.current !== null && typeof window !== "undefined") {
        window.clearTimeout(undoFlashTimerRef.current);
      }
    };
  }, []);

  return (
    <article
      className={clsx(
        "action-card rounded-lg border border-border-default bg-surface-primary p-3 space-y-2 transition-shadow hover:shadow-md",
        `action-intent-${actionIntent}`,
        isPending && "ring-2 ring-accent-primary/30",
        isPreviewActive && "ring-2 ring-accent-primary/20 bg-accent-secondary/5 is-preview-active",
        action.status === "applied" && "opacity-75 is-applied",
        undoFlash && "animate-pulse",
      )}
      onMouseEnter={() => {
        if (canPreviewBlastRadius) onPreviewEnter(action.id);
      }}
      onFocusCapture={() => {
        if (canPreviewBlastRadius) onPreviewEnter(action.id);
      }}
    >
      {/* Summary header */}
      <button
        type="button"
        className="w-full text-left flex flex-col gap-1"
        onClick={() => void onLoadLogs(action.id)}
      >
        <div className="flex items-center gap-2 text-xs text-text-secondary">
          <span className="font-mono">#{action.id}</span>
          <strong className="text-text-primary text-sm">{`${actionIntentLabel} · ${action.action_type}`}</strong>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={clsx(
              "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium",
              action.status === "proposed" && "bg-accent-secondary text-accent-primary",
              action.status === "applied" && "bg-ok-bg text-ok",
              action.status === "rejected" && "bg-danger-bg text-danger",
              action.status === "undone" && "bg-warning-bg text-warning",
            )}
          >
            {actionStatusLabel}
          </span>
          {action.status === "applied" ? (
            <span className="text-xs text-ok font-medium">&#10003; 已应用</span>
          ) : null}
        </div>
      </button>

      {/* Summary body */}
      <div className="space-y-1">
        {summaryLines.map((line, idx) => (
          <p key={`${action.id}-summary-${idx}`} className="text-sm text-text-primary leading-relaxed">
            {line}
          </p>
        ))}
        {riskHints.map((line, idx) => (
          <p key={`${action.id}-risk-${idx}`} className="text-xs text-warning font-medium">
            风险提示：{line}
          </p>
        ))}
        {blastRadiusSummary ? (
          <p className="text-xs text-text-secondary">爆炸半径：{blastRadiusSummary}</p>
        ) : null}
      </div>

      {/* Diff rows */}
      <div className="space-y-2">
        {diffRows.map((row, idx) => {
          const isAliasRow = isAliasDiffField(row.field);
          const beforeTags = row.beforeTags ?? (isAliasRow ? parseAliasTags(row.before) : []);
          const afterTags = row.afterTags ?? (isAliasRow ? parseAliasTags(row.after) : []);
          return (
            <article
              key={`${action.id}-diff-${idx}`}
              className="grid grid-cols-[auto_1fr_1fr] gap-2 rounded-md border border-border-default bg-surface-base p-2 text-xs"
            >
              <div className="font-medium text-text-secondary self-start pt-0.5 min-w-[60px]">{row.field}</div>
              <div className="min-w-0">
                <small className="block text-text-tertiary mb-0.5">原值</small>
                {isAliasRow && beforeTags.length > 0 ? (
                  <div className="flex flex-wrap gap-1">
                    {beforeTags.map((tag) => (
                      <span
                        key={`before-${action.id}-${idx}-${tag}`}
                        className="inline-flex items-center rounded-full bg-surface-elevated px-2 py-0.5 text-xs text-text-secondary border border-border-default"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                ) : (
                  <pre className="whitespace-pre-wrap break-words text-text-primary font-mono text-xs">{row.before}</pre>
                )}
              </div>
              <div className="min-w-0">
                <small className="block text-text-tertiary mb-0.5">新值</small>
                {isAliasRow && afterTags.length > 0 ? (
                  <div className="flex flex-wrap gap-1">
                    {afterTags.map((tag) => (
                      <span
                        key={`after-${action.id}-${idx}-${tag}`}
                        className="inline-flex items-center rounded-full bg-accent-secondary px-2 py-0.5 text-xs text-accent-primary border border-accent-primary/20"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                ) : (
                  <pre className="whitespace-pre-wrap break-words text-text-primary font-mono text-xs">{row.after}</pre>
                )}
              </div>
            </article>
          );
        })}
      </div>

      {/* Debug JSON */}
      <details className="group">
        <summary className="text-xs text-text-tertiary cursor-pointer hover:text-text-secondary transition-colors">
          调试 JSON
        </summary>
        <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-3">
          <article className="rounded-md bg-surface-base border border-border-default p-2">
            <small className="block text-xs text-text-tertiary mb-1">payload</small>
            <pre className="text-xs font-mono text-text-primary whitespace-pre-wrap break-all max-h-40 overflow-y-auto">
              {formatJson(action.payload)}
            </pre>
          </article>
          <article className="rounded-md bg-surface-base border border-border-default p-2">
            <small className="block text-xs text-text-tertiary mb-1">apply_result</small>
            <pre className="text-xs font-mono text-text-primary whitespace-pre-wrap break-all max-h-40 overflow-y-auto">
              {formatJson(action.apply_result)}
            </pre>
          </article>
          <article className="rounded-md bg-surface-base border border-border-default p-2">
            <small className="block text-xs text-text-tertiary mb-1">undo_payload</small>
            <pre className="text-xs font-mono text-text-primary whitespace-pre-wrap break-all max-h-40 overflow-y-auto">
              {formatJson(action.undo_payload)}
            </pre>
          </article>
        </div>
      </details>

      {/* Action buttons */}
      <div className="flex items-center gap-2 pt-1">
        {action.status === "proposed" ? (
          <>
            {applyConfirmOpen ? (
              <>
                <button
                  type="button"
                  className="rounded-md bg-accent-primary px-3 py-1.5 text-xs font-medium text-white hover:bg-accent-primary-hover disabled:opacity-40 transition-colors"
                  onClick={() => void onMutateAction(action, "apply")}
                  disabled={controlsDisabled}
                >
                  确认应用
                </button>
                <button
                  type="button"
                  className="rounded-md border border-border-default px-3 py-1.5 text-xs text-text-secondary hover:text-text-primary hover:bg-surface-elevated disabled:opacity-40 transition-colors"
                  onClick={() => setApplyConfirmOpen(false)}
                  disabled={controlsDisabled}
                >
                  取消
                </button>
              </>
            ) : (
              <button
                type="button"
                className="rounded-md bg-accent-primary px-3 py-1.5 text-xs font-medium text-white hover:bg-accent-primary-hover disabled:opacity-40 transition-colors"
                onClick={() => {
                  setApplyConfirmOpen(true);
                  if (canPreviewBlastRadius) onPreviewEnter(action.id);
                }}
                disabled={controlsDisabled}
              >
                应用到项目
              </button>
            )}
            <button
              type="button"
              className="rounded-md border border-border-default px-3 py-1.5 text-xs text-text-secondary hover:text-text-primary hover:bg-surface-elevated disabled:opacity-40 transition-colors"
              onClick={() => void onMutateAction(action, "reject")}
              disabled={controlsDisabled}
            >
              暂不采用
            </button>
          </>
        ) : null}
        {action.status === "applied" ? (
          <button
            type="button"
            className="rounded-md border border-border-default px-3 py-1.5 text-xs text-text-secondary hover:text-danger hover:border-danger/30 hover:bg-danger-bg disabled:opacity-40 transition-colors"
            onClick={() => void onMutateAction(action, "undo")}
            disabled={controlsDisabled}
          >
            撤销应用
          </button>
        ) : null}
      </div>

      {/* Pre-apply notice */}
      {action.status === "proposed" ? (
        <p className="text-xs text-warning">应用后会立即写入设定/卡片，并记录审计日志。</p>
      ) : null}

      {/* Apply confirmation panel */}
      {action.status === "proposed" && applyConfirmOpen ? (
        <div
          className="rounded-lg border border-border-default bg-surface-base p-3 space-y-2"
          aria-live="polite"
        >
          <div className="flex items-center justify-between">
            <strong className="text-sm text-text-primary">应用前总览</strong>
            <small className="text-xs text-text-tertiary">确认前不会写入项目</small>
          </div>

          {blastRadius ? (
            <div className="flex flex-wrap gap-1.5">
              {blastRadiusChips.length > 0
                ? blastRadiusChips.map((item) => (
                    <span
                      key={item.key}
                      className={clsx(
                        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium",
                        item.tone === "add" && "bg-ok-bg text-ok",
                        item.tone === "update" && "bg-accent-secondary text-accent-primary",
                        item.tone === "delete" && "bg-danger-bg text-danger",
                      )}
                    >
                      {item.label}
                    </span>
                  ))
                : (
                    <span className="inline-flex items-center rounded-full bg-accent-secondary px-2 py-0.5 text-xs font-medium text-accent-primary">
                      仅锚点波及
                    </span>
                  )}
            </div>
          ) : (
            <p className="text-xs text-text-tertiary">当前动作未提供图谱爆炸半径预览</p>
          )}

          {blastRadius ? (
            <>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  className="rounded-md border border-border-default px-2 py-1 text-xs text-text-secondary hover:text-text-primary hover:bg-surface-elevated transition-colors"
                  onClick={() =>
                    void navigator.clipboard?.writeText(
                      formatBlastRadiusMarkdown(blastRadius, `动作 #${action.id} · ${action.action_type}`),
                    )
                  }
                >
                  复制清单
                </button>
                <button
                  type="button"
                  className="rounded-md border border-border-default px-2 py-1 text-xs text-text-secondary hover:text-text-primary hover:bg-surface-elevated transition-colors"
                  onClick={() => void navigator.clipboard?.writeText(formatJson(blastRadius))}
                >
                  复制 JSON
                </button>
              </div>

              <BlastRadiusDetailDisclosure
                preview={blastRadius}
                resetKey={`${action.id}:${action.status}`}
                summaryLabel="查看图谱明细"
                className="blast-radius-detail action-apply-detail"
              />
            </>
          ) : null}
        </div>
      ) : null}
    </article>
  );
});
