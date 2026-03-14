import { memo } from "react";
import { formatJson } from "../utils/formatting";
import type { ActionAuditLog } from "../types";

export type ActionLogsListProps = {
  selectedActionId: number | null;
  actionLogs: ActionAuditLog[];
};

export const ActionLogsList = memo(function ActionLogsList({ selectedActionId, actionLogs }: ActionLogsListProps) {
  return (
    <>
      <div className="flex items-center justify-between mb-2 px-3 pt-3">
        <h3>动作日志</h3>
        <small>{selectedActionId ? `action #${selectedActionId}` : "未选择动作"}</small>
      </div>
      <div className="space-y-2 px-3 pb-3">
        {actionLogs.length === 0 ? <p className="text-sm text-text-tertiary italic px-3">点一条动作查看审计日志</p> : null}
        {actionLogs.map((log) => (
          <article key={log.id} className="rounded-md border border-border-default bg-surface-base p-3">
            <div className="flex items-center justify-between mb-1 text-xs text-text-secondary">
              <span>{log.event_type}</span>
              <small>{log.operator_id}</small>
            </div>
            <pre className="text-xs font-mono text-text-primary whitespace-pre-wrap break-all mt-1 max-h-40 overflow-y-auto">{formatJson(log.event_payload)}</pre>
          </article>
        ))}
      </div>
    </>
  );
});
