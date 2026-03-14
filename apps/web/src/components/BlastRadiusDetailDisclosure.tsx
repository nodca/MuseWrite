import { memo, useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { resolveBlastRadiusTone } from "../lib/blastRadius";
import type { GraphBlastRadiusPreview } from "../types";

export type BlastRadiusDetailDisclosureProps = {
  preview: GraphBlastRadiusPreview | null;
  resetKey: unknown;
  summaryLabel?: string;
  className?: string;
  maxNodes?: number;
  maxEdges?: number;
};

const inputClass =
  "w-full rounded-md border border-border-default bg-surface-primary px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent-primary/30 disabled:opacity-40";

export const BlastRadiusDetailDisclosure = memo(function BlastRadiusDetailDisclosure({
  preview,
  resetKey,
  summaryLabel = "查看明细",
  className,
  maxNodes = 80,
  maxEdges = 120,
}: BlastRadiusDetailDisclosureProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<"all" | "add" | "update" | "delete">("all");

  const nodeItems = preview?.nodes ?? [];
  const edgeItems = preview?.edges ?? [];
  const token = query.trim().toLowerCase();

  const filteredNodes = useMemo(() => {
    if (!token && filter === "all") return nodeItems;
    return nodeItems.filter((item) => {
      const tone = resolveBlastRadiusTone(item.change);
      if (filter !== "all" && tone !== filter) return false;
      if (!token) return true;
      const label = String(item.label || item.id || "").toLowerCase();
      const id = String(item.id || "").toLowerCase();
      const role = String(item.role || "").toLowerCase();
      return label.includes(token) || id.includes(token) || role.includes(token);
    });
  }, [filter, nodeItems, token]);

  const filteredEdges = useMemo(() => {
    if (!token && filter === "all") return edgeItems;
    return edgeItems.filter((item) => {
      const tone = resolveBlastRadiusTone(item.change);
      if (filter !== "all" && tone !== filter) return false;
      if (!token) return true;
      const source = String(item.source || "").toLowerCase();
      const target = String(item.target || "").toLowerCase();
      const relation = String(item.relation || "").toLowerCase();
      return source.includes(token) || target.includes(token) || relation.includes(token);
    });
  }, [edgeItems, filter, token]);

  useEffect(() => {
    setQuery("");
    setFilter("all");
    setOpen(false);
  }, [resetKey]);

  if (!preview) return null;

  return (
    <details className={clsx("rounded-lg border border-border-default", className)} open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-text-secondary hover:text-text-primary select-none">{summaryLabel}</summary>
      <div className="flex items-center gap-3 px-3 pb-2">
        <label className="flex flex-col gap-1 text-xs">
          <span>检索</span>
          <input
            className={inputClass}
            type="search"
            placeholder="节点/关系/角色"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <label className="flex flex-col gap-1 text-xs">
          <span>变更</span>
          <select className={inputClass} value={filter} onChange={(event) => setFilter(event.target.value as typeof filter)}>
            <option value="all">全部</option>
            <option value="add">新增</option>
            <option value="update">更新/波及</option>
            <option value="delete">删除</option>
          </select>
        </label>
      </div>
      <div className="grid grid-cols-2 gap-3 px-3 pb-3">
        <section className="space-y-1">
          <div className="flex items-center justify-between text-xs text-text-secondary">
            <strong>{`节点 ${filteredNodes.length}/${nodeItems.length}`}</strong>
            <small>点击可复制标题</small>
          </div>
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {nodeItems.length === 0 ? <p className="text-xs text-text-tertiary italic">暂无节点变更</p> : null}
            {nodeItems.length > 0 && filteredNodes.length === 0 ? <p className="text-xs text-text-tertiary italic">筛选后无节点命中</p> : null}
            {filteredNodes.slice(0, maxNodes).map((item) => {
              const tone = resolveBlastRadiusTone(item.change);
              const existsLabel = item.in_current_graph ? "已存在" : "投影";
              const roleLabel = String(item.role || "related");
              return (
                <button
                  key={`${item.id}:${item.label}:${roleLabel}`}
                  type="button"
                  className={clsx(
                    "w-full text-left flex items-center justify-between px-2 py-1 rounded text-xs hover:bg-surface-elevated transition-colors",
                    tone === "add" && "text-ok",
                    tone === "update" && "text-accent-primary",
                    tone === "delete" && "text-danger",
                  )}
                  onClick={() => void navigator.clipboard?.writeText(item.label || item.id)}
                >
                  <span className="truncate">{item.label}</span>
                  <small>{`${roleLabel} · ${existsLabel}`}</small>
                </button>
              );
            })}
            {filteredNodes.length > maxNodes ? (
              <p className="text-xs text-text-tertiary italic">{`已截断展示明细：仅显示前 ${maxNodes} 条（当前命中 ${filteredNodes.length} 条）`}</p>
            ) : null}
          </div>
        </section>

        <section className="space-y-1">
          <div className="flex items-center justify-between text-xs text-text-secondary">
            <strong>{`边 ${filteredEdges.length}/${edgeItems.length}`}</strong>
            <small>点击可复制三元组</small>
          </div>
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {edgeItems.length === 0 ? <p className="text-xs text-text-tertiary italic">暂无关系变更</p> : null}
            {edgeItems.length > 0 && filteredEdges.length === 0 ? <p className="text-xs text-text-tertiary italic">筛选后无关系命中</p> : null}
            {filteredEdges.slice(0, maxEdges).map((item) => {
              const tone = resolveBlastRadiusTone(item.change);
              const existsLabel = item.in_current_graph ? "已存在" : "投影";
              const triple = `${item.source} -[${item.relation}]-> ${item.target}`;
              return (
                <button
                  key={item.key}
                  type="button"
                  className={clsx(
                    "w-full text-left flex items-center justify-between px-2 py-1 rounded text-xs hover:bg-surface-elevated transition-colors",
                    tone === "add" && "text-ok",
                    tone === "update" && "text-accent-primary",
                    tone === "delete" && "text-danger",
                  )}
                  onClick={() => void navigator.clipboard?.writeText(triple)}
                >
                  <span className="truncate">{triple}</span>
                  <small>{existsLabel}</small>
                </button>
              );
            })}
            {filteredEdges.length > maxEdges ? (
              <p className="text-xs text-text-tertiary italic">{`已截断展示明细：仅显示前 ${maxEdges} 条（当前命中 ${filteredEdges.length} 条）`}</p>
            ) : null}
          </div>
        </section>
      </div>
    </details>
  );
});
