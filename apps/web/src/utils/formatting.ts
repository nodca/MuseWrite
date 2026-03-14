import type { UiMessage } from "../types";

export function formatRole(role: UiMessage["role"]): string {
  if (role === "user") return "你";
  if (role === "assistant") return "助手";
  return "系统";
}

export function formatJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

export function formatDateTime(value?: string | null): string {
  if (!value) return "未保存";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

export function isSameLocalDate(value: string | null | undefined, anchorDate: Date): boolean {
  if (!value) return false;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return false;
  return (
    parsed.getFullYear() === anchorDate.getFullYear() &&
    parsed.getMonth() === anchorDate.getMonth() &&
    parsed.getDate() === anchorDate.getDate()
  );
}

export function toFiniteNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

export function computePercentile(values: number[], percentile: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((percentile / 100) * sorted.length) - 1));
  return sorted[index];
}

export function formatMs(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "--";
  return `${Math.round(value)}ms`;
}

export function formatPercent(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0%";
  return `${Math.round(value)}%`;
}

export function normalizeEditorText(value: string): string {
  const normalized = (value || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n").replace(/\u00a0/g, " ");
  if (normalized.endsWith("\n")) {
    return normalized.slice(0, -1);
  }
  return normalized;
}

export function parseReferenceProjectIds(value: string, currentProjectId: number): number[] {
  const tokens = (value || "")
    .split(/[,\s]+/)
    .map((item) => Number(item))
    .filter((item) => Number.isFinite(item) && item > 0 && item !== currentProjectId);
  const deduped = Array.from(new Set(tokens));
  return deduped.slice(0, 5);
}
