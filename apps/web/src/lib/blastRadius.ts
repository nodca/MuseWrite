import { safeToRecord } from "./actionHelpers";
import type { ChatAction, GraphBlastRadiusPreview } from "../types";

export function normalizeGraphPreviewToken(value: string): string {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[\s_-]+/g, "");
}

export function buildGraphPreviewEdgeKey(source: string, relation: string, target: string): string {
  const sourceToken = normalizeGraphPreviewToken(source);
  const targetToken = normalizeGraphPreviewToken(target);
  const relationToken = String(relation || "").trim().toUpperCase();
  if (!sourceToken || !targetToken || !relationToken) return "";
  return `${sourceToken}|${relationToken}|${targetToken}`;
}

export function getActionBlastRadius(action: ChatAction | null | undefined): GraphBlastRadiusPreview | null {
  const payload = safeToRecord(action?.payload);
  const raw = safeToRecord(payload?._graph_blast_radius);
  if (!raw) return null;

  const nodes = (Array.isArray(raw.nodes) ? raw.nodes : [])
    .map((item) => safeToRecord(item))
    .filter((item): item is Record<string, unknown> => item !== null)
    .map((item) => {
      const label =
        typeof item.label === "string"
          ? item.label.trim()
          : typeof item.id === "string"
            ? item.id.trim()
            : "";
      const id =
        typeof item.id === "string" && item.id.trim().length > 0
          ? item.id.trim()
          : label;
      if (!id || !label) return null;
      return {
        id,
        label,
        change: typeof item.change === "string" ? item.change : "touch",
        role: typeof item.role === "string" ? item.role : "related",
        in_current_graph: Boolean(item.in_current_graph),
      };
    })
    .filter((item): item is GraphBlastRadiusPreview["nodes"][number] => item !== null);

  const edges = (Array.isArray(raw.edges) ? raw.edges : [])
    .map((item) => safeToRecord(item))
    .filter((item): item is Record<string, unknown> => item !== null)
    .map((item) => {
      const source = typeof item.source === "string" ? item.source.trim() : "";
      const target = typeof item.target === "string" ? item.target.trim() : "";
      const relation = typeof item.relation === "string" ? item.relation.trim().toUpperCase() : "";
      const key =
        typeof item.key === "string" && item.key.trim().length > 0
          ? item.key.trim()
          : buildGraphPreviewEdgeKey(source, relation, target);
      if (!source || !target || !relation || !key) return null;
      return {
        key,
        source,
        target,
        relation,
        change: typeof item.change === "string" ? item.change : "add",
        in_current_graph: Boolean(item.in_current_graph),
      };
    })
    .filter((item): item is GraphBlastRadiusPreview["edges"][number] => item !== null);

  const summaryWrap = safeToRecord(raw.summary);
  const summaryNodes = safeToRecord(summaryWrap?.nodes) ?? {};
  const summaryEdges = safeToRecord(summaryWrap?.edges) ?? {};

  return {
    source: typeof raw.source === "string" ? raw.source : "none",
    action_type: typeof raw.action_type === "string" ? raw.action_type : String(action?.action_type || ""),
    chapter_index:
      typeof raw.chapter_index === "number"
        ? raw.chapter_index
        : typeof raw.chapter_index === "string" && raw.chapter_index.trim().length > 0
          ? Number(raw.chapter_index)
          : null,
    nodes,
    edges,
    summary: {
      nodes: Object.fromEntries(
        Object.entries(summaryNodes).map(([key, value]) => [key, Number(value) || 0])
      ),
      edges: Object.fromEntries(
        Object.entries(summaryEdges).map(([key, value]) => [key, Number(value) || 0])
      ),
    },
    notes: Array.isArray(raw.notes)
      ? raw.notes.map((item) => String(item || "").trim()).filter((item) => item.length > 0)
      : [],
  };
}

export function summarizeBlastRadius(preview: GraphBlastRadiusPreview | null): string | null {
  if (!preview) return null;
  const nodeSummary = safeToRecord(preview.summary?.nodes) ?? {};
  const edgeSummary = safeToRecord(preview.summary?.edges) ?? {};
  const createNodes = Number(nodeSummary.create ?? 0) || 0;
  const updateNodes = Number(nodeSummary.update ?? 0) || 0;
  const deleteNodes = Number(nodeSummary.delete ?? 0) || 0;
  const touchNodes = Number(nodeSummary.touch ?? 0) || 0;
  const addEdges = Number(edgeSummary.add ?? 0) || 0;
  const updateEdges = Number(edgeSummary.update ?? 0) || 0;
  const deleteEdges = Number(edgeSummary.delete ?? 0) || 0;
  const parts: string[] = [];
  if (createNodes > 0) parts.push(`+${createNodes} 节点`);
  if (updateNodes > 0) parts.push(`~${updateNodes} 节点`);
  if (deleteNodes > 0) parts.push(`-${deleteNodes} 节点`);
  if (touchNodes > 0) parts.push(`${touchNodes} 波及`);
  if (addEdges > 0) parts.push(`+${addEdges} 边`);
  if (updateEdges > 0) parts.push(`~${updateEdges} 边`);
  if (deleteEdges > 0) parts.push(`-${deleteEdges} 边`);
  if (parts.length > 0) return parts.join(" / ");
  return preview.notes?.[0] ?? "当前动作不直接改写图谱关系";
}

export function resolveBlastRadiusTone(change: string): "add" | "update" | "delete" {
  const normalized = String(change || "").trim().toLowerCase();
  if (normalized === "create" || normalized === "add") return "add";
  if (normalized === "delete" || normalized === "remove") return "delete";
  return "update";
}

export function buildBlastRadiusSummaryChips(preview: GraphBlastRadiusPreview | null): Array<{
  key: string;
  tone: "add" | "update" | "delete";
  label: string;
}> {
  if (!preview) return [];
  const nodeSummary = safeToRecord(preview.summary?.nodes) ?? {};
  const edgeSummary = safeToRecord(preview.summary?.edges) ?? {};
  const chips: Array<{ key: string; tone: "add" | "update" | "delete"; label: string }> = [];
  const nodeCreate = Number(nodeSummary.create ?? 0) || 0;
  const nodeUpdate = (Number(nodeSummary.update ?? 0) || 0) + (Number(nodeSummary.touch ?? 0) || 0);
  const nodeDelete = Number(nodeSummary.delete ?? 0) || 0;
  const edgeAdd = Number(edgeSummary.add ?? 0) || 0;
  const edgeUpdate = Number(edgeSummary.update ?? 0) || 0;
  const edgeDelete = Number(edgeSummary.delete ?? 0) || 0;
  if (nodeCreate > 0) chips.push({ key: "node-create", tone: "add", label: `+${nodeCreate} 节点` });
  if (nodeUpdate > 0) chips.push({ key: "node-update", tone: "update", label: `~${nodeUpdate} 节点` });
  if (nodeDelete > 0) chips.push({ key: "node-delete", tone: "delete", label: `-${nodeDelete} 节点` });
  if (edgeAdd > 0) chips.push({ key: "edge-add", tone: "add", label: `+${edgeAdd} 边` });
  if (edgeUpdate > 0) chips.push({ key: "edge-update", tone: "update", label: `~${edgeUpdate} 边` });
  if (edgeDelete > 0) chips.push({ key: "edge-delete", tone: "delete", label: `-${edgeDelete} 边` });
  return chips;
}

export function formatBlastRadiusMarkdown(preview: GraphBlastRadiusPreview | null, title: string): string {
  const heading = String(title || "").trim();
  const lines: string[] = [];
  if (heading) lines.push(`# ${heading}`);
  if (!preview) {
    lines.push("");
    lines.push("（无图谱爆炸半径预览）");
    return lines.join("\n");
  }
  const summary = summarizeBlastRadius(preview);
  if (summary) lines.push(`- 摘要：${summary}`);
  if (preview.action_type) lines.push(`- 动作：${preview.action_type}`);
  if (preview.chapter_index !== null) lines.push(`- 章节：${preview.chapter_index}`);
  if (preview.source) lines.push(`- 来源：${preview.source}`);
  if (preview.notes?.[0]) lines.push(`- 备注：${preview.notes[0]}`);

  lines.push("");
  lines.push("## 节点");
  if ((preview.nodes ?? []).length === 0) {
    lines.push("- （无）");
  } else {
    preview.nodes.forEach((node) => {
      const tone = resolveBlastRadiusTone(node.change);
      const existsLabel = node.in_current_graph ? "已存在" : "投影";
      const role = String(node.role || "related");
      lines.push(`- [${tone}] ${node.label} (${role} · ${existsLabel})`);
    });
  }

  lines.push("");
  lines.push("## 边");
  if ((preview.edges ?? []).length === 0) {
    lines.push("- （无）");
  } else {
    preview.edges.forEach((edge) => {
      const tone = resolveBlastRadiusTone(edge.change);
      const existsLabel = edge.in_current_graph ? "已存在" : "投影";
      lines.push(`- [${tone}] ${edge.source} -[${edge.relation}]-> ${edge.target} (${existsLabel})`);
    });
  }

  return lines.join("\n");
}
