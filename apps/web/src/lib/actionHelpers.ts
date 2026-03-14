import { formatJson } from "../utils/formatting";
import type { ChatAction } from "../types";

export function isEntityMergeActionType(actionType: string): boolean {
  const normalized = (actionType || "").trim().toLowerCase().replace(/[_-]/g, ".");
  return normalized.startsWith("entity.merge") || normalized.startsWith("graph.entity.merge");
}

export type ActionIntent = "create" | "update" | "delete";

export function resolveActionIntent(actionType: string): ActionIntent {
  const normalized = (actionType || "").trim().toLowerCase();
  if (normalized.includes("delete")) return "delete";
  if (normalized.includes("create")) return "create";
  return "update";
}

export function resolveActionIntentLabel(intent: ActionIntent): string {
  if (intent === "create") return "创建";
  if (intent === "delete") return "删除";
  return "修改";
}

export function resolveActionStatusLabel(status: string): string {
  const normalized = (status || "").trim().toLowerCase();
  if (normalized === "proposed") return "AI 建议";
  if (normalized === "applied" || normalized === "confirmed") return "已应用";
  if (normalized === "undone") return "已撤销";
  if (normalized === "rejected") return "已忽略";
  if (normalized === "failed") return "执行失败";
  return status;
}

export function collectEntityMergePayloadAliases(payload: Record<string, unknown>): string[] {
  const rawItems: unknown[] = [];
  for (const key of ["alias", "source_entity", "source_alias", "candidate_alias", "from_entity"]) {
    const value = payload[key];
    if (value !== undefined && value !== null) rawItems.push(value);
  }
  if (payload.aliases !== undefined && payload.aliases !== null) {
    rawItems.push(payload.aliases);
  }

  const aliases: string[] = [];
  const seen = new Set<string>();
  for (const item of rawItems) {
    if (typeof item === "string") {
      const token = item.trim();
      if (!token || seen.has(token)) continue;
      seen.add(token);
      aliases.push(token);
      continue;
    }
    if (Array.isArray(item)) {
      for (const entry of item) {
        const token = typeof entry === "string" ? entry.trim() : "";
        if (!token || seen.has(token)) continue;
        seen.add(token);
        aliases.push(token);
      }
    }
  }
  return aliases;
}

export function summarizeAction(action: ChatAction): string[] {
  const payload = action.payload ?? {};
  if (isEntityMergeActionType(action.action_type)) {
    const source =
      typeof payload.source_entity === "string"
        ? payload.source_entity
        : typeof payload.alias === "string"
          ? payload.alias
          : "候选别名";
    const target =
      typeof payload.target_title === "string"
        ? payload.target_title
        : typeof payload.canonical_name === "string"
          ? payload.canonical_name
          : typeof payload.target_card_id === "number"
            ? `卡片 #${payload.target_card_id}`
            : typeof payload.card_id === "number"
              ? `卡片 #${payload.card_id}`
              : "目标实体";
    return [`实体合并提案：将「${source}」并入「${target}」`];
  }
  if (action.action_type === "setting.upsert") {
    const key = typeof payload.key === "string" ? payload.key : "未命名设定";
    return [`更新设定：${key}`];
  }
  if (action.action_type === "setting.delete") {
    const key = typeof payload.key === "string" ? payload.key : "未命名设定";
    return [`删除设定：${key}`];
  }
  if (action.action_type === "card.create") {
    const title = typeof payload.title === "string" ? payload.title : "未命名卡片";
    return [`创建卡片：${title}`];
  }
  if (action.action_type === "card.update") {
    const cardId = typeof payload.card_id === "number" ? payload.card_id : "未知";
    const title = typeof payload.title === "string" ? `，新标题：${payload.title}` : "";
    return [`更新卡片 #${cardId}${title}`];
  }
  return [`执行动作：${action.action_type}`];
}

export function actionRiskHints(action: ChatAction): string[] {
  const payload = action.payload ?? {};
  const hints: string[] = [];
  if (isEntityMergeActionType(action.action_type)) {
    hints.push("该动作仅写入目标卡片 aliases，用于后续归一化；不会自动执行图谱硬合并。");
    hints.push("请先确认是否存在身份反转剧情，避免过早绑定别名。");
  }
  if (action.action_type === "setting.delete") {
    hints.push("删除后会立即影响检索与注入，但可通过动作回滚恢复。");
  }
  if (action.action_type === "card.update" && payload.merge === false) {
    hints.push("覆盖式更新：可能替换掉卡片原有字段。");
  }
  if (action.action_type === "setting.upsert") {
    const valueLength = JSON.stringify(payload.value ?? {}).length;
    if (valueLength > 2000) {
      hints.push("写入内容较大，可能增加检索与注入延迟。");
    }
  }
  return hints;
}

export type ActionDiffRow = {
  field: string;
  before: string;
  after: string;
  beforeTags?: string[];
  afterTags?: string[];
};

export function safeToRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

export function diffValuePreview(value: unknown, maxLength = 220): string {
  let serialized = "";
  if (value === undefined) {
    serialized = "（未设置）";
  } else if (value === null) {
    serialized = "null";
  } else if (typeof value === "string") {
    serialized = value;
  } else {
    serialized = formatJson(value);
  }
  const normalized = serialized.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, maxLength).trimEnd()}...`;
}

export function parseAliasTags(value: string): string[] {
  const raw = value.trim();
  if (!raw) return [];

  const collect = (items: unknown[]): string[] => {
    const seen = new Set<string>();
    const tags: string[] = [];
    for (const item of items) {
      const text = String(item ?? "").trim();
      if (!text) continue;
      if (seen.has(text)) continue;
      seen.add(text);
      tags.push(text);
    }
    return tags.slice(0, 24);
  };

  if (raw.startsWith("[") && raw.endsWith("]")) {
    try {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        return collect(parsed);
      }
    } catch {
      // noop
    }
  }

  const normalized = raw
    .replace(/^\[|\]$/g, "")
    .replace(/["']/g, "")
    .trim();
  if (!normalized) return [];
  return collect(normalized.split(/[,\n，、;；]+/));
}

export function isAliasDiffField(field: string): boolean {
  const normalized = field.trim().toLowerCase();
  return (
    normalized === "卡片.aliases" ||
    normalized.endsWith(".aliases") ||
    normalized.endsWith(".alias") ||
    normalized.includes("别名")
  );
}

export function buildObjectDiffRows(
  beforeObj: Record<string, unknown>,
  afterObj: Record<string, unknown>,
  prefix: string,
  limit = 8
): ActionDiffRow[] {
  const keys = Array.from(new Set([...Object.keys(beforeObj), ...Object.keys(afterObj)])).sort();
  const rows: ActionDiffRow[] = [];
  for (const key of keys) {
    const beforeText = diffValuePreview(beforeObj[key]);
    const afterText = diffValuePreview(afterObj[key]);
    if (beforeText === afterText) continue;
    const aliasField = isAliasDiffField(`${prefix}.${key}`);
    rows.push({
      field: `${prefix}.${key}`,
      before: beforeText,
      after: afterText,
      beforeTags: aliasField ? parseAliasTags(beforeText) : undefined,
      afterTags: aliasField ? parseAliasTags(afterText) : undefined,
    });
    if (rows.length >= limit) break;
  }
  if (keys.length > limit) {
    rows.push({
      field: `${prefix}.*`,
      before: "（更多字段已折叠）",
      after: `共 ${keys.length} 个字段，请展开调试 JSON 查看完整内容`,
    });
  }
  return rows;
}

export function buildActionDiffRows(action: ChatAction): ActionDiffRow[] {
  const payload = action.payload ?? {};
  const undoPayload = action.undo_payload ?? {};
  const applyResult = action.apply_result ?? {};

  if (isEntityMergeActionType(action.action_type)) {
    const beforeWrap = safeToRecord(undoPayload.before) ?? {};
    const beforeAliases = Array.isArray(beforeWrap.aliases) ? beforeWrap.aliases : [];
    const appliedAliases = Array.isArray(applyResult.aliases) ? applyResult.aliases : [];
    const pendingAliases = collectEntityMergePayloadAliases(payload);
    const afterAliases = appliedAliases.length > 0 ? appliedAliases : pendingAliases;
    const target =
      typeof payload.target_title === "string"
        ? payload.target_title
        : typeof applyResult.title === "string"
          ? applyResult.title
          : typeof payload.target_card_id === "number"
            ? `卡片 #${payload.target_card_id}`
            : "目标实体";

    return [
      {
        field: "实体目标",
        before: diffValuePreview(target),
        after: diffValuePreview(target),
      },
      {
        field: "卡片.aliases",
        before: diffValuePreview(beforeAliases.length > 0 ? beforeAliases : "（应用时读取当前值）"),
        after: diffValuePreview(afterAliases.length > 0 ? afterAliases : "（无候选别名）"),
        beforeTags: beforeAliases.map((item) => String(item).trim()).filter((item) => item.length > 0),
        afterTags: afterAliases.map((item) => String(item).trim()).filter((item) => item.length > 0),
      },
    ];
  }

  if (action.action_type === "setting.upsert") {
    const key = typeof payload.key === "string" ? payload.key : "未命名设定";
    const beforeWrap = safeToRecord(undoPayload.before);
    const beforeValue = beforeWrap?.exists === true ? beforeWrap.value : "（新增设定）";
    const afterValue = payload.value ?? payload.content ?? {};
    return [
      {
        field: `设定.${key}`,
        before: diffValuePreview(beforeValue),
        after: diffValuePreview(afterValue),
      },
    ];
  }

  if (action.action_type === "setting.delete") {
    const key = typeof payload.key === "string" ? payload.key : "未命名设定";
    const beforeWrap = safeToRecord(undoPayload.before);
    const beforeValue = beforeWrap?.exists === true ? beforeWrap.value : "（设定不存在）";
    return [
      {
        field: `设定.${key}`,
        before: diffValuePreview(beforeValue),
        after: "（已删除）",
      },
    ];
  }

  if (action.action_type === "card.create") {
    const title = typeof payload.title === "string" ? payload.title : "未命名卡片";
    const content = safeToRecord(payload.content) ?? {};
    return [
      { field: "卡片.title", before: "（新建）", after: diffValuePreview(title) },
      { field: "卡片.content", before: "（空）", after: diffValuePreview(content) },
    ];
  }

  if (action.action_type === "card.update") {
    const beforeWrap = safeToRecord(undoPayload.before) ?? {};
    const beforeTitle = typeof beforeWrap.title === "string" ? beforeWrap.title : "";
    const beforeContent = safeToRecord(beforeWrap.content) ?? {};
    const patchContent = safeToRecord(payload.content) ?? {};
    const merge = payload.merge !== false;
    const afterContent = merge ? { ...beforeContent, ...patchContent } : patchContent;
    const nextTitle = typeof payload.title === "string" ? payload.title : beforeTitle;

    const rows: ActionDiffRow[] = [];
    if (beforeTitle !== nextTitle) {
      rows.push({
        field: "卡片.title",
        before: diffValuePreview(beforeTitle || "（空）"),
        after: diffValuePreview(nextTitle || "（空）"),
      });
    }
    rows.push(...buildObjectDiffRows(beforeContent, afterContent, "卡片.content"));
    if (rows.length === 0) {
      rows.push({
        field: "卡片",
        before: "（无显著变化）",
        after: "（无显著变化）",
      });
    }
    return rows;
  }

  return [
    {
      field: action.action_type,
      before: "（未知）",
      after: diffValuePreview(payload),
    },
  ];
}