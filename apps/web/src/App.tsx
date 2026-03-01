import { Suspense, lazy, memo, useEffect, useMemo, useRef, useState } from "react";
import { Extension } from "@tiptap/core";
import Placeholder from "@tiptap/extension-placeholder";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import { EditorContent, useEditor, type Editor, type JSONContent } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import { useShallow } from "zustand/react/shallow";
import {
  createForeshadowingCard,
  createProjectPromptTemplate,
  createProjectVolume,
  createSceneBeat,
  createProjectChapter,
  deleteForeshadowingCard,
  deleteProjectPromptTemplate,
  deleteProjectVolume,
  deleteSceneBeat,
  deleteProjectChapter,
  decideAction,
  generateGhostText,
  getActionLogs,
  getForeshadowingCards,
  getProjectConsistencyAudits,
  getProjectGraphTimeline,
  getProjectPromptTemplates,
  getProjectPromptTemplateRevisions,
  getProjectVolumes,
  deleteProjectModelProfile,
  getProjectChapter,
  getProjectChapterRevisions,
  getProjectChapters,
  getProjectCards,
  getProjectModelProfiles,
  getProjectSessions,
  getSceneBeats,
  getProjectSettings,
  moveProjectChapter,
  preheatContextPack,
  reorderProjectChapters,
  rollbackProjectPromptTemplate,
  rollbackProjectChapter,
  runProjectConsistencyAudit,
  saveProjectChapter,
  updateProjectModelProfile,
  activateProjectModelProfile,
  createProjectModelProfile,
  updateForeshadowingCard,
  updateProjectPromptTemplate,
  updateProjectVolume,
  updateSceneBeat,
  getSessionActions,
  getSessionMessages,
  type ChatStreamTimingMetrics,
} from "./api/chatApi";
import { useChatStore } from "./store/chatStore";
import { ModeSwitch } from "./components/ModeSwitch";
import {
  clearDraftRecoverySnapshot,
  readDraftRecoverySnapshot,
  shouldRestoreDraftRecovery,
  useDraftWorkspaceFlow,
} from "./hooks/useDraftWorkspaceFlow";
import { useAssistantSessionFlow } from "./hooks/useAssistantSessionFlow";
import { useTimelineGraphFlow } from "./hooks/useTimelineGraphFlow";
import type {
  ActionAuditLog,
  ChatAction,
  ChatStreamTraceEvent,
  ConsistencyAuditReport,
  DraftAutoSaveState,
  EvidencePayload,
  ForeshadowingCard,
  GraphTimelineSnapshot,
  ChatSessionSummary,
  ProjectChapter,
  ProjectChapterRevision,
  ProjectVolume,
  ModelProfile,
  PromptTemplate,
  PromptTemplateRevision,
  SceneBeat,
  SettingEntry,
  StoryCard,
  UiMessage,
} from "./types";

type PostChatSnapshotData = {
  messagesData: Awaited<ReturnType<typeof getSessionMessages>>;
  actionsData: Awaited<ReturnType<typeof getSessionActions>>;
  sessionsData: ChatSessionSummary[];
};

type FullSessionSnapshotData = {
  messagesData: Awaited<ReturnType<typeof getSessionMessages>>;
  actionsData: Awaited<ReturnType<typeof getSessionActions>>;
  settingsData: Awaited<ReturnType<typeof getProjectSettings>>;
  auditsData: ConsistencyAuditReport[];
  modelProfilesData: Awaited<ReturnType<typeof getProjectModelProfiles>>;
  cardsData: Awaited<ReturnType<typeof getProjectCards>>;
  templatesData: Awaited<ReturnType<typeof getProjectPromptTemplates>>;
  volumesData: ProjectVolume[];
  foreshadowData: ForeshadowingCard[];
  sessionsData: ChatSessionSummary[];
};

type ProjectSnapshotData = {
  settingsData: Awaited<ReturnType<typeof getProjectSettings>>;
  auditsData: ConsistencyAuditReport[];
  modelProfilesData: Awaited<ReturnType<typeof getProjectModelProfiles>>;
  cardsData: Awaited<ReturnType<typeof getProjectCards>>;
  templatesData: Awaited<ReturnType<typeof getProjectPromptTemplates>>;
  volumesData: ProjectVolume[];
  foreshadowData: ForeshadowingCard[];
  sessionsData: ChatSessionSummary[];
};

type ChapterSnapshotData = {
  chapter: Awaited<ReturnType<typeof getProjectChapter>>;
  revisions: Awaited<ReturnType<typeof getProjectChapterRevisions>>;
  beats: SceneBeat[];
  overdueForeshadows: ForeshadowingCard[];
};

type DraftSnapshotData = {
  chapterList: Awaited<ReturnType<typeof getProjectChapters>>;
  volumeList: ProjectVolume[];
  foreshadowList: ForeshadowingCard[];
};

type PlanningSnapshotData = {
  volumeList: ProjectVolume[];
  foreshadowList: ForeshadowingCard[];
  beats: SceneBeat[];
  overdue: ForeshadowingCard[];
};

const POST_CHAT_SNAPSHOT_TTL_MS = 400;
type WritingTheme = "paper" | "wenkai" | "modern" | "contrast";

function toUiMessage(message: {
  id: number;
  role: "user" | "assistant" | "system";
  content: string;
}): UiMessage {
  return {
    id: `msg-${message.id}`,
    role: message.role,
    content: message.content,
  };
}

function formatRole(role: UiMessage["role"]): string {
  if (role === "user") return "你";
  if (role === "assistant") return "助手";
  return "系统";
}

function formatJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function formatDateTime(value?: string | null): string {
  if (!value) return "未保存";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function normalizeEditorText(value: string): string {
  const normalized = (value || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n").replace(/\u00a0/g, " ");
  if (normalized.endsWith("\n")) {
    return normalized.slice(0, -1);
  }
  return normalized;
}

function toEditorDoc(text: string): JSONContent {
  const normalized = normalizeEditorText(text);
  const lines = normalized.length > 0 ? normalized.split("\n") : [""];
  return {
    type: "doc",
    content: lines.map((line) => {
      if (!line) {
        return { type: "paragraph" };
      }
      return {
        type: "paragraph",
        content: [{ type: "text", text: line }],
      };
    }),
  };
}

function readEditorText(editor: Editor): string {
  return normalizeEditorText(editor.getText({ blockSeparator: "\n" }));
}

function readSelectedText(editor: Editor): string {
  const { from, to, empty } = editor.state.selection;
  if (empty) return "";
  const content = editor.state.doc.textBetween(from, to, "\n", "\n");
  return normalizeEditorText(content).trim();
}

function parseReferenceProjectIds(value: string, currentProjectId: number): number[] {
  const tokens = (value || "")
    .split(/[,\s]+/)
    .map((item) => Number(item))
    .filter((item) => Number.isFinite(item) && item > 0 && item !== currentProjectId);
  const deduped = Array.from(new Set(tokens));
  return deduped.slice(0, 5);
}

type EntityHighlightHint = {
  token: string;
  canonical: string;
  summary: string;
};

type EntityHintPluginState = {
  hints: EntityHighlightHint[];
  regex: RegExp | null;
  hintMap: Map<string, EntityHighlightHint>;
  decorations: DecorationSet;
};

const ENTITY_HINT_LIMIT = 120;
const ENTITY_ALIAS_FIELD_KEYS = new Set([
  "alias",
  "aliases",
  "aka",
  "别名",
  "称呼",
  "曾用名",
  "外号",
  "昵称",
  "简称",
]);
const ENTITY_NAME_FIELD_KEYS = new Set(["name", "title", "名称", "姓名", "角色名", "实体名"]);
const entityHintPluginKey = new PluginKey<EntityHintPluginState>("entity-inline-hints");

function normalizeEntityToken(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "")
    .replace(/[^\w\u4e00-\u9fff·-]/g, "");
}

function summarizeKnowledgeSnippet(value: unknown, maxLength = 64): string {
  let raw = "";
  if (typeof value === "string") {
    raw = value;
  } else if (value === null || value === undefined) {
    raw = "";
  } else {
    raw = formatJson(value);
  }
  const normalized = raw.replace(/\s+/g, " ").trim();
  if (!normalized) return "";
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, maxLength).trimEnd()}...`;
}

function splitAliasText(value: string): string[] {
  return value
    .split(/[,\s，、;；|/]+/)
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
}

function flattenAliasValues(value: unknown, depth = 0): string[] {
  if (depth > 2) return [];
  if (typeof value === "string") return splitAliasText(value);
  if (typeof value === "number" || typeof value === "boolean") return [String(value)];
  if (Array.isArray(value)) {
    const values: string[] = [];
    value.forEach((item) => values.push(...flattenAliasValues(item, depth + 1)));
    return values;
  }
  if (value && typeof value === "object") {
    const values: string[] = [];
    Object.values(value as Record<string, unknown>).forEach((item) => {
      values.push(...flattenAliasValues(item, depth + 1));
    });
    return values;
  }
  return [];
}

function extractAliasesFromRecord(record: Record<string, unknown>): string[] {
  const aliases: string[] = [];
  Object.entries(record).forEach(([key, value]) => {
    const normalizedKey = key.trim().toLowerCase();
    if (ENTITY_ALIAS_FIELD_KEYS.has(key.trim()) || ENTITY_ALIAS_FIELD_KEYS.has(normalizedKey)) {
      aliases.push(...flattenAliasValues(value));
    }
  });
  return aliases;
}

function extractNameCandidatesFromRecord(record: Record<string, unknown>): string[] {
  const names: string[] = [];
  Object.entries(record).forEach(([key, value]) => {
    const normalizedKey = key.trim().toLowerCase();
    if (!ENTITY_NAME_FIELD_KEYS.has(key.trim()) && !ENTITY_NAME_FIELD_KEYS.has(normalizedKey)) {
      return;
    }
    if (typeof value === "string") {
      names.push(value.trim());
    }
  });
  return names;
}

function collectEntityHighlightHints(settingsList: SettingEntry[], cardsList: StoryCard[]): EntityHighlightHint[] {
  const hints: EntityHighlightHint[] = [];
  const seen = new Set<string>();

  const pushHint = (token: string, canonical: string, summary: string) => {
    const trimmedToken = String(token || "").trim();
    const canonicalText = String(canonical || "").trim();
    if (!trimmedToken || !canonicalText) return;
    const normalized = normalizeEntityToken(trimmedToken);
    if (normalized.length < 2 || seen.has(normalized)) return;
    seen.add(normalized);
    hints.push({
      token: trimmedToken,
      canonical: canonicalText,
      summary,
    });
  };

  settingsList.forEach((item) => {
    const canonical = item.key.trim();
    const summary = summarizeKnowledgeSnippet(item.value);
    pushHint(canonical, canonical, summary);
    const valueObj = item.value && typeof item.value === "object" ? (item.value as Record<string, unknown>) : {};
    extractNameCandidatesFromRecord(valueObj).forEach((name) => pushHint(name, canonical, summary));
    extractAliasesFromRecord(valueObj).forEach((alias) => pushHint(alias, canonical, summary));
  });

  cardsList.forEach((card) => {
    const canonical = (card.title || "").trim();
    if (!canonical) return;
    const summary = summarizeKnowledgeSnippet(card.content);
    pushHint(canonical, canonical, summary);
    const contentObj =
      card.content && typeof card.content === "object" ? (card.content as Record<string, unknown>) : {};
    extractNameCandidatesFromRecord(contentObj).forEach((name) => pushHint(name, canonical, summary));
    extractAliasesFromRecord(contentObj).forEach((alias) => pushHint(alias, canonical, summary));
  });

  return hints.slice(0, ENTITY_HINT_LIMIT);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function buildEntityHintLookup(hints: EntityHighlightHint[]): {
  regex: RegExp | null;
  hintMap: Map<string, EntityHighlightHint>;
} {
  const hintMap = new Map<string, EntityHighlightHint>();
  const tokens: string[] = [];
  hints.forEach((item) => {
    const token = item.token.trim();
    if (!token) return;
    const lookupKey = token.toLowerCase();
    if (!hintMap.has(lookupKey)) {
      hintMap.set(lookupKey, item);
      tokens.push(token);
    }
  });
  if (tokens.length === 0) {
    return { regex: null, hintMap };
  }
  const pattern = tokens
    .sort((a, b) => b.length - a.length)
    .map((token) => escapeRegExp(token))
    .join("|");
  try {
    return {
      regex: new RegExp(pattern, "giu"),
      hintMap,
    };
  } catch {
    return { regex: null, hintMap };
  }
}

function buildEntityDecorations(
  doc: ProseMirrorNode,
  regex: RegExp | null,
  hintMap: Map<string, EntityHighlightHint>
): DecorationSet {
  if (!regex || hintMap.size === 0) {
    return DecorationSet.empty;
  }
  const decorations: Decoration[] = [];
  doc.descendants((node, pos) => {
    if (!node.isText || !node.text) return;
    const text = node.text;
    regex.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = regex.exec(text)) !== null) {
      const matched = match[0];
      if (!matched) {
        regex.lastIndex += 1;
        continue;
      }
      const from = pos + match.index;
      const to = from + matched.length;
      if (to <= from) continue;

      const hint = hintMap.get(matched.toLowerCase()) ?? hintMap.get(matched);
      if (!hint) continue;
      const title = hint.summary ? `${hint.canonical}：${hint.summary}` : hint.canonical;
      decorations.push(
        Decoration.inline(from, to, {
          class: "entity-inline-hint",
          "aria-label": title,
          "data-entity-canonical": hint.canonical,
          "data-entity-tooltip": title,
        })
      );
    }
  });
  return DecorationSet.create(doc, decorations);
}

const EntityInlineHintExtension = Extension.create({
  name: "entityInlineHint",
  addProseMirrorPlugins() {
    return [
      new Plugin<EntityHintPluginState>({
        key: entityHintPluginKey,
        state: {
          init(_, state) {
            const lookup = buildEntityHintLookup([]);
            return {
              hints: [],
              regex: lookup.regex,
              hintMap: lookup.hintMap,
              decorations: buildEntityDecorations(state.doc, lookup.regex, lookup.hintMap),
            };
          },
          apply(tr, value, _oldState, newState) {
            const meta = tr.getMeta(entityHintPluginKey);
            let hints = value.hints;
            let regex = value.regex;
            let hintMap = value.hintMap;
            let shouldRebuild = tr.docChanged;

            if (Array.isArray(meta)) {
              hints = meta
                .filter((item): item is EntityHighlightHint => {
                  if (!item || typeof item !== "object") return false;
                  const record = item as Record<string, unknown>;
                  return (
                    typeof record.token === "string" &&
                    typeof record.canonical === "string" &&
                    typeof record.summary === "string"
                  );
                })
                .slice(0, ENTITY_HINT_LIMIT);
              const lookup = buildEntityHintLookup(hints);
              regex = lookup.regex;
              hintMap = lookup.hintMap;
              shouldRebuild = true;
            }

            if (!shouldRebuild) {
              return value;
            }

            return {
              hints,
              regex,
              hintMap,
              decorations: buildEntityDecorations(newState.doc, regex, hintMap),
            };
          },
        },
        props: {
          decorations(state) {
            const pluginState = entityHintPluginKey.getState(state);
            return pluginState?.decorations ?? DecorationSet.empty;
          },
        },
      }),
    ];
  },
});

function normalizeAwarenessTag(value: string): string {
  const cleaned = value
    .replace(/^[#\[\(【]+/, "")
    .replace(/[\]\)】]+$/, "")
    .trim();
  if (!cleaned) return "";
  return cleaned.replace(/\s+/g, " ").slice(0, 24);
}

function collectAwarenessTags(
  evidence: EvidencePayload | null,
  options: { includeDebugSignals: boolean }
): string[] {
  const { includeDebugSignals } = options;
  if (!evidence) return [];
  const tags: string[] = [];
  const seen = new Set<string>();

  const pushTag = (raw: string | null | undefined) => {
    const cleaned = normalizeAwarenessTag(String(raw || ""));
    if (!cleaned || cleaned.length < 2 || seen.has(cleaned)) return;
    seen.add(cleaned);
    tags.push(cleaned);
  };

  pushTag(evidence.policy.anchor);
  evidence.sources.dsl.slice(0, 3).forEach((item) => pushTag(item.title));
  evidence.sources.graph.slice(0, 3).forEach((item) => {
    pushTag(item.title);
    if (includeDebugSignals) {
      pushTag(item.fact);
    }
  });
  evidence.sources.rag.slice(0, 2).forEach((item) => pushTag(item.title));
  return tags.slice(0, includeDebugSignals ? 8 : 5);
}

function isEntityMergeActionType(actionType: string): boolean {
  const normalized = (actionType || "").trim().toLowerCase().replace(/[_-]/g, ".");
  return normalized.startsWith("entity.merge") || normalized.startsWith("graph.entity.merge");
}

function collectEntityMergePayloadAliases(payload: Record<string, unknown>): string[] {
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

function summarizeAction(action: ChatAction): string[] {
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

function actionRiskHints(action: ChatAction): string[] {
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

type ActionDiffRow = {
  field: string;
  before: string;
  after: string;
  beforeTags?: string[];
  afterTags?: string[];
};

function safeToRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function diffValuePreview(value: unknown, maxLength = 220): string {
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

function parseAliasTags(value: string): string[] {
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

function isAliasDiffField(field: string): boolean {
  const normalized = field.trim().toLowerCase();
  return (
    normalized === "卡片.aliases" ||
    normalized.endsWith(".aliases") ||
    normalized.endsWith(".alias") ||
    normalized.includes("别名")
  );
}

function buildObjectDiffRows(
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

function buildActionDiffRows(action: ChatAction): ActionDiffRow[] {
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

const FOCUSABLE_SELECTOR = [
  "a[href]",
  "area[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(",");

function getFocusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter((element) => {
    if (element.getAttribute("aria-hidden") === "true") return false;
    if (element.hasAttribute("disabled")) return false;
    if (element.tabIndex < 0) return false;
    return element.getClientRects().length > 0;
  });
}

function focusFirstDialogElement(container: HTMLElement): void {
  const preferred = container.querySelector<HTMLElement>("[data-autofocus]");
  if (preferred && !preferred.hasAttribute("disabled") && preferred.getClientRects().length > 0) {
    preferred.focus();
    return;
  }
  const firstField = container.querySelector<HTMLElement>(
    "input:not([disabled]), select:not([disabled]), textarea:not([disabled])"
  );
  if (firstField) {
    firstField.focus();
    return;
  }
  const focusables = getFocusableElements(container);
  if (focusables.length > 0) {
    focusables[0].focus();
    return;
  }
  container.focus();
}

type WorkspaceStatusBarProps = {
  uiMode: "writing" | "pro";
  sessionId: number | null;
  ghostAutoEnabled: boolean;
  referenceProjectIds: number[];
  retrievalDegraded: boolean;
  degradedReasons: string[];
  lastStreamMetrics: ChatStreamTimingMetrics | null;
};

const WorkspaceStatusBar = memo(function WorkspaceStatusBar({
  uiMode,
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
        <strong>{uiMode === "writing" ? "写作模式" : "工作台模式"}</strong>
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
        {uiMode === "pro" && degradedReasons.length > 0 ? (
          <small>{degradedReasons.slice(0, 2).join(" / ")}</small>
        ) : null}
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

type LazyPanelFallbackProps = {
  title: string;
  detail: string;
  className?: string;
};

const LazyPanelFallback = memo(function LazyPanelFallback({ title, detail, className }: LazyPanelFallbackProps) {
  return (
    <section className={className ?? "panel"}>
      <div className="panel-title">
        <h2>{title}</h2>
        <small>加载中...</small>
      </div>
      <p className="empty">{detail}</p>
    </section>
  );
});

const LazyDebugSnapshotGrid = lazy(async () => {
  const module = await import("./debugPanels");
  return { default: module.DebugSnapshotGrid };
});

const LazyPromptWorkshopPanel = lazy(async () => {
  const module = await import("./debugPanels");
  return { default: module.PromptWorkshopPanel };
});

const LazyGraphCandidateReviewPanel = lazy(async () => {
  const module = await import("./debugPanels");
  return { default: module.GraphCandidateReviewPanel };
});

type AssistantChatPanelProps = {
  usage: Record<string, unknown> | null;
  messages: UiMessage[];
  input: string;
  streaming: boolean;
  composerInputRef: { current: HTMLTextAreaElement | null };
  setInput: (value: string) => void;
  handleSendRef: { current: () => Promise<void> };
};

const AssistantChatPanel = memo(function AssistantChatPanel({
  usage,
  messages,
  input,
  streaming,
  composerInputRef,
  setInput,
  handleSendRef,
}: AssistantChatPanelProps) {
  return (
    <section className="panel chat-panel">
      <div className="panel-title">
        <h2>聊天流</h2>
        {usage ? (
          <small className="usage">{formatJson(usage)}</small>
        ) : (
          <small>尚未返回 usage</small>
        )}
      </div>

      <div className="chat-log">
        {messages.length === 0 ? <p className="empty">先发一条消息，验证 SSE + 动作提议。</p> : null}
        {messages.map((message) => (
          <article key={message.id} className={`msg ${message.role}`}>
            <div className="msg-head">
              <span>{formatRole(message.role)}</span>
              {message.streaming ? <em>streaming...</em> : null}
            </div>
            <pre>{message.content}</pre>
          </article>
        ))}
      </div>

      <div className="composer">
        <textarea
          aria-label="给助手的输入"
          ref={composerInputRef}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="例：请补充设定，主角的第一动机是复仇但不能越过底线。"
          rows={4}
          disabled={streaming}
        />
        <button className="btn primary" onClick={() => void handleSendRef.current()} disabled={streaming || !input.trim()}>
          {streaming ? "发送中..." : "发送"}
        </button>
      </div>
    </section>
  );
});

type ActionCardProps = {
  action: ChatAction;
  isPending: boolean;
  controlsDisabled: boolean;
  loadLogsRef: { current: (actionId: number) => Promise<void> };
  mutateActionRef: { current: (action: ChatAction, decision: "apply" | "reject" | "undo") => Promise<void> };
};

const ActionCard = memo(function ActionCard({
  action,
  isPending,
  controlsDisabled,
  loadLogsRef,
  mutateActionRef,
}: ActionCardProps) {
  const summaryLines = useMemo(() => summarizeAction(action), [action]);
  const riskHints = useMemo(() => actionRiskHints(action), [action]);
  const diffRows = useMemo(() => buildActionDiffRows(action), [action]);
  const [undoFlash, setUndoFlash] = useState(false);
  const previousStatusRef = useRef(action.status);
  const undoFlashTimerRef = useRef<number | null>(null);

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

  const cardClassName = `action-card${isPending ? " highlight" : ""}${undoFlash ? " undo-flash" : ""}`;

  return (
    <article className={cardClassName}>
      <button type="button" className="action-summary" onClick={() => void loadLogsRef.current(action.id)}>
        <span>#{action.id}</span>
        <strong>{action.action_type}</strong>
        <span className={`status ${action.status}`}>{action.status}</span>
      </button>
      <div className="action-summary-body">
        {summaryLines.map((line, idx) => (
          <p key={`${action.id}-summary-${idx}`} className="action-summary-line">
            {line}
          </p>
        ))}
        {riskHints.map((line, idx) => (
          <p key={`${action.id}-risk-${idx}`} className="action-risk-line">
            风险提示：{line}
          </p>
        ))}
      </div>
      <div className="action-diff">
        {diffRows.map((row, idx) => {
          const isAliasRow = isAliasDiffField(row.field);
          const beforeTags = row.beforeTags ?? (isAliasRow ? parseAliasTags(row.before) : []);
          const afterTags = row.afterTags ?? (isAliasRow ? parseAliasTags(row.after) : []);
          return (
            <article key={`${action.id}-diff-${idx}`} className="action-diff-row">
              <div className="action-diff-field">{row.field}</div>
              <div className="action-diff-before">
                <small>原值</small>
                {isAliasRow && beforeTags.length > 0 ? (
                  <div className="action-alias-tags">
                    {beforeTags.map((tag) => (
                      <span key={`before-${action.id}-${idx}-${tag}`} className="awareness-tag">
                        {tag}
                      </span>
                    ))}
                  </div>
                ) : (
                  <pre>{row.before}</pre>
                )}
              </div>
              <div className="action-diff-after">
                <small>新值</small>
                {isAliasRow && afterTags.length > 0 ? (
                  <div className="action-alias-tags">
                    {afterTags.map((tag) => (
                      <span key={`after-${action.id}-${idx}-${tag}`} className="awareness-tag">
                        {tag}
                      </span>
                    ))}
                  </div>
                ) : (
                  <pre>{row.after}</pre>
                )}
              </div>
            </article>
          );
        })}
      </div>
      <details className="action-raw-details">
        <summary>调试 JSON</summary>
        <div className="action-raw-grid">
          <article>
            <small>payload</small>
            <pre>{formatJson(action.payload)}</pre>
          </article>
          <article>
            <small>apply_result</small>
            <pre>{formatJson(action.apply_result)}</pre>
          </article>
          <article>
            <small>undo_payload</small>
            <pre>{formatJson(action.undo_payload)}</pre>
          </article>
        </div>
      </details>
      <div className="action-ops">
        {action.status === "proposed" ? (
          <>
            <button
              className="btn primary tiny"
              onClick={() => void mutateActionRef.current(action, "apply")}
              disabled={controlsDisabled}
            >
              应用并记录
            </button>
            <button
              className="btn ghost tiny"
              onClick={() => void mutateActionRef.current(action, "reject")}
              disabled={controlsDisabled}
            >
              不应用
            </button>
          </>
        ) : null}
        {action.status === "applied" ? (
          <button
            className="btn ghost tiny"
            onClick={() => void mutateActionRef.current(action, "undo")}
            disabled={controlsDisabled}
          >
            撤销应用
          </button>
        ) : null}
      </div>
      {action.status === "proposed" ? (
        <p className="action-risk-line">应用后会立即写入设定/卡片，并记录审计日志。</p>
      ) : null}
    </article>
  );
});

type ActionLogsListProps = {
  selectedActionId: number | null;
  actionLogs: ActionAuditLog[];
};

const ActionLogsList = memo(function ActionLogsList({ selectedActionId, actionLogs }: ActionLogsListProps) {
  return (
    <>
      <div className="panel-title sub">
        <h3>动作日志</h3>
        <small>{selectedActionId ? `action #${selectedActionId}` : "未选择动作"}</small>
      </div>
      <div className="log-list">
        {actionLogs.length === 0 ? <p className="empty">点一条动作查看审计日志</p> : null}
        {actionLogs.map((log) => (
          <article key={log.id} className="log-card">
            <div className="msg-head">
              <span>{log.event_type}</span>
              <small>{log.operator_id}</small>
            </div>
            <pre>{formatJson(log.event_payload)}</pre>
          </article>
        ))}
      </div>
    </>
  );
});

type ChapterOutlineEntry = {
  id: number;
  chapterIndex: number;
  title: string;
  wordCount: number;
  preview: string;
};

type ChapterOutlineListProps = {
  chapterOutlines: ChapterOutlineEntry[];
  activeChapterId: number | null;
  dragChapterId: number | null;
  disabled: boolean;
  onDragStartRef: { current: (chapterId: number) => void };
  onDragEndRef: { current: () => void };
  onReorderRef: { current: (targetChapterId: number) => Promise<void> };
  onSelectRef: { current: (chapterId: number) => Promise<void> };
};

const ChapterOutlineList = memo(function ChapterOutlineList({
  chapterOutlines,
  activeChapterId,
  dragChapterId,
  disabled,
  onDragStartRef,
  onDragEndRef,
  onReorderRef,
  onSelectRef,
}: ChapterOutlineListProps) {
  return (
    <div className="chapter-outline-list">
      {chapterOutlines.length === 0 ? <p className="empty">暂无章节</p> : null}
      {chapterOutlines.map((item) => (
        <button
          key={item.id}
          type="button"
          draggable
          className={`chapter-outline-item ${item.id === activeChapterId ? "active" : ""} ${
            item.id === dragChapterId ? "dragging" : ""
          }`}
          onDragStart={() => onDragStartRef.current(item.id)}
          onDragEnd={() => onDragEndRef.current()}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            void onReorderRef.current(item.id);
          }}
          onClick={() => void onSelectRef.current(item.id)}
          disabled={disabled}
        >
          <strong>
            {item.chapterIndex}. {item.title}
          </strong>
          <small>{item.wordCount} 字</small>
          <span>{item.preview}</span>
        </button>
      ))}
    </div>
  );
});

type DraftRevisionListProps = {
  draftRevisions: ProjectChapterRevision[];
  disabled: boolean;
  rollbackDraftToVersionRef: { current: (targetVersion: number) => Promise<void> };
};

const DraftRevisionList = memo(function DraftRevisionList({
  draftRevisions,
  disabled,
  rollbackDraftToVersionRef,
}: DraftRevisionListProps) {
  return (
    <details className="draft-history">
      <summary>版本历史（最近 {draftRevisions.length} 条）</summary>
      <div className="draft-revision-list">
        {draftRevisions.length === 0 ? <p className="empty">暂无版本历史</p> : null}
        {draftRevisions.map((revision) => (
          <article key={revision.id} className="draft-revision-card">
            <div className="msg-head">
              <span>
                v{revision.version} · {revision.source}
              </span>
              <small>{formatDateTime(revision.created_at)}</small>
            </div>
            {(revision.semantic_summary ?? []).length > 0 ? (
              <ul className="revision-semantic-list">
                {(revision.semantic_summary ?? []).map((line, idx) => (
                  <li key={`${revision.id}-semantic-${idx}`}>{line}</li>
                ))}
              </ul>
            ) : null}
            <pre>{revision.content.slice(0, 220) || "(空正文)"}</pre>
            <div className="action-ops">
              <button
                className="btn ghost tiny"
                onClick={() => void rollbackDraftToVersionRef.current(revision.version)}
                disabled={disabled}
              >
                回滚到此版本
              </button>
            </div>
          </article>
        ))}
      </div>
    </details>
  );
});

type StoryPlanningPanelProps = {
  activeChapterId: number | null;
  volumes: ProjectVolume[];
  activeVolumeId: number | null;
  onSelectVolume: (volumeId: number) => void;
  onCreateVolume: () => Promise<void>;
  onBindChapterToVolume: (volumeId: number) => Promise<void>;
  volumeOutlineDraft: string;
  setVolumeOutlineDraft: (value: string) => void;
  onSaveVolumeOutline: () => Promise<void>;
  sceneBeats: SceneBeat[];
  activeSceneBeatId: number | null;
  onSelectSceneBeat: (beatId: number | null) => void;
  newBeatContent: string;
  setNewBeatContent: (value: string) => void;
  onCreateSceneBeat: () => Promise<void>;
  onToggleSceneBeatStatus: (beatId: number, done: boolean) => Promise<void>;
  onDeleteSceneBeat: (beatId: number) => Promise<void>;
  foreshadowCards: ForeshadowingCard[];
  overdueForeshadowCards: ForeshadowingCard[];
  foreshadowDraftTitle: string;
  setForeshadowDraftTitle: (value: string) => void;
  foreshadowDraftDescription: string;
  setForeshadowDraftDescription: (value: string) => void;
  onCreateForeshadowCard: () => Promise<void>;
  onToggleForeshadowStatus: (card: ForeshadowingCard, nextStatus: "open" | "resolved") => Promise<void>;
  onDeleteForeshadowCard: (cardId: number) => Promise<void>;
  busy: boolean;
};

const StoryPlanningPanel = memo(function StoryPlanningPanel({
  activeChapterId,
  volumes,
  activeVolumeId,
  onSelectVolume,
  onCreateVolume,
  onBindChapterToVolume,
  volumeOutlineDraft,
  setVolumeOutlineDraft,
  onSaveVolumeOutline,
  sceneBeats,
  activeSceneBeatId,
  onSelectSceneBeat,
  newBeatContent,
  setNewBeatContent,
  onCreateSceneBeat,
  onToggleSceneBeatStatus,
  onDeleteSceneBeat,
  foreshadowCards,
  overdueForeshadowCards,
  foreshadowDraftTitle,
  setForeshadowDraftTitle,
  foreshadowDraftDescription,
  setForeshadowDraftDescription,
  onCreateForeshadowCard,
  onToggleForeshadowStatus,
  onDeleteForeshadowCard,
  busy,
}: StoryPlanningPanelProps) {
  return (
    <section className="panel planning-panel">
      <div className="panel-title">
        <h2>结构化大纲与伏笔</h2>
        <small>Volume / Scene Beat / Foreshadow</small>
      </div>
      <div className="planning-grid">
        <article className="planning-card">
          <div className="panel-title sub">
            <h3>卷纲</h3>
            <small>{activeVolumeId ? `卷 #${activeVolumeId}` : "未绑定"}</small>
          </div>
          <div className="planning-row">
            <select
              value={activeVolumeId ?? ""}
              onChange={(event) => {
                const nextId = Number(event.target.value || 0);
                if (!nextId) return;
                onSelectVolume(nextId);
                if (activeChapterId) {
                  void onBindChapterToVolume(nextId);
                }
              }}
              disabled={busy}
            >
              <option value="">选择卷</option>
              {volumes.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.volume_index}. {item.title}
                </option>
              ))}
            </select>
            <button className="btn ghost tiny" onClick={() => void onCreateVolume()} disabled={busy}>
              新建卷
            </button>
          </div>
          <textarea
            rows={4}
            value={volumeOutlineDraft}
            onChange={(event) => setVolumeOutlineDraft(event.target.value)}
            placeholder="卷纲：本卷核心冲突、推进目标与收束点。"
            disabled={busy || !activeVolumeId}
          />
          <div className="planning-row">
            <button className="btn ghost tiny" onClick={() => void onSaveVolumeOutline()} disabled={busy || !activeVolumeId}>
              保存卷纲
            </button>
          </div>
        </article>

        <article className="planning-card">
          <div className="panel-title sub">
            <h3>Scene Beats</h3>
            <small>{sceneBeats.length} 条</small>
          </div>
          <div className="scene-beat-list">
            {sceneBeats.length === 0 ? <p className="empty">当前章节还没有 Beat</p> : null}
            {sceneBeats.map((beat) => (
              <article
                key={beat.id}
                className={`scene-beat-item ${beat.id === activeSceneBeatId ? "active" : ""}`}
                role="button"
                tabIndex={0}
                aria-pressed={beat.id === activeSceneBeatId}
                onClick={() => onSelectSceneBeat(beat.id)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelectSceneBeat(beat.id);
                  }
                }}
              >
                <strong>
                  Beat {beat.beat_index} · {beat.status === "done" ? "已完成" : "进行中"}
                </strong>
                <p>{beat.content || "（空）"}</p>
                <div className="action-ops">
                  <button
                    className="btn ghost tiny"
                    onClick={(event) => {
                      event.stopPropagation();
                      void onToggleSceneBeatStatus(beat.id, beat.status !== "done");
                    }}
                    disabled={busy}
                  >
                    {beat.status === "done" ? "标记进行中" : "标记已完成"}
                  </button>
                  <button
                    className="btn ghost tiny"
                    onClick={(event) => {
                      event.stopPropagation();
                      void onDeleteSceneBeat(beat.id);
                    }}
                    disabled={busy}
                  >
                    删除
                  </button>
                </div>
              </article>
            ))}
          </div>
          <textarea
            rows={3}
            value={newBeatContent}
            onChange={(event) => setNewBeatContent(event.target.value)}
            placeholder="新增 Beat：例如「男主发现破绽并留下悬念」"
            disabled={busy || !activeChapterId}
          />
          <div className="planning-row">
            <button className="btn ghost tiny" onClick={() => void onCreateSceneBeat()} disabled={busy || !activeChapterId}>
              添加 Beat
            </button>
            <button className="btn ghost tiny" onClick={() => onSelectSceneBeat(null)} disabled={busy}>
              不使用 Beat 约束
            </button>
          </div>
        </article>

        <article className="planning-card">
          <div className="panel-title sub">
            <h3>伏笔追踪</h3>
            <small>{foreshadowCards.length} 条</small>
          </div>
          {overdueForeshadowCards.length > 0 ? (
            <p className="draft-hint warning">提醒：有 {overdueForeshadowCards.length} 条伏笔已超 50 章未收束。</p>
          ) : null}
          <div className="foreshadow-list">
            {foreshadowCards.length === 0 ? <p className="empty">暂无伏笔卡</p> : null}
            {foreshadowCards.map((item) => (
              <article key={item.id} className="foreshadow-item">
                <strong>
                  {item.title} · {item.status === "resolved" ? "已收束" : "未收束"}
                </strong>
                <p>{item.description || "（无描述）"}</p>
                <div className="action-ops">
                  <button
                    className="btn ghost tiny"
                    onClick={() =>
                      void onToggleForeshadowStatus(item, item.status === "resolved" ? "open" : "resolved")
                    }
                    disabled={busy}
                  >
                    {item.status === "resolved" ? "改为未收束" : "标记已收束"}
                  </button>
                  <button className="btn ghost tiny" onClick={() => void onDeleteForeshadowCard(item.id)} disabled={busy}>
                    删除
                  </button>
                </div>
              </article>
            ))}
          </div>
          <input
            type="text"
            value={foreshadowDraftTitle}
            onChange={(event) => setForeshadowDraftTitle(event.target.value)}
            placeholder="伏笔标题：如「半块玉佩」"
            disabled={busy}
          />
          <textarea
            rows={3}
            value={foreshadowDraftDescription}
            onChange={(event) => setForeshadowDraftDescription(event.target.value)}
            placeholder="伏笔描述：埋入信息、预期收束方向。"
            disabled={busy}
          />
          <div className="planning-row">
            <button className="btn ghost tiny" onClick={() => void onCreateForeshadowCard()} disabled={busy}>
              新建伏笔卡
            </button>
          </div>
        </article>
      </div>
    </section>
  );
});

type TopBarProps = {
  uiMode: "writing" | "pro";
  zenMode: boolean;
  streaming: boolean;
  settingsDialogOpen: boolean;
  assistantDrawerOpen: boolean;
  onToggleUiMode: () => void;
  onToggleZenMode: () => void;
  onOpenSettingsDialog: () => void;
  onOpenAssistantDrawer: () => void;
  onRefreshSnapshot: () => Promise<void>;
  onStartNewSession: () => void;
};

const TopBar = memo(function TopBar({
  uiMode,
  zenMode,
  streaming,
  settingsDialogOpen,
  assistantDrawerOpen,
  onToggleUiMode,
  onToggleZenMode,
  onOpenSettingsDialog,
  onOpenAssistantDrawer,
  onRefreshSnapshot,
  onStartNewSession,
}: TopBarProps) {
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">Novel Platform</p>
        <h1>AI 辅助写作工作台</h1>
      </div>
      <div className="top-actions">
        <ModeSwitch uiMode={uiMode} onToggle={onToggleUiMode} disabled={streaming} />
        <button
          className="btn ghost"
          onClick={onToggleZenMode}
          disabled={streaming || uiMode !== "writing"}
          aria-pressed={zenMode}
        >
          {zenMode ? "退出沉浸" : "进入沉浸"}
        </button>
        <button
          type="button"
          className="btn ghost"
          onClick={onOpenSettingsDialog}
          aria-haspopup="dialog"
          aria-expanded={settingsDialogOpen}
          aria-controls="settings-dialog"
          disabled={streaming}
        >
          写作设置
        </button>
        <button
          type="button"
          className="btn primary"
          onClick={onOpenAssistantDrawer}
          aria-haspopup="dialog"
          aria-expanded={assistantDrawerOpen}
          aria-controls="assistant-drawer"
        >
          助手抽屉
        </button>
        <button className="btn ghost" onClick={() => void onRefreshSnapshot()} disabled={streaming}>
          刷新快照
        </button>
        <button className="btn ghost" onClick={onStartNewSession} disabled={streaming}>
          新会话
        </button>
      </div>
    </header>
  );
});

type WorkbenchPanelVisibility = {
  prompt: boolean;
  planning: boolean;
  snapshot: boolean;
};

type WorkbenchPanelKey = keyof WorkbenchPanelVisibility;

const WORKBENCH_PANEL_LABELS: Record<WorkbenchPanelKey, string> = {
  prompt: "Prompt + 知识库",
  planning: "结构化大纲",
  snapshot: "检索快照",
};

type WorkbenchPanelBarProps = {
  visibility: WorkbenchPanelVisibility;
  onToggle: (panelKey: WorkbenchPanelKey) => void;
};

const WorkbenchPanelBar = memo(function WorkbenchPanelBar({
  visibility,
  onToggle,
}: WorkbenchPanelBarProps) {
  const allHidden = !visibility.prompt && !visibility.planning && !visibility.snapshot;
  return (
    <section className="panel workbench-panel-bar">
      <div className="panel-title sub">
        <h3>工作台面板</h3>
        <small>默认仅展示核心面板，可按需展开</small>
      </div>
      <div className="workbench-panel-toggles">
        {(Object.keys(WORKBENCH_PANEL_LABELS) as WorkbenchPanelKey[]).map((panelKey) => (
          <label key={panelKey} className="workbench-panel-toggle">
            <input
              type="checkbox"
              checked={visibility[panelKey]}
              onChange={() => onToggle(panelKey)}
            />
            <span>{WORKBENCH_PANEL_LABELS[panelKey]}</span>
          </label>
        ))}
      </div>
      {allHidden ? <p className="workbench-panel-hint">当前未显示任何工作台面板，建议至少开启一个。</p> : null}
    </section>
  );
});

type DraftWorkspacePanelProps = {
  draftWordCount: number;
  draftVersion: number;
  draftUpdatedAt: string | null;
  activeChapterId: number | null;
  chapters: ProjectChapter[];
  switchChapterRef: { current: (chapterId: number) => Promise<void> };
  draftLoading: boolean;
  draftSaving: boolean;
  draftTitle: string;
  setDraftTitle: (value: string) => void;
  createChapterAndSwitchRef: { current: () => Promise<void> };
  moveActiveChapterRef: { current: (direction: "up" | "down") => Promise<void> };
  canMoveChapterUp: boolean;
  canMoveChapterDown: boolean;
  deleteActiveChapterRef: { current: () => Promise<void> };
  awarenessTags: string[];
  draftFocusMode: boolean;
  autoSaveState: DraftAutoSaveState;
  autoSaveAt: string | null;
  typewriterModeEnabled: boolean;
  localRecoveryNotice: string | null;
  onToggleTypewriterMode: () => void;
  onToggleDraftFocusMode: () => void;
  onToggleZenMode: () => void;
  zenMode: boolean;
  uiMode: "writing" | "pro";
  draftEditorRef: { current: HTMLDivElement | null };
  editor: Editor | null;
  ghostLoading: boolean;
  ghostText: string;
  ghostError: string | null;
  ghostAutoEnabled: boolean;
  onRequestGhostSuggestion: (forceRefresh?: boolean) => Promise<void>;
  acceptGhostTextRef: { current: () => void };
  rejectGhostTextRef: { current: () => void };
  regenerateGhostTextRef: { current: () => Promise<void> };
  onToggleGhostAuto: () => void;
  saveDraftSnapshotRef: { current: () => Promise<void> };
  refreshDraftSnapshotRef: { current: (nextProjectId: number, preferredChapterId?: number | null) => Promise<void> };
  projectId: number;
  fillPromptFromSelectionRef: { current: (mode: "polish" | "expand") => void };
  applyAssistantToDraftRef: { current: (mode: "insert" | "replace") => void };
  selectedDraftText: string;
  latestAssistantReply: string;
  chapterOutlines: ChapterOutlineEntry[];
  dragChapterId: number | null;
  handleOutlineDragStartRef: { current: (chapterId: number) => void };
  handleOutlineDragEndRef: { current: () => void };
  reorderByDragRef: { current: (targetChapterId: number) => Promise<void> };
  draftRevisions: ProjectChapterRevision[];
  rollbackDraftToVersionRef: { current: (targetVersion: number) => Promise<void> };
};

const DraftWorkspacePanel = memo(function DraftWorkspacePanel({
  draftWordCount,
  draftVersion,
  draftUpdatedAt,
  activeChapterId,
  chapters,
  switchChapterRef,
  draftLoading,
  draftSaving,
  draftTitle,
  setDraftTitle,
  createChapterAndSwitchRef,
  moveActiveChapterRef,
  canMoveChapterUp,
  canMoveChapterDown,
  deleteActiveChapterRef,
  awarenessTags,
  draftFocusMode,
  autoSaveState,
  autoSaveAt,
  typewriterModeEnabled,
  localRecoveryNotice,
  onToggleTypewriterMode,
  onToggleDraftFocusMode,
  onToggleZenMode,
  zenMode,
  uiMode,
  draftEditorRef,
  editor,
  ghostLoading,
  ghostText,
  ghostError,
  ghostAutoEnabled,
  onRequestGhostSuggestion,
  acceptGhostTextRef,
  rejectGhostTextRef,
  regenerateGhostTextRef,
  onToggleGhostAuto,
  saveDraftSnapshotRef,
  refreshDraftSnapshotRef,
  projectId,
  fillPromptFromSelectionRef,
  applyAssistantToDraftRef,
  selectedDraftText,
  latestAssistantReply,
  chapterOutlines,
  dragChapterId,
  handleOutlineDragStartRef,
  handleOutlineDragEndRef,
  reorderByDragRef,
  draftRevisions,
  rollbackDraftToVersionRef,
}: DraftWorkspacePanelProps) {
  const ghostPreviewKey = useMemo(() => {
    const trimmed = ghostText.trim();
    if (!trimmed) return "ghost-placeholder";
    return `ghost-${trimmed.length}-${trimmed.slice(0, 24)}-${trimmed.slice(-24)}`;
  }, [ghostText]);

  return (
    <section className="panel draft-panel">
      <div className="panel-title">
        <h2>正文工作区</h2>
        <small>
          {draftWordCount} 字 · v{draftVersion} · {formatDateTime(draftUpdatedAt)}
        </small>
      </div>
      <div className="draft-chapter-row">
        <label className="inline-field">
          章节
          <select
            value={activeChapterId ?? ""}
            onChange={(event) => void switchChapterRef.current(Number(event.target.value || 0))}
            disabled={draftLoading || draftSaving}
          >
            {chapters.map((chapter) => (
              <option key={chapter.id} value={chapter.id}>
                {chapter.chapter_index}. {chapter.title}
              </option>
            ))}
          </select>
        </label>
        <label className="inline-field">
          章节标题
          <input
            type="text"
            value={draftTitle}
            onChange={(event) => setDraftTitle(event.target.value)}
            disabled={draftLoading || draftSaving || !activeChapterId}
          />
        </label>
        <button className="btn ghost tiny" onClick={() => void createChapterAndSwitchRef.current()} disabled={draftLoading || draftSaving}>
          新建章节
        </button>
        <button
          className="btn ghost tiny"
          onClick={() => void moveActiveChapterRef.current("up")}
          disabled={draftLoading || draftSaving || !canMoveChapterUp}
        >
          上移
        </button>
        <button
          className="btn ghost tiny"
          onClick={() => void moveActiveChapterRef.current("down")}
          disabled={draftLoading || draftSaving || !canMoveChapterDown}
        >
          下移
        </button>
        <button
          className="btn ghost tiny"
          onClick={() => void deleteActiveChapterRef.current()}
          disabled={draftLoading || draftSaving || !activeChapterId}
        >
          删除章节
        </button>
      </div>
      <div className="awareness-strip" aria-live="polite">
        <small>AI 当前认知</small>
        <div className="awareness-tags">
          {awarenessTags.length === 0 ? (
            <span className="awareness-empty">等待会话建立上下文</span>
          ) : (
            awarenessTags.map((tag) => (
              <span key={tag} className="awareness-tag">
                #{tag}
              </span>
            ))
          )}
        </div>
      </div>
      <div className={`draft-editor-shell ${draftFocusMode ? "focus-mode" : ""}`}>
        <div className="draft-editor-toolbar">
          <div className="draft-editor-status">
            <small>
              编辑模式：{draftFocusMode ? "专注" : "标准"} · 零感保存：
              {autoSaveState === "pending" ? "等待中" : null}
              {autoSaveState === "saving" ? "保存中" : null}
              {autoSaveState === "saved" ? `已保存(${formatDateTime(autoSaveAt)})` : null}
              {autoSaveState === "error" ? "失败" : null}
              {autoSaveState === "idle" ? "空闲" : null}
            </small>
            <small>打字机滚动：{typewriterModeEnabled ? "开启" : "关闭"}</small>
            {localRecoveryNotice ? <small className="recovery-note">{localRecoveryNotice}</small> : null}
          </div>
          <div className="draft-toolbar-actions">
            <button
              type="button"
              className="btn ghost tiny"
              onClick={onToggleTypewriterMode}
              disabled={draftLoading || !activeChapterId}
            >
              {typewriterModeEnabled ? "关闭打字机滚动" : "开启打字机滚动"}
            </button>
            <button
              type="button"
              className="btn ghost tiny"
              onClick={onToggleDraftFocusMode}
              disabled={draftLoading || !activeChapterId}
            >
              {draftFocusMode ? "退出专注" : "进入专注"}
            </button>
            <button
              type="button"
              className="btn ghost tiny"
              onClick={onToggleZenMode}
              disabled={draftLoading || !activeChapterId || uiMode !== "writing"}
              aria-pressed={zenMode}
            >
              {zenMode ? "退出沉浸" : "进入沉浸"}
            </button>
          </div>
        </div>
        <EditorContent
          ref={draftEditorRef}
          editor={editor}
          className={`draft-editor ${draftLoading || !activeChapterId ? "disabled" : ""}`}
        />
      </div>
      <div className="ghost-panel">
        <div className="ghost-head">
          <strong>Ghost Text</strong>
          <small>
            {ghostLoading ? "生成中..." : ghostText ? "已就绪" : "等待输入"}
            {ghostError ? ` · ${ghostError}` : ""}
          </small>
        </div>
        <pre key={ghostPreviewKey} className={`ghost-preview ${ghostText.trim() ? "ready" : ""}`}>
          {ghostText ||
            (ghostAutoEnabled
              ? "继续输入正文，系统会自动给出下一句建议。"
              : "当前为手动触发，点击“生成建议”获取 Ghost Text。")}
        </pre>
        <div className="action-ops">
          <button
            className="btn ghost tiny"
            onClick={() => void onRequestGhostSuggestion(false)}
            disabled={ghostLoading || draftLoading || draftSaving || !activeChapterId}
          >
            生成建议
          </button>
          <button
            className="btn primary tiny"
            onClick={acceptGhostTextRef.current}
            disabled={!ghostText.trim() || ghostLoading || draftLoading || draftSaving}
          >
            接受
          </button>
          <button
            className="btn ghost tiny"
            onClick={rejectGhostTextRef.current}
            disabled={!ghostText.trim() || ghostLoading}
          >
            拒绝
          </button>
          <button
            className="btn ghost tiny"
            onClick={() => void regenerateGhostTextRef.current()}
            disabled={ghostLoading || !activeChapterId}
          >
            重生
          </button>
          <button className="btn ghost tiny" onClick={onToggleGhostAuto} disabled={ghostLoading}>
            {ghostAutoEnabled ? "改为手动" : "改为自动"}
          </button>
        </div>
        <p className="ghost-shortcuts">快捷键：Tab 接受 · Esc 拒绝 · Alt + ] 重生</p>
      </div>
      <div className="draft-actions">
        <button className="btn primary tiny" onClick={() => void saveDraftSnapshotRef.current()} disabled={draftSaving || draftLoading}>
          {draftSaving ? "保存中..." : "保存正文"}
        </button>
        <button
          className="btn ghost tiny"
          onClick={() => void refreshDraftSnapshotRef.current(projectId, activeChapterId)}
          disabled={draftSaving || draftLoading}
        >
          拉取服务器版本
        </button>
        <button className="btn ghost tiny" onClick={() => fillPromptFromSelectionRef.current("polish")}>
          润色选中
        </button>
        <button className="btn ghost tiny" onClick={() => fillPromptFromSelectionRef.current("expand")}>
          扩写选中
        </button>
        <button className="btn primary tiny" onClick={() => applyAssistantToDraftRef.current("insert")}>
          插入助手回复
        </button>
        <button className="btn ghost tiny" onClick={() => applyAssistantToDraftRef.current("replace")}>
          替换选中为助手回复
        </button>
      </div>
      <p className="draft-hint">
        {draftLoading ? "正在加载服务器正文..." : `已选 ${selectedDraftText.length} 字`}
        {latestAssistantReply ? " · 最近一条助手回复可写回正文" : " · 暂无助手回复可写回"}
      </p>
      <div className="chapter-outline">
        <div className="panel-title sub">
          <h3>章节大纲</h3>
          <small>拖拽排序 + 点击切章</small>
        </div>
        <ChapterOutlineList
          chapterOutlines={chapterOutlines}
          activeChapterId={activeChapterId}
          dragChapterId={dragChapterId}
          disabled={draftLoading || draftSaving}
          onDragStartRef={handleOutlineDragStartRef}
          onDragEndRef={handleOutlineDragEndRef}
          onReorderRef={reorderByDragRef}
          onSelectRef={switchChapterRef}
        />
      </div>
      <DraftRevisionList
        draftRevisions={draftRevisions}
        disabled={draftSaving || draftLoading}
        rollbackDraftToVersionRef={rollbackDraftToVersionRef}
      />
    </section>
  );
});

type SettingsDialogProps = {
  onCloseSettingsDialog: () => void;
  settingsDialogRef: { current: HTMLElement | null };
  projectId: number;
  setProjectId: (projectId: number) => void;
  model: string;
  setModel: (value: string) => void;
  modelProfiles: ModelProfile[];
  selectedModelProfileId: string | null;
  setSelectedModelProfileId: (value: string | null) => void;
  modelProfileDraftIdInput: string;
  setModelProfileDraftIdInput: (value: string) => void;
  modelProfileName: string;
  setModelProfileName: (value: string) => void;
  modelProfileProvider: "openai_compatible" | "deepseek" | "claude" | "gemini";
  setModelProfileProvider: (value: "openai_compatible" | "deepseek" | "claude" | "gemini") => void;
  modelProfileBaseUrl: string;
  setModelProfileBaseUrl: (value: string) => void;
  modelProfileApiKey: string;
  setModelProfileApiKey: (value: string) => void;
  modelProfileApiKeyMasked: string | null;
  clearModelProfileApiKey: boolean;
  setClearModelProfileApiKey: (value: boolean) => void;
  modelProfileModel: string;
  setModelProfileModel: (value: string) => void;
  modelProfileSaving: boolean;
  onSaveModelProfile: () => Promise<void>;
  onDeleteModelProfile: () => Promise<void>;
  onActivateModelProfile: () => Promise<void>;
  onResetModelProfileDraft: () => void;
  chatTemperatureProfile: "action" | "chat" | "brainstorm";
  setChatTemperatureProfile: (value: "action" | "chat" | "brainstorm") => void;
  ghostTemperatureProfile: "ghost" | "chat" | "action" | "brainstorm";
  setGhostTemperatureProfile: (value: "ghost" | "chat" | "action" | "brainstorm") => void;
  temperatureOverrideInput: string;
  setTemperatureOverrideInput: (value: string) => void;
  contextWindowProfile: "balanced" | "chapter_focus" | "world_focus" | "minimal";
  setContextWindowProfile: (value: "balanced" | "chapter_focus" | "world_focus" | "minimal") => void;
  povMode: "global" | "character";
  setPovMode: (value: "global" | "character") => void;
  povAnchor: string;
  setPovAnchor: (value: string) => void;
  ragMode: "local" | "global" | "hybrid" | "mix";
  setRagMode: (value: "local" | "global" | "hybrid" | "mix") => void;
  deterministicFirst: boolean;
  setDeterministicFirst: (value: boolean) => void;
  thinkingEnabled: boolean;
  setThinkingEnabled: (value: boolean) => void;
  referenceProjectInput: string;
  setReferenceProjectInput: (value: string) => void;
  ghostAutoEnabled: boolean;
  setGhostAutoEnabled: (value: boolean) => void;
  typewriterModeEnabled: boolean;
  setTypewriterModeEnabled: (value: boolean) => void;
  writingTheme: WritingTheme;
  setWritingTheme: (value: WritingTheme) => void;
  streaming: boolean;
};

const SettingsDialog = memo(function SettingsDialog({
  onCloseSettingsDialog,
  settingsDialogRef,
  projectId,
  setProjectId,
  model,
  setModel,
  modelProfiles,
  selectedModelProfileId,
  setSelectedModelProfileId,
  modelProfileDraftIdInput,
  setModelProfileDraftIdInput,
  modelProfileName,
  setModelProfileName,
  modelProfileProvider,
  setModelProfileProvider,
  modelProfileBaseUrl,
  setModelProfileBaseUrl,
  modelProfileApiKey,
  setModelProfileApiKey,
  modelProfileApiKeyMasked,
  clearModelProfileApiKey,
  setClearModelProfileApiKey,
  modelProfileModel,
  setModelProfileModel,
  modelProfileSaving,
  onSaveModelProfile,
  onDeleteModelProfile,
  onActivateModelProfile,
  onResetModelProfileDraft,
  chatTemperatureProfile,
  setChatTemperatureProfile,
  ghostTemperatureProfile,
  setGhostTemperatureProfile,
  temperatureOverrideInput,
  setTemperatureOverrideInput,
  contextWindowProfile,
  setContextWindowProfile,
  povMode,
  setPovMode,
  povAnchor,
  setPovAnchor,
  ragMode,
  setRagMode,
  deterministicFirst,
  setDeterministicFirst,
  thinkingEnabled,
  setThinkingEnabled,
  referenceProjectInput,
  setReferenceProjectInput,
  ghostAutoEnabled,
  setGhostAutoEnabled,
  typewriterModeEnabled,
  setTypewriterModeEnabled,
  writingTheme,
  setWritingTheme,
  streaming,
}: SettingsDialogProps) {
  return (
    <div className="settings-backdrop" role="presentation" onClick={onCloseSettingsDialog}>
      <section
        id="settings-dialog"
        ref={settingsDialogRef}
        className="settings-modal panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-dialog-title"
        tabIndex={-1}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="panel-title">
          <h2 id="settings-dialog-title">写作设置</h2>
          <button
            type="button"
            className="btn ghost tiny"
            onClick={onCloseSettingsDialog}
            aria-label="关闭写作设置"
          >
            关闭
          </button>
        </div>
        <div className="settings-sections">
          <details className="settings-section" open>
            <summary>
              <strong>基础写作（推荐）</strong>
              <small>只改这组也能直接开始写作</small>
            </summary>
            <div className="settings-section-body">
              <div className="settings-grid">
                <label>
                  项目 ID
                  <input
                    data-autofocus
                    type="number"
                    value={projectId}
                    min={1}
                    onChange={(event) => setProjectId(Number(event.target.value || 1))}
                    disabled={streaming}
                  />
                </label>
                <label>
                  写作主题
                  <select value={writingTheme} onChange={(event) => setWritingTheme(event.target.value as WritingTheme)}>
                    <option value="paper">paper（纸感衬线）</option>
                    <option value="wenkai">wenkai（文楷复古）</option>
                    <option value="modern">modern（现代简洁）</option>
                    <option value="contrast">contrast（高对比）</option>
                  </select>
                </label>
                <label>
                  Ghost 触发策略
                  <select
                    value={ghostAutoEnabled ? "auto" : "manual"}
                    onChange={(event) => setGhostAutoEnabled(event.target.value === "auto")}
                  >
                    <option value="manual">手动触发（默认）</option>
                    <option value="auto">自动触发（900ms 去抖）</option>
                  </select>
                </label>
                <label>
                  打字机滚动
                  <select
                    value={typewriterModeEnabled ? "on" : "off"}
                    onChange={(event) => setTypewriterModeEnabled(event.target.value === "on")}
                  >
                    <option value="on">开启（光标行居中）</option>
                    <option value="off">关闭</option>
                  </select>
                </label>
              </div>
            </div>
          </details>

          <details className="settings-section">
            <summary>
              <strong>模型与检索</strong>
              <small>Profile、RAG、上下文策略</small>
            </summary>
            <div className="settings-section-body">
              <div className="settings-grid">
                <label>
                  模型覆盖（可空）
                  <small className="field-help">临时指定本次写作用哪个模型；留空时使用当前激活 Profile。</small>
                  <input
                    type="text"
                    placeholder="gpt-4o-mini"
                    value={model}
                    onChange={(event) => setModel(event.target.value)}
                    disabled={streaming}
                  />
                </label>
                <label>
                  模型配置中心（Profile）
                  <small className="field-help">切换你预设好的模型连接配置，适合在不同服务间快速切换。</small>
                  <select
                    value={selectedModelProfileId ?? ""}
                    onChange={(event) => setSelectedModelProfileId(event.target.value || null)}
                    disabled={streaming || modelProfileSaving}
                  >
                    <option value="">新建 profile</option>
                    {modelProfiles.map((profile) => (
                      <option key={profile.profile_id} value={profile.profile_id}>
                        {`${profile.is_active ? "★ " : ""}${profile.name || profile.profile_id} (${profile.provider})`}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Profile ID（新建可填）
                  <small className="field-help">给这套配置起一个机器可识别的代号，后续便于复用。</small>
                  <input
                    type="text"
                    placeholder="relay-main"
                    value={modelProfileDraftIdInput}
                    onChange={(event) => setModelProfileDraftIdInput(event.target.value)}
                    disabled={streaming || modelProfileSaving || Boolean(selectedModelProfileId)}
                  />
                </label>
                <label>
                  Profile 名称
                  <small className="field-help">给自己看的名称，建议写成“用途 + 场景”，便于识别。</small>
                  <input
                    type="text"
                    placeholder="主中转"
                    value={modelProfileName}
                    onChange={(event) => setModelProfileName(event.target.value)}
                    disabled={streaming || modelProfileSaving}
                  />
                </label>
                <label>
                  Provider
                  <small className="field-help">模型服务类型；不知道怎么选时，优先保持默认即可。</small>
                  <select
                    value={modelProfileProvider}
                    onChange={(event) =>
                      setModelProfileProvider(
                        event.target.value as "openai_compatible" | "deepseek" | "claude" | "gemini"
                      )
                    }
                    disabled={streaming || modelProfileSaving}
                  >
                    <option value="openai_compatible">openai_compatible（推荐中转）</option>
                    <option value="deepseek">deepseek</option>
                    <option value="claude">claude</option>
                    <option value="gemini">gemini</option>
                  </select>
                </label>
                <label>
                  Base URL
                  <small className="field-help">模型服务入口地址，通常由服务商或中转服务提供。</small>
                  <input
                    type="text"
                    placeholder="https://api.example.com/v1"
                    value={modelProfileBaseUrl}
                    onChange={(event) => setModelProfileBaseUrl(event.target.value)}
                    disabled={streaming || modelProfileSaving}
                  />
                </label>
                <label>
                  API Key
                  <small className="field-help">访问模型服务的密钥，平台仅在调用时使用，不会展示明文。</small>
                  <input
                    type="password"
                    placeholder={selectedModelProfileId && modelProfileApiKeyMasked ? modelProfileApiKeyMasked : "sk-..."}
                    value={modelProfileApiKey}
                    onChange={(event) => {
                      setClearModelProfileApiKey(false);
                      setModelProfileApiKey(event.target.value);
                    }}
                    disabled={streaming || modelProfileSaving}
                  />
                </label>
                <label>
                  模型名
                  <small className="field-help">服务端可用的模型标识，不确定时使用服务商推荐值。</small>
                  <input
                    type="text"
                    placeholder="gpt-5-mini"
                    value={modelProfileModel}
                    onChange={(event) => setModelProfileModel(event.target.value)}
                    disabled={streaming || modelProfileSaving}
                  />
                </label>
                <label>
                  Key 操作
                  <small className="field-help">更新 Profile 时，选择保留现有密钥还是清空重设。</small>
                  <select
                    value={clearModelProfileApiKey ? "clear" : "keep"}
                    onChange={(event) => setClearModelProfileApiKey(event.target.value === "clear")}
                    disabled={streaming || modelProfileSaving || !selectedModelProfileId}
                  >
                    <option value="keep">保持现有 Key</option>
                    <option value="clear">清空现有 Key</option>
                  </select>
                </label>
                <div className="action-ops">
                  <button
                    type="button"
                    className="btn ghost tiny"
                    onClick={onResetModelProfileDraft}
                    disabled={streaming || modelProfileSaving}
                  >
                    新建草稿
                  </button>
                  <button
                    type="button"
                    className="btn primary tiny"
                    onClick={() => void onSaveModelProfile()}
                    disabled={streaming || modelProfileSaving}
                  >
                    {modelProfileSaving ? "保存中..." : selectedModelProfileId ? "更新 Profile" : "创建 Profile"}
                  </button>
                  <button
                    type="button"
                    className="btn ghost tiny"
                    onClick={() => void onActivateModelProfile()}
                    disabled={streaming || modelProfileSaving || !selectedModelProfileId}
                  >
                    设为激活
                  </button>
                  <button
                    type="button"
                    className="btn ghost tiny"
                    onClick={() => void onDeleteModelProfile()}
                    disabled={streaming || modelProfileSaving || !selectedModelProfileId}
                  >
                    删除
                  </button>
                </div>
                <p className="field-help settings-actions-hint">保存后即可生效；“设为激活”会作为默认模型配置。</p>
                <label>
                  上下文滑窗策略
                  <small className="field-help">控制助手优先关注“当前章节”还是“全局世界观”。</small>
                  <select
                    value={contextWindowProfile}
                    onChange={(event) =>
                      setContextWindowProfile(
                        event.target.value as "balanced" | "chapter_focus" | "world_focus" | "minimal"
                      )
                    }
                    disabled={streaming}
                  >
                    <option value="balanced">balanced（默认均衡）</option>
                    <option value="chapter_focus">chapter_focus（章节优先）</option>
                    <option value="world_focus">world_focus（世界观优先）</option>
                    <option value="minimal">minimal（最小上下文）</option>
                  </select>
                </label>
                <label>
                  POV 模式
                  <small className="field-help">决定叙事视角是全局叙述，还是围绕某个角色展开。</small>
                  <select
                    value={povMode}
                    onChange={(event) => setPovMode(event.target.value as "global" | "character")}
                    disabled={streaming}
                  >
                    <option value="global">全局视角</option>
                    <option value="character">角色沙箱</option>
                  </select>
                </label>
                <label>
                  POV 锚点（角色名）
                  <small className="field-help">仅在角色视角时填写，用来指定“围绕谁来写”。</small>
                  <input
                    type="text"
                    placeholder={povMode === "character" ? "例如：林澈" : "全局模式可留空"}
                    value={povAnchor}
                    onChange={(event) => setPovAnchor(event.target.value)}
                    disabled={streaming}
                  />
                </label>
                <label>
                  RAG 路由
                  <small className="field-help">决定检索资料来自当前项目、全局知识，或两者混合。</small>
                  <select
                    value={ragMode}
                    onChange={(event) => setRagMode(event.target.value as "local" | "global" | "hybrid" | "mix")}
                    disabled={streaming}
                  >
                    <option value="local">local</option>
                    <option value="global">global</option>
                    <option value="hybrid">hybrid</option>
                    <option value="mix">mix</option>
                  </select>
                </label>
                <label>
                  事实短路
                  <small className="field-help">开启后优先采用已确认事实，减少跑偏但灵活度会降低。</small>
                  <select
                    value={deterministicFirst ? "on" : "off"}
                    onChange={(event) => setDeterministicFirst(event.target.value === "on")}
                    disabled={streaming}
                  >
                    <option value="off">关闭</option>
                    <option value="on">开启（DSL+GRAPH优先）</option>
                  </select>
                </label>
                <label>
                  跨项目引用（逗号分隔）
                  <small className="field-help">把其他项目当作参考资料源，适合系列文共享设定。</small>
                  <input
                    type="text"
                    placeholder="例如：2,3"
                    value={referenceProjectInput}
                    onChange={(event) => setReferenceProjectInput(event.target.value)}
                    disabled={streaming}
                  />
                </label>
              </div>
            </div>
          </details>

          <details className="settings-section">
            <summary>
              <strong>高级调优</strong>
              <small>温度与 Thinking 配置</small>
            </summary>
            <div className="settings-section-body">
              <div className="settings-grid">
                <label>
                  聊天温度策略
                  <small className="field-help">影响助手回答风格：越保守越稳定，越发散越有新意。</small>
                  <select
                    value={chatTemperatureProfile}
                    onChange={(event) => setChatTemperatureProfile(event.target.value as "action" | "chat" | "brainstorm")}
                    disabled={streaming}
                  >
                    <option value="action">action（稳健提案）</option>
                    <option value="chat">chat（常规写作）</option>
                    <option value="brainstorm">brainstorm（发散灵感）</option>
                  </select>
                </label>
                <label>
                  Ghost 温度策略
                  <small className="field-help">控制续写建议的创造程度，建议与当前写作阶段匹配。</small>
                  <select
                    value={ghostTemperatureProfile}
                    onChange={(event) =>
                      setGhostTemperatureProfile(event.target.value as "ghost" | "chat" | "action" | "brainstorm")
                    }
                    disabled={streaming}
                  >
                    <option value="ghost">ghost（续写）</option>
                    <option value="chat">chat（常规）</option>
                    <option value="action">action（保守）</option>
                    <option value="brainstorm">brainstorm（发散）</option>
                  </select>
                </label>
                <label>
                  温度覆盖（可空 0~2）
                  <small className="field-help">手动覆盖所有温度策略；留空时按上面的策略自动决定。</small>
                  <input
                    type="number"
                    min={0}
                    max={2}
                    step={0.05}
                    value={temperatureOverrideInput}
                    onChange={(event) => setTemperatureOverrideInput(event.target.value)}
                    disabled={streaming}
                  />
                </label>
                <label>
                  Thinking
                  <small className="field-help">开启后会更谨慎地组织答案，通常更稳但速度略慢。</small>
                  <select
                    value={thinkingEnabled ? "on" : "off"}
                    onChange={(event) => setThinkingEnabled(event.target.value === "on")}
                    disabled={streaming}
                  >
                    <option value="off">关闭</option>
                    <option value="on">开启（更稳健）</option>
                  </select>
                </label>
              </div>
            </div>
          </details>
        </div>
      </section>
    </div>
  );
});

type AssistantActionsPanelProps = {
  sortedActions: ChatAction[];
  pendingActionIds: number[];
  mutatingActionId: number | null;
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
  loadLogsRef: { current: (actionId: number) => Promise<void> };
  mutateActionRef: { current: (action: ChatAction, decision: "apply" | "reject" | "undo") => Promise<void> };
  runConsistencyAuditRef: { current: () => Promise<void> };
};

const AssistantActionsPanel = memo(function AssistantActionsPanel({
  sortedActions,
  pendingActionIds,
  mutatingActionId,
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
  loadLogsRef,
  mutateActionRef,
  runConsistencyAuditRef,
}: AssistantActionsPanelProps) {
  const pendingActionSet = useMemo(() => new Set(pendingActionIds), [pendingActionIds]);
  const latestAudits = useMemo(() => consistencyAudits.slice(0, 3), [consistencyAudits]);
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
  } = useTimelineGraphFlow({
    graphTimeline,
    graphTimelineChapterIndex,
    maxChapterIndex,
  });
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

  return (
    <section className="panel side-panel">
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
            onClick={() => void runConsistencyAuditRef.current()}
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
            {graphEdges.map((edge) => {
              const source = graphLayout.positions[edge.source];
              const target = graphLayout.positions[edge.target];
              if (!source || !target) return null;
              const edgeClassName = selectedTimelineNodeId
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
            })}
            {graphNodes.map((node) => {
              const point = graphLayout.positions[node.id];
              if (!point) return null;
              const isSelected = selectedTimelineNodeId === node.id;
              const isNeighbor = highlightedNodeIdSet.has(node.id);
              const nodeClassName = selectedTimelineNodeId
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
            })}
          </svg>
        ) : null}
        {selectedTimelineNodeId ? (
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
            controlsDisabled={mutatingActionId !== null}
            loadLogsRef={loadLogsRef}
            mutateActionRef={mutateActionRef}
          />
        ))}
      </div>

      <ActionLogsList selectedActionId={selectedActionId} actionLogs={actionLogs} />
    </section>
  );
});

type AssistantDrawerProps = {
  projectId: number;
  assistantDrawerOpen: boolean;
  onOpenAssistantDrawer: () => void;
  onCloseAssistantDrawer: () => void;
  onStartNewSession: () => void;
  onSwitchSession: (sessionId: number) => Promise<void>;
  onRenameSession: () => Promise<void>;
  onDeleteSession: () => Promise<void>;
  assistantDrawerRef: { current: HTMLElement | null };
  sessionId: number | null;
  projectSessions: ChatSessionSummary[];
  usage: Record<string, unknown> | null;
  messages: UiMessage[];
  input: string;
  streaming: boolean;
  composerInputRef: { current: HTMLTextAreaElement | null };
  setInput: (value: string) => void;
  handleSendRef: { current: () => Promise<void> };
  sortedActions: ChatAction[];
  pendingActionIds: number[];
  mutatingActionId: number | null;
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
  loadLogsRef: { current: (actionId: number) => Promise<void> };
  mutateActionRef: { current: (action: ChatAction, decision: "apply" | "reject" | "undo") => Promise<void> };
  runConsistencyAuditRef: { current: () => Promise<void> };
};

const AssistantDrawer = memo(function AssistantDrawer({
  projectId,
  assistantDrawerOpen,
  onOpenAssistantDrawer,
  onCloseAssistantDrawer,
  onStartNewSession,
  onSwitchSession,
  onRenameSession,
  onDeleteSession,
  assistantDrawerRef,
  sessionId,
  projectSessions,
  usage,
  messages,
  input,
  streaming,
  composerInputRef,
  setInput,
  handleSendRef,
  sortedActions,
  pendingActionIds,
  mutatingActionId,
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
  loadLogsRef,
  mutateActionRef,
  runConsistencyAuditRef,
}: AssistantDrawerProps) {
  const [sideTab, setSideTab] = useState<"actions" | "candidates">("actions");

  useEffect(() => {
    if (!assistantDrawerOpen) {
      setSideTab("actions");
    }
  }, [assistantDrawerOpen]);

  const handleSessionChange = (value: string) => {
    const raw = value.trim();
    if (!raw) {
      onStartNewSession();
      return;
    }
    const nextSessionId = Number(raw);
    if (!Number.isFinite(nextSessionId) || nextSessionId <= 0) return;
    if (sessionId === nextSessionId) return;
    void onSwitchSession(nextSessionId);
  };

  return (
    <>
      <button
        type="button"
        className="assistant-fab"
        onClick={onOpenAssistantDrawer}
        aria-haspopup="dialog"
        aria-expanded={assistantDrawerOpen}
        aria-controls="assistant-drawer"
      >
        助手
      </button>
      <div
        className={`assistant-drawer-backdrop ${assistantDrawerOpen ? "open" : ""}`}
        role="presentation"
        aria-hidden={!assistantDrawerOpen}
        onClick={onCloseAssistantDrawer}
      />
      <aside
        id="assistant-drawer"
        ref={assistantDrawerRef}
        className={`assistant-drawer ${assistantDrawerOpen ? "open" : ""}`}
        role="dialog"
        aria-modal={assistantDrawerOpen ? true : undefined}
        aria-labelledby="assistant-drawer-title"
        aria-describedby="assistant-drawer-desc"
        aria-hidden={!assistantDrawerOpen}
        tabIndex={-1}
      >
        <div className="assistant-drawer-top">
          <div>
            <h2 id="assistant-drawer-title">助手抽屉</h2>
            <small id="assistant-drawer-desc">默认折叠，按 `Ctrl/Cmd + Shift + A` 快速呼出</small>
            <div className="assistant-session-switch">
              <label htmlFor="assistant-session-select">会话</label>
              <select
                id="assistant-session-select"
                value={sessionId ?? ""}
                onChange={(event) => handleSessionChange(event.target.value)}
                disabled={streaming}
              >
                <option value="">新会话（未创建）</option>
                {projectSessions.map((item) => (
                  <option key={item.id} value={item.id}>
                    {`${item.title || `会话 #${item.id}`} · ${formatDateTime(item.updated_at)}`}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="assistant-drawer-top-actions">
            <button
              type="button"
              className="btn ghost tiny"
              onClick={onStartNewSession}
              disabled={streaming}
            >
              新会话
            </button>
            {sessionId ? (
              <>
                <button
                  type="button"
                  className="btn ghost tiny"
                  onClick={() => void onRenameSession()}
                  disabled={streaming}
                >
                  重命名
                </button>
                <button
                  type="button"
                  className="btn ghost tiny"
                  onClick={() => void onDeleteSession()}
                  disabled={streaming}
                >
                  删除会话
                </button>
              </>
            ) : null}
            <button
              type="button"
              className="btn ghost tiny"
              onClick={onCloseAssistantDrawer}
              aria-label="关闭助手抽屉"
            >
              关闭
            </button>
          </div>
        </div>

        <div className="assistant-drawer-grid">
          <AssistantChatPanel
            usage={usage}
            messages={messages}
            input={input}
            streaming={streaming}
            composerInputRef={composerInputRef}
            setInput={setInput}
            handleSendRef={handleSendRef}
          />

          <div className="assistant-tools-column">
            <div className="assistant-tools-tabs">
              <button
                type="button"
                className={`btn ghost tiny ${sideTab === "actions" ? "active" : ""}`}
                onClick={() => setSideTab("actions")}
              >
                动作提议
              </button>
              <button
                type="button"
                className={`btn ghost tiny ${sideTab === "candidates" ? "active" : ""}`}
                onClick={() => setSideTab("candidates")}
              >
                候选审核
              </button>
            </div>

            {sideTab === "actions" ? (
              <AssistantActionsPanel
                sortedActions={sortedActions}
                pendingActionIds={pendingActionIds}
                mutatingActionId={mutatingActionId}
                consistencyAudits={consistencyAudits}
                consistencyAuditRunning={consistencyAuditRunning}
                traceEvents={traceEvents}
                graphTimeline={graphTimeline}
                graphTimelineLoading={graphTimelineLoading}
                graphTimelineChapterIndex={graphTimelineChapterIndex}
                maxChapterIndex={maxChapterIndex}
                setGraphTimelineChapterIndex={setGraphTimelineChapterIndex}
                selectedActionId={selectedActionId}
                actionLogs={actionLogs}
                loadLogsRef={loadLogsRef}
                mutateActionRef={mutateActionRef}
                runConsistencyAuditRef={runConsistencyAuditRef}
              />
            ) : (
              <Suspense
                fallback={
                  <LazyPanelFallback
                    className="panel side-panel candidate-review-panel"
                    title="候选列表查询 / 批量审核"
                    detail="正在加载候选审核面板..."
                  />
                }
              >
                <LazyGraphCandidateReviewPanel projectId={projectId} />
              </Suspense>
            )}
          </div>
        </div>
      </aside>
    </>
  );
});

export default function App() {
  const {
    uiMode,
    projectId,
    model,
    povMode,
    povAnchor,
    ragMode,
    deterministicFirst,
    thinkingEnabled,
    sessionId,
    streaming,
    error,
    usage,
    messages,
    actions,
    pendingActionIds,
    settings,
    cards,
    selectedActionId,
    actionLogs,
    evidence,
  } = useChatStore(
    useShallow((state) => ({
      uiMode: state.uiMode,
      projectId: state.projectId,
      model: state.model,
      povMode: state.povMode,
      povAnchor: state.povAnchor,
      ragMode: state.ragMode,
      deterministicFirst: state.deterministicFirst,
      thinkingEnabled: state.thinkingEnabled,
      sessionId: state.sessionId,
      streaming: state.streaming,
      error: state.error,
      usage: state.usage,
      messages: state.messages,
      actions: state.actions,
      pendingActionIds: state.pendingActionIds,
      settings: state.settings,
      cards: state.cards,
      selectedActionId: state.selectedActionId,
      actionLogs: state.actionLogs,
      evidence: state.evidence,
    }))
  );

  const {
    setUiMode,
    setProjectId,
    setModel,
    setPovMode,
    setPovAnchor,
    setRagMode,
    setDeterministicFirst,
    setThinkingEnabled,
    setSessionId,
    setStreaming,
    setError,
    setUsage,
    setMessages,
    appendMessage,
    appendMessageDelta,
    updateMessage,
    setActions,
    setPendingActionIds,
    setSettings,
    setCards,
    setSelectedActionId,
    setActionLogs,
    setEvidence,
    resetSessionState,
  } = useChatStore(
    useShallow((state) => ({
      setUiMode: state.setUiMode,
      setProjectId: state.setProjectId,
      setModel: state.setModel,
      setPovMode: state.setPovMode,
      setPovAnchor: state.setPovAnchor,
      setRagMode: state.setRagMode,
      setDeterministicFirst: state.setDeterministicFirst,
      setThinkingEnabled: state.setThinkingEnabled,
      setSessionId: state.setSessionId,
      setStreaming: state.setStreaming,
      setError: state.setError,
      setUsage: state.setUsage,
      setMessages: state.setMessages,
      appendMessage: state.appendMessage,
      appendMessageDelta: state.appendMessageDelta,
      updateMessage: state.updateMessage,
      setActions: state.setActions,
      setPendingActionIds: state.setPendingActionIds,
      setSettings: state.setSettings,
      setCards: state.setCards,
      setSelectedActionId: state.setSelectedActionId,
      setActionLogs: state.setActionLogs,
      setEvidence: state.setEvidence,
      resetSessionState: state.resetSessionState,
    }))
  );

  const [input, setInput] = useState("");
  const [projectSessions, setProjectSessions] = useState<ChatSessionSummary[]>([]);
  const [consistencyAudits, setConsistencyAudits] = useState<ConsistencyAuditReport[]>([]);
  const [consistencyAuditRunning, setConsistencyAuditRunning] = useState(false);
  const [traceEvents, setTraceEvents] = useState<ChatStreamTraceEvent[]>([]);
  const [graphTimeline, setGraphTimeline] = useState<GraphTimelineSnapshot | null>(null);
  const [graphTimelineLoading, setGraphTimelineLoading] = useState(false);
  const [graphTimelineChapterIndex, setGraphTimelineChapterIndex] = useState(0);
  const [draftText, setDraftText] = useState("");
  const [selectedDraftText, setSelectedDraftText] = useState("");
  const [mutatingActionId, setMutatingActionId] = useState<number | null>(null);
  const [chapters, setChapters] = useState<ProjectChapter[]>([]);
  const [activeChapterId, setActiveChapterId] = useState<number | null>(null);
  const [volumes, setVolumes] = useState<ProjectVolume[]>([]);
  const [activeVolumeId, setActiveVolumeId] = useState<number | null>(null);
  const [volumeOutlineDraft, setVolumeOutlineDraft] = useState("");
  const [sceneBeats, setSceneBeats] = useState<SceneBeat[]>([]);
  const [activeSceneBeatId, setActiveSceneBeatId] = useState<number | null>(null);
  const [newBeatContent, setNewBeatContent] = useState("");
  const [foreshadowCards, setForeshadowCards] = useState<ForeshadowingCard[]>([]);
  const [overdueForeshadowCards, setOverdueForeshadowCards] = useState<ForeshadowingCard[]>([]);
  const [foreshadowDraftTitle, setForeshadowDraftTitle] = useState("");
  const [foreshadowDraftDescription, setForeshadowDraftDescription] = useState("");
  const [planningBusy, setPlanningBusy] = useState(false);
  const [draftTitle, setDraftTitle] = useState("第1章");
  const [draftVersion, setDraftVersion] = useState(0);
  const [draftUpdatedAt, setDraftUpdatedAt] = useState<string | null>(null);
  const [draftRevisions, setDraftRevisions] = useState<ProjectChapterRevision[]>([]);
  const [draftLoading, setDraftLoading] = useState(false);
  const [draftSaving, setDraftSaving] = useState(false);
  const [dragChapterId, setDragChapterId] = useState<number | null>(null);
  const [draftFocusMode, setDraftFocusMode] = useState(false);
  const [typewriterModeEnabled, setTypewriterModeEnabled] = useState(true);
  const [writingTheme, setWritingTheme] = useState<WritingTheme>("paper");
  const [promptTemplates, setPromptTemplates] = useState<PromptTemplate[]>([]);
  const [activePromptTemplateId, setActivePromptTemplateId] = useState<number | null>(null);
  const [templateDraftId, setTemplateDraftId] = useState<number | null>(null);
  const [templateName, setTemplateName] = useState("默认模板");
  const [templateSystemPrompt, setTemplateSystemPrompt] = useState("");
  const [templateUserPromptPrefix, setTemplateUserPromptPrefix] = useState("");
  const [templateKnowledgeSettingKeys, setTemplateKnowledgeSettingKeys] = useState<string[]>([]);
  const [templateKnowledgeCardIds, setTemplateKnowledgeCardIds] = useState<number[]>([]);
  const [templateSaving, setTemplateSaving] = useState(false);
  const [templateRevisions, setTemplateRevisions] = useState<PromptTemplateRevision[]>([]);
  const [templateRevisionsLoading, setTemplateRevisionsLoading] = useState(false);
  const [modelProfiles, setModelProfiles] = useState<ModelProfile[]>([]);
  const [selectedModelProfileId, setSelectedModelProfileId] = useState<string | null>(null);
  const [activeModelProfileId, setActiveModelProfileId] = useState<string | null>(null);
  const [modelProfileDraftIdInput, setModelProfileDraftIdInput] = useState("");
  const [modelProfileName, setModelProfileName] = useState("");
  const [modelProfileProvider, setModelProfileProvider] = useState<
    "openai_compatible" | "deepseek" | "claude" | "gemini"
  >("openai_compatible");
  const [modelProfileBaseUrl, setModelProfileBaseUrl] = useState("");
  const [modelProfileApiKey, setModelProfileApiKey] = useState("");
  const [modelProfileApiKeyMasked, setModelProfileApiKeyMasked] = useState<string | null>(null);
  const [clearModelProfileApiKey, setClearModelProfileApiKey] = useState(false);
  const [modelProfileModel, setModelProfileModel] = useState("");
  const [modelProfileSaving, setModelProfileSaving] = useState(false);
  const [referenceProjectInput, setReferenceProjectInput] = useState("");
  const [ghostText, setGhostText] = useState("");
  const [ghostLoading, setGhostLoading] = useState(false);
  const [ghostError, setGhostError] = useState<string | null>(null);
  const [ghostAutoEnabled, setGhostAutoEnabled] = useState(false);
  const [chatTemperatureProfile, setChatTemperatureProfile] = useState<"action" | "chat" | "brainstorm">("action");
  const [ghostTemperatureProfile, setGhostTemperatureProfile] = useState<
    "ghost" | "chat" | "action" | "brainstorm"
  >("ghost");
  const [temperatureOverrideInput, setTemperatureOverrideInput] = useState("");
  const [contextWindowProfile, setContextWindowProfile] = useState<
    "balanced" | "chapter_focus" | "world_focus" | "minimal"
  >("balanced");
  const [workbenchPanelVisibility, setWorkbenchPanelVisibility] = useState<WorkbenchPanelVisibility>({
    prompt: false,
    planning: true,
    snapshot: false,
  });
  const [zenMode, setZenMode] = useState(false);
  const [assistantDrawerOpen, setAssistantDrawerOpen] = useState(false);
  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false);
  const [lastStreamMetrics, setLastStreamMetrics] = useState<ChatStreamTimingMetrics | null>(null);
  const [autoSaveState, setAutoSaveState] = useState<DraftAutoSaveState>("idle");
  const [autoSaveAt, setAutoSaveAt] = useState<string | null>(null);
  const [localRecoveryNotice, setLocalRecoveryNotice] = useState<string | null>(null);
  const autoSaveTimerRef = useRef<number | null>(null);
  const localRecoveryTimerRef = useRef<number | null>(null);
  const composerInputRef = useRef<HTMLTextAreaElement | null>(null);
  const draftEditorRef = useRef<HTMLDivElement | null>(null);
  const assistantDrawerRef = useRef<HTMLElement | null>(null);
  const settingsDialogRef = useRef<HTMLElement | null>(null);
  const assistantDrawerReturnFocusRef = useRef<HTMLElement | null>(null);
  const settingsDialogReturnFocusRef = useRef<HTMLElement | null>(null);
  const previousAssistantDrawerOpenRef = useRef(false);
  const previousSettingsDialogOpenRef = useRef(false);
  const lastSavedDraftRef = useRef<{
    chapterId: number | null;
    volumeId: number | null;
    title: string;
    content: string;
  }>({
    chapterId: null,
    volumeId: null,
    title: "第1章",
    content: "",
  });
  const ghostCacheRef = useRef<Map<string, string>>(new Map());
  const ghostRequestRef = useRef(0);
  const typewriterRafRef = useRef<number | null>(null);
  const typewriterModeEnabledRef = useRef(typewriterModeEnabled);
  const typewriterDimmingEnabledRef = useRef(false);
  const activeTypewriterParagraphRef = useRef<HTMLParagraphElement | null>(null);
  const writingShortcutStateRef = useRef<{
    uiMode: "writing" | "pro";
    assistantDrawerOpen: boolean;
    settingsDialogOpen: boolean;
    hasGhostText: boolean;
    ghostLoading: boolean;
    draftLoading: boolean;
    draftSaving: boolean;
    activeChapterId: number | null;
  }>({
    uiMode: "writing",
    assistantDrawerOpen: false,
    settingsDialogOpen: false,
    hasGhostText: false,
    ghostLoading: false,
    draftLoading: false,
    draftSaving: false,
    activeChapterId: null,
  });
  const acceptGhostTextRef = useRef<() => void>(() => undefined);
  const rejectGhostTextRef = useRef<() => void>(() => undefined);
  const regenerateGhostTextRef = useRef<() => Promise<void>>(async () => undefined);
  const handleSendRef = useRef<() => Promise<void>>(async () => undefined);
  const loadLogsRef = useRef<(actionId: number) => Promise<void>>(async () => undefined);
  const mutateActionRef = useRef<
    (action: ChatAction, decision: "apply" | "reject" | "undo") => Promise<void>
  >(async () => undefined);
  const switchChapterRef = useRef<(chapterId: number) => Promise<void>>(async () => undefined);
  const reorderByDragRef = useRef<(targetChapterId: number) => Promise<void>>(async () => undefined);
  const handleOutlineDragStartRef = useRef<(chapterId: number) => void>(() => undefined);
  const handleOutlineDragEndRef = useRef<() => void>(() => undefined);
  const rollbackDraftToVersionRef = useRef<(targetVersion: number) => Promise<void>>(async () => undefined);
  const createChapterAndSwitchRef = useRef<() => Promise<void>>(async () => undefined);
  const moveActiveChapterRef = useRef<(direction: "up" | "down") => Promise<void>>(async () => undefined);
  const deleteActiveChapterRef = useRef<() => Promise<void>>(async () => undefined);
  const saveDraftSnapshotRef = useRef<() => Promise<void>>(async () => undefined);
  const refreshDraftSnapshotRef = useRef<
    (nextProjectId: number, preferredChapterId?: number | null) => Promise<void>
  >(async () => undefined);
  const fillPromptFromSelectionRef = useRef<(mode: "polish" | "expand") => void>(() => undefined);
  const applyAssistantToDraftRef = useRef<(mode: "insert" | "replace") => void>(() => undefined);
  const runConsistencyAuditRef = useRef<() => Promise<void>>(async () => undefined);
  const refreshGraphTimelineRef = useRef<(chapterIndex: number) => Promise<void>>(async () => undefined);
  const actionLogsCacheRef = useRef<Map<number, ActionAuditLog[]>>(new Map());
  const actionLogsInFlightRef = useRef<Map<number, Promise<ActionAuditLog[]>>>(new Map());
  const actionLogsRequestSeqRef = useRef(0);
  const chapterSnapshotInFlightRef = useRef<Map<string, Promise<ChapterSnapshotData>>>(new Map());
  const chapterSnapshotRequestSeqRef = useRef(0);
  const draftSnapshotInFlightRef = useRef<Map<string, Promise<DraftSnapshotData>>>(new Map());
  const draftSnapshotRequestSeqRef = useRef(0);
  const planningSnapshotInFlightRef = useRef<Map<string, Promise<PlanningSnapshotData>>>(new Map());
  const planningSnapshotRequestSeqRef = useRef(0);
  const projectSnapshotInFlightRef = useRef<Map<string, Promise<ProjectSnapshotData>>>(new Map());
  const projectSnapshotRequestSeqRef = useRef(0);
  const fullSessionSnapshotInFlightRef = useRef<Map<string, Promise<FullSessionSnapshotData>>>(new Map());
  const fullSessionSnapshotRequestSeqRef = useRef(0);
  const templateRevisionsInFlightRef = useRef<Map<string, Promise<PromptTemplateRevision[]>>>(new Map());
  const templateRevisionsRequestSeqRef = useRef(0);
  const postChatSnapshotCacheRef = useRef<Map<string, { at: number; data: PostChatSnapshotData }>>(new Map());
  const postChatSnapshotInFlightRef = useRef<Map<string, Promise<PostChatSnapshotData>>>(new Map());
  const postChatSnapshotRequestSeqRef = useRef(0);
  const graphTimelineRequestSeqRef = useRef(0);

  const scheduleTypewriterScroll = (currentEditor: Editor) => {
    if (!typewriterModeEnabledRef.current) return;
    if (typeof window === "undefined") return;
    if (typewriterRafRef.current) {
      window.cancelAnimationFrame(typewriterRafRef.current);
      typewriterRafRef.current = null;
    }

    typewriterRafRef.current = window.requestAnimationFrame(() => {
      typewriterRafRef.current = null;
      const proseMirror = draftEditorRef.current?.querySelector<HTMLElement>(".ProseMirror");
      if (!proseMirror) return;
      if (proseMirror.scrollHeight <= proseMirror.clientHeight + 8) return;

      const pos = currentEditor.state.selection.$head.pos;
      let coords: { top: number; bottom: number };
      try {
        coords = currentEditor.view.coordsAtPos(pos);
      } catch {
        return;
      }
      const containerRect = proseMirror.getBoundingClientRect();
      if (containerRect.height <= 0) return;
      const cursorCenter = (coords.top + coords.bottom) / 2;
      const expectedCenter = containerRect.top + containerRect.height / 2;
      const delta = cursorCenter - expectedCenter;
      if (Math.abs(delta) < 6) return;
      proseMirror.scrollTop += delta;
    });
  };

  const clearTypewriterParagraphFocus = () => {
    const proseMirror = draftEditorRef.current?.querySelector<HTMLElement>(".ProseMirror");
    if (proseMirror) {
      proseMirror.classList.remove("typewriter-dimming-active");
    }
    if (activeTypewriterParagraphRef.current) {
      activeTypewriterParagraphRef.current.classList.remove("is-typewriter-active-paragraph");
      activeTypewriterParagraphRef.current = null;
    }
  };

  const syncTypewriterParagraphFocus = (currentEditor: Editor) => {
    const proseMirror = draftEditorRef.current?.querySelector<HTMLElement>(".ProseMirror");
    if (!proseMirror) return;
    if (!typewriterDimmingEnabledRef.current) {
      clearTypewriterParagraphFocus();
      return;
    }

    proseMirror.classList.add("typewriter-dimming-active");
    const domAtSelection = currentEditor.view.domAtPos(currentEditor.state.selection.$head.pos);
    const baseNode = domAtSelection.node instanceof HTMLElement ? domAtSelection.node : domAtSelection.node.parentElement;
    const nextParagraph = baseNode?.closest("p") as HTMLParagraphElement | null;

    if (activeTypewriterParagraphRef.current && activeTypewriterParagraphRef.current !== nextParagraph) {
      activeTypewriterParagraphRef.current.classList.remove("is-typewriter-active-paragraph");
    }

    if (nextParagraph && proseMirror.contains(nextParagraph)) {
      nextParagraph.classList.add("is-typewriter-active-paragraph");
      activeTypewriterParagraphRef.current = nextParagraph;
      return;
    }

    activeTypewriterParagraphRef.current = null;
  };

  const sortedActions = useMemo(
    () => [...actions].sort((a, b) => b.id - a.id),
    [actions]
  );
  const activeChapter = useMemo(
    () => chapters.find((item) => item.id === activeChapterId) ?? null,
    [chapters, activeChapterId]
  );
  const maxChapterIndex = useMemo(
    () => Math.max(1, ...chapters.map((item) => Number(item.chapter_index || 0))),
    [chapters]
  );
  const activeChapterIndex = useMemo(
    () => Number(activeChapter?.chapter_index || 0),
    [activeChapter]
  );
  const activeVolume = useMemo(
    () => volumes.find((item) => item.id === activeVolumeId) ?? null,
    [volumes, activeVolumeId]
  );
  const activeChapterPos = useMemo(
    () => chapters.findIndex((item) => item.id === activeChapterId),
    [chapters, activeChapterId]
  );
  const canMoveChapterUp = activeChapterPos > 0;
  const canMoveChapterDown = activeChapterPos >= 0 && activeChapterPos < chapters.length - 1;
  const chapterOutlines = useMemo<ChapterOutlineEntry[]>(
    () =>
      chapters.map((chapter) => {
        const cleaned = (chapter.content || "").replace(/\s+/g, "");
        const wordCount = cleaned.length;
        const previewLine = (chapter.content || "")
          .split("\n")
          .map((line) => line.trim())
          .find((line) => line.length > 0);
        const preview = previewLine ? previewLine.slice(0, 42) : "（暂无正文）";
        return {
          id: chapter.id,
          chapterIndex: chapter.chapter_index,
          title: chapter.title,
          wordCount,
          preview,
        };
      }),
    [chapters]
  );

  useEffect(() => {
    if (activeChapter?.volume_id && volumes.some((item) => item.id === activeChapter.volume_id)) {
      setActiveVolumeId(activeChapter.volume_id);
      return;
    }
    if (volumes.length > 0) {
      setActiveVolumeId(volumes[0].id);
      return;
    }
    setActiveVolumeId(null);
  }, [activeChapter?.volume_id, volumes]);

  useEffect(() => {
    setVolumeOutlineDraft(activeVolume?.outline ?? "");
  }, [activeVolume?.id, activeVolume?.outline]);

  const latestAssistantReply = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const message = messages[i];
      if (message.role === "assistant" && !message.streaming) {
        const content = message.content.trim();
        if (content) {
          return content;
        }
      }
    }
    return "";
  }, [messages]);

  const draftWordCount = useMemo(() => {
    return draftText.replace(/\s+/g, "").length;
  }, [draftText]);

  const activePromptTemplate = useMemo(
    () => promptTemplates.find((item) => item.id === activePromptTemplateId) ?? null,
    [promptTemplates, activePromptTemplateId]
  );
  const debugPromptPanelReady = uiMode === "pro";
  const proSnapshotPanelReady = uiMode === "pro";
  const referenceProjectIds = useMemo(
    () => parseReferenceProjectIds(referenceProjectInput, projectId),
    [referenceProjectInput, projectId]
  );
  const selectedKnowledgeSettings = useMemo(
    () => (debugPromptPanelReady ? settings.filter((item) => templateKnowledgeSettingKeys.includes(item.key)) : []),
    [debugPromptPanelReady, settings, templateKnowledgeSettingKeys]
  );
  const selectedKnowledgeCards = useMemo(
    () => (debugPromptPanelReady ? cards.filter((item) => templateKnowledgeCardIds.includes(item.id)) : []),
    [debugPromptPanelReady, cards, templateKnowledgeCardIds]
  );
  const missingSettingKeys = useMemo(
    () =>
      debugPromptPanelReady
        ? templateKnowledgeSettingKeys.filter((key) => !settings.some((item) => item.key === key))
        : [],
    [debugPromptPanelReady, templateKnowledgeSettingKeys, settings]
  );
  const missingCardIds = useMemo(
    () =>
      debugPromptPanelReady
        ? templateKnowledgeCardIds.filter((id) => !cards.some((item) => item.id === id))
        : [],
    [debugPromptPanelReady, templateKnowledgeCardIds, cards]
  );
  const entityHighlightHints = useMemo(
    () => collectEntityHighlightHints(settings, cards),
    [settings, cards]
  );
  const estimatedPromptChars = useMemo(() => {
    if (!debugPromptPanelReady) return 0;
    const settingsChars = selectedKnowledgeSettings.reduce(
      (acc, item) => acc + JSON.stringify(item.value ?? {}).length,
      0
    );
    const cardsChars = selectedKnowledgeCards.reduce(
      (acc, item) => acc + JSON.stringify(item.content ?? {}).length,
      0
    );
    return templateSystemPrompt.length + templateUserPromptPrefix.length + settingsChars + cardsChars;
  }, [debugPromptPanelReady, selectedKnowledgeCards, selectedKnowledgeSettings, templateSystemPrompt, templateUserPromptPrefix]);
  const degradedReasons = useMemo(() => {
    const gate = evidence?.policy?.quality_gate;
    if (!gate || typeof gate !== "object" || !Array.isArray(gate.degrade_reasons)) {
      return [];
    }
    return gate.degrade_reasons
      .map((item) => String(item || "").trim())
      .filter((item) => item.length > 0);
  }, [evidence]);
  const retrievalDegraded = useMemo(() => {
    const gate = evidence?.policy?.quality_gate;
    if (!gate || typeof gate !== "object") return false;
    return Boolean(gate.degraded);
  }, [evidence]);
  const temperatureOverride = useMemo(() => {
    const raw = temperatureOverrideInput.trim();
    if (!raw) return null;
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return null;
    return Math.min(2, Math.max(0, parsed));
  }, [temperatureOverrideInput]);
  const awarenessTags = useMemo(
    () => collectAwarenessTags(evidence, { includeDebugSignals: uiMode === "pro" }),
    [evidence, uiMode]
  );
  const ghostTextShortcutExtension = useMemo(
    () =>
      Extension.create({
        name: "ghostTextShortcuts",
        priority: 1000,
        addKeyboardShortcuts() {
          const tryAcceptGhost = () => {
            const shortcutState = writingShortcutStateRef.current;
            if (shortcutState.uiMode !== "writing") return false;
            if (shortcutState.assistantDrawerOpen || shortcutState.settingsDialogOpen) return false;
            if (
              !shortcutState.hasGhostText ||
              shortcutState.ghostLoading ||
              shortcutState.draftLoading ||
              shortcutState.draftSaving
            ) {
              return false;
            }
            acceptGhostTextRef.current();
            return true;
          };
          return {
            Tab: () => tryAcceptGhost(),
            "Shift-Tab": () => tryAcceptGhost(),
            Escape: () => {
              const shortcutState = writingShortcutStateRef.current;
              if (shortcutState.uiMode !== "writing") return false;
              if (shortcutState.assistantDrawerOpen || shortcutState.settingsDialogOpen) return false;
              if (!shortcutState.hasGhostText || shortcutState.ghostLoading) return false;
              rejectGhostTextRef.current();
              return true;
            },
            "Alt-]": () => {
              const shortcutState = writingShortcutStateRef.current;
              if (shortcutState.uiMode !== "writing") return false;
              if (shortcutState.assistantDrawerOpen || shortcutState.settingsDialogOpen) return false;
              if (
                shortcutState.ghostLoading ||
                shortcutState.draftLoading ||
                shortcutState.draftSaving ||
                !shortcutState.activeChapterId
              ) {
                return false;
              }
              void regenerateGhostTextRef.current();
              return true;
            },
          };
        },
      }),
    []
  );

  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        heading: false,
        blockquote: false,
        bulletList: false,
        orderedList: false,
        listItem: false,
        codeBlock: false,
        horizontalRule: false,
      }),
      Placeholder.configure({
        placeholder:
          "在这里写正文。你可以选中一段文字后让 AI 润色/扩写，也可以把助手回复一键写回正文。",
      }),
      ghostTextShortcutExtension,
      EntityInlineHintExtension,
    ],
    content: toEditorDoc(""),
    editable: false,
    onUpdate: ({ editor: currentEditor }) => {
      setDraftText(readEditorText(currentEditor));
      setSelectedDraftText(readSelectedText(currentEditor));
      scheduleTypewriterScroll(currentEditor);
      syncTypewriterParagraphFocus(currentEditor);
    },
    onSelectionUpdate: ({ editor: currentEditor }) => {
      setSelectedDraftText(readSelectedText(currentEditor));
      scheduleTypewriterScroll(currentEditor);
      syncTypewriterParagraphFocus(currentEditor);
    },
    onBlur: ({ editor: currentEditor }) => {
      setSelectedDraftText(readSelectedText(currentEditor));
    },
  });

  const resetTemplateDraft = () => {
    setTemplateDraftId(null);
    setTemplateName("默认模板");
    setTemplateSystemPrompt("");
    setTemplateUserPromptPrefix("");
    setTemplateKnowledgeSettingKeys([]);
    setTemplateKnowledgeCardIds([]);
  };

  const loadTemplateIntoDraft = (template: PromptTemplate | null) => {
    if (!template) {
      resetTemplateDraft();
      return;
    }
    setTemplateDraftId(template.id);
    setTemplateName(template.name);
    setTemplateSystemPrompt(template.system_prompt);
    setTemplateUserPromptPrefix(template.user_prompt_prefix);
    setTemplateKnowledgeSettingKeys(template.knowledge_setting_keys ?? []);
    setTemplateKnowledgeCardIds(template.knowledge_card_ids ?? []);
  };

  const resetModelProfileDraft = () => {
    setSelectedModelProfileId(null);
    setModelProfileDraftIdInput("");
    setModelProfileName("");
    setModelProfileProvider("openai_compatible");
    setModelProfileBaseUrl("");
    setModelProfileApiKey("");
    setModelProfileApiKeyMasked(null);
    setClearModelProfileApiKey(false);
    setModelProfileModel("");
  };

  const loadModelProfileIntoDraft = (profile: ModelProfile | null) => {
    if (!profile) {
      resetModelProfileDraft();
      return;
    }
    setSelectedModelProfileId(profile.profile_id);
    setModelProfileDraftIdInput(profile.profile_id);
    setModelProfileName((profile.name || "").trim());
    if (profile.provider === "deepseek" || profile.provider === "claude" || profile.provider === "gemini") {
      setModelProfileProvider(profile.provider);
    } else {
      setModelProfileProvider("openai_compatible");
    }
    setModelProfileBaseUrl((profile.base_url || "").trim());
    setModelProfileApiKey("");
    setModelProfileApiKeyMasked((profile.api_key_masked || "").trim() || null);
    setClearModelProfileApiKey(false);
    setModelProfileModel((profile.model || "").trim());
  };

  const loadChapterSnapshot = async (nextProjectId: number, chapterId: number) => {
    const requestSeq = chapterSnapshotRequestSeqRef.current + 1;
    chapterSnapshotRequestSeqRef.current = requestSeq;
    const cacheKey = `${nextProjectId}:${chapterId}`;

    let snapshotPromise = chapterSnapshotInFlightRef.current.get(cacheKey);
    if (!snapshotPromise) {
      snapshotPromise = (async (): Promise<ChapterSnapshotData> => {
        const [chapter, revisions, beats, overdueForeshadows] = await Promise.all([
          getProjectChapter(nextProjectId, chapterId),
          getProjectChapterRevisions(nextProjectId, chapterId, 20),
          getSceneBeats(nextProjectId, chapterId).catch(() => [] as SceneBeat[]),
          getForeshadowingCards(nextProjectId, {
            overdue_for_chapter_id: chapterId,
            chapter_gap: 50,
          }).catch(() => [] as ForeshadowingCard[]),
        ]);
        return { chapter, revisions, beats, overdueForeshadows };
      })();
      chapterSnapshotInFlightRef.current.set(cacheKey, snapshotPromise);
    }

    try {
      const snapshotData = await snapshotPromise;
      if (chapterSnapshotRequestSeqRef.current !== requestSeq) {
        return;
      }
      const { chapter, revisions, beats, overdueForeshadows } = snapshotData;
      const localSnapshot = readDraftRecoverySnapshot(nextProjectId, chapter.id);
      const shouldRestore = localSnapshot ? shouldRestoreDraftRecovery(localSnapshot, chapter) : false;
      const resolvedTitle = shouldRestore && localSnapshot ? localSnapshot.title : chapter.title;
      const resolvedContent = shouldRestore && localSnapshot ? localSnapshot.content : chapter.content;

      setDraftTitle(resolvedTitle);
      setDraftText(resolvedContent);
      setDraftVersion(chapter.version);
      setDraftUpdatedAt(chapter.updated_at);
      setDraftRevisions(revisions);
      setSceneBeats(beats);
      const pendingBeat = beats.find((item) => item.status !== "done");
      setActiveSceneBeatId(pendingBeat?.id ?? beats[0]?.id ?? null);
      setOverdueForeshadowCards(overdueForeshadows);
      setSelectedDraftText("");
      lastSavedDraftRef.current = {
        chapterId: chapter.id,
        volumeId: chapter.volume_id ?? null,
        title: chapter.title,
        content: chapter.content,
      };
      if (shouldRestore && localSnapshot) {
        setAutoSaveState("pending");
        setAutoSaveAt(localSnapshot.saved_at);
        setLocalRecoveryNotice(`已恢复本地快照（${formatDateTime(localSnapshot.saved_at)}），请继续写作或手动保存。`);
      } else {
        setAutoSaveState("idle");
        setLocalRecoveryNotice(null);
        if (localSnapshot) {
          clearDraftRecoverySnapshot(nextProjectId, chapter.id);
        }
      }
      setGhostText("");
      setGhostError(null);
    } finally {
      if (chapterSnapshotInFlightRef.current.get(cacheKey) === snapshotPromise) {
        chapterSnapshotInFlightRef.current.delete(cacheKey);
      }
    }
  };

  const refreshDraftSnapshot = async (nextProjectId: number, preferredChapterId?: number | null) => {
    const requestSeq = draftSnapshotRequestSeqRef.current + 1;
    draftSnapshotRequestSeqRef.current = requestSeq;
    const cacheKey = `${nextProjectId}`;

    let snapshotPromise = draftSnapshotInFlightRef.current.get(cacheKey);
    if (!snapshotPromise) {
      snapshotPromise = (async (): Promise<DraftSnapshotData> => {
        const [chapterList, volumeList, foreshadowList] = await Promise.all([
          getProjectChapters(nextProjectId),
          getProjectVolumes(nextProjectId).catch(() => [] as ProjectVolume[]),
          getForeshadowingCards(nextProjectId).catch(() => [] as ForeshadowingCard[]),
        ]);
        return { chapterList, volumeList, foreshadowList };
      })();
      draftSnapshotInFlightRef.current.set(cacheKey, snapshotPromise);
    }

    try {
      const snapshotData = await snapshotPromise;
      if (draftSnapshotRequestSeqRef.current !== requestSeq) {
        return;
      }

      const { chapterList, volumeList, foreshadowList } = snapshotData;
      setChapters(chapterList);
      setVolumes(volumeList);
      setForeshadowCards(foreshadowList);
      if (chapterList.length === 0) {
        setActiveChapterId(null);
        setDraftTitle("第1章");
        setDraftText("");
        setDraftVersion(0);
        setDraftUpdatedAt(null);
        setDraftRevisions([]);
        setSceneBeats([]);
        setActiveSceneBeatId(null);
        setOverdueForeshadowCards([]);
        setSelectedDraftText("");
        lastSavedDraftRef.current = {
          chapterId: null,
          volumeId: null,
          title: "第1章",
          content: "",
        };
        setAutoSaveState("idle");
        setLocalRecoveryNotice(null);
        return;
      }

      const preferred = preferredChapterId ?? activeChapterId;
      const resolved =
        preferred && chapterList.some((item) => item.id === preferred)
          ? preferred
          : chapterList[0].id;
      setActiveChapterId(resolved);
      if (draftSnapshotRequestSeqRef.current !== requestSeq) {
        return;
      }
      await loadChapterSnapshot(nextProjectId, resolved);
    } finally {
      if (draftSnapshotInFlightRef.current.get(cacheKey) === snapshotPromise) {
        draftSnapshotInFlightRef.current.delete(cacheKey);
      }
    }
  };

  const { isDraftDirty, persistDraftSnapshot, saveDraftSnapshot, switchChapter } = useDraftWorkspaceFlow({
    projectId,
    activeChapterId,
    activeVolumeId,
    draftTitle,
    draftText,
    draftVersion,
    draftLoading,
    draftSaving,
    setDraftLoading,
    setDraftSaving,
    setActiveChapterId,
    setError,
    setGhostError,
    setDraftTitle,
    setDraftVersion,
    setDraftUpdatedAt,
    setDraftRevisions,
    setChapters,
    setAutoSaveState,
    setAutoSaveAt,
    setLocalRecoveryNotice,
    lastSavedDraftRef,
    autoSaveTimerRef,
    localRecoveryTimerRef,
    loadChapterSnapshot,
  });

  const refreshPlanningData = async (nextProjectId: number, chapterId: number | null) => {
    const requestSeq = planningSnapshotRequestSeqRef.current + 1;
    planningSnapshotRequestSeqRef.current = requestSeq;
    const cacheKey = `${nextProjectId}:${chapterId ?? "none"}`;

    let snapshotPromise = planningSnapshotInFlightRef.current.get(cacheKey);
    if (!snapshotPromise) {
      snapshotPromise = (async (): Promise<PlanningSnapshotData> => {
        const [volumeList, foreshadowList] = await Promise.all([
          getProjectVolumes(nextProjectId).catch(() => [] as ProjectVolume[]),
          getForeshadowingCards(nextProjectId).catch(() => [] as ForeshadowingCard[]),
        ]);
        if (!chapterId) {
          return {
            volumeList,
            foreshadowList,
            beats: [],
            overdue: [],
          };
        }
        const [beats, overdue] = await Promise.all([
          getSceneBeats(nextProjectId, chapterId).catch(() => [] as SceneBeat[]),
          getForeshadowingCards(nextProjectId, {
            overdue_for_chapter_id: chapterId,
            chapter_gap: 50,
          }).catch(() => [] as ForeshadowingCard[]),
        ]);
        return { volumeList, foreshadowList, beats, overdue };
      })();
      planningSnapshotInFlightRef.current.set(cacheKey, snapshotPromise);
    }

    try {
      const snapshotData = await snapshotPromise;
      if (planningSnapshotRequestSeqRef.current !== requestSeq) {
        return;
      }
      setVolumes(snapshotData.volumeList);
      setForeshadowCards(snapshotData.foreshadowList);
      if (chapterId) {
        const pendingBeat = snapshotData.beats.find((item) => item.status !== "done");
        setSceneBeats(snapshotData.beats);
        setActiveSceneBeatId((prev) => {
          if (prev && snapshotData.beats.some((item) => item.id === prev)) return prev;
          return pendingBeat?.id ?? snapshotData.beats[0]?.id ?? null;
        });
        setOverdueForeshadowCards(snapshotData.overdue);
      } else {
        setSceneBeats([]);
        setActiveSceneBeatId(null);
        setOverdueForeshadowCards([]);
      }
    } finally {
      if (planningSnapshotInFlightRef.current.get(cacheKey) === snapshotPromise) {
        planningSnapshotInFlightRef.current.delete(cacheKey);
      }
    }
  };

  useEffect(() => {
    let cancelled = false;
    setDraftLoading(true);
    (async () => {
      try {
        await Promise.all([refreshDraftSnapshot(projectId), refreshProjectSnapshot(projectId)]);
        if (cancelled) return;
      } catch (loadError) {
        if (cancelled) return;
        const message = loadError instanceof Error ? loadError.message : "读取章节正文失败";
        setError(message);
      } finally {
        if (!cancelled) {
          setDraftLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId, setError]);

  useEffect(() => {
    if (!editor) return;
    editor.setEditable(!draftLoading && !!activeChapterId);
  }, [editor, draftLoading, activeChapterId]);

  useEffect(() => {
    if (!editor) return;
    editor.view.dispatch(editor.state.tr.setMeta(entityHintPluginKey, entityHighlightHints));
  }, [editor, entityHighlightHints]);

  useEffect(() => {
    typewriterModeEnabledRef.current = typewriterModeEnabled;
  }, [typewriterModeEnabled]);

  useEffect(() => {
    typewriterDimmingEnabledRef.current =
      uiMode === "writing" && typewriterModeEnabled && (draftFocusMode || zenMode);
    if (!editor) {
      clearTypewriterParagraphFocus();
      return;
    }
    syncTypewriterParagraphFocus(editor);
  }, [editor, uiMode, typewriterModeEnabled, draftFocusMode, zenMode, activeChapterId]);

  useEffect(() => {
    return () => {
      clearTypewriterParagraphFocus();
    };
  }, []);

  useEffect(() => {
    if (!editor) return;
    const normalized = normalizeEditorText(draftText);
    if (readEditorText(editor) !== normalized) {
      editor.commands.setContent(toEditorDoc(normalized), { emitUpdate: false });
      scheduleTypewriterScroll(editor);
      syncTypewriterParagraphFocus(editor);
    }
  }, [draftText, editor]);

  useEffect(() => {
    if (!editor || !typewriterModeEnabled) return;
    scheduleTypewriterScroll(editor);
  }, [editor, typewriterModeEnabled, activeChapterId]);

  useEffect(() => {
    if (uiMode === "writing") return;
    if (!zenMode) return;
    setZenMode(false);
  }, [uiMode, zenMode]);

  const buildGhostCacheKey = () => {
    const tail = draftText.slice(Math.max(0, draftText.length - 320));
    const promptPart = activePromptTemplateId ?? 0;
    return [
      projectId,
      activeChapterId ?? 0,
      activeSceneBeatId ?? 0,
      promptPart,
      activeModelProfileId ?? "no-profile",
      model.trim() || "default",
      ghostTemperatureProfile,
      temperatureOverride ?? "auto",
      tail,
    ].join("|");
  };

  const requestGhostSuggestion = async (forceRefresh = false) => {
    if (!activeChapterId || draftLoading || streaming) return;
    const prefixText = draftText.trimEnd();
    if (!prefixText) {
      setGhostText("");
      setGhostError(null);
      return;
    }
    const cacheKey = buildGhostCacheKey();
    if (!forceRefresh && ghostCacheRef.current.has(cacheKey)) {
      setGhostText(ghostCacheRef.current.get(cacheKey) ?? "");
      setGhostError(null);
      setGhostLoading(false);
      return;
    }

    const requestId = ghostRequestRef.current + 1;
    ghostRequestRef.current = requestId;
    setGhostLoading(true);
    setGhostError(null);
    try {
      const result = await generateGhostText({
        project_id: projectId,
        chapter_id: activeChapterId,
        scene_beat_id: activeSceneBeatId,
        prompt_template_id: activePromptTemplateId,
        prefix_text: prefixText.slice(Math.max(0, prefixText.length - 1600)),
        model: model.trim() ? model.trim() : null,
        model_profile_id: activeModelProfileId,
        temperature_profile: ghostTemperatureProfile,
        temperature_override: temperatureOverride,
      });
      if (ghostRequestRef.current !== requestId) return;
      const nextText = (result.suggestion || "").trim();
      setGhostText(nextText);
      if (nextText) {
        ghostCacheRef.current.set(cacheKey, nextText);
      }
    } catch (ghostErr) {
      if (ghostRequestRef.current !== requestId) return;
      const message = ghostErr instanceof Error ? ghostErr.message : "Ghost Text 生成失败";
      setGhostError(message);
      setGhostText("");
    } finally {
      if (ghostRequestRef.current === requestId) {
        setGhostLoading(false);
      }
    }
  };

  useEffect(() => {
    if (!ghostAutoEnabled) {
      if (!ghostLoading) {
        return;
      }
      setGhostLoading(false);
      return;
    }
    if (!activeChapterId || draftLoading) {
      setGhostText("");
      setGhostError(null);
      setGhostLoading(false);
      return;
    }
    const timer = window.setTimeout(() => {
      void requestGhostSuggestion(false);
    }, 900);
    return () => window.clearTimeout(timer);
  }, [
    activeChapterId,
    activeSceneBeatId,
    draftText,
    draftLoading,
    projectId,
    activePromptTemplateId,
    activeModelProfileId,
    model,
    streaming,
    ghostAutoEnabled,
    ghostTemperatureProfile,
    temperatureOverride,
  ]);

  useEffect(() => {
    return () => {
      if (typewriterRafRef.current) {
        window.cancelAnimationFrame(typewriterRafRef.current);
        typewriterRafRef.current = null;
      }
    };
  }, []);

  const openSettingsDialog = () => {
    if (!settingsDialogOpen && typeof document !== "undefined") {
      const active = document.activeElement;
      settingsDialogReturnFocusRef.current = active instanceof HTMLElement ? active : null;
    }
    setSettingsDialogOpen(true);
  };

  const closeSettingsDialog = () => setSettingsDialogOpen(false);

  const toggleUiMode = () => {
    setUiMode(uiMode === "writing" ? "pro" : "writing");
  };

  const toggleZenMode = () => {
    if (uiMode !== "writing") return;
    setZenMode((prev) => {
      const next = !prev;
      if (next) {
        setAssistantDrawerOpen(false);
        setSettingsDialogOpen(false);
      }
      return next;
    });
  };

  const toggleTypewriterMode = () => {
    setTypewriterModeEnabled((prev) => !prev);
  };

  const toggleDraftFocusMode = () => {
    setDraftFocusMode((prev) => !prev);
  };

  const toggleGhostAuto = () => {
    setGhostAutoEnabled((prev) => !prev);
  };

  const bindActiveChapterToVolume = async (volumeId: number) => {
    if (!activeChapterId) return;
    if (draftSaving || draftLoading) return;
    setPlanningBusy(true);
    setError(null);
    try {
      await saveProjectChapter(projectId, activeChapterId, {
        title: draftTitle.trim() || "未命名章节",
        content: draftText,
        volume_id: volumeId,
        expected_version: null,
      });
      await refreshDraftSnapshot(projectId, activeChapterId);
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "绑定章节到卷失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const createVolume = async () => {
    if (planningBusy) return;
    setPlanningBusy(true);
    setError(null);
    try {
      const created = await createProjectVolume(projectId, {
        title: null,
        outline: "",
      });
      await refreshPlanningData(projectId, activeChapterId);
      setActiveVolumeId(created.id);
      setVolumeOutlineDraft(created.outline);
      if (activeChapterId) {
        await bindActiveChapterToVolume(created.id);
      }
    } catch (createError) {
      const message = createError instanceof Error ? createError.message : "创建卷失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const saveVolumeOutline = async () => {
    if (!activeVolumeId) {
      setError("请先选择卷");
      return;
    }
    if (planningBusy) return;
    const target = volumes.find((item) => item.id === activeVolumeId);
    if (!target) {
      setError("卷不存在，请刷新后重试");
      return;
    }
    setPlanningBusy(true);
    setError(null);
    try {
      await updateProjectVolume(projectId, activeVolumeId, {
        title: target.title,
        outline: volumeOutlineDraft,
      });
      await refreshPlanningData(projectId, activeChapterId);
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "保存卷纲失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const createBeatForActiveChapter = async () => {
    if (!activeChapterId) {
      setError("请先选择章节");
      return;
    }
    const content = newBeatContent.trim();
    if (!content) {
      setError("请先填写 Beat 内容");
      return;
    }
    if (planningBusy) return;
    setPlanningBusy(true);
    setError(null);
    try {
      const created = await createSceneBeat(projectId, activeChapterId, {
        content,
        status: "pending",
      });
      setNewBeatContent("");
      await refreshPlanningData(projectId, activeChapterId);
      setActiveSceneBeatId(created.id);
    } catch (createError) {
      const message = createError instanceof Error ? createError.message : "创建 Beat 失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const toggleBeatStatus = async (beatId: number, done: boolean) => {
    if (!activeChapterId || planningBusy) return;
    const target = sceneBeats.find((item) => item.id === beatId);
    if (!target) return;
    setPlanningBusy(true);
    setError(null);
    try {
      await updateSceneBeat(projectId, activeChapterId, beatId, {
        content: target.content,
        status: done ? "done" : "pending",
      });
      await refreshPlanningData(projectId, activeChapterId);
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "更新 Beat 失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const deleteBeat = async (beatId: number) => {
    if (!activeChapterId || planningBusy) return;
    if (!window.confirm("确认删除这个 Beat 吗？")) return;
    setPlanningBusy(true);
    setError(null);
    try {
      await deleteSceneBeat(projectId, activeChapterId, beatId);
      await refreshPlanningData(projectId, activeChapterId);
    } catch (deleteError) {
      const message = deleteError instanceof Error ? deleteError.message : "删除 Beat 失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const createForeshadow = async () => {
    const title = foreshadowDraftTitle.trim();
    if (!title) {
      setError("请先填写伏笔标题");
      return;
    }
    if (planningBusy) return;
    setPlanningBusy(true);
    setError(null);
    try {
      await createForeshadowingCard(projectId, {
        title,
        description: foreshadowDraftDescription,
        planted_in_chapter_id: activeChapterId,
      });
      setForeshadowDraftTitle("");
      setForeshadowDraftDescription("");
      await refreshPlanningData(projectId, activeChapterId);
    } catch (createError) {
      const message = createError instanceof Error ? createError.message : "创建伏笔卡失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const toggleForeshadowStatus = async (card: ForeshadowingCard, nextStatus: "open" | "resolved") => {
    if (planningBusy) return;
    setPlanningBusy(true);
    setError(null);
    try {
      await updateForeshadowingCard(projectId, card.id, {
        title: card.title,
        description: card.description,
        status: nextStatus,
        planted_in_chapter_id: card.planted_in_chapter_id,
        resolved_in_chapter_id: nextStatus === "resolved" ? (activeChapterId ?? card.resolved_in_chapter_id) : null,
      });
      await refreshPlanningData(projectId, activeChapterId);
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "更新伏笔卡失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const deleteForeshadow = async (cardId: number) => {
    if (planningBusy) return;
    if (!window.confirm("确认删除这个伏笔卡吗？")) return;
    setPlanningBusy(true);
    setError(null);
    try {
      await deleteForeshadowingCard(projectId, cardId);
      await refreshPlanningData(projectId, activeChapterId);
    } catch (deleteError) {
      const message = deleteError instanceof Error ? deleteError.message : "删除伏笔卡失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const applyAssistantToDraft = (mode: "insert" | "replace") => {
    if (!editor) return;
    const assistantText = latestAssistantReply;
    if (!assistantText) {
      setError("暂无可写回的助手回复。先让助手生成内容。");
      return;
    }

    setError(null);

    const { from, to, empty } = editor.state.selection;
    const shouldReplace = mode === "replace" && !empty;
    let insertion = assistantText;
    if (!shouldReplace) {
      const beforeChar = from > 1 ? editor.state.doc.textBetween(from - 1, from, "\n", "\n") : "";
      const prefix = from > 1 && !/\s/.test(beforeChar) ? "\n\n" : "";
      insertion = `${prefix}${assistantText}`;
    }

    const insertionContent = toEditorDoc(insertion).content ?? [];
    editor.chain().focus().insertContentAt({ from, to }, insertionContent).run();
    setDraftText(readEditorText(editor));
    setSelectedDraftText("");
  };

  const acceptGhostText = () => {
    if (!editor || !ghostText.trim()) return;
    const { from, to } = editor.state.selection;
    const insertionContent = toEditorDoc(ghostText).content ?? [];
    editor.chain().focus().insertContentAt({ from, to }, insertionContent).run();
    setDraftText(readEditorText(editor));
    setGhostText("");
    setGhostError(null);
  };

  const rejectGhostText = () => {
    setGhostText("");
    setGhostError(null);
  };

  const regenerateGhostText = async () => {
    const key = buildGhostCacheKey();
    ghostCacheRef.current.delete(key);
    await requestGhostSuggestion(true);
  };
  acceptGhostTextRef.current = acceptGhostText;
  rejectGhostTextRef.current = rejectGhostText;
  regenerateGhostTextRef.current = regenerateGhostText;

  const fillPromptFromSelection = (mode: "polish" | "expand") => {
    if (!selectedDraftText) {
      setError("请先在正文工作区选中一段文本。");
      return;
    }
    setError(null);
    const nextPrompt =
      mode === "polish"
        ? `请在不改变剧情事实、人设和时间线的前提下润色这段正文，输出润色后的完整段落：\n\n${selectedDraftText}`
        : `请基于这段正文进行扩写，保持同一人称和语气，不引入越权设定，输出扩写后的完整段落：\n\n${selectedDraftText}`;
    setInput(nextPrompt);
    openAssistantDrawerAndFocusComposer();
  };

  const rollbackDraftToVersion = async (targetVersion: number) => {
    if (!activeChapterId) {
      setError("请先选择章节");
      return;
    }
    if (draftSaving) return;
    setDraftSaving(true);
    setError(null);
    try {
      await rollbackProjectChapter(projectId, activeChapterId, {
        target_version: targetVersion,
      });
      await refreshDraftSnapshot(projectId, activeChapterId);
    } catch (rollbackError) {
      const message = rollbackError instanceof Error ? rollbackError.message : "回滚正文失败";
      setError(message);
    } finally {
      setDraftSaving(false);
    }
  };

  const createChapterAndSwitch = async () => {
    if (draftSaving || draftLoading) return;
    setDraftSaving(true);
    setError(null);
    try {
      const created = await createProjectChapter(projectId, {
        title: null,
        volume_id: activeVolumeId,
      });
      await refreshDraftSnapshot(projectId, created.id);
    } catch (createError) {
      const message = createError instanceof Error ? createError.message : "创建章节失败";
      setError(message);
    } finally {
      setDraftSaving(false);
    }
  };

  const moveActiveChapter = async (direction: "up" | "down") => {
    if (!activeChapterId) {
      setError("请先选择章节");
      return;
    }
    if (draftSaving || draftLoading) return;
    setDraftSaving(true);
    setError(null);
    try {
      await moveProjectChapter(projectId, activeChapterId, {
        direction,
      });
      await refreshDraftSnapshot(projectId, activeChapterId);
    } catch (moveError) {
      const message = moveError instanceof Error ? moveError.message : "调整章节顺序失败";
      setError(message);
    } finally {
      setDraftSaving(false);
    }
  };

  const deleteActiveChapter = async () => {
    if (!activeChapterId) {
      setError("请先选择章节");
      return;
    }
    const label = activeChapter ? `${activeChapter.chapter_index}. ${activeChapter.title}` : `#${activeChapterId}`;
    if (!window.confirm(`确认删除章节「${label}」吗？`)) {
      return;
    }
    if (draftSaving || draftLoading) return;
    setDraftSaving(true);
    setError(null);
    try {
      const result = await deleteProjectChapter(projectId, activeChapterId);
      await refreshDraftSnapshot(projectId, result.active_chapter_id);
    } catch (deleteError) {
      const message = deleteError instanceof Error ? deleteError.message : "删除章节失败";
      setError(message);
    } finally {
      setDraftSaving(false);
    }
  };

  const handleOutlineDragStart = (chapterId: number) => {
    setDragChapterId(chapterId);
  };

  const handleOutlineDragEnd = () => {
    setDragChapterId(null);
  };

  const reorderByDrag = async (targetChapterId: number) => {
    if (!dragChapterId || dragChapterId === targetChapterId) {
      setDragChapterId(null);
      return;
    }
    const fromIndex = chapters.findIndex((item) => item.id === dragChapterId);
    const toIndex = chapters.findIndex((item) => item.id === targetChapterId);
    if (fromIndex < 0 || toIndex < 0) {
      setDragChapterId(null);
      return;
    }

    const reordered = [...chapters];
    const [moved] = reordered.splice(fromIndex, 1);
    reordered.splice(toIndex, 0, moved);
    const orderedIds = reordered.map((item) => item.id);

    setDraftSaving(true);
    setError(null);
    try {
      await reorderProjectChapters(projectId, {
        ordered_ids: orderedIds,
      });
      await refreshDraftSnapshot(projectId, activeChapterId ?? dragChapterId);
    } catch (reorderError) {
      const message = reorderError instanceof Error ? reorderError.message : "拖拽排序失败";
      setError(message);
    } finally {
      setDraftSaving(false);
      setDragChapterId(null);
    }
  };

  const applyTemplatesSnapshot = (
    templates: PromptTemplate[],
    preferredActiveTemplateId?: number | null,
    preferredDraftTemplateId?: number | null
  ) => {
    setPromptTemplates(templates);
    if (templates.length === 0) {
      setActivePromptTemplateId(null);
      resetTemplateDraft();
      setTemplateRevisions([]);
      return;
    }

    const resolvedActive =
      preferredActiveTemplateId && templates.some((item) => item.id === preferredActiveTemplateId)
        ? preferredActiveTemplateId
        : templates[0].id;
    setActivePromptTemplateId(resolvedActive);

    const resolvedDraftId =
      preferredDraftTemplateId && templates.some((item) => item.id === preferredDraftTemplateId)
        ? preferredDraftTemplateId
        : resolvedActive;
    const draftTemplate = templates.find((item) => item.id === resolvedDraftId) ?? null;
    loadTemplateIntoDraft(draftTemplate);
  };

  const applyModelProfilesSnapshot = (
    profiles: ModelProfile[],
    preferredSelectedProfileId?: string | null
  ) => {
    setModelProfiles(profiles);
    const resolvedActive =
      profiles.find((item) => Boolean(item.is_active))?.profile_id ?? null;
    setActiveModelProfileId(resolvedActive);

    if (profiles.length === 0) {
      resetModelProfileDraft();
      return;
    }

    const selectedCandidate =
      (preferredSelectedProfileId && profiles.some((item) => item.profile_id === preferredSelectedProfileId)
        ? preferredSelectedProfileId
        : null) ??
      (selectedModelProfileId && profiles.some((item) => item.profile_id === selectedModelProfileId)
        ? selectedModelProfileId
        : null) ??
      (resolvedActive && profiles.some((item) => item.profile_id === resolvedActive) ? resolvedActive : null) ??
      profiles[0].profile_id;

    const draftProfile = profiles.find((item) => item.profile_id === selectedCandidate) ?? null;
    loadModelProfileIntoDraft(draftProfile);
  };

  const refreshTemplateRevisions = async (nextProjectId: number, templateId: number | null) => {
    const requestSeq = templateRevisionsRequestSeqRef.current + 1;
    templateRevisionsRequestSeqRef.current = requestSeq;

    if (!templateId) {
      setTemplateRevisions([]);
      setTemplateRevisionsLoading(false);
      return;
    }

    const cacheKey = `${nextProjectId}:${templateId}`;
    let revisionsPromise = templateRevisionsInFlightRef.current.get(cacheKey);
    if (!revisionsPromise) {
      revisionsPromise = getProjectPromptTemplateRevisions(nextProjectId, templateId, 20);
      templateRevisionsInFlightRef.current.set(cacheKey, revisionsPromise);
    }

    setTemplateRevisionsLoading(true);
    try {
      const revisions = await revisionsPromise;
      if (templateRevisionsRequestSeqRef.current !== requestSeq) {
        return;
      }
      setTemplateRevisions(revisions);
    } catch (revisionError) {
      if (templateRevisionsRequestSeqRef.current !== requestSeq) {
        return;
      }
      const message = revisionError instanceof Error ? revisionError.message : "读取模板历史失败";
      setError(message);
      setTemplateRevisions([]);
    } finally {
      if (templateRevisionsInFlightRef.current.get(cacheKey) === revisionsPromise) {
        templateRevisionsInFlightRef.current.delete(cacheKey);
      }
      if (templateRevisionsRequestSeqRef.current === requestSeq) {
        setTemplateRevisionsLoading(false);
      }
    }
  };

  const refreshGraphTimeline = async (requestedChapterIndex: number) => {
    const chapter = Number.isFinite(requestedChapterIndex) ? Math.max(0, Math.floor(requestedChapterIndex)) : 0;
    if (projectId <= 0 || chapter <= 0) {
      setGraphTimeline(null);
      return;
    }
    const requestSeq = graphTimelineRequestSeqRef.current + 1;
    graphTimelineRequestSeqRef.current = requestSeq;
    setGraphTimelineLoading(true);
    try {
      const snapshot = await getProjectGraphTimeline(projectId, chapter, 260);
      if (graphTimelineRequestSeqRef.current !== requestSeq) return;
      setGraphTimeline(snapshot);
    } catch {
      if (graphTimelineRequestSeqRef.current !== requestSeq) return;
      setGraphTimeline({
        project_id: projectId,
        chapter_index: chapter,
        nodes: [],
        edges: [],
        stats: { source: "fallback", nodes: 0, edges: 0 },
      });
    } finally {
      if (graphTimelineRequestSeqRef.current === requestSeq) {
        setGraphTimelineLoading(false);
      }
    }
  };

  useEffect(() => {
    if (activeChapterIndex > 0) {
      setGraphTimelineChapterIndex(activeChapterIndex);
    }
  }, [activeChapterIndex]);

  useEffect(() => {
    const targetChapter = graphTimelineChapterIndex > 0 ? graphTimelineChapterIndex : activeChapterIndex;
    if (projectId <= 0 || targetChapter <= 0) {
      setGraphTimeline(null);
      return;
    }
    void refreshGraphTimeline(targetChapter);
  }, [projectId, graphTimelineChapterIndex, activeChapterIndex]);

  const refreshProjectSnapshot = async (nextProjectId: number) => {
    const requestSeq = projectSnapshotRequestSeqRef.current + 1;
    projectSnapshotRequestSeqRef.current = requestSeq;
    const cacheKey = `${nextProjectId}`;

    let snapshotPromise = projectSnapshotInFlightRef.current.get(cacheKey);
    if (!snapshotPromise) {
      snapshotPromise = (async (): Promise<ProjectSnapshotData> => {
        const [
          settingsData,
          auditsData,
          modelProfilesData,
          cardsData,
          templatesData,
          volumesData,
          foreshadowData,
          sessionsData,
        ] = await Promise.all([
          getProjectSettings(nextProjectId),
          getProjectConsistencyAudits(nextProjectId, 8).catch(() => [] as ConsistencyAuditReport[]),
          getProjectModelProfiles(nextProjectId).catch(() => [] as ModelProfile[]),
          getProjectCards(nextProjectId),
          getProjectPromptTemplates(nextProjectId),
          getProjectVolumes(nextProjectId).catch(() => [] as ProjectVolume[]),
          getForeshadowingCards(nextProjectId).catch(() => [] as ForeshadowingCard[]),
          getProjectSessions(nextProjectId).catch(() => [] as ChatSessionSummary[]),
        ]);
        return {
          settingsData,
          auditsData,
          modelProfilesData,
          cardsData,
          templatesData,
          volumesData,
          foreshadowData,
          sessionsData,
        };
      })();
      projectSnapshotInFlightRef.current.set(cacheKey, snapshotPromise);
    }

    try {
      const snapshotData = await snapshotPromise;
      if (projectSnapshotRequestSeqRef.current !== requestSeq) {
        return;
      }
      setSettings(snapshotData.settingsData);
      setConsistencyAudits(snapshotData.auditsData);
      applyModelProfilesSnapshot(snapshotData.modelProfilesData, selectedModelProfileId);
      setCards(snapshotData.cardsData);
      setVolumes(snapshotData.volumesData);
      setForeshadowCards(snapshotData.foreshadowData);
      setProjectSessions(snapshotData.sessionsData);
      applyTemplatesSnapshot(snapshotData.templatesData, activePromptTemplateId, templateDraftId);
    } finally {
      if (projectSnapshotInFlightRef.current.get(cacheKey) === snapshotPromise) {
        projectSnapshotInFlightRef.current.delete(cacheKey);
      }
    }
  };

  const refreshSessionSnapshot = async (nextSessionId: number, nextProjectId: number) => {
    const requestSeq = fullSessionSnapshotRequestSeqRef.current + 1;
    fullSessionSnapshotRequestSeqRef.current = requestSeq;
    const cacheKey = `${nextProjectId}:${nextSessionId}`;

    let snapshotPromise = fullSessionSnapshotInFlightRef.current.get(cacheKey);
    if (!snapshotPromise) {
      snapshotPromise = (async (): Promise<FullSessionSnapshotData> => {
        const [
          messagesData,
          actionsData,
          settingsData,
          auditsData,
          modelProfilesData,
          cardsData,
          templatesData,
          volumesData,
          foreshadowData,
          sessionsData,
        ] =
          await Promise.all([
            getSessionMessages(nextSessionId),
            getSessionActions(nextSessionId),
            getProjectSettings(nextProjectId),
            getProjectConsistencyAudits(nextProjectId, 8).catch(() => [] as ConsistencyAuditReport[]),
            getProjectModelProfiles(nextProjectId).catch(() => [] as ModelProfile[]),
            getProjectCards(nextProjectId),
            getProjectPromptTemplates(nextProjectId),
            getProjectVolumes(nextProjectId).catch(() => [] as ProjectVolume[]),
            getForeshadowingCards(nextProjectId).catch(() => [] as ForeshadowingCard[]),
            getProjectSessions(nextProjectId).catch(() => [] as ChatSessionSummary[]),
          ]);
        return {
          messagesData,
          actionsData,
          settingsData,
          auditsData,
          modelProfilesData,
          cardsData,
          templatesData,
          volumesData,
          foreshadowData,
          sessionsData,
        };
      })();
      fullSessionSnapshotInFlightRef.current.set(cacheKey, snapshotPromise);
    }

    try {
      const snapshotData = await snapshotPromise;
      if (fullSessionSnapshotRequestSeqRef.current !== requestSeq) {
        return;
      }
      setMessages(snapshotData.messagesData.map(toUiMessage));
      setActions(snapshotData.actionsData);
      setSettings(snapshotData.settingsData);
      setConsistencyAudits(snapshotData.auditsData);
      applyModelProfilesSnapshot(snapshotData.modelProfilesData, selectedModelProfileId);
      setCards(snapshotData.cardsData);
      setVolumes(snapshotData.volumesData);
      setForeshadowCards(snapshotData.foreshadowData);
      setProjectSessions(snapshotData.sessionsData);
      applyTemplatesSnapshot(snapshotData.templatesData, activePromptTemplateId, templateDraftId);
    } finally {
      if (fullSessionSnapshotInFlightRef.current.get(cacheKey) === snapshotPromise) {
        fullSessionSnapshotInFlightRef.current.delete(cacheKey);
      }
    }
  };

  const refreshSessionPostChatSnapshot = async (nextSessionId: number, nextProjectId: number) => {
    const requestSeq = postChatSnapshotRequestSeqRef.current + 1;
    postChatSnapshotRequestSeqRef.current = requestSeq;
    const cacheKey = `${nextProjectId}:${nextSessionId}`;
    const cached = postChatSnapshotCacheRef.current.get(cacheKey);
    const now = Date.now();

    const applySnapshotData = (snapshotData: PostChatSnapshotData) => {
      if (postChatSnapshotRequestSeqRef.current !== requestSeq) {
        return;
      }
      setMessages(snapshotData.messagesData.map(toUiMessage));
      setActions(snapshotData.actionsData);
      setProjectSessions(snapshotData.sessionsData);
    };

    if (cached && now - cached.at <= POST_CHAT_SNAPSHOT_TTL_MS) {
      applySnapshotData(cached.data);
      return;
    }

    let snapshotPromise = postChatSnapshotInFlightRef.current.get(cacheKey);
    if (!snapshotPromise) {
      snapshotPromise = (async (): Promise<PostChatSnapshotData> => {
        const [messagesData, actionsData, sessionsData] = await Promise.all([
          getSessionMessages(nextSessionId),
          getSessionActions(nextSessionId),
          getProjectSessions(nextProjectId).catch(() => [] as ChatSessionSummary[]),
        ]);
        return { messagesData, actionsData, sessionsData };
      })();
      postChatSnapshotInFlightRef.current.set(cacheKey, snapshotPromise);
    }

    try {
      const snapshotData = await snapshotPromise;
      postChatSnapshotCacheRef.current.set(cacheKey, {
        at: Date.now(),
        data: snapshotData,
      });
      applySnapshotData(snapshotData);
    } finally {
      if (postChatSnapshotInFlightRef.current.get(cacheKey) === snapshotPromise) {
        postChatSnapshotInFlightRef.current.delete(cacheKey);
      }
    }
  };

  useEffect(() => {
    void refreshTemplateRevisions(projectId, templateDraftId);
  }, [projectId, templateDraftId]);

  useEffect(() => {
    if (!selectedModelProfileId) {
      if (modelProfiles.length === 0) {
        resetModelProfileDraft();
      } else {
        setModelProfileDraftIdInput("");
        setModelProfileName("");
        setModelProfileProvider("openai_compatible");
        setModelProfileBaseUrl("");
        setModelProfileApiKey("");
        setModelProfileApiKeyMasked(null);
        setClearModelProfileApiKey(false);
        setModelProfileModel("");
      }
      return;
    }
    const profile = modelProfiles.find((item) => item.profile_id === selectedModelProfileId) ?? null;
    if (!profile) return;
    loadModelProfileIntoDraft(profile);
  }, [modelProfiles, selectedModelProfileId]);

  const typewriterDimmingEnabled =
    uiMode === "writing" && typewriterModeEnabled && (draftFocusMode || zenMode);
  typewriterDimmingEnabledRef.current = typewriterDimmingEnabled;

  writingShortcutStateRef.current = {
    uiMode,
    assistantDrawerOpen,
    settingsDialogOpen,
    hasGhostText: ghostText.trim().length > 0,
    ghostLoading,
    draftLoading,
    draftSaving,
    activeChapterId,
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === "z") {
        if (uiMode !== "writing") return;
        event.preventDefault();
        toggleZenMode();
        return;
      }
      if (event.key === "F11") {
        if (uiMode !== "writing") return;
        event.preventDefault();
        toggleZenMode();
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === "a") {
        event.preventDefault();
        toggleAssistantDrawer();
      }
      if (event.key === "Escape") {
        closeAssistantDrawer();
        closeSettingsDialog();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [uiMode]);

  useEffect(() => {
    if (typeof document === "undefined") return;
    const zenClass = "zen-mode-active";
    document.body.classList.toggle(zenClass, zenMode);
    return () => {
      document.body.classList.remove(zenClass);
    };
  }, [zenMode]);

  useEffect(() => {
    actionLogsCacheRef.current.clear();
    actionLogsInFlightRef.current.clear();
    actionLogsRequestSeqRef.current = 0;
    chapterSnapshotInFlightRef.current.clear();
    chapterSnapshotRequestSeqRef.current = 0;
    draftSnapshotInFlightRef.current.clear();
    draftSnapshotRequestSeqRef.current = 0;
    planningSnapshotInFlightRef.current.clear();
    planningSnapshotRequestSeqRef.current = 0;
    projectSnapshotInFlightRef.current.clear();
    projectSnapshotRequestSeqRef.current = 0;
    fullSessionSnapshotInFlightRef.current.clear();
    fullSessionSnapshotRequestSeqRef.current = 0;
    templateRevisionsInFlightRef.current.clear();
    templateRevisionsRequestSeqRef.current = 0;
    postChatSnapshotCacheRef.current.clear();
    postChatSnapshotInFlightRef.current.clear();
    postChatSnapshotRequestSeqRef.current = 0;
  }, [projectId, sessionId]);

  useEffect(() => {
    const wasOpen = previousSettingsDialogOpenRef.current;
    previousSettingsDialogOpenRef.current = settingsDialogOpen;
    if (!wasOpen || settingsDialogOpen) return;
    const target = settingsDialogReturnFocusRef.current;
    if (!target || typeof document === "undefined" || !document.contains(target)) return;
    window.setTimeout(() => target.focus(), 0);
  }, [settingsDialogOpen]);

  useEffect(() => {
    if (!assistantDrawerOpen || settingsDialogOpen) return;
    if (typeof document === "undefined") return;
    const container = assistantDrawerRef.current;
    if (!container) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const focusables = getFocusableElements(container);
      if (focusables.length === 0) return;

      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      const active = document.activeElement instanceof HTMLElement ? document.activeElement : null;

      if (event.shiftKey) {
        if (!active || active === first || !container.contains(active)) {
          event.preventDefault();
          last.focus();
        }
        return;
      }

      if (!active || active === last || !container.contains(active)) {
        event.preventDefault();
        first.focus();
      }
    };

    const onFocusIn = (event: FocusEvent) => {
      const target = event.target;
      if (target instanceof Node && container.contains(target)) return;
      const focusables = getFocusableElements(container);
      if (focusables.length > 0) {
        focusables[0].focus();
        return;
      }
      container.focus();
    };

    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("focusin", onFocusIn);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("focusin", onFocusIn);
    };
  }, [assistantDrawerOpen, settingsDialogOpen]);

  useEffect(() => {
    if (!settingsDialogOpen) return;
    if (typeof document === "undefined") return;
    const container = settingsDialogRef.current;
    if (!container) return;

    focusFirstDialogElement(container);

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const focusables = getFocusableElements(container);
      if (focusables.length === 0) return;

      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      const active = document.activeElement instanceof HTMLElement ? document.activeElement : null;

      if (event.shiftKey) {
        if (!active || active === first || !container.contains(active)) {
          event.preventDefault();
          last.focus();
        }
        return;
      }

      if (!active || active === last || !container.contains(active)) {
        event.preventDefault();
        first.focus();
      }
    };

    const onFocusIn = (event: FocusEvent) => {
      const target = event.target;
      if (target instanceof Node && container.contains(target)) return;
      focusFirstDialogElement(container);
    };

    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("focusin", onFocusIn);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("focusin", onFocusIn);
    };
  }, [settingsDialogOpen]);

  const {
    toggleAssistantDrawer,
    closeAssistantDrawer,
    openAssistantDrawerAndFocusComposer,
    startNewSession,
    switchSession,
    renameSession,
    deleteSession,
    handleSend,
  } = useAssistantSessionFlow({
    projectSessions,
    input,
    streaming,
    assistantDrawerOpen,
    setAssistantDrawerOpen,
    composerInputRef,
    assistantDrawerReturnFocusRef,
    previousAssistantDrawerOpenRef,
    streamOptions: {
      projectId,
      sessionId,
      activeChapterId,
      activeSceneBeatId,
      activePromptTemplateId,
      activeModelProfileId,
      model,
      povMode,
      povAnchor,
      ragMode,
      deterministicFirst,
      thinkingEnabled,
      referenceProjectIds,
      chatTemperatureProfile,
      temperatureOverride,
      contextWindowProfile,
    },
    setInput,
    setSessionId,
    setStreaming,
    setError,
    setUsage,
    setPendingActionIds,
    setSelectedActionId,
    setActionLogs,
    setTraceEvents,
    setEvidence,
    setLastStreamMetrics,
    appendMessage,
    appendMessageDelta,
    updateMessage,
    resetSessionState,
    refreshSessionSnapshot,
    refreshSessionPostChatSnapshot,
    refreshProjectSnapshot,
    isDraftDirty,
    persistDraftSnapshot,
  });

  const loadLogs = async (actionId: number, options?: { force?: boolean }) => {
    const force = Boolean(options?.force);
    const requestSeq = actionLogsRequestSeqRef.current + 1;
    actionLogsRequestSeqRef.current = requestSeq;
    setSelectedActionId(actionId);
    if (!force) {
      const cached = actionLogsCacheRef.current.get(actionId);
      if (cached) {
        setActionLogs(cached);
        return;
      }
    }

    let logsPromise = actionLogsInFlightRef.current.get(actionId);
    if (!logsPromise || force) {
      logsPromise = getActionLogs(actionId);
      actionLogsInFlightRef.current.set(actionId, logsPromise);
    }

    try {
      const logs = await logsPromise;
      actionLogsCacheRef.current.set(actionId, logs);
      if (actionLogsRequestSeqRef.current !== requestSeq) {
        return;
      }
      setActionLogs(logs);
    } catch (loadError) {
      if (actionLogsRequestSeqRef.current !== requestSeq) {
        return;
      }
      const message =
        loadError instanceof Error ? loadError.message : "读取日志失败";
      setError(message);
      setActionLogs([]);
    } finally {
      if (actionLogsInFlightRef.current.get(actionId) === logsPromise) {
        actionLogsInFlightRef.current.delete(actionId);
      }
    }
  };

  const mutateAction = async (
    action: ChatAction,
    decision: "apply" | "reject" | "undo"
  ) => {
    if (mutatingActionId !== null) return;
    setMutatingActionId(action.id);
    setError(null);
    try {
      const eventPayload =
        decision === "apply" && isEntityMergeActionType(action.action_type)
          ? { manual_confirmed: true, review_surface: "action_drawer" }
          : {};
      const updatedAction = await decideAction(action.id, decision, eventPayload);
      const nextStatus = String(updatedAction.status || "").trim().toLowerCase();
      const canUseLightSessionRefresh =
        decision === "reject"
        || nextStatus === "rejected"
        || nextStatus === "failed"
        || (decision === "apply" && nextStatus !== "applied")
        || (decision === "undo" && nextStatus !== "undone");
      if (sessionId) {
        if (canUseLightSessionRefresh) {
          await refreshSessionPostChatSnapshot(sessionId, projectId);
        } else {
          await refreshSessionSnapshot(sessionId, projectId);
        }
      } else {
        await refreshProjectSnapshot(projectId);
      }
      if (selectedActionId === action.id) {
        actionLogsCacheRef.current.delete(action.id);
        await loadLogs(action.id, { force: true });
      }
    } catch (mutationError) {
      const message =
        mutationError instanceof Error ? mutationError.message : "动作执行失败";
      setError(message);
    } finally {
      setMutatingActionId(null);
    }
  };
  const handleRefresh = async () => {
    setError(null);
    try {
      void preheatContextPack(projectId).catch(() => undefined);
      if (sessionId) {
        await Promise.all([refreshSessionSnapshot(sessionId, projectId), refreshDraftSnapshot(projectId)]);
      } else {
        await Promise.all([refreshProjectSnapshot(projectId), refreshDraftSnapshot(projectId)]);
      }
    } catch (refreshError) {
      const message =
        refreshError instanceof Error ? refreshError.message : "刷新失败";
      setError(message);
    }
  };

  const handleActiveTemplateChange = (value: string) => {
    const nextId = value ? Number(value) : null;
    setActivePromptTemplateId(nextId);
    const template = promptTemplates.find((item) => item.id === nextId) ?? null;
    loadTemplateIntoDraft(template);
  };

  const startCreateTemplateDraft = () => {
    setError(null);
    resetTemplateDraft();
  };

  const saveTemplateDraft = async () => {
    const name = templateName.trim();
    if (!name) {
      setError("模板名称不能为空");
      return;
    }
    if (templateSaving) return;
    setTemplateSaving(true);
    setError(null);
    try {
      const payload = {
        name,
        system_prompt: templateSystemPrompt,
        user_prompt_prefix: templateUserPromptPrefix,
        knowledge_setting_keys: templateKnowledgeSettingKeys,
        knowledge_card_ids: templateKnowledgeCardIds,
      };
      const saved = templateDraftId
        ? await updateProjectPromptTemplate(projectId, templateDraftId, payload)
        : await createProjectPromptTemplate(projectId, payload);
      const templates = await getProjectPromptTemplates(projectId);
      const preferredActive = templateDraftId ? activePromptTemplateId : saved.id;
      applyTemplatesSnapshot(templates, preferredActive, saved.id);
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "保存模板失败";
      setError(message);
    } finally {
      setTemplateSaving(false);
    }
  };

  const deleteTemplateDraft = async () => {
    if (!templateDraftId) {
      setError("请先选择要删除的模板");
      return;
    }
    const target = promptTemplates.find((item) => item.id === templateDraftId);
    const label = target?.name ?? `#${templateDraftId}`;
    if (!window.confirm(`确认删除模板「${label}」吗？`)) return;
    if (templateSaving) return;
    setTemplateSaving(true);
    setError(null);
    try {
      await deleteProjectPromptTemplate(projectId, templateDraftId);
      const templates = await getProjectPromptTemplates(projectId);
      const preferredActive =
        activePromptTemplateId === templateDraftId ? null : activePromptTemplateId;
      applyTemplatesSnapshot(templates, preferredActive, null);
    } catch (deleteError) {
      const message = deleteError instanceof Error ? deleteError.message : "删除模板失败";
      setError(message);
    } finally {
      setTemplateSaving(false);
    }
  };

  const copyTemplateDraft = async () => {
    const name = templateName.trim();
    if (!name) {
      setError("请先填写模板名称");
      return;
    }
    if (templateSaving) return;
    setTemplateSaving(true);
    setError(null);
    try {
      const copied = await createProjectPromptTemplate(projectId, {
        name: `${name}-副本`.slice(0, 128),
        system_prompt: templateSystemPrompt,
        user_prompt_prefix: templateUserPromptPrefix,
        knowledge_setting_keys: templateKnowledgeSettingKeys,
        knowledge_card_ids: templateKnowledgeCardIds,
      });
      const templates = await getProjectPromptTemplates(projectId);
      applyTemplatesSnapshot(templates, copied.id, copied.id);
    } catch (copyError) {
      const message = copyError instanceof Error ? copyError.message : "复制模板失败";
      setError(message);
    } finally {
      setTemplateSaving(false);
    }
  };

  const rollbackTemplateToVersion = async (targetVersion: number) => {
    if (!templateDraftId) {
      setError("请先选择模板");
      return;
    }
    if (!window.confirm(`确认回滚模板到 v${targetVersion} 吗？`)) {
      return;
    }
    if (templateSaving) return;
    setTemplateSaving(true);
    setError(null);
    try {
      await rollbackProjectPromptTemplate(projectId, templateDraftId, {
        target_version: targetVersion,
      });
      const templates = await getProjectPromptTemplates(projectId);
      applyTemplatesSnapshot(templates, activePromptTemplateId, templateDraftId);
      await refreshTemplateRevisions(projectId, templateDraftId);
    } catch (rollbackError) {
      const message = rollbackError instanceof Error ? rollbackError.message : "模板回滚失败";
      setError(message);
    } finally {
      setTemplateSaving(false);
    }
  };

  const saveModelProfile = async () => {
    if (modelProfileSaving) return;
    setModelProfileSaving(true);
    setError(null);
    try {
      const payload: Record<string, string | null> = {
        name: modelProfileName.trim() || null,
        provider: modelProfileProvider,
        base_url: modelProfileBaseUrl.trim() || null,
        model: modelProfileModel.trim() || null,
      };
      const apiKeyInput = modelProfileApiKey.trim();
      if (selectedModelProfileId) {
        if (clearModelProfileApiKey) {
          payload.api_key = "";
        } else if (apiKeyInput) {
          payload.api_key = apiKeyInput;
        }
        const saved = await updateProjectModelProfile(projectId, selectedModelProfileId, payload);
        const profiles = await getProjectModelProfiles(projectId);
        applyModelProfilesSnapshot(profiles, saved.profile_id);
      } else {
        if (apiKeyInput) {
          payload.api_key = apiKeyInput;
        }
        const profileIdInput = modelProfileDraftIdInput.trim();
        if (profileIdInput) {
          payload.profile_id = profileIdInput;
        }
        const saved = await createProjectModelProfile(projectId, payload);
        const profiles = await getProjectModelProfiles(projectId);
        applyModelProfilesSnapshot(profiles, saved.profile_id);
      }
    } catch (profileError) {
      const message = profileError instanceof Error ? profileError.message : "保存模型 profile 失败";
      setError(message);
    } finally {
      setModelProfileSaving(false);
    }
  };

  const deleteModelProfile = async () => {
    if (!selectedModelProfileId) {
      setError("请先选择要删除的模型 profile");
      return;
    }
    if (!window.confirm(`确认删除模型 profile「${selectedModelProfileId}」吗？`)) return;
    if (modelProfileSaving) return;
    setModelProfileSaving(true);
    setError(null);
    try {
      await deleteProjectModelProfile(projectId, selectedModelProfileId);
      const profiles = await getProjectModelProfiles(projectId);
      applyModelProfilesSnapshot(profiles, null);
    } catch (profileError) {
      const message = profileError instanceof Error ? profileError.message : "删除模型 profile 失败";
      setError(message);
    } finally {
      setModelProfileSaving(false);
    }
  };

  const activateModelProfile = async () => {
    if (!selectedModelProfileId) {
      setError("请先选择要激活的模型 profile");
      return;
    }
    if (modelProfileSaving) return;
    setModelProfileSaving(true);
    setError(null);
    try {
      await activateProjectModelProfile(projectId, selectedModelProfileId);
      const profiles = await getProjectModelProfiles(projectId);
      applyModelProfilesSnapshot(profiles, selectedModelProfileId);
    } catch (profileError) {
      const message = profileError instanceof Error ? profileError.message : "激活模型 profile 失败";
      setError(message);
    } finally {
      setModelProfileSaving(false);
    }
  };

  const runConsistencyAudit = async () => {
    if (consistencyAuditRunning) return;
    setConsistencyAuditRunning(true);
    setError(null);
    try {
      const result = await runProjectConsistencyAudit(projectId, {
        run_mode: "sync",
        reason: "manual_ui",
        force: true,
        max_chapters: 3,
      });
      const report = result.report ?? null;
      if (report) {
        setConsistencyAudits((prev) => {
          const next = [report, ...prev.filter((item) => item.report_id !== report.report_id)];
          return next.slice(0, 8);
        });
      } else {
        const latest = await getProjectConsistencyAudits(projectId, 8);
        setConsistencyAudits(latest);
      }
    } catch (auditError) {
      const message = auditError instanceof Error ? auditError.message : "运行创作体检失败";
      setError(message);
    } finally {
      setConsistencyAuditRunning(false);
    }
  };

  handleSendRef.current = handleSend;
  loadLogsRef.current = loadLogs;
  mutateActionRef.current = mutateAction;
  switchChapterRef.current = switchChapter;
  reorderByDragRef.current = reorderByDrag;
  handleOutlineDragStartRef.current = handleOutlineDragStart;
  handleOutlineDragEndRef.current = handleOutlineDragEnd;
  rollbackDraftToVersionRef.current = rollbackDraftToVersion;
  createChapterAndSwitchRef.current = createChapterAndSwitch;
  moveActiveChapterRef.current = moveActiveChapter;
  deleteActiveChapterRef.current = deleteActiveChapter;
  saveDraftSnapshotRef.current = saveDraftSnapshot;
  refreshDraftSnapshotRef.current = refreshDraftSnapshot;
  fillPromptFromSelectionRef.current = fillPromptFromSelection;
  applyAssistantToDraftRef.current = applyAssistantToDraft;
  runConsistencyAuditRef.current = runConsistencyAudit;
  refreshGraphTimelineRef.current = refreshGraphTimeline;

  return (
    <div
      className={`page-shell mode-${uiMode} ${zenMode ? "zen-mode" : ""}`}
      data-writing-theme={writingTheme}
      data-typewriter-focus={typewriterDimmingEnabled ? "on" : "off"}
    >
      <TopBar
        uiMode={uiMode}
        zenMode={zenMode}
        streaming={streaming}
        settingsDialogOpen={settingsDialogOpen}
        assistantDrawerOpen={assistantDrawerOpen}
        onToggleUiMode={toggleUiMode}
        onToggleZenMode={toggleZenMode}
        onOpenSettingsDialog={openSettingsDialog}
        onOpenAssistantDrawer={openAssistantDrawerAndFocusComposer}
        onRefreshSnapshot={handleRefresh}
        onStartNewSession={startNewSession}
      />

      {uiMode === "pro" ? (
        <>
          <WorkspaceStatusBar
            uiMode={uiMode}
            sessionId={sessionId}
            ghostAutoEnabled={ghostAutoEnabled}
            referenceProjectIds={referenceProjectIds}
            retrievalDegraded={retrievalDegraded}
            degradedReasons={degradedReasons}
            lastStreamMetrics={lastStreamMetrics}
          />

          <WorkbenchPanelBar
            visibility={workbenchPanelVisibility}
            onToggle={(panelKey) =>
              setWorkbenchPanelVisibility((prev) => ({
                ...prev,
                [panelKey]: !prev[panelKey],
              }))
            }
          />

          {workbenchPanelVisibility.prompt && debugPromptPanelReady ? (
            <Suspense
              fallback={
                <LazyPanelFallback
                  className="panel prompt-panel"
                  title="Prompt + 知识库面板"
                  detail="正在加载工作台模块..."
                />
              }
            >
              <LazyPromptWorkshopPanel
                activePromptTemplate={activePromptTemplate}
                activePromptTemplateId={activePromptTemplateId}
                templateSaving={templateSaving}
                promptTemplates={promptTemplates}
                onHandleActiveTemplateChange={handleActiveTemplateChange}
                onStartCreateTemplateDraft={startCreateTemplateDraft}
                onCopyTemplateDraft={copyTemplateDraft}
                templateName={templateName}
                setTemplateName={setTemplateName}
                templateSystemPrompt={templateSystemPrompt}
                setTemplateSystemPrompt={setTemplateSystemPrompt}
                templateUserPromptPrefix={templateUserPromptPrefix}
                setTemplateUserPromptPrefix={setTemplateUserPromptPrefix}
                settings={settings}
                templateKnowledgeSettingKeys={templateKnowledgeSettingKeys}
                setTemplateKnowledgeSettingKeys={setTemplateKnowledgeSettingKeys}
                cards={cards}
                templateKnowledgeCardIds={templateKnowledgeCardIds}
                setTemplateKnowledgeCardIds={setTemplateKnowledgeCardIds}
                onSaveTemplateDraft={saveTemplateDraft}
                templateDraftId={templateDraftId}
                onDeleteTemplateDraft={deleteTemplateDraft}
                onRefreshProjectSnapshot={refreshProjectSnapshot}
                projectId={projectId}
                selectedKnowledgeSettings={selectedKnowledgeSettings}
                selectedKnowledgeCards={selectedKnowledgeCards}
                estimatedPromptChars={estimatedPromptChars}
                missingSettingKeys={missingSettingKeys}
                missingCardIds={missingCardIds}
                templateRevisions={templateRevisions}
                templateRevisionsLoading={templateRevisionsLoading}
                onRollbackTemplateToVersion={rollbackTemplateToVersion}
              />
            </Suspense>
          ) : null}

          {workbenchPanelVisibility.planning ? (
            <StoryPlanningPanel
              activeChapterId={activeChapterId}
              volumes={volumes}
              activeVolumeId={activeVolumeId}
              onSelectVolume={setActiveVolumeId}
              onCreateVolume={createVolume}
              onBindChapterToVolume={bindActiveChapterToVolume}
              volumeOutlineDraft={volumeOutlineDraft}
              setVolumeOutlineDraft={setVolumeOutlineDraft}
              onSaveVolumeOutline={saveVolumeOutline}
              sceneBeats={sceneBeats}
              activeSceneBeatId={activeSceneBeatId}
              onSelectSceneBeat={setActiveSceneBeatId}
              newBeatContent={newBeatContent}
              setNewBeatContent={setNewBeatContent}
              onCreateSceneBeat={createBeatForActiveChapter}
              onToggleSceneBeatStatus={toggleBeatStatus}
              onDeleteSceneBeat={deleteBeat}
              foreshadowCards={foreshadowCards}
              overdueForeshadowCards={overdueForeshadowCards}
              foreshadowDraftTitle={foreshadowDraftTitle}
              setForeshadowDraftTitle={setForeshadowDraftTitle}
              foreshadowDraftDescription={foreshadowDraftDescription}
              setForeshadowDraftDescription={setForeshadowDraftDescription}
              onCreateForeshadowCard={createForeshadow}
              onToggleForeshadowStatus={toggleForeshadowStatus}
              onDeleteForeshadowCard={deleteForeshadow}
              busy={planningBusy}
            />
          ) : null}

          {workbenchPanelVisibility.snapshot && proSnapshotPanelReady ? (
            <Suspense
              fallback={
                <LazyPanelFallback className="panel" title="检索与知识快照" detail="正在加载快照视图..." />
              }
            >
              <LazyDebugSnapshotGrid evidence={evidence} settings={settings} cards={cards} />
            </Suspense>
          ) : null}
        </>
      ) : null}

      {uiMode === "writing" ? (
        <>
          <DraftWorkspacePanel
            draftWordCount={draftWordCount}
            draftVersion={draftVersion}
            draftUpdatedAt={draftUpdatedAt}
            activeChapterId={activeChapterId}
            chapters={chapters}
            switchChapterRef={switchChapterRef}
            draftLoading={draftLoading}
            draftSaving={draftSaving}
            draftTitle={draftTitle}
            setDraftTitle={setDraftTitle}
            createChapterAndSwitchRef={createChapterAndSwitchRef}
            moveActiveChapterRef={moveActiveChapterRef}
            canMoveChapterUp={canMoveChapterUp}
            canMoveChapterDown={canMoveChapterDown}
            deleteActiveChapterRef={deleteActiveChapterRef}
            awarenessTags={awarenessTags}
            draftFocusMode={draftFocusMode}
            autoSaveState={autoSaveState}
            autoSaveAt={autoSaveAt}
            typewriterModeEnabled={typewriterModeEnabled}
            localRecoveryNotice={localRecoveryNotice}
            onToggleTypewriterMode={toggleTypewriterMode}
            onToggleDraftFocusMode={toggleDraftFocusMode}
            onToggleZenMode={toggleZenMode}
            zenMode={zenMode}
            uiMode={uiMode}
            draftEditorRef={draftEditorRef}
            editor={editor}
            ghostLoading={ghostLoading}
            ghostText={ghostText}
            ghostError={ghostError}
            ghostAutoEnabled={ghostAutoEnabled}
            onRequestGhostSuggestion={requestGhostSuggestion}
            acceptGhostTextRef={acceptGhostTextRef}
            rejectGhostTextRef={rejectGhostTextRef}
            regenerateGhostTextRef={regenerateGhostTextRef}
            onToggleGhostAuto={toggleGhostAuto}
            saveDraftSnapshotRef={saveDraftSnapshotRef}
            refreshDraftSnapshotRef={refreshDraftSnapshotRef}
            projectId={projectId}
            fillPromptFromSelectionRef={fillPromptFromSelectionRef}
            applyAssistantToDraftRef={applyAssistantToDraftRef}
            selectedDraftText={selectedDraftText}
            latestAssistantReply={latestAssistantReply}
            chapterOutlines={chapterOutlines}
            dragChapterId={dragChapterId}
            handleOutlineDragStartRef={handleOutlineDragStartRef}
            handleOutlineDragEndRef={handleOutlineDragEndRef}
            reorderByDragRef={reorderByDragRef}
            draftRevisions={draftRevisions}
            rollbackDraftToVersionRef={rollbackDraftToVersionRef}
          />
        </>
      ) : null}

      <AssistantDrawer
        projectId={projectId}
        assistantDrawerOpen={assistantDrawerOpen}
        onOpenAssistantDrawer={openAssistantDrawerAndFocusComposer}
        onCloseAssistantDrawer={closeAssistantDrawer}
        onStartNewSession={startNewSession}
        onSwitchSession={switchSession}
        onRenameSession={renameSession}
        onDeleteSession={deleteSession}
        assistantDrawerRef={assistantDrawerRef}
        sessionId={sessionId}
        projectSessions={projectSessions}
        usage={usage}
        messages={messages}
        input={input}
        streaming={streaming}
        composerInputRef={composerInputRef}
        setInput={setInput}
        handleSendRef={handleSendRef}
        sortedActions={sortedActions}
        pendingActionIds={pendingActionIds}
        mutatingActionId={mutatingActionId}
        consistencyAudits={consistencyAudits}
        consistencyAuditRunning={consistencyAuditRunning}
        traceEvents={traceEvents}
        graphTimeline={graphTimeline}
        graphTimelineLoading={graphTimelineLoading}
        graphTimelineChapterIndex={graphTimelineChapterIndex}
        maxChapterIndex={maxChapterIndex}
        setGraphTimelineChapterIndex={setGraphTimelineChapterIndex}
        selectedActionId={selectedActionId}
        actionLogs={actionLogs}
        loadLogsRef={loadLogsRef}
        mutateActionRef={mutateActionRef}
        runConsistencyAuditRef={runConsistencyAuditRef}
      />

      {settingsDialogOpen ? (
        <SettingsDialog
          onCloseSettingsDialog={closeSettingsDialog}
          settingsDialogRef={settingsDialogRef}
          projectId={projectId}
          setProjectId={setProjectId}
          model={model}
          setModel={setModel}
          modelProfiles={modelProfiles}
          selectedModelProfileId={selectedModelProfileId}
          setSelectedModelProfileId={setSelectedModelProfileId}
          modelProfileDraftIdInput={modelProfileDraftIdInput}
          setModelProfileDraftIdInput={setModelProfileDraftIdInput}
          modelProfileName={modelProfileName}
          setModelProfileName={setModelProfileName}
          modelProfileProvider={modelProfileProvider}
          setModelProfileProvider={setModelProfileProvider}
          modelProfileBaseUrl={modelProfileBaseUrl}
          setModelProfileBaseUrl={setModelProfileBaseUrl}
          modelProfileApiKey={modelProfileApiKey}
          setModelProfileApiKey={setModelProfileApiKey}
          modelProfileApiKeyMasked={modelProfileApiKeyMasked}
          clearModelProfileApiKey={clearModelProfileApiKey}
          setClearModelProfileApiKey={setClearModelProfileApiKey}
          modelProfileModel={modelProfileModel}
          setModelProfileModel={setModelProfileModel}
          modelProfileSaving={modelProfileSaving}
          onSaveModelProfile={saveModelProfile}
          onDeleteModelProfile={deleteModelProfile}
          onActivateModelProfile={activateModelProfile}
          onResetModelProfileDraft={resetModelProfileDraft}
          chatTemperatureProfile={chatTemperatureProfile}
          setChatTemperatureProfile={setChatTemperatureProfile}
          ghostTemperatureProfile={ghostTemperatureProfile}
          setGhostTemperatureProfile={setGhostTemperatureProfile}
          temperatureOverrideInput={temperatureOverrideInput}
          setTemperatureOverrideInput={setTemperatureOverrideInput}
          contextWindowProfile={contextWindowProfile}
          setContextWindowProfile={setContextWindowProfile}
          povMode={povMode}
          setPovMode={setPovMode}
          povAnchor={povAnchor}
          setPovAnchor={setPovAnchor}
          ragMode={ragMode}
          setRagMode={setRagMode}
          deterministicFirst={deterministicFirst}
          setDeterministicFirst={setDeterministicFirst}
          thinkingEnabled={thinkingEnabled}
          setThinkingEnabled={setThinkingEnabled}
          referenceProjectInput={referenceProjectInput}
          setReferenceProjectInput={setReferenceProjectInput}
          ghostAutoEnabled={ghostAutoEnabled}
          setGhostAutoEnabled={setGhostAutoEnabled}
          typewriterModeEnabled={typewriterModeEnabled}
          setTypewriterModeEnabled={setTypewriterModeEnabled}
          writingTheme={writingTheme}
          setWritingTheme={setWritingTheme}
          streaming={streaming}
        />
      ) : null}

      {error ? <div className="error-banner">错误：{error}</div> : null}
    </div>
  );
}
