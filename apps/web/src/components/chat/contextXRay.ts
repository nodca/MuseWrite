import type { ContextXRayPayload, EvidenceItem, SettingEntry, StoryCard, UiMessage } from "../../types";

export type ContextXRaySegment =
  | {
      kind: "text";
      text: string;
    }
  | {
      kind: "token";
      text: string;
      binding: {
        canonical: string;
        excerpt: string;
        source: "evidence" | "fallback";
        sourceLabel: "本轮引用" | "设定回退";
      };
    };

type ContextXRayBinding = {
  text: string;
  binding: Extract<ContextXRaySegment, { kind: "token" }>["binding"];
  priority: number;
};

const MIN_ENTITY_TOKEN_LENGTH = 2;
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

export function normalizeEntityToken(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "")
    .replace(/[^\w\u4e00-\u9fff·-]/g, "");
}

export function summarizeKnowledgeSnippet(value: unknown, maxLength = 80): string {
  let raw = "";
  if (typeof value === "string") {
    raw = value;
  } else if (value !== null && value !== undefined) {
    raw = JSON.stringify(value);
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
    return value.flatMap((item) => flattenAliasValues(item, depth + 1));
  }
  if (value && typeof value === "object") {
    return Object.values(value as Record<string, unknown>).flatMap((item) => flattenAliasValues(item, depth + 1));
  }
  return [];
}

export function extractAliasesFromRecord(record: Record<string, unknown>): string[] {
  const aliases: string[] = [];
  Object.entries(record).forEach(([key, value]) => {
    const normalizedKey = key.trim().toLowerCase();
    if (ENTITY_ALIAS_FIELD_KEYS.has(key.trim()) || ENTITY_ALIAS_FIELD_KEYS.has(normalizedKey)) {
      aliases.push(...flattenAliasValues(value));
    }
  });
  return aliases;
}

export function extractNameCandidatesFromRecord(record: Record<string, unknown>): string[] {
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

function pushBinding(
  target: Map<string, ContextXRayBinding>,
  token: string,
  canonical: string,
  excerpt: string,
  source: "evidence" | "fallback",
  priority: number
): void {
  const cleanedToken = String(token || "").trim();
  const cleanedCanonical = String(canonical || "").trim();
  const normalized = normalizeEntityToken(cleanedToken);
  if (!cleanedToken || !cleanedCanonical || normalized.length < MIN_ENTITY_TOKEN_LENGTH) return;

  const candidate: ContextXRayBinding = {
    text: cleanedToken,
    binding: {
      canonical: cleanedCanonical,
      excerpt: excerpt || cleanedCanonical,
      source,
      sourceLabel: source === "evidence" ? "本轮引用" : "设定回退",
    },
    priority,
  };
  const existing = target.get(normalized);
  if (
    !existing ||
    priority > existing.priority ||
    (priority === existing.priority && cleanedToken.length > existing.text.length)
  ) {
    target.set(normalized, candidate);
  }
}

function addEvidenceItemBindings(
  target: Map<string, ContextXRayBinding>,
  item: EvidenceItem,
  sourcePriority: number
): void {
  const title = String(item.title || "").trim();
  const excerpt = String(item.snippet || item.fact || item.title || "").trim();
  if (!title) return;
  pushBinding(target, title, title, excerpt, "evidence", sourcePriority);
}

function buildEvidenceBindings(
  contextXRay: ContextXRayPayload | null | undefined
): Map<string, ContextXRayBinding> {
  const bindings = new Map<string, ContextXRayBinding>();
  const evidence = contextXRay?.evidence;
  if (!evidence) return bindings;

  evidence.sources.dsl.forEach((item) => addEvidenceItemBindings(bindings, item, 400));
  evidence.sources.graph.forEach((item) => addEvidenceItemBindings(bindings, item, 360));
  evidence.sources.rag.forEach((item) => addEvidenceItemBindings(bindings, item, 320));
  if (typeof evidence.policy.anchor === "string" && evidence.policy.anchor.trim()) {
    pushBinding(bindings, evidence.policy.anchor, evidence.policy.anchor, evidence.policy.anchor, "evidence", 340);
  }
  return bindings;
}

function buildFallbackBindings(settings: SettingEntry[], cards: StoryCard[]): Map<string, ContextXRayBinding> {
  const bindings = new Map<string, ContextXRayBinding>();

  settings.forEach((item) => {
    const canonical = item.key.trim();
    const summary = summarizeKnowledgeSnippet(item.value);
    pushBinding(bindings, canonical, canonical, summary, "fallback", 220);
    const valueObj = item.value && typeof item.value === "object" ? (item.value as Record<string, unknown>) : {};
    extractNameCandidatesFromRecord(valueObj).forEach((name) =>
      pushBinding(bindings, name, canonical, summary, "fallback", 210)
    );
    extractAliasesFromRecord(valueObj).forEach((alias) =>
      pushBinding(bindings, alias, canonical, summary, "fallback", 205)
    );
  });

  cards.forEach((card) => {
    const canonical = (card.title || "").trim();
    if (!canonical) return;
    const summary = summarizeKnowledgeSnippet(card.content);
    pushBinding(bindings, canonical, canonical, summary, "fallback", 260);
    const contentObj =
      card.content && typeof card.content === "object" ? (card.content as Record<string, unknown>) : {};
    extractNameCandidatesFromRecord(contentObj).forEach((name) =>
      pushBinding(bindings, name, canonical, summary, "fallback", 250)
    );
    extractAliasesFromRecord(contentObj).forEach((alias) =>
      pushBinding(bindings, alias, canonical, summary, "fallback", 245)
    );
  });

  return bindings;
}

function resolveBindings(
  message: UiMessage,
  settings: SettingEntry[],
  cards: StoryCard[]
): ContextXRayBinding[] {
  const evidenceBindings = buildEvidenceBindings(message.contextXRay);
  const fallbackBindings = buildFallbackBindings(settings, cards);
  fallbackBindings.forEach((binding, key) => {
    if (!evidenceBindings.has(key)) {
      evidenceBindings.set(key, binding);
    }
  });
  return Array.from(evidenceBindings.values()).sort((left, right) => {
    if (right.text.length !== left.text.length) {
      return right.text.length - left.text.length;
    }
    return right.priority - left.priority;
  });
}

export function buildContextXRaySegments(
  message: UiMessage,
  settings: SettingEntry[],
  cards: StoryCard[]
): ContextXRaySegment[] {
  const content = String(message.content || "");
  if (!content) return [{ kind: "text", text: "" }];

  const bindings = resolveBindings(message, settings, cards);
  if (bindings.length === 0) {
    return [{ kind: "text", text: content }];
  }

  const segments: ContextXRaySegment[] = [];
  let cursor = 0;
  while (cursor < content.length) {
    let matched: ContextXRayBinding | null = null;
    for (const binding of bindings) {
      if (content.startsWith(binding.text, cursor)) {
        matched = binding;
        break;
      }
    }
    if (!matched) {
      const previous = segments[segments.length - 1];
      if (previous && previous.kind === "text") {
        previous.text += content[cursor];
      } else {
        segments.push({ kind: "text", text: content[cursor] });
      }
      cursor += 1;
      continue;
    }
    segments.push({
      kind: "token",
      text: matched.text,
      binding: matched.binding,
    });
    cursor += matched.text.length;
  }

  return segments;
}
