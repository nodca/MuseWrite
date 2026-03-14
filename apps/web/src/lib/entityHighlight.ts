import { Extension } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import {
  extractAliasesFromRecord,
  extractNameCandidatesFromRecord,
  normalizeEntityToken,
  summarizeKnowledgeSnippet,
} from "../components/chat/contextXRay";
import type { EvidencePayload, SettingEntry, StoryCard } from "../types";

export type EntityHighlightHint = {
  token: string;
  canonical: string;
  summary: string;
};

export type EntityHintPluginState = {
  hints: EntityHighlightHint[];
  regex: RegExp | null;
  hintMap: Map<string, EntityHighlightHint>;
  decorations: DecorationSet;
};

export const ENTITY_HINT_LIMIT = 120;
export const entityHintPluginKey = new PluginKey<EntityHintPluginState>("entity-inline-hints");

export function collectEntityHighlightHints(settingsList: SettingEntry[], cardsList: StoryCard[]): EntityHighlightHint[] {
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

export function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function buildEntityHintLookup(hints: EntityHighlightHint[]): {
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

export function buildEntityDecorations(
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

export const EntityInlineHintExtension = Extension.create({
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

export function normalizeAwarenessTag(value: string): string {
  const cleaned = value
    .replace(/^[#\[\(【]+/, "")
    .replace(/[\]\)】]+$/, "")
    .trim();
  if (!cleaned) return "";
  return cleaned.replace(/\s+/g, " ").slice(0, 24);
}

export function collectAwarenessTags(
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