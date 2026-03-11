import { Extension } from "@tiptap/core";
import { Plugin, PluginKey, type Transaction } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";

export type DiffSuggestion = {
  id: string;
  from: number;
  to: number;
  originalText: string;
  suggestedText: string;
};

type DiffPluginState = {
  suggestions: DiffSuggestion[];
};

type DiffPluginMeta =
  | {
      type: "set";
      suggestions: DiffSuggestion[];
    }
  | {
      type: "clear";
    }
  | {
      type: "reject";
      id: string;
    };

const diffPluginKey = new PluginKey<DiffPluginState>("semantic-diff-suggestions");

function normalizeEditorText(value: string): string {
  const normalized = (value || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n").replace(/\u00a0/g, " ");
  if (normalized.endsWith("\n")) {
    return normalized.slice(0, -1);
  }
  return normalized;
}

function toParagraphContent(text: string): Array<Record<string, unknown>> {
  const normalized = normalizeEditorText(text);
  const lines = normalized.length > 0 ? normalized.split("\n") : [""];
  return lines.map((line) => {
    if (!line) return { type: "paragraph" };
    return {
      type: "paragraph",
      content: [{ type: "text", text: line }],
    };
  });
}

function buildSuggestionId(): string {
  const rnd = Math.random().toString(16).slice(2, 10);
  return `diff-${Date.now().toString(16)}-${rnd}`;
}

export function buildDiffSuggestions(
  from: number,
  to: number,
  originalText: string,
  suggestedText: string
): DiffSuggestion[] {
  const safeFrom = Number.isFinite(from) ? Math.max(0, Math.floor(from)) : 0;
  const safeTo = Number.isFinite(to) ? Math.max(safeFrom, Math.floor(to)) : safeFrom;
  const original = normalizeEditorText(originalText);
  const suggested = normalizeEditorText(suggestedText);
  if (!original && !suggested) return [];
  if (original === suggested) return [];
  if (safeTo <= safeFrom) return [];
  return [
    {
      id: buildSuggestionId(),
      from: safeFrom,
      to: safeTo,
      originalText: original,
      suggestedText: suggested,
    },
  ];
}

function mapSuggestionsThroughTransaction(suggestions: DiffSuggestion[], tr: Transaction): DiffSuggestion[] {
  if (!tr.docChanged) return suggestions;
  const next: DiffSuggestion[] = [];
  for (const item of suggestions) {
    const mappedFrom = tr.mapping.mapResult(item.from, 1);
    const mappedTo = tr.mapping.mapResult(item.to, -1);
    if (mappedFrom.deleted || mappedTo.deleted) continue;
    const from = mappedFrom.pos;
    const to = mappedTo.pos;
    if (to <= from) continue;
    next.push({
      ...item,
      from,
      to,
    });
  }
  return next;
}

function dispatchDiffActionEvent(target: HTMLElement, action: "accept" | "reject", id: string) {
  target.dispatchEvent(
    new CustomEvent("semantic-diff-action", {
      bubbles: true,
      detail: {
        action,
        id,
      },
    })
  );
}

declare module "@tiptap/core" {
  interface Commands<ReturnType> {
    semanticDiff: {
      setDiffSuggestions: (suggestions: DiffSuggestion[]) => ReturnType;
      clearDiffSuggestions: () => ReturnType;
      acceptDiffSuggestion: (id: string) => ReturnType;
      rejectDiffSuggestion: (id: string) => ReturnType;
    };
  }
}

export const SemanticDiffExtension = Extension.create({
  name: "semanticDiff",
  addCommands() {
    return {
      setDiffSuggestions:
        (suggestions: DiffSuggestion[]) =>
        ({ tr, dispatch }) => {
          if (!dispatch) return true;
          tr.setMeta(diffPluginKey, { type: "set", suggestions } satisfies DiffPluginMeta);
          dispatch(tr);
          return true;
        },
      clearDiffSuggestions:
        () =>
        ({ tr, dispatch }) => {
          if (!dispatch) return true;
          tr.setMeta(diffPluginKey, { type: "clear" } satisfies DiffPluginMeta);
          dispatch(tr);
          return true;
        },
      acceptDiffSuggestion:
        (id: string) =>
        ({ state, chain }) => {
          const pluginState = diffPluginKey.getState(state);
          const target = pluginState?.suggestions.find((item) => item.id === id) ?? null;
          if (!target) return false;
          const insertion = toParagraphContent(target.suggestedText);
          return chain()
            .focus()
            .insertContentAt({ from: target.from, to: target.to }, insertion)
            .command(({ tr }) => {
              tr.setMeta(diffPluginKey, { type: "reject", id } satisfies DiffPluginMeta);
              return true;
            })
            .run();
        },
      rejectDiffSuggestion:
        (id: string) =>
        ({ tr, dispatch }) => {
          if (!dispatch) return true;
          tr.setMeta(diffPluginKey, { type: "reject", id } satisfies DiffPluginMeta);
          dispatch(tr);
          return true;
        },
    };
  },
  addProseMirrorPlugins() {
    return [
      new Plugin<DiffPluginState>({
        key: diffPluginKey,
        state: {
          init: () => ({ suggestions: [] }),
          apply(tr, prev) {
            const meta = tr.getMeta(diffPluginKey) as DiffPluginMeta | undefined;
            if (meta) {
              if (meta.type === "clear") {
                return { suggestions: [] };
              }
              if (meta.type === "set") {
                return { suggestions: [...meta.suggestions] };
              }
              if (meta.type === "reject") {
                return { suggestions: prev.suggestions.filter((item) => item.id !== meta.id) };
              }
            }
            return {
              suggestions: mapSuggestionsThroughTransaction(prev.suggestions, tr),
            };
          },
        },
        props: {
          decorations(state) {
            const pluginState = diffPluginKey.getState(state);
            const suggestions = pluginState?.suggestions ?? [];
            if (suggestions.length === 0) return null;
            const decos: Decoration[] = [];
            for (const item of suggestions) {
              if (item.to <= item.from) continue;
              decos.push(Decoration.inline(item.from, item.to, { class: "diff-original-text" }));
              decos.push(
                Decoration.widget(
                  item.to,
                  () => {
                    const widget = document.createElement("span");
                    widget.className = "diff-suggestion-widget";
                    widget.setAttribute("data-diff-suggestion-id", item.id);

                    const suggested = document.createElement("span");
                    suggested.className = "diff-suggested-text";
                    suggested.textContent = item.suggestedText.length > 1600 ? `${item.suggestedText.slice(0, 1600)}…` : item.suggestedText;
                    widget.appendChild(suggested);

                    const actions = document.createElement("span");
                    actions.className = "diff-suggestion-actions";

                    const acceptBtn = document.createElement("button");
                    acceptBtn.type = "button";
                    acceptBtn.className = "diff-suggestion-action accept";
                    acceptBtn.textContent = "接受";
                    acceptBtn.addEventListener("click", (event) => {
                      event.preventDefault();
                      dispatchDiffActionEvent(widget, "accept", item.id);
                    });

                    const rejectBtn = document.createElement("button");
                    rejectBtn.type = "button";
                    rejectBtn.className = "diff-suggestion-action reject";
                    rejectBtn.textContent = "忽略";
                    rejectBtn.addEventListener("click", (event) => {
                      event.preventDefault();
                      dispatchDiffActionEvent(widget, "reject", item.id);
                    });

                    actions.appendChild(acceptBtn);
                    actions.appendChild(rejectBtn);
                    widget.appendChild(actions);
                    return widget;
                  },
                  {
                    key: `diff-suggestion:${item.id}`,
                    side: 1,
                    stopEvent: () => true,
                  }
                )
              );
            }
            return DecorationSet.create(state.doc, decos);
          },
        },
      }),
    ];
  },
});

