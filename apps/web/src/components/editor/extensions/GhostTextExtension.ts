import { Extension } from "@tiptap/core";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";

type GhostTextPluginState = {
  text: string;
};

type GhostTextMeta =
  | {
      type: "set";
      text: string;
    }
  | {
      type: "clear";
    };

const ghostTextPluginKey = new PluginKey<GhostTextPluginState>("ghost-text-widget");

function normalizeGhostText(value: string): string {
  return String(value || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
}

function ghostWidgetKey(text: string): string {
  // Keep this cheap and stable: we only need to avoid stale DOM reuse when the
  // ghost text changes (e.g. accept-one-word leaves the remainder).
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash * 31 + text.charCodeAt(i)) | 0;
  }
  return `ghost-text:${hash >>> 0}:${text.length}`;
}

declare module "@tiptap/core" {
  interface Commands<ReturnType> {
    ghostText: {
      setGhostText: (text: string) => ReturnType;
      clearGhostText: () => ReturnType;
    };
  }
}

export const GhostTextExtension = Extension.create({
  name: "ghostText",
  addCommands() {
    return {
      setGhostText:
        (text: string) =>
        ({ tr, dispatch }) => {
          const normalized = normalizeGhostText(text);
          if (!dispatch) return true;
          tr.setMeta(ghostTextPluginKey, { type: "set", text: normalized } satisfies GhostTextMeta);
          dispatch(tr);
          return true;
        },
      clearGhostText:
        () =>
        ({ tr, dispatch }) => {
          if (!dispatch) return true;
          tr.setMeta(ghostTextPluginKey, { type: "clear" } satisfies GhostTextMeta);
          dispatch(tr);
          return true;
        },
    };
  },
  addProseMirrorPlugins() {
    return [
      new Plugin<GhostTextPluginState>({
        key: ghostTextPluginKey,
        state: {
          init: () => ({ text: "" }),
          apply(tr, prev) {
            const meta = tr.getMeta(ghostTextPluginKey) as GhostTextMeta | undefined;
            if (!meta) return prev;
            if (meta.type === "clear") return { text: "" };
            return { text: meta.text };
          },
        },
        props: {
          decorations(state) {
            const pluginState = ghostTextPluginKey.getState(state);
            const ghostText = pluginState?.text ?? "";
            if (!ghostText) return null;

            const selection = state.selection;
            if (!selection.empty) return null;

            const displayText = ghostText.length > 2400 ? `${ghostText.slice(0, 2400)}…` : ghostText;
            const deco = Decoration.widget(
              selection.from,
              () => {
                const dom = document.createElement("span");
                dom.className = "ghost-text-widget";
                dom.textContent = displayText;
                dom.setAttribute("contenteditable", "false");
                return dom;
              },
              {
                // ProseMirror will reuse widget DOM nodes with the same key even
                // when their content changes. Include the displayed text to
                // prevent stale widgets.
                key: ghostWidgetKey(displayText),
                side: 1,
                stopEvent: () => true,
              }
            );
            return DecorationSet.create(state.doc, [deco]);
          },
        },
      }),
    ];
  },
});
