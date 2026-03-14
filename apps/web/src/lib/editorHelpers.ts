import { Editor, type JSONContent } from "@tiptap/react";
import { normalizeEditorText } from "../utils/formatting";

export function toEditorDoc(text: string): JSONContent {
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

export function readEditorText(editor: Editor): string {
  return normalizeEditorText(editor.getText({ blockSeparator: "\n" }));
}

export function readSelectedText(editor: Editor): string {
  const { from, to, empty } = editor.state.selection;
  if (empty) return "";
  const content = editor.state.doc.textBetween(from, to, "\n", "\n");
  return normalizeEditorText(content).trim();
}
