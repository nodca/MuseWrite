import { memo } from "react";
import { Plus, ChevronUp, ChevronDown, Trash2 } from "lucide-react";

import { IconButton } from "../shared/IconButton";
import type { ProjectChapter } from "../../types";

export type EditorToolbarProps = {
  activeChapterId: number | null;
  chapters: ProjectChapter[];
  onSwitchChapter: (chapterId: number) => Promise<void>;
  draftTitle: string;
  setDraftTitle: (value: string) => void;
  onCreateChapterAndSwitch: () => Promise<void>;
  onMoveActiveChapter: (direction: "up" | "down") => Promise<void>;
  canMoveChapterUp: boolean;
  canMoveChapterDown: boolean;
  onDeleteActiveChapter: () => Promise<void>;
  onSaveDraftSnapshot: () => Promise<void>;
  draftLoading: boolean;
  draftSaving: boolean;
};

export const EditorToolbar = memo(function EditorToolbar({
  activeChapterId,
  chapters,
  onSwitchChapter,
  draftTitle,
  setDraftTitle,
  onCreateChapterAndSwitch,
  onMoveActiveChapter,
  canMoveChapterUp,
  canMoveChapterDown,
  onDeleteActiveChapter,
  onSaveDraftSnapshot,
  draftLoading,
  draftSaving,
}: EditorToolbarProps) {
  const busy = draftLoading || draftSaving;

  return (
    <div className="flex items-center gap-2 px-3 py-2 border-b border-border-default bg-surface-primary/80">
      <select
        className="text-sm border border-border-default rounded-md px-2 py-1 bg-surface-primary text-text-primary"
        value={activeChapterId ?? ""}
        onChange={(e) => void onSwitchChapter(Number(e.target.value || 0))}
        disabled={busy}
      >
        {chapters.map((ch) => (
          <option key={ch.id} value={ch.id}>
            {ch.chapter_index}. {ch.title}
          </option>
        ))}
      </select>

      <input
        type="text"
        className="flex-1 text-sm border-0 bg-transparent text-text-primary font-medium placeholder:text-text-tertiary focus:outline-none"
        placeholder="章节标题"
        value={draftTitle}
        onChange={(e) => setDraftTitle(e.target.value)}
        disabled={busy || !activeChapterId}
      />

      <div className="flex items-center gap-1">
        <IconButton
          icon={<Plus size={16} />}
          label="新建章节"
          onClick={() => void onCreateChapterAndSwitch()}
          disabled={busy}
        />
        <IconButton
          icon={<ChevronUp size={16} />}
          label="上移"
          onClick={() => void onMoveActiveChapter("up")}
          disabled={busy || !canMoveChapterUp}
        />
        <IconButton
          icon={<ChevronDown size={16} />}
          label="下移"
          onClick={() => void onMoveActiveChapter("down")}
          disabled={busy || !canMoveChapterDown}
        />
        <IconButton
          icon={<Trash2 size={16} />}
          label="删除章节"
          onClick={() => void onDeleteActiveChapter()}
          disabled={busy || !activeChapterId}
        />
        <button
          type="button"
          className="rounded-md bg-accent-primary px-3 py-1.5 text-xs font-medium text-white hover:bg-accent-primary-hover disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          onClick={() => void onSaveDraftSnapshot()}
          disabled={busy}
        >
          {draftSaving ? "保存中..." : "保存"}
        </button>
      </div>
    </div>
  );
});
