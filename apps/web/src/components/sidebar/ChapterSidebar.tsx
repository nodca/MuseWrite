import { memo } from "react";
import { PanelLeftOpen, PanelLeftClose } from "lucide-react";
import clsx from "clsx";

import { ChapterOutlineList } from "../ChapterOutlineList";
import type { ChapterOutlineEntry } from "../ChapterOutlineList";
import { DraftRevisionList } from "../DraftRevisionList";
import type { ProjectChapterRevision } from "../../types";

export type ChapterSidebarProps = {
  collapsed: boolean;
  onToggleCollapse: () => void;
  totalChapterWords: number;
  todayAddedWords: number;
  chapterCount: number;
  chapterOutlines: ChapterOutlineEntry[];
  activeChapterId: number | null;
  dragChapterId: number | null;
  disabled: boolean;
  onOutlineDragStart: (chapterId: number) => void;
  onOutlineDragEnd: () => void;
  onReorderByDrag: (targetChapterId: number) => Promise<void>;
  onSwitchChapter: (chapterId: number) => Promise<void>;
  draftRevisions: ProjectChapterRevision[];
  onRollbackDraftToVersion: (targetVersion: number) => Promise<void>;
};

export const ChapterSidebar = memo(function ChapterSidebar({
  collapsed,
  onToggleCollapse,
  totalChapterWords,
  todayAddedWords,
  chapterCount,
  chapterOutlines,
  activeChapterId,
  dragChapterId,
  disabled,
  onOutlineDragStart,
  onOutlineDragEnd,
  onReorderByDrag,
  onSwitchChapter,
  draftRevisions,
  onRollbackDraftToVersion,
}: ChapterSidebarProps) {
  return (
    <aside
      className={clsx(
        "flex flex-col border-r border-border-default bg-surface-base transition-all duration-250",
        collapsed ? "w-12" : "w-64",
      )}
    >
      {/* Toggle button */}
      <button
        type="button"
        className="p-3 flex justify-center text-text-secondary hover:text-text-primary"
        onClick={onToggleCollapse}
        aria-label={collapsed ? "展开侧栏" : "收起侧栏"}
      >
        {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
      </button>

      {!collapsed && (
        <>
          {/* KPI grid */}
          <div className="grid grid-cols-3 gap-2 px-3 pb-3">
            <div className="rounded-md bg-surface-primary p-2 text-center border border-border-default">
              <div className="text-lg font-semibold text-text-primary">{totalChapterWords}</div>
              <div className="text-[10px] text-text-secondary">总字数</div>
            </div>
            <div className="rounded-md bg-surface-primary p-2 text-center border border-border-default">
              <div className="text-lg font-semibold text-text-primary">{todayAddedWords}</div>
              <div className="text-[10px] text-text-secondary">今日新增</div>
            </div>
            <div className="rounded-md bg-surface-primary p-2 text-center border border-border-default">
              <div className="text-lg font-semibold text-text-primary">{chapterCount}</div>
              <div className="text-[10px] text-text-secondary">章节数</div>
            </div>
          </div>

          {/* Chapter outline */}
          <div className="flex-1 overflow-y-auto px-3">
            <h3 className="text-xs font-medium text-text-secondary uppercase tracking-wider mb-2">
              章节大纲
            </h3>
            <ChapterOutlineList
              chapterOutlines={chapterOutlines}
              activeChapterId={activeChapterId}
              dragChapterId={dragChapterId}
              disabled={disabled}
              onDragStart={onOutlineDragStart}
              onDragEnd={onOutlineDragEnd}
              onReorder={onReorderByDrag}
              onSelect={onSwitchChapter}
            />
          </div>

          {/* Revision history */}
          <div className="border-t border-border-default px-3 py-2">
            <DraftRevisionList
              draftRevisions={draftRevisions}
              disabled={disabled}
              onRollbackDraftToVersion={onRollbackDraftToVersion}
            />
          </div>
        </>
      )}
    </aside>
  );
});
