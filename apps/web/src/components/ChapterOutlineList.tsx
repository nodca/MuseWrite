import { memo, useMemo } from "react";
import { FixedSizeList, type ListChildComponentProps } from "react-window";

export const CHAPTER_OUTLINE_ROW_HEIGHT = 106;
export const CHAPTER_OUTLINE_MAX_HEIGHT = 424;
export const CHAPTER_OUTLINE_OVERSCAN = 6;

export type ChapterOutlineEntry = {
  id: number;
  chapterIndex: number;
  title: string;
  wordCount: number;
  preview: string;
  updatedAt: string;
  progressPercent: number;
};

export type ChapterOutlineListProps = {
  chapterOutlines: ChapterOutlineEntry[];
  activeChapterId: number | null;
  dragChapterId: number | null;
  disabled: boolean;
  onDragStart: (chapterId: number) => void;
  onDragEnd: () => void;
  onReorder: (targetChapterId: number) => Promise<void>;
  onSelect: (chapterId: number) => Promise<void>;
};

export type ChapterOutlineRowData = {
  chapterOutlines: ChapterOutlineEntry[];
  activeChapterId: number | null;
  dragChapterId: number | null;
  disabled: boolean;
  onDragStart: (chapterId: number) => void;
  onDragEnd: () => void;
  onReorder: (targetChapterId: number) => Promise<void>;
  onSelect: (chapterId: number) => Promise<void>;
};

export const ChapterOutlineRow = memo(function ChapterOutlineRow({
  index,
  style,
  data,
}: ListChildComponentProps<ChapterOutlineRowData>) {
  const item = data.chapterOutlines[index];
  if (!item) return null;
  return (
    <div style={style} className="chapter-outline-row">
      <button
        type="button"
        draggable
        className={`chapter-outline-item ${item.id === data.activeChapterId ? "active" : ""} ${
          item.id === data.dragChapterId ? "dragging" : ""
        }`}
        onDragStart={() => data.onDragStart(item.id)}
        onDragEnd={data.onDragEnd}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          void data.onReorder(item.id);
        }}
        onClick={() => void data.onSelect(item.id)}
        disabled={data.disabled}
      >
        <strong>
          {item.chapterIndex}. {item.title}
        </strong>
        <small>{item.wordCount} 字</small>
        <span>{item.preview}</span>
        <div className="chapter-outline-progress" aria-hidden="true">
          <span style={{ width: `${item.progressPercent}%` }} />
        </div>
      </button>
    </div>
  );
});

export const ChapterOutlineList = memo(function ChapterOutlineList({
  chapterOutlines,
  activeChapterId,
  dragChapterId,
  disabled,
  onDragStart,
  onDragEnd,
  onReorder,
  onSelect,
}: ChapterOutlineListProps) {
  const listHeight = Math.min(
    CHAPTER_OUTLINE_MAX_HEIGHT,
    Math.max(CHAPTER_OUTLINE_ROW_HEIGHT, chapterOutlines.length * CHAPTER_OUTLINE_ROW_HEIGHT)
  );
  const listData = useMemo<ChapterOutlineRowData>(
    () => ({
      chapterOutlines,
      activeChapterId,
      dragChapterId,
      disabled,
      onDragStart,
      onDragEnd,
      onReorder,
      onSelect,
    }),
    [chapterOutlines, activeChapterId, dragChapterId, disabled, onDragStart, onDragEnd, onReorder, onSelect]
  );

  return (
    <div className="chapter-outline-list">
      {chapterOutlines.length === 0 ? <p className="empty">暂无章节</p> : null}
      {chapterOutlines.length > 0 ? (
        <FixedSizeList
          className="chapter-outline-virtual-list"
          height={listHeight}
          width="100%"
          itemCount={chapterOutlines.length}
          itemData={listData}
          itemSize={CHAPTER_OUTLINE_ROW_HEIGHT}
          itemKey={(index: number, data: ChapterOutlineRowData) => data.chapterOutlines[index]?.id ?? index}
          overscanCount={CHAPTER_OUTLINE_OVERSCAN}
        >
          {ChapterOutlineRow}
        </FixedSizeList>
      ) : null}
    </div>
  );
});
