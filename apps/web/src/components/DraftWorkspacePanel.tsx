import { memo, useEffect, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent } from "react";
import { EditorContent, type Editor } from "@tiptap/react";
import { FixedSizeList, type ListChildComponentProps } from "react-window";

import type { DraftAutoSaveState, ProjectChapter, ProjectChapterRevision } from "../types";

const CHAPTER_OUTLINE_ROW_HEIGHT = 106;
const CHAPTER_OUTLINE_MAX_HEIGHT = 424;
const CHAPTER_OUTLINE_OVERSCAN = 6;

function formatDateTime(value?: string | null): string {
  if (!value) return "未保存";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

export type ChapterOutlineEntry = {
  id: number;
  chapterIndex: number;
  title: string;
  wordCount: number;
  preview: string;
  updatedAt: string;
  progressPercent: number;
};

type ChapterOutlineListProps = {
  chapterOutlines: ChapterOutlineEntry[];
  activeChapterId: number | null;
  dragChapterId: number | null;
  disabled: boolean;
  onDragStart: (chapterId: number) => void;
  onDragEnd: () => void;
  onReorder: (targetChapterId: number) => Promise<void>;
  onSelect: (chapterId: number) => Promise<void>;
};

type ChapterOutlineRowData = {
  chapterOutlines: ChapterOutlineEntry[];
  activeChapterId: number | null;
  dragChapterId: number | null;
  disabled: boolean;
  onDragStart: (chapterId: number) => void;
  onDragEnd: () => void;
  onReorder: (targetChapterId: number) => Promise<void>;
  onSelect: (chapterId: number) => Promise<void>;
};

const ChapterOutlineRow = memo(function ChapterOutlineRow({
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

const ChapterOutlineList = memo(function ChapterOutlineList({
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

type DraftRevisionListProps = {
  draftRevisions: ProjectChapterRevision[];
  disabled: boolean;
  onRollbackDraftToVersion: (targetVersion: number) => Promise<void>;
};

const DraftRevisionList = memo(function DraftRevisionList({
  draftRevisions,
  disabled,
  onRollbackDraftToVersion,
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
                onClick={() => void onRollbackDraftToVersion(revision.version)}
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

export type DraftWorkspacePanelProps = {
  draftWordCount: number;
  draftVersion: number;
  draftUpdatedAt: string | null;
  totalChapterWords: number;
  todayAddedWords: number;
  onboardingChecklist: {
    hasRoleCard: boolean;
    hasDraftParagraph: boolean;
    hasOpenedAssistant: boolean;
    hasAppliedSuggestion: boolean;
  };
  activeChapterId: number | null;
  chapters: ProjectChapter[];
  onSwitchChapter: (chapterId: number) => Promise<void>;
  draftLoading: boolean;
  draftSaving: boolean;
  draftTitle: string;
  setDraftTitle: (value: string) => void;
  onCreateChapterAndSwitch: () => Promise<void>;
  onMoveActiveChapter: (direction: "up" | "down") => Promise<void>;
  canMoveChapterUp: boolean;
  canMoveChapterDown: boolean;
  onDeleteActiveChapter: () => Promise<void>;
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
  onSaveDraftSnapshot: () => Promise<void>;
  onRefreshDraftSnapshot: (nextProjectId: number, preferredChapterId?: number | null) => Promise<void>;
  projectId: number;
  onFillPromptFromSelection: (mode: "polish" | "expand") => Promise<void>;
  onApplyAssistantToDraft: (mode: "insert" | "replace") => void;
  selectedDraftText: string;
  latestAssistantReply: string;
  chapterOutlines: ChapterOutlineEntry[];
  dragChapterId: number | null;
  onOutlineDragStart: (chapterId: number) => void;
  onOutlineDragEnd: () => void;
  onReorderByDrag: (targetChapterId: number) => Promise<void>;
  draftRevisions: ProjectChapterRevision[];
  onRollbackDraftToVersion: (targetVersion: number) => Promise<void>;
};

export const DraftWorkspacePanel = memo(function DraftWorkspacePanel({
  draftWordCount,
  draftVersion,
  draftUpdatedAt,
  totalChapterWords,
  todayAddedWords,
  onboardingChecklist,
  activeChapterId,
  chapters,
  onSwitchChapter,
  draftLoading,
  draftSaving,
  draftTitle,
  setDraftTitle,
  onCreateChapterAndSwitch,
  onMoveActiveChapter,
  canMoveChapterUp,
  canMoveChapterDown,
  onDeleteActiveChapter,
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
  onSaveDraftSnapshot,
  onRefreshDraftSnapshot,
  projectId,
  onFillPromptFromSelection,
  onApplyAssistantToDraft,
  selectedDraftText,
  latestAssistantReply,
  chapterOutlines,
  dragChapterId,
  onOutlineDragStart,
  onOutlineDragEnd,
  onReorderByDrag,
  draftRevisions,
  onRollbackDraftToVersion,
}: DraftWorkspacePanelProps) {
  const [chapterSwitching, setChapterSwitching] = useState(false);
  const [selectionMenu, setSelectionMenu] = useState<{ x: number; y: number } | null>(null);
  const selectionMenuRef = useRef<HTMLDivElement | null>(null);
  const selectionMenuButtonsRef = useRef<Array<HTMLButtonElement | null>>([]);

  useEffect(() => {
    if (!activeChapterId || typeof window === "undefined") return;
    setChapterSwitching(true);
    const timer = window.setTimeout(() => setChapterSwitching(false), 200);
    return () => window.clearTimeout(timer);
  }, [activeChapterId]);

  useEffect(() => {
    if (selectedDraftText.trim()) return;
    setSelectionMenu(null);
  }, [selectedDraftText]);

  useEffect(() => {
    if (!selectionMenu || typeof window === "undefined") return;
    const close = () => setSelectionMenu(null);
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    window.addEventListener("click", close);
    return () => {
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
      window.removeEventListener("click", close);
    };
  }, [selectionMenu]);

  useEffect(() => {
    if (!selectionMenu) return;
    const timer = window.setTimeout(() => {
      selectionMenuButtonsRef.current[0]?.focus();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [selectionMenu]);

  const handleEditorContextMenu = (event: MouseEvent<HTMLDivElement>) => {
    if (!selectedDraftText.trim()) {
      setSelectionMenu(null);
      return;
    }
    event.preventDefault();
    setSelectionMenu({ x: event.clientX, y: event.clientY });
  };

  const runSelectionTool = (mode: "polish" | "expand") => {
    void onFillPromptFromSelection(mode);
    setSelectionMenu(null);
  };

  const handleSelectionMenuKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!selectionMenu) return;
    const buttons = selectionMenuButtonsRef.current.filter((item): item is HTMLButtonElement => Boolean(item));
    if (buttons.length === 0) return;
    const currentIndex = Math.max(
      0,
      buttons.findIndex((item) => item === (document.activeElement as HTMLButtonElement | null))
    );

    if (event.key === "Escape") {
      event.preventDefault();
      setSelectionMenu(null);
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      const next = (currentIndex + 1) % buttons.length;
      buttons[next]?.focus();
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      const next = (currentIndex - 1 + buttons.length) % buttons.length;
      buttons[next]?.focus();
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      (document.activeElement as HTMLButtonElement | null)?.click();
    }
  };

  const onboardingItems = [
    { id: "role", label: "创建第一个角色卡片", done: onboardingChecklist.hasRoleCard },
    { id: "draft", label: "写下第一段正文", done: onboardingChecklist.hasDraftParagraph },
    { id: "assistant", label: "尝试呼出助手", done: onboardingChecklist.hasOpenedAssistant },
    { id: "apply", label: "接受一次 AI 建议", done: onboardingChecklist.hasAppliedSuggestion },
  ];
  const onboardingCompletedCount = onboardingItems.filter((item) => item.done).length;
  const showOnboardingChecklist = onboardingCompletedCount < onboardingItems.length;

  return (
    <section className="panel draft-panel">
      <div className="panel-title">
        <h2>正文工作区</h2>
        <small>
          {draftWordCount} 字 · v{draftVersion} · {formatDateTime(draftUpdatedAt)}
        </small>
      </div>
      <div className="writing-progress-summary" aria-live="polite">
        <article className="writing-progress-kpi"><small>章节总字数</small><strong>{totalChapterWords}</strong></article>
        <article className="writing-progress-kpi"><small>今日新增字数</small><strong>{todayAddedWords}</strong></article>
        <article className="writing-progress-kpi"><small>章节数</small><strong>{chapterOutlines.length}</strong></article>
      </div>
      {showOnboardingChecklist ? <section className="onboarding-checklist" aria-live="polite"><div className="onboarding-checklist-head"><strong>新手任务清单</strong><small>{`${onboardingCompletedCount}/${onboardingItems.length} 已完成`}</small></div><ul>{onboardingItems.map((item) => <li key={item.id} className={item.done ? "done" : ""}><span>{item.done ? "✓" : "○"}</span><span>{item.label}</span></li>)}</ul></section> : null}
      <div className="draft-chapter-row">
        <label className="inline-field">章节<select value={activeChapterId ?? ""} onChange={(event) => void onSwitchChapter(Number(event.target.value || 0))} disabled={draftLoading || draftSaving}>{chapters.map((chapter) => <option key={chapter.id} value={chapter.id}>{chapter.chapter_index}. {chapter.title}</option>)}</select></label>
        <label className="inline-field">章节标题<input type="text" value={draftTitle} onChange={(event) => setDraftTitle(event.target.value)} disabled={draftLoading || draftSaving || !activeChapterId} /></label>
        <button className="btn ghost tiny" onClick={() => void onCreateChapterAndSwitch()} disabled={draftLoading || draftSaving}>新建章节</button>
        <button className="btn ghost tiny" onClick={() => void onMoveActiveChapter("up")} disabled={draftLoading || draftSaving || !canMoveChapterUp}>上移</button>
        <button className="btn ghost tiny" onClick={() => void onMoveActiveChapter("down")} disabled={draftLoading || draftSaving || !canMoveChapterDown}>下移</button>
        <button className="btn ghost tiny" onClick={() => void onDeleteActiveChapter()} disabled={draftLoading || draftSaving || !activeChapterId}>删除章节</button>
      </div>
      {chapters.length === 0 ? <section className="draft-empty-state" aria-live="polite"><h3>从第一章开始</h3><p>当前项目还没有章节。点击下面按钮即可创建第一章并进入写作。</p><button type="button" className="btn primary" onClick={() => void onCreateChapterAndSwitch()} disabled={draftLoading || draftSaving}>点击新建章节开始写作</button></section> : <><div className="awareness-strip" aria-live="polite"><small>AI 当前认知</small><div className="awareness-tags">{awarenessTags.length === 0 ? <span className="awareness-empty">等待会话建立上下文</span> : awarenessTags.map((tag) => <span key={tag} className="awareness-tag">#{tag}</span>)}</div></div><div className={`draft-editor-shell ${draftFocusMode ? "focus-mode" : ""} ${chapterSwitching ? "chapter-switching" : ""}`}><div className="draft-editor-toolbar"><div className="draft-editor-status"><small>编辑模式：{draftFocusMode ? "专注" : "标准"} · 零感保存：{autoSaveState === "pending" ? "等待中" : null}{autoSaveState === "saving" ? "保存中" : null}{autoSaveState === "saved" ? `已保存(${formatDateTime(autoSaveAt)})` : null}{autoSaveState === "error" ? "失败" : null}{autoSaveState === "idle" ? "空闲" : null}</small><small>打字机滚动：{typewriterModeEnabled ? "开启" : "关闭"}</small>{localRecoveryNotice ? <small className="recovery-note">{localRecoveryNotice}</small> : null}</div><div className="draft-toolbar-actions"><button type="button" className="btn ghost tiny" onClick={onToggleTypewriterMode} disabled={draftLoading || !activeChapterId}>{typewriterModeEnabled ? "关闭打字机滚动" : "开启打字机滚动"}</button><button type="button" className="btn ghost tiny" onClick={onToggleDraftFocusMode} disabled={draftLoading || !activeChapterId}>{draftFocusMode ? "退出专注" : "进入专注"}</button><button type="button" className="btn ghost tiny" onClick={onToggleZenMode} disabled={draftLoading || !activeChapterId || uiMode !== "writing"} aria-pressed={zenMode}>{zenMode ? "退出沉浸" : "进入沉浸"}</button></div></div><div className="draft-editor-context" onContextMenu={handleEditorContextMenu}><EditorContent ref={draftEditorRef} editor={editor} className={`draft-editor ${draftLoading || !activeChapterId ? "disabled" : ""}`} />{selectionMenu ? <div ref={selectionMenuRef} className="selection-context-menu" role="menu" tabIndex={-1} onKeyDown={handleSelectionMenuKeyDown} style={{ left: `${selectionMenu.x}px`, top: `${selectionMenu.y}px` }}><button ref={(el) => { selectionMenuButtonsRef.current[0] = el; }} className="btn ghost tiny" onClick={() => runSelectionTool("polish")} role="menuitem">✨ 润色选中</button><button ref={(el) => { selectionMenuButtonsRef.current[1] = el; }} className="btn ghost tiny" onClick={() => runSelectionTool("expand")} role="menuitem">🧩 扩写选中</button></div> : null}</div></div><div className="draft-actions"><button className="btn primary tiny" onClick={() => void onSaveDraftSnapshot()} disabled={draftSaving || draftLoading}>{draftSaving ? "保存中..." : "保存正文"}</button><button className="btn ghost tiny" onClick={() => void onRefreshDraftSnapshot(projectId, activeChapterId)} disabled={draftSaving || draftLoading}>拉取服务器版本</button><button className="btn primary tiny" onClick={() => onApplyAssistantToDraft("insert")}>插入助手回复</button><button className="btn ghost tiny" onClick={() => onApplyAssistantToDraft("replace")}>替换选中为助手回复</button></div><p className="draft-hint">{draftLoading ? "正在加载服务器正文..." : `已选 ${selectedDraftText.length} 字`}{latestAssistantReply ? " · 最近一条助手回复可写回正文" : " · 暂无助手回复可写回"}</p><div className="chapter-outline"><div className="panel-title sub"><h3>章节大纲</h3><small>拖拽排序 + 点击切章</small></div><ChapterOutlineList chapterOutlines={chapterOutlines} activeChapterId={activeChapterId} dragChapterId={dragChapterId} disabled={draftLoading || draftSaving} onDragStart={onOutlineDragStart} onDragEnd={onOutlineDragEnd} onReorder={onReorderByDrag} onSelect={onSwitchChapter} /></div><DraftRevisionList draftRevisions={draftRevisions} disabled={draftSaving || draftLoading} onRollbackDraftToVersion={onRollbackDraftToVersion} /></>}
    </section>
  );
});
