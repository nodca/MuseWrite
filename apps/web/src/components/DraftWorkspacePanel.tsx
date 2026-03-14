import { memo, useEffect, useRef, useState, type KeyboardEvent, type MouseEvent } from "react";
import { EditorContent, type Editor } from "@tiptap/react";

import type { DraftAutoSaveState, ProjectChapter, ProjectChapterRevision } from "../types";

import { EditorLayout } from "./layout/EditorLayout";
import { ChapterSidebar } from "./sidebar/ChapterSidebar";
import { EditorToolbar } from "./editor/EditorToolbar";
import { EditorStatusBar } from "./editor/EditorStatusBar";

export type { ChapterOutlineEntry } from "./ChapterOutlineList";
import type { ChapterOutlineEntry } from "./ChapterOutlineList";

function formatDateTime(value?: string | null): string {
  if (!value) return "未保存";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

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
  canEnterZenMode: boolean;
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
  canEnterZenMode,
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
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
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
    <EditorLayout
      sidebar={
        <ChapterSidebar
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed((p) => !p)}
          totalChapterWords={totalChapterWords}
          todayAddedWords={todayAddedWords}
          chapterCount={chapterOutlines.length}
          chapterOutlines={chapterOutlines}
          activeChapterId={activeChapterId}
          dragChapterId={dragChapterId}
          disabled={draftLoading || draftSaving}
          onOutlineDragStart={onOutlineDragStart}
          onOutlineDragEnd={onOutlineDragEnd}
          onReorderByDrag={onReorderByDrag}
          onSwitchChapter={onSwitchChapter}
          draftRevisions={draftRevisions}
          onRollbackDraftToVersion={onRollbackDraftToVersion}
        />
      }
      toolbar={
        <EditorToolbar
          activeChapterId={activeChapterId}
          chapters={chapters}
          onSwitchChapter={onSwitchChapter}
          draftTitle={draftTitle}
          setDraftTitle={setDraftTitle}
          onCreateChapterAndSwitch={onCreateChapterAndSwitch}
          onMoveActiveChapter={onMoveActiveChapter}
          canMoveChapterUp={canMoveChapterUp}
          canMoveChapterDown={canMoveChapterDown}
          onDeleteActiveChapter={onDeleteActiveChapter}
          onSaveDraftSnapshot={onSaveDraftSnapshot}
          draftLoading={draftLoading}
          draftSaving={draftSaving}
        />
      }
      editor={
        <>
          {showOnboardingChecklist ? (
            <section className="onboarding-checklist" aria-live="polite">
              <div className="onboarding-checklist-head">
                <strong>新手任务清单</strong>
                <small>{`${onboardingCompletedCount}/${onboardingItems.length} 已完成`}</small>
              </div>
              <ul>
                {onboardingItems.map((item) => (
                  <li key={item.id} className={item.done ? "done" : ""}>
                    <span>{item.done ? "\u2713" : "\u25CB"}</span>
                    <span>{item.label}</span>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}

          {chapters.length === 0 ? (
            <section className="draft-empty-state" aria-live="polite">
              <h3>从第一章开始</h3>
              <p>当前项目还没有章节。点击下面按钮即可创建第一章并进入写作。</p>
              <button
                type="button"
                className="btn primary"
                onClick={() => void onCreateChapterAndSwitch()}
                disabled={draftLoading || draftSaving}
              >
                点击新建章节开始写作
              </button>
            </section>
          ) : (
            <>
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

              <div
                className={`draft-editor-shell ${draftFocusMode ? "focus-mode" : ""} ${chapterSwitching ? "chapter-switching" : ""}`}
              >
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
                      disabled={draftLoading || !activeChapterId || !canEnterZenMode}
                      aria-pressed={zenMode}
                    >
                      {zenMode ? "退出沉浸" : "进入沉浸"}
                    </button>
                  </div>
                </div>

                <div className="draft-editor-context" onContextMenu={handleEditorContextMenu}>
                  <EditorContent
                    ref={draftEditorRef}
                    editor={editor}
                    className={`draft-editor ${draftLoading || !activeChapterId ? "disabled" : ""}`}
                  />
                  {selectionMenu ? (
                    <div
                      ref={selectionMenuRef}
                      className="selection-context-menu"
                      role="menu"
                      tabIndex={-1}
                      onKeyDown={handleSelectionMenuKeyDown}
                      style={{ left: `${selectionMenu.x}px`, top: `${selectionMenu.y}px` }}
                    >
                      <button
                        ref={(el) => {
                          selectionMenuButtonsRef.current[0] = el;
                        }}
                        className="btn ghost tiny"
                        onClick={() => runSelectionTool("polish")}
                        role="menuitem"
                      >
                        ✨ 润色选中
                      </button>
                      <button
                        ref={(el) => {
                          selectionMenuButtonsRef.current[1] = el;
                        }}
                        className="btn ghost tiny"
                        onClick={() => runSelectionTool("expand")}
                        role="menuitem"
                      >
                        🧩 扩写选中
                      </button>
                    </div>
                  ) : null}
                </div>
              </div>

              <div className="draft-actions">
                <button
                  className="btn primary tiny"
                  onClick={() => void onSaveDraftSnapshot()}
                  disabled={draftSaving || draftLoading}
                >
                  {draftSaving ? "保存中..." : "保存正文"}
                </button>
                <button
                  className="btn ghost tiny"
                  onClick={() => void onRefreshDraftSnapshot(projectId, activeChapterId)}
                  disabled={draftSaving || draftLoading}
                >
                  拉取服务器版本
                </button>
                <button className="btn primary tiny" onClick={() => onApplyAssistantToDraft("insert")}>
                  插入助手回复
                </button>
                <button className="btn ghost tiny" onClick={() => onApplyAssistantToDraft("replace")}>
                  替换选中为助手回复
                </button>
              </div>

              <p className="draft-hint">
                {draftLoading ? "正在加载服务器正文..." : `已选 ${selectedDraftText.length} 字`}
                {latestAssistantReply ? " · 最近一条助手回复可写回正文" : " · 暂无助手回复可写回"}
              </p>
            </>
          )}
        </>
      }
      statusBar={
        <EditorStatusBar
          draftWordCount={draftWordCount}
          draftVersion={draftVersion}
          draftUpdatedAt={draftUpdatedAt}
          autoSaveState={autoSaveState}
          autoSaveAt={autoSaveAt}
          draftFocusMode={draftFocusMode}
          typewriterModeEnabled={typewriterModeEnabled}
          selectedDraftText={selectedDraftText}
          latestAssistantReply={latestAssistantReply}
        />
      }
    />
  );
});
