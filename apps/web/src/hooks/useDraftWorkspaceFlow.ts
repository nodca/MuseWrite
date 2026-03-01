import { useCallback, useEffect, type MutableRefObject } from "react";
import { getProjectChapterRevisions, getProjectChapters, saveProjectChapter } from "../api/chatApi";
import type { DraftAutoSaveState, ProjectChapter, ProjectChapterRevision } from "../types";

export type DraftRecoverySnapshot = {
  project_id: number;
  chapter_id: number;
  title: string;
  content: string;
  base_version: number;
  saved_at: string;
};

const DRAFT_RECOVERY_PREFIX = "novel-platform:draft-recovery:v1";

export function buildDraftRecoveryKey(projectId: number, chapterId: number): string {
  return `${DRAFT_RECOVERY_PREFIX}:${projectId}:${chapterId}`;
}

export function readDraftRecoverySnapshot(projectId: number, chapterId: number): DraftRecoverySnapshot | null {
  try {
    const raw = window.localStorage.getItem(buildDraftRecoveryKey(projectId, chapterId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as DraftRecoverySnapshot;
    if (
      parsed.project_id !== projectId ||
      parsed.chapter_id !== chapterId ||
      typeof parsed.title !== "string" ||
      typeof parsed.content !== "string" ||
      typeof parsed.base_version !== "number" ||
      typeof parsed.saved_at !== "string"
    ) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function writeDraftRecoverySnapshot(snapshot: DraftRecoverySnapshot): void {
  try {
    window.localStorage.setItem(
      buildDraftRecoveryKey(snapshot.project_id, snapshot.chapter_id),
      JSON.stringify(snapshot)
    );
  } catch {
    // ignore localStorage quota/permission failures
  }
}

export function clearDraftRecoverySnapshot(projectId: number, chapterId: number): void {
  try {
    window.localStorage.removeItem(buildDraftRecoveryKey(projectId, chapterId));
  } catch {
    // ignore localStorage quota/permission failures
  }
}

export function shouldRestoreDraftRecovery(snapshot: DraftRecoverySnapshot, chapter: ProjectChapter): boolean {
  if (snapshot.title === chapter.title && snapshot.content === chapter.content) {
    return false;
  }
  const savedAtMs = new Date(snapshot.saved_at).getTime();
  const updatedAtMs = new Date(chapter.updated_at).getTime();
  const newerThanServer =
    Number.isFinite(savedAtMs) && Number.isFinite(updatedAtMs) ? savedAtMs >= updatedAtMs : false;
  return snapshot.base_version >= chapter.version || newerThanServer;
}

type DraftPersistOptions = {
  silent?: boolean;
  auto?: boolean;
};

type LastSavedDraftSnapshot = {
  chapterId: number | null;
  volumeId: number | null;
  title: string;
  content: string;
};

type UseDraftWorkspaceFlowArgs = {
  projectId: number;
  activeChapterId: number | null;
  activeVolumeId: number | null;
  draftTitle: string;
  draftText: string;
  draftVersion: number;
  draftLoading: boolean;
  draftSaving: boolean;
  setDraftLoading: (loading: boolean) => void;
  setDraftSaving: (saving: boolean) => void;
  setActiveChapterId: (chapterId: number | null) => void;
  setError: (error: string | null) => void;
  setGhostError: (error: string | null) => void;
  setDraftTitle: (title: string) => void;
  setDraftVersion: (version: number) => void;
  setDraftUpdatedAt: (updatedAt: string | null) => void;
  setDraftRevisions: (revisions: ProjectChapterRevision[]) => void;
  setChapters: (chapters: ProjectChapter[]) => void;
  setAutoSaveState: (state: DraftAutoSaveState) => void;
  setAutoSaveAt: (value: string | null) => void;
  setLocalRecoveryNotice: (notice: string | null) => void;
  lastSavedDraftRef: MutableRefObject<LastSavedDraftSnapshot>;
  autoSaveTimerRef: MutableRefObject<number | null>;
  localRecoveryTimerRef: MutableRefObject<number | null>;
  loadChapterSnapshot: (nextProjectId: number, chapterId: number) => Promise<void>;
};

export function useDraftWorkspaceFlow({
  projectId,
  activeChapterId,
  activeVolumeId,
  draftTitle,
  draftText,
  draftVersion,
  draftLoading,
  draftSaving,
  setDraftLoading,
  setDraftSaving,
  setActiveChapterId,
  setError,
  setGhostError,
  setDraftTitle,
  setDraftVersion,
  setDraftUpdatedAt,
  setDraftRevisions,
  setChapters,
  setAutoSaveState,
  setAutoSaveAt,
  setLocalRecoveryNotice,
  lastSavedDraftRef,
  autoSaveTimerRef,
  localRecoveryTimerRef,
  loadChapterSnapshot,
}: UseDraftWorkspaceFlowArgs) {
  const isDraftDirty = useCallback(() => {
    const currentTitle = draftTitle.trim() || "未命名章节";
    const snapshot = lastSavedDraftRef.current;
    return (
      snapshot.chapterId !== activeChapterId ||
      snapshot.volumeId !== (activeVolumeId ?? null) ||
      snapshot.title !== currentTitle ||
      snapshot.content !== draftText
    );
  }, [activeChapterId, activeVolumeId, draftText, draftTitle, lastSavedDraftRef]);

  const persistDraftSnapshot = useCallback(
    async (options?: DraftPersistOptions) => {
      if (!activeChapterId) {
        if (!options?.silent) {
          setError("请先选择章节");
        }
        return false;
      }
      if (draftSaving) return false;
      setDraftSaving(true);
      if (!options?.silent) {
        setError(null);
      }
      if (options?.auto) {
        setAutoSaveState("saving");
        setGhostError(null);
      }
      try {
        const normalizedTitle = draftTitle.trim() || "未命名章节";
        const saved = await saveProjectChapter(projectId, activeChapterId, {
          title: normalizedTitle,
          content: draftText,
          volume_id: activeVolumeId,
          expected_version: draftVersion > 0 ? draftVersion : null,
        });
        setDraftTitle(saved.title);
        setDraftVersion(saved.version);
        setDraftUpdatedAt(saved.updated_at);
        const [revisions, chapterList] = await Promise.all([
          getProjectChapterRevisions(projectId, activeChapterId, 20),
          getProjectChapters(projectId),
        ]);
        setDraftRevisions(revisions);
        setChapters(chapterList);
        lastSavedDraftRef.current = {
          chapterId: activeChapterId,
          volumeId: saved.volume_id ?? null,
          title: saved.title,
          content: draftText,
        };
        clearDraftRecoverySnapshot(projectId, activeChapterId);
        setLocalRecoveryNotice(null);
        setAutoSaveAt(new Date().toISOString());
        setAutoSaveState("saved");
        return true;
      } catch (saveError) {
        const rawMessage = saveError instanceof Error ? saveError.message : "保存正文失败";
        const isVersionConflict =
          rawMessage.toLowerCase().includes("version conflict") ||
          rawMessage.toLowerCase().includes("409");
        const message = isVersionConflict
          ? "检测到章节版本冲突：已阻止覆盖，请刷新章节后重试。"
          : rawMessage;
        if (!options?.silent || isVersionConflict) {
          setError(message);
        }
        if (isVersionConflict) {
          setLocalRecoveryNotice("本地草稿已保留，请刷新章节后手动合并。");
        }
        if (options?.auto) {
          setAutoSaveState("error");
        }
        return false;
      } finally {
        setDraftSaving(false);
      }
    },
    [
      activeChapterId,
      activeVolumeId,
      draftVersion,
      draftSaving,
      draftText,
      draftTitle,
      projectId,
      setAutoSaveAt,
      setAutoSaveState,
      setChapters,
      setDraftRevisions,
      setDraftSaving,
      setDraftTitle,
      setDraftUpdatedAt,
      setDraftVersion,
      setError,
      setGhostError,
      setLocalRecoveryNotice,
      lastSavedDraftRef,
    ]
  );

  const saveDraftSnapshot = useCallback(async () => {
    await persistDraftSnapshot({ silent: false, auto: false });
  }, [persistDraftSnapshot]);

  const switchChapter = useCallback(
    async (chapterId: number) => {
      if (!chapterId || chapterId === activeChapterId) return;
      setDraftLoading(true);
      setError(null);
      try {
        if (isDraftDirty()) {
          await persistDraftSnapshot({ silent: true, auto: true });
        }
        setActiveChapterId(chapterId);
        await loadChapterSnapshot(projectId, chapterId);
      } catch (switchError) {
        const message = switchError instanceof Error ? switchError.message : "切换章节失败";
        setError(message);
      } finally {
        setDraftLoading(false);
      }
    },
    [
      activeChapterId,
      isDraftDirty,
      loadChapterSnapshot,
      persistDraftSnapshot,
      projectId,
      setActiveChapterId,
      setDraftLoading,
      setError,
    ]
  );

  useEffect(() => {
    if (autoSaveTimerRef.current) {
      window.clearTimeout(autoSaveTimerRef.current);
      autoSaveTimerRef.current = null;
    }
    if (!activeChapterId || draftLoading || draftSaving) return;
    if (!isDraftDirty()) return;

    setAutoSaveState("pending");
    autoSaveTimerRef.current = window.setTimeout(() => {
      void persistDraftSnapshot({ silent: true, auto: true });
    }, 1200);

    return () => {
      if (autoSaveTimerRef.current) {
        window.clearTimeout(autoSaveTimerRef.current);
        autoSaveTimerRef.current = null;
      }
    };
  }, [
    activeChapterId,
    autoSaveTimerRef,
    draftLoading,
    draftSaving,
    isDraftDirty,
    persistDraftSnapshot,
    setAutoSaveState,
  ]);

  useEffect(() => {
    if (localRecoveryTimerRef.current) {
      window.clearTimeout(localRecoveryTimerRef.current);
      localRecoveryTimerRef.current = null;
    }
    if (!activeChapterId || draftLoading) return;
    if (!isDraftDirty()) {
      clearDraftRecoverySnapshot(projectId, activeChapterId);
      return;
    }

    localRecoveryTimerRef.current = window.setTimeout(() => {
      writeDraftRecoverySnapshot({
        project_id: projectId,
        chapter_id: activeChapterId,
        title: draftTitle.trim() || "未命名章节",
        content: draftText,
        base_version: draftVersion,
        saved_at: new Date().toISOString(),
      });
    }, 450);

    return () => {
      if (localRecoveryTimerRef.current) {
        window.clearTimeout(localRecoveryTimerRef.current);
        localRecoveryTimerRef.current = null;
      }
    };
  }, [
    activeChapterId,
    draftLoading,
    draftText,
    draftTitle,
    draftVersion,
    isDraftDirty,
    localRecoveryTimerRef,
    projectId,
  ]);

  useEffect(() => {
    return () => {
      if (autoSaveTimerRef.current) {
        window.clearTimeout(autoSaveTimerRef.current);
        autoSaveTimerRef.current = null;
      }
      if (localRecoveryTimerRef.current) {
        window.clearTimeout(localRecoveryTimerRef.current);
        localRecoveryTimerRef.current = null;
      }
    };
  }, [autoSaveTimerRef, localRecoveryTimerRef]);

  return {
    isDraftDirty,
    persistDraftSnapshot,
    saveDraftSnapshot,
    switchChapter,
  };
}
