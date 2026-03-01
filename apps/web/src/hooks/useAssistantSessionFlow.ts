import {
  useCallback,
  useEffect,
  useRef,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import {
  deleteProjectSession,
  preheatContextPack,
  streamChat,
  updateProjectSession,
  type ChatStreamTimingMetrics,
} from "../api/chatApi";
import type {
  ActionAuditLog,
  ChatSessionSummary,
  ChatStreamEvent,
  ChatStreamTraceEvent,
  EvidencePayload,
  UiMessage,
} from "../types";

type StreamRequestOptions = {
  projectId: number;
  sessionId: number | null;
  activeChapterId: number | null;
  activeSceneBeatId: number | null;
  activePromptTemplateId: number | null;
  activeModelProfileId: string | null;
  model: string;
  povMode: "global" | "character";
  povAnchor: string;
  ragMode: "local" | "global" | "hybrid" | "mix";
  deterministicFirst: boolean;
  thinkingEnabled: boolean;
  referenceProjectIds: number[];
  chatTemperatureProfile: "action" | "chat" | "brainstorm";
  temperatureOverride: number | null;
  contextWindowProfile: "balanced" | "chapter_focus" | "world_focus" | "minimal";
};

type UseAssistantSessionFlowArgs = {
  projectSessions: ChatSessionSummary[];
  input: string;
  streaming: boolean;
  assistantDrawerOpen: boolean;
  setAssistantDrawerOpen: Dispatch<SetStateAction<boolean>>;
  composerInputRef: MutableRefObject<HTMLTextAreaElement | null>;
  assistantDrawerReturnFocusRef: MutableRefObject<HTMLElement | null>;
  previousAssistantDrawerOpenRef: MutableRefObject<boolean>;

  streamOptions: StreamRequestOptions;
  setInput: (value: string) => void;
  setSessionId: (sessionId: number | null) => void;
  setStreaming: (streaming: boolean) => void;
  setError: (error: string | null) => void;
  setUsage: (usage: Record<string, unknown> | null) => void;
  setPendingActionIds: (actionIds: number[]) => void;
  setSelectedActionId: (actionId: number | null) => void;
  setActionLogs: (logs: ActionAuditLog[]) => void;
  setTraceEvents: Dispatch<SetStateAction<ChatStreamTraceEvent[]>>;
  setEvidence: (evidence: EvidencePayload | null) => void;
  setLastStreamMetrics: (metrics: ChatStreamTimingMetrics | null) => void;
  appendMessage: (message: UiMessage) => void;
  appendMessageDelta: (messageId: string, delta: string) => void;
  updateMessage: (messageId: string, updater: (message: UiMessage) => UiMessage) => void;
  resetSessionState: () => void;
  refreshSessionSnapshot: (nextSessionId: number, nextProjectId: number) => Promise<void>;
  refreshSessionPostChatSnapshot: (nextSessionId: number, nextProjectId: number) => Promise<void>;
  refreshProjectSnapshot: (nextProjectId: number) => Promise<void>;
  isDraftDirty: () => boolean;
  persistDraftSnapshot: (options?: { silent?: boolean; auto?: boolean }) => Promise<boolean>;
};

export function useAssistantSessionFlow({
  projectSessions,
  input,
  streaming,
  assistantDrawerOpen,
  setAssistantDrawerOpen,
  composerInputRef,
  assistantDrawerReturnFocusRef,
  previousAssistantDrawerOpenRef,
  streamOptions,
  setInput,
  setSessionId,
  setStreaming,
  setError,
  setUsage,
  setPendingActionIds,
  setSelectedActionId,
  setActionLogs,
  setTraceEvents,
  setEvidence,
  setLastStreamMetrics,
  appendMessage,
  appendMessageDelta,
  updateMessage,
  resetSessionState,
  refreshSessionSnapshot,
  refreshSessionPostChatSnapshot,
  refreshProjectSnapshot,
  isDraftDirty,
  persistDraftSnapshot,
}: UseAssistantSessionFlowArgs) {
  const assistantFocusTimerRef = useRef<number | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const streamDeltaFrameRef = useRef<number | null>(null);
  const streamDeltaBufferRef = useRef<{
    assistantLocalId: string | null;
    text: string;
  }>({
    assistantLocalId: null,
    text: "",
  });

  const focusAssistantComposer = useCallback((fallbackDelayMs = 140) => {
    const focus = () => {
      const inputEl = composerInputRef.current;
      if (!inputEl) return;
      inputEl.focus();
      const end = inputEl.value.length;
      inputEl.setSelectionRange(end, end);
    };
    focus();
    if (assistantFocusTimerRef.current) {
      window.clearTimeout(assistantFocusTimerRef.current);
      assistantFocusTimerRef.current = null;
    }
    assistantFocusTimerRef.current = window.setTimeout(() => {
      focus();
      assistantFocusTimerRef.current = null;
    }, fallbackDelayMs);
  }, [composerInputRef]);

  const rememberActiveElement = useCallback((target: { current: HTMLElement | null }) => {
    if (typeof document === "undefined") return;
    const active = document.activeElement;
    target.current = active instanceof HTMLElement ? active : null;
  }, []);

  const closeAssistantDrawer = useCallback(() => {
    setAssistantDrawerOpen(false);
  }, [setAssistantDrawerOpen]);

  const openAssistantDrawerAndFocusComposer = useCallback(() => {
    if (!assistantDrawerOpen) {
      rememberActiveElement(assistantDrawerReturnFocusRef);
    }
    setAssistantDrawerOpen(true);
    focusAssistantComposer();
  }, [
    assistantDrawerOpen,
    assistantDrawerReturnFocusRef,
    focusAssistantComposer,
    rememberActiveElement,
    setAssistantDrawerOpen,
  ]);

  const toggleAssistantDrawer = useCallback(() => {
    setAssistantDrawerOpen((prev) => {
      const next = !prev;
      if (next) {
        rememberActiveElement(assistantDrawerReturnFocusRef);
        window.setTimeout(() => focusAssistantComposer(0), 0);
      }
      return next;
    });
  }, [assistantDrawerReturnFocusRef, focusAssistantComposer, rememberActiveElement, setAssistantDrawerOpen]);

  const startNewSession = useCallback(() => {
    resetSessionState();
    setTraceEvents([]);
    setInput("");
  }, [resetSessionState, setInput, setTraceEvents]);

  const switchSession = useCallback(
    async (nextSessionId: number) => {
      if (streaming) return;
      if (nextSessionId <= 0 || streamOptions.sessionId === nextSessionId) return;
      setError(null);
      setSelectedActionId(null);
      setActionLogs([]);
      setTraceEvents([]);
      try {
        await refreshSessionSnapshot(nextSessionId, streamOptions.projectId);
        setSessionId(nextSessionId);
      } catch (switchError) {
        const message = switchError instanceof Error ? switchError.message : "切换会话失败";
        setError(message);
      }
    },
    [
      refreshSessionSnapshot,
      setActionLogs,
      setError,
      setSelectedActionId,
      setSessionId,
      setTraceEvents,
      streamOptions.projectId,
      streamOptions.sessionId,
      streaming,
    ]
  );

  const renameSession = useCallback(async () => {
    if (streaming || !streamOptions.sessionId) return;
    const active = projectSessions.find((item) => item.id === streamOptions.sessionId);
    const currentTitle = (active?.title || "").trim() || `会话 #${streamOptions.sessionId}`;
    const nextTitle = window.prompt("输入新的会话标题", currentTitle);
    if (nextTitle === null) return;
    const normalized = nextTitle.trim();
    if (!normalized || normalized === currentTitle) return;
    setError(null);
    try {
      await updateProjectSession(streamOptions.projectId, streamOptions.sessionId, { title: normalized });
      await refreshSessionSnapshot(streamOptions.sessionId, streamOptions.projectId);
    } catch (renameError) {
      const message = renameError instanceof Error ? renameError.message : "重命名会话失败";
      setError(message);
    }
  }, [projectSessions, refreshSessionSnapshot, setError, streamOptions.projectId, streamOptions.sessionId, streaming]);

  const deleteSession = useCallback(async () => {
    if (streaming || !streamOptions.sessionId) return;
    const active = projectSessions.find((item) => item.id === streamOptions.sessionId);
    const label = (active?.title || "").trim() || `#${streamOptions.sessionId}`;
    if (!window.confirm(`确认删除会话「${label}」吗？此操作不可恢复。`)) {
      return;
    }
    setError(null);
    try {
      await deleteProjectSession(streamOptions.projectId, streamOptions.sessionId);
      startNewSession();
      await refreshProjectSnapshot(streamOptions.projectId);
    } catch (deleteError) {
      const message = deleteError instanceof Error ? deleteError.message : "删除会话失败";
      setError(message);
    }
  }, [
    projectSessions,
    refreshProjectSnapshot,
    setError,
    startNewSession,
    streamOptions.projectId,
    streamOptions.sessionId,
    streaming,
  ]);

  const flushBufferedStreamDelta = useCallback(() => {
    const pending = streamDeltaBufferRef.current;
    if (!pending.assistantLocalId || !pending.text) return;
    appendMessageDelta(pending.assistantLocalId, pending.text);
    pending.text = "";
  }, [appendMessageDelta]);

  const scheduleBufferedStreamDeltaFlush = useCallback(() => {
    if (streamDeltaFrameRef.current !== null) return;
    if (typeof window === "undefined") {
      flushBufferedStreamDelta();
      return;
    }
    streamDeltaFrameRef.current = window.requestAnimationFrame(() => {
      streamDeltaFrameRef.current = null;
      flushBufferedStreamDelta();
    });
  }, [flushBufferedStreamDelta]);

  const queueStreamDelta = useCallback(
    (assistantLocalId: string, text: string) => {
      if (!text) return;
      const pending = streamDeltaBufferRef.current;
      if (pending.assistantLocalId && pending.assistantLocalId !== assistantLocalId && pending.text) {
        flushBufferedStreamDelta();
      }
      pending.assistantLocalId = assistantLocalId;
      pending.text += text;
      scheduleBufferedStreamDeltaFlush();
    },
    [flushBufferedStreamDelta, scheduleBufferedStreamDeltaFlush]
  );

  const cancelBufferedStreamDeltaFlush = useCallback(() => {
    if (streamDeltaFrameRef.current !== null && typeof window !== "undefined") {
      window.cancelAnimationFrame(streamDeltaFrameRef.current);
      streamDeltaFrameRef.current = null;
    }
  }, []);

  const handleStreamEvent = useCallback(
    (event: ChatStreamEvent, assistantLocalId: string, onMeta: (resolvedSessionId: number) => void) => {
      if (event.type === "meta") {
        setSessionId(event.session_id);
        setPendingActionIds(event.proposed_action_ids ?? []);
        onMeta(event.session_id);
        return;
      }
      if (event.type === "evidence") {
        setEvidence(event);
        return;
      }
      if (event.type === "trace") {
        setTraceEvents((prev) => {
          const next = [...prev, event];
          if (next.length <= 48) return next;
          return next.slice(next.length - 48);
        });
        return;
      }
      if (event.type === "delta") {
        queueStreamDelta(assistantLocalId, event.text);
        return;
      }
      if (event.type === "done") {
        flushBufferedStreamDelta();
        setUsage(event.usage ?? null);
        updateMessage(assistantLocalId, (message) => ({ ...message, streaming: false }));
        return;
      }
      if (event.type === "error") {
        flushBufferedStreamDelta();
        setError(event.message);
        updateMessage(assistantLocalId, (message) => ({
          ...message,
          streaming: false,
          content: message.content || `模型返回错误：${event.message}`,
        }));
      }
    },
    [
      flushBufferedStreamDelta,
      queueStreamDelta,
      setError,
      setEvidence,
      setPendingActionIds,
      setSessionId,
      setTraceEvents,
      setUsage,
      updateMessage,
    ]
  );

  const handleSend = useCallback(async () => {
    const content = input.trim();
    if (!content || streaming) return;

    setInput("");
    setError(null);
    setTraceEvents([]);
    setStreaming(true);
    setLastStreamMetrics(null);

    const localSeed = Date.now();
    const userLocalId = `local-user-${localSeed}`;
    const assistantLocalId = `local-assistant-${localSeed}`;
    appendMessage({ id: userLocalId, role: "user", content, streaming: false });
    appendMessage({ id: assistantLocalId, role: "assistant", content: "", streaming: true });

    let resolvedSessionId = streamOptions.sessionId;
    streamAbortRef.current?.abort();
    const streamAbortController = new AbortController();
    streamAbortRef.current = streamAbortController;
    cancelBufferedStreamDeltaFlush();
    streamDeltaBufferRef.current = { assistantLocalId, text: "" };

    try {
      if (isDraftDirty()) {
        await persistDraftSnapshot({ silent: true, auto: true });
      }
      void preheatContextPack(streamOptions.projectId).catch(() => undefined);
      await streamChat(
        {
          project_id: streamOptions.projectId,
          content,
          session_id: streamOptions.sessionId,
          chapter_id: streamOptions.activeChapterId,
          scene_beat_id: streamOptions.activeSceneBeatId,
          prompt_template_id: streamOptions.activePromptTemplateId,
          model: streamOptions.model.trim() ? streamOptions.model.trim() : null,
          model_profile_id: streamOptions.activeModelProfileId,
          pov_mode: streamOptions.povMode,
          pov_anchor: streamOptions.povAnchor.trim() ? streamOptions.povAnchor.trim() : null,
          rag_mode: streamOptions.ragMode,
          deterministic_first: streamOptions.deterministicFirst,
          thinking_enabled: streamOptions.thinkingEnabled,
          reference_project_ids: streamOptions.referenceProjectIds,
          temperature_profile: streamOptions.chatTemperatureProfile,
          temperature_override: streamOptions.temperatureOverride,
          context_window_profile: streamOptions.contextWindowProfile,
        },
        (event) => {
          handleStreamEvent(event, assistantLocalId, (nextSessionId) => {
            resolvedSessionId = nextSessionId;
          });
        },
        {
          signal: streamAbortController.signal,
          onMetrics: (metrics) => {
            setLastStreamMetrics(metrics);
            if (typeof window !== "undefined" && import.meta.env.DEV) {
              const metricWindow = window as Window & {
                __NOVEL_STREAM_METRICS__?: Array<
                  ChatStreamTimingMetrics & {
                    at: string;
                    projectId: number;
                    sessionId: number | null;
                  }
                >;
              };
              const samples = metricWindow.__NOVEL_STREAM_METRICS__ ?? [];
              samples.push({
                ...metrics,
                at: new Date().toISOString(),
                projectId: streamOptions.projectId,
                sessionId: resolvedSessionId ?? streamOptions.sessionId ?? null,
              });
              if (samples.length > 60) {
                samples.splice(0, samples.length - 60);
              }
              metricWindow.__NOVEL_STREAM_METRICS__ = samples;
            }
          },
        }
      );

      if (resolvedSessionId) {
        await refreshSessionPostChatSnapshot(resolvedSessionId, streamOptions.projectId);
      } else {
        await refreshProjectSnapshot(streamOptions.projectId);
      }
    } catch (streamError) {
      if (streamError instanceof DOMException && streamError.name === "AbortError") {
        return;
      }
      const message = streamError instanceof Error ? streamError.message : "发送失败，请稍后再试";
      flushBufferedStreamDelta();
      setError(message);
      updateMessage(assistantLocalId, (item) => ({
        ...item,
        streaming: false,
        content: item.content || `请求失败：${message}`,
      }));
    } finally {
      if (streamAbortRef.current === streamAbortController) {
        streamAbortRef.current = null;
      }
      flushBufferedStreamDelta();
      cancelBufferedStreamDeltaFlush();
      streamDeltaBufferRef.current = { assistantLocalId: null, text: "" };
      setStreaming(false);
    }
  }, [
    appendMessage,
    cancelBufferedStreamDeltaFlush,
    flushBufferedStreamDelta,
    handleStreamEvent,
    input,
    isDraftDirty,
    persistDraftSnapshot,
    refreshProjectSnapshot,
    refreshSessionPostChatSnapshot,
    setError,
    setInput,
    setLastStreamMetrics,
    setStreaming,
    setTraceEvents,
    streamOptions,
    streaming,
    updateMessage,
  ]);

  useEffect(() => {
    return () => {
      streamAbortRef.current?.abort();
      streamAbortRef.current = null;
      cancelBufferedStreamDeltaFlush();
      streamDeltaBufferRef.current = { assistantLocalId: null, text: "" };
      if (assistantFocusTimerRef.current) {
        window.clearTimeout(assistantFocusTimerRef.current);
        assistantFocusTimerRef.current = null;
      }
    };
  }, [cancelBufferedStreamDeltaFlush]);

  useEffect(() => {
    const wasOpen = previousAssistantDrawerOpenRef.current;
    previousAssistantDrawerOpenRef.current = assistantDrawerOpen;
    if (!wasOpen || assistantDrawerOpen) return;
    const target = assistantDrawerReturnFocusRef.current;
    if (!target || typeof document === "undefined" || !document.contains(target)) return;
    window.setTimeout(() => target.focus(), 0);
  }, [assistantDrawerOpen, assistantDrawerReturnFocusRef, previousAssistantDrawerOpenRef]);

  return {
    focusAssistantComposer,
    rememberActiveElement,
    closeAssistantDrawer,
    openAssistantDrawerAndFocusComposer,
    toggleAssistantDrawer,
    startNewSession,
    switchSession,
    renameSession,
    deleteSession,
    handleSend,
  };
}
