import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import Placeholder from "@tiptap/extension-placeholder";
import { buildDiffSuggestions, SemanticDiffExtension } from "./components/editor/extensions/DiffExtension";
import { EditorContent, useEditor, type Editor, type JSONContent } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import { useShallow } from "zustand/react/shallow";
import {
  createForeshadowingCard,
  createProjectPromptTemplate,
  createProjectVolume,
  createSceneBeat,
  createProjectChapter,
  deleteForeshadowingCard,
  deleteProjectPromptTemplate,
  deleteProjectVolume,
  deleteSceneBeat,
  deleteProjectChapter,
  decideAction,
  polishSelection,
  expandSelection,
  getActionLogs,
  getForeshadowingCards,
  getProjectConsistencyAudits,
  getProjectGraphTimeline,
  getProjectPromptTemplates,
  getProjectPromptTemplateRevisions,
  getProjectVolumes,
  deleteProjectModelProfile,
  getProjectChapter,
  getProjectChapterRevisions,
  getProjectChapters,
  getProjectCards,
  getProjectModelProfiles,
  getProjectSessions,
  getSceneBeats,
  getProjectSettings,
  moveProjectChapter,
  preheatContextPack,
  reorderProjectChapters,
  rollbackProjectPromptTemplate,
  rollbackProjectChapter,
  runProjectConsistencyAudit,
  saveProjectChapter,
  updateProjectModelProfile,
  activateProjectModelProfile,
  createProjectModelProfile,
  updateForeshadowingCard,
  updateProjectPromptTemplate,
  updateProjectVolume,
  updateSceneBeat,
  getSessionActions,
  getSessionMessages,
  type ChatStreamTimingMetrics,
} from "./api/chatApi";
import { useChatStore } from "./store/chatStore";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { DraftWorkspacePanel } from "./components/DraftWorkspacePanel";
import { toUiMessage } from "./components/chat/messageMapping";
import { StoryPlanningPanel, type StoryPlanningPanelProps } from "./components/StoryPlanningPanel";
import { DebugSnapshotGrid, GraphCandidateReviewPanel, PromptWorkshopPanel } from "./debugPanels";
import {
  clearDraftRecoverySnapshot,
  readDraftRecoverySnapshot,
  shouldRestoreDraftRecovery,
  useDraftWorkspaceFlow,
} from "./hooks/useDraftWorkspaceFlow";
import { useAssistantSessionFlow } from "./hooks/useAssistantSessionFlow";
import { useTimelineGraphFlow } from "./hooks/useTimelineGraphFlow";
import { useSettingsStorage } from "./hooks/useSettingsStorage";
import type {
  ActionAuditLog,
  ChatAction,
  ChatStreamTraceEvent,
  ConsistencyAuditReport,
  DraftAutoSaveState,
  EvidencePayload,
  ForeshadowingCard,
  GraphBlastRadiusPreview,
  GraphTimelineSnapshot,
  ChatSessionSummary,
  ProjectChapter,
  ProjectChapterRevision,
  ProjectVolume,
  ModelProfile,
  PromptTemplate,
  PromptTemplateRevision,
  SceneBeat,
  SettingEntry,
  StoryCard,
  UiMessage,
} from "./types";

// --- Extracted modules ---
import { formatRole, formatJson, formatDateTime, isSameLocalDate, toFiniteNumber, computePercentile, formatMs, formatPercent, normalizeEditorText, parseReferenceProjectIds } from "./utils/formatting";
import { toEditorDoc, readEditorText, readSelectedText } from "./lib/editorHelpers";
import { entityHintPluginKey, collectEntityHighlightHints, EntityInlineHintExtension, collectAwarenessTags } from "./lib/entityHighlight";
import { isEntityMergeActionType, summarizeAction, safeToRecord } from "./lib/actionHelpers";
import { normalizeGraphPreviewToken, buildGraphPreviewEdgeKey, getActionBlastRadius, summarizeBlastRadius, resolveBlastRadiusTone, buildBlastRadiusSummaryChips, formatBlastRadiusMarkdown } from "./lib/blastRadius";
import { getFocusableElements, focusFirstDialogElement } from "./lib/focusTrap";
import { BlastRadiusDetailDisclosure } from "./components/BlastRadiusDetailDisclosure";
import { AssistantChatPanel } from "./components/AssistantChatPanel";
import { ActionCard } from "./components/ActionCard";
import { ActionLogsList } from "./components/ActionLogsList";
import { ChapterOutlineList, CHAPTER_OUTLINE_ROW_HEIGHT, CHAPTER_OUTLINE_MAX_HEIGHT, CHAPTER_OUTLINE_OVERSCAN } from "./components/ChapterOutlineList";
import type { ChapterOutlineEntry } from "./components/ChapterOutlineList";
import { DraftRevisionList } from "./components/DraftRevisionList";
import { TopBar } from "./components/TopBar";
import { useTheme } from "./hooks/useTheme";
import { SettingsDialog } from "./components/SettingsDialog";
import type { WritingTheme } from "./components/SettingsDialog";
import { AssistantActionsPanel } from "./components/AssistantActionsPanel";
import type { StreamLatencySample, TokenUsageSample, RetrievalHitSample } from "./components/AssistantActionsPanel";
import { AssistantDrawer } from "./components/AssistantDrawer";

type PostChatSnapshotData = {
  messagesData: Awaited<ReturnType<typeof getSessionMessages>>;
  actionsData: Awaited<ReturnType<typeof getSessionActions>>;
  settingsData: Awaited<ReturnType<typeof getProjectSettings>>;
  cardsData: Awaited<ReturnType<typeof getProjectCards>>;
  sessionsData: ChatSessionSummary[];
};

type FullSessionSnapshotData = {
  messagesData: Awaited<ReturnType<typeof getSessionMessages>>;
  actionsData: Awaited<ReturnType<typeof getSessionActions>>;
  settingsData: Awaited<ReturnType<typeof getProjectSettings>>;
  auditsData: ConsistencyAuditReport[];
  modelProfilesData: Awaited<ReturnType<typeof getProjectModelProfiles>>;
  cardsData: Awaited<ReturnType<typeof getProjectCards>>;
  templatesData: Awaited<ReturnType<typeof getProjectPromptTemplates>>;
  volumesData: ProjectVolume[];
  foreshadowData: ForeshadowingCard[];
  sessionsData: ChatSessionSummary[];
};

type ProjectSnapshotData = {
  settingsData: Awaited<ReturnType<typeof getProjectSettings>>;
  auditsData: ConsistencyAuditReport[];
  modelProfilesData: Awaited<ReturnType<typeof getProjectModelProfiles>>;
  cardsData: Awaited<ReturnType<typeof getProjectCards>>;
  templatesData: Awaited<ReturnType<typeof getProjectPromptTemplates>>;
  volumesData: ProjectVolume[];
  foreshadowData: ForeshadowingCard[];
  sessionsData: ChatSessionSummary[];
};

type ChapterSnapshotData = {
  chapter: Awaited<ReturnType<typeof getProjectChapter>>;
  revisions: Awaited<ReturnType<typeof getProjectChapterRevisions>>;
  beats: SceneBeat[];
  overdueForeshadows: ForeshadowingCard[];
};

type DraftSnapshotData = {
  chapterList: Awaited<ReturnType<typeof getProjectChapters>>;
  volumeList: ProjectVolume[];
  foreshadowList: ForeshadowingCard[];
};

type PlanningSnapshotData = {
  volumeList: ProjectVolume[];
  foreshadowList: ForeshadowingCard[];
  beats: SceneBeat[];
  overdue: ForeshadowingCard[];
};

const POST_CHAT_SNAPSHOT_TTL_MS = 400;
const ACTION_FLOW_GUIDE_SEEN_KEY = "novel-platform:action-flow-guide:seen:v1";
const PERFORMANCE_SAMPLE_LIMIT = 60;
type WorkbenchPanelVisibility = {
  actions: boolean;
  prompt: boolean;
  planning: boolean;
  snapshot: boolean;
};

const WORKBENCH_PANEL_LABELS: Record<keyof WorkbenchPanelVisibility, string> = {
  actions: "动作提议 + 图谱",
  prompt: "Prompt + 知识库",
  planning: "结构化规划",
  snapshot: "检索快照",
};


export default function App() {
  const {
    assistantDrawerOpen,
    advancedPanelOpen,
    assistantSection,
    projectId,
    model,
    povMode,
    povAnchor,
    ragMode,
    deterministicFirst,
    thinkingEnabled,
    sessionId,
    streaming,
    error,
    usage,
    messages,
    actions,
    pendingActionIds,
    settings,
    cards,
    selectedActionId,
    actionLogs,
    evidence,
  } = useChatStore(
    useShallow((state) => ({
      assistantDrawerOpen: state.assistantDrawerOpen,
      advancedPanelOpen: state.advancedPanelOpen,
      assistantSection: state.assistantSection,
      projectId: state.projectId,
      model: state.model,
      povMode: state.povMode,
      povAnchor: state.povAnchor,
      ragMode: state.ragMode,
      deterministicFirst: state.deterministicFirst,
      thinkingEnabled: state.thinkingEnabled,
      sessionId: state.sessionId,
      streaming: state.streaming,
      error: state.error,
      usage: state.usage,
      messages: state.messages,
      actions: state.actions,
      pendingActionIds: state.pendingActionIds,
      settings: state.settings,
      cards: state.cards,
      selectedActionId: state.selectedActionId,
      actionLogs: state.actionLogs,
      evidence: state.evidence,
    }))
  );

  const {
    setAssistantDrawerOpen: setAssistantDrawerOpenState,
    setAdvancedPanelOpen,
    setAssistantSection,
    setProjectId,
    setModel,
    setPovMode,
    setPovAnchor,
    setRagMode,
    setDeterministicFirst,
    setThinkingEnabled,
    setSessionId,
    setStreaming,
    setError,
    setUsage,
    setMessages,
    appendMessage,
    appendMessageDelta,
    updateMessage,
    setActions,
    setPendingActionIds,
    setSettings,
    setCards,
    setSelectedActionId,
    setActionLogs,
    setEvidence,
    resetSessionState,
  } = useChatStore(
    useShallow((state) => ({
      setAssistantDrawerOpen: state.setAssistantDrawerOpen,
      setAdvancedPanelOpen: state.setAdvancedPanelOpen,
      setAssistantSection: state.setAssistantSection,
      setProjectId: state.setProjectId,
      setModel: state.setModel,
      setPovMode: state.setPovMode,
      setPovAnchor: state.setPovAnchor,
      setRagMode: state.setRagMode,
      setDeterministicFirst: state.setDeterministicFirst,
      setThinkingEnabled: state.setThinkingEnabled,
      setSessionId: state.setSessionId,
      setStreaming: state.setStreaming,
      setError: state.setError,
      setUsage: state.setUsage,
      setMessages: state.setMessages,
      appendMessage: state.appendMessage,
      appendMessageDelta: state.appendMessageDelta,
      updateMessage: state.updateMessage,
      setActions: state.setActions,
      setPendingActionIds: state.setPendingActionIds,
      setSettings: state.setSettings,
      setCards: state.setCards,
      setSelectedActionId: state.setSelectedActionId,
      setActionLogs: state.setActionLogs,
      setEvidence: state.setEvidence,
      resetSessionState: state.resetSessionState,
    }))
  );

  const { theme, setTheme } = useTheme();

  const [input, setInput] = useState("");
  const [projectSessions, setProjectSessions] = useState<ChatSessionSummary[]>([]);
  const [consistencyAudits, setConsistencyAudits] = useState<ConsistencyAuditReport[]>([]);
  const [consistencyAuditRunning, setConsistencyAuditRunning] = useState(false);
  const [traceEvents, setTraceEvents] = useState<ChatStreamTraceEvent[]>([]);
  const [graphTimeline, setGraphTimeline] = useState<GraphTimelineSnapshot | null>(null);
  const [graphTimelineLoading, setGraphTimelineLoading] = useState(false);
  const [graphTimelineChapterIndex, setGraphTimelineChapterIndex] = useState(0);
  const [draftText, setDraftText] = useState("");
  const [selectedDraftText, setSelectedDraftText] = useState("");
  const [mutatingActionId, setMutatingActionId] = useState<number | null>(null);
  const [chapters, setChapters] = useState<ProjectChapter[]>([]);
  const [activeChapterId, setActiveChapterId] = useState<number | null>(null);
  const [volumes, setVolumes] = useState<ProjectVolume[]>([]);
  const [activeVolumeId, setActiveVolumeId] = useState<number | null>(null);
  const [volumeOutlineDraft, setVolumeOutlineDraft] = useState("");
  const [sceneBeats, setSceneBeats] = useState<SceneBeat[]>([]);
  const [activeSceneBeatId, setActiveSceneBeatId] = useState<number | null>(null);
  const [newBeatContent, setNewBeatContent] = useState("");
  const [foreshadowCards, setForeshadowCards] = useState<ForeshadowingCard[]>([]);
  const [overdueForeshadowCards, setOverdueForeshadowCards] = useState<ForeshadowingCard[]>([]);
  const [foreshadowDraftTitle, setForeshadowDraftTitle] = useState("");
  const [foreshadowDraftDescription, setForeshadowDraftDescription] = useState("");
  const [planningBusy, setPlanningBusy] = useState(false);
  const [draftTitle, setDraftTitle] = useState("第1章");
  const [draftVersion, setDraftVersion] = useState(0);
  const [draftUpdatedAt, setDraftUpdatedAt] = useState<string | null>(null);
  const [draftRevisions, setDraftRevisions] = useState<ProjectChapterRevision[]>([]);
  const [draftLoading, setDraftLoading] = useState(false);
  const [draftSaving, setDraftSaving] = useState(false);
  const [dragChapterId, setDragChapterId] = useState<number | null>(null);
  const [draftFocusMode, setDraftFocusMode] = useState(false);
  const [typewriterModeEnabled, setTypewriterModeEnabled] = useState(true);
  const [writingTheme, setWritingTheme] = useState<WritingTheme>("paper");
  const [promptTemplates, setPromptTemplates] = useState<PromptTemplate[]>([]);
  const [activePromptTemplateId, setActivePromptTemplateId] = useState<number | null>(null);
  const [templateDraftId, setTemplateDraftId] = useState<number | null>(null);
  const [templateName, setTemplateName] = useState("默认模板");
  const [templateSystemPrompt, setTemplateSystemPrompt] = useState("");
  const [templateUserPromptPrefix, setTemplateUserPromptPrefix] = useState("");
  const [templateKnowledgeSettingKeys, setTemplateKnowledgeSettingKeys] = useState<string[]>([]);
  const [templateKnowledgeCardIds, setTemplateKnowledgeCardIds] = useState<number[]>([]);
  const [templateSaving, setTemplateSaving] = useState(false);
  const [templateRevisions, setTemplateRevisions] = useState<PromptTemplateRevision[]>([]);
  const [templateRevisionsLoading, setTemplateRevisionsLoading] = useState(false);
  const [modelProfiles, setModelProfiles] = useState<ModelProfile[]>([]);
  const [selectedModelProfileId, setSelectedModelProfileId] = useState<string | null>(null);
  const [activeModelProfileId, setActiveModelProfileId] = useState<string | null>(null);
  const [suggestionModelProfileId, setSuggestionModelProfileId] = useState<string | null>(null);
  const [chatTemperatureProfile, setChatTemperatureProfile] = useState<"action" | "chat" | "brainstorm">("chat");
  const [suggestionTemperatureProfile, setSuggestionTemperatureProfile] = useState<"suggestion" | "chat" | "action" | "brainstorm">(
    "suggestion"
  );
  const [modelProfileDraftIdInput, setModelProfileDraftIdInput] = useState("");
  const [modelProfileName, setModelProfileName] = useState("");
  const [modelProfileProvider, setModelProfileProvider] = useState<
    "openai_compatible" | "deepseek" | "claude" | "gemini"
  >("openai_compatible");
  const [modelProfileBaseUrl, setModelProfileBaseUrl] = useState("");
  const [modelProfileApiKey, setModelProfileApiKey] = useState("");
  const [modelProfileApiKeyMasked, setModelProfileApiKeyMasked] = useState<string | null>(null);
  const [clearModelProfileApiKey, setClearModelProfileApiKey] = useState(false);
  const [modelProfileModel, setModelProfileModel] = useState("");
  const [modelProfileSaving, setModelProfileSaving] = useState(false);
  const [referenceProjectInput, setReferenceProjectInput] = useState("");
  const [temperatureOverrideInput, setTemperatureOverrideInput] = useState("");
  const [contextWindowProfile, setContextWindowProfile] = useState<
    "balanced" | "chapter_focus" | "world_focus" | "minimal"
  >("balanced");
  const [workbenchPanelVisibility, setWorkbenchPanelVisibility] = useState<WorkbenchPanelVisibility>({
    actions: true,
    prompt: false,
    planning: true,
    snapshot: false,
  });
  const [zenMode, setZenMode] = useState(false);
  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false);
  const [lastStreamMetrics, setLastStreamMetrics] = useState<ChatStreamTimingMetrics | null>(null);
  const [streamLatencySamples, setStreamLatencySamples] = useState<StreamLatencySample[]>([]);
  const [tokenUsageSamples, setTokenUsageSamples] = useState<TokenUsageSample[]>([]);
  const [retrievalHitSamples, setRetrievalHitSamples] = useState<RetrievalHitSample[]>([]);
  const [hasOpenedAssistantOnce, setHasOpenedAssistantOnce] = useState(false);
  const [autoSaveState, setAutoSaveState] = useState<DraftAutoSaveState>("idle");
  const [autoSaveAt, setAutoSaveAt] = useState<string | null>(null);
  const [localRecoveryNotice, setLocalRecoveryNotice] = useState<string | null>(null);
  const autoSaveTimerRef = useRef<number | null>(null);
  const localRecoveryTimerRef = useRef<number | null>(null);
  const composerInputRef = useRef<HTMLTextAreaElement | null>(null);
  const draftEditorRef = useRef<HTMLDivElement | null>(null);
  const assistantDrawerRef = useRef<HTMLElement | null>(null);
  const settingsDialogRef = useRef<HTMLElement | null>(null);
  const assistantDrawerReturnFocusRef = useRef<HTMLElement | null>(null);
  const settingsDialogReturnFocusRef = useRef<HTMLElement | null>(null);
  const previousAssistantDrawerOpenRef = useRef(false);
  const previousSettingsDialogOpenRef = useRef(false);
  const writingShortcutStateRef = useRef<{
    advancedPanelOpen: boolean;
    assistantDrawerOpen: boolean;
    settingsDialogOpen: boolean;
    draftLoading: boolean;
    draftSaving: boolean;
    activeChapterId: number | null;
  }>({
    advancedPanelOpen,
    assistantDrawerOpen: false,
    settingsDialogOpen: false,
    draftLoading: false,
    draftSaving: false,
    activeChapterId: null,
  });
  const lastSavedDraftRef = useRef<{
    chapterId: number | null;
    volumeId: number | null;
    title: string;
    content: string;
  }>({
    chapterId: null,
    volumeId: null,
    title: "第1章",
    content: "",
  });

  useSettingsStorage({
    projectId,
    modelProfiles,
    suggestionModelProfileId,
    setSuggestionModelProfileId,
    chatTemperatureProfile,
    setChatTemperatureProfile,
    suggestionTemperatureProfile,
    setSuggestionTemperatureProfile,
  });

  const typewriterRafRef = useRef<number | null>(null);
  const typewriterModeEnabledRef = useRef(typewriterModeEnabled);
  const typewriterDimmingEnabledRef = useRef(false);
  const activeTypewriterParagraphRef = useRef<HTMLParagraphElement | null>(null);
  const actionLogsCacheRef = useRef<Map<number, ActionAuditLog[]>>(new Map());
  const actionLogsInFlightRef = useRef<Map<number, Promise<ActionAuditLog[]>>>(new Map());
  const actionLogsRequestSeqRef = useRef(0);
  const chapterSnapshotInFlightRef = useRef<Map<string, Promise<ChapterSnapshotData>>>(new Map());
  const chapterSnapshotRequestSeqRef = useRef(0);
  const draftSnapshotInFlightRef = useRef<Map<string, Promise<DraftSnapshotData>>>(new Map());
  const draftSnapshotRequestSeqRef = useRef(0);
  const planningSnapshotInFlightRef = useRef<Map<string, Promise<PlanningSnapshotData>>>(new Map());
  const planningSnapshotRequestSeqRef = useRef(0);
  const projectSnapshotInFlightRef = useRef<Map<string, Promise<ProjectSnapshotData>>>(new Map());
  const projectSnapshotRequestSeqRef = useRef(0);
  const fullSessionSnapshotInFlightRef = useRef<Map<string, Promise<FullSessionSnapshotData>>>(new Map());
  const fullSessionSnapshotRequestSeqRef = useRef(0);
  const templateRevisionsInFlightRef = useRef<Map<string, Promise<PromptTemplateRevision[]>>>(new Map());
  const templateRevisionsRequestSeqRef = useRef(0);
  const postChatSnapshotCacheRef = useRef<Map<string, { at: number; data: PostChatSnapshotData }>>(new Map());
  const postChatSnapshotInFlightRef = useRef<Map<string, Promise<PostChatSnapshotData>>>(new Map());
  const postChatSnapshotRequestSeqRef = useRef(0);
  const graphTimelineRequestSeqRef = useRef(0);

  const scheduleTypewriterScroll = (currentEditor: Editor) => {
    if (!typewriterModeEnabledRef.current) return;
    if (typeof window === "undefined") return;
    if (typewriterRafRef.current) {
      window.cancelAnimationFrame(typewriterRafRef.current);
      typewriterRafRef.current = null;
    }

    typewriterRafRef.current = window.requestAnimationFrame(() => {
      typewriterRafRef.current = null;
      const proseMirror = draftEditorRef.current?.querySelector<HTMLElement>(".ProseMirror");
      if (!proseMirror) return;
      if (proseMirror.scrollHeight <= proseMirror.clientHeight + 8) return;

      const pos = currentEditor.state.selection.$head.pos;
      let coords: { top: number; bottom: number };
      try {
        coords = currentEditor.view.coordsAtPos(pos);
      } catch {
        return;
      }
      const containerRect = proseMirror.getBoundingClientRect();
      if (containerRect.height <= 0) return;
      const cursorCenter = (coords.top + coords.bottom) / 2;
      const expectedCenter = containerRect.top + containerRect.height / 2;
      const delta = cursorCenter - expectedCenter;
      if (Math.abs(delta) < 6) return;
      proseMirror.scrollTop += delta;
    });
  };

  const clearTypewriterParagraphFocus = () => {
    const proseMirror = draftEditorRef.current?.querySelector<HTMLElement>(".ProseMirror");
    if (proseMirror) {
      proseMirror.classList.remove("typewriter-dimming-active");
    }
    if (activeTypewriterParagraphRef.current) {
      activeTypewriterParagraphRef.current.classList.remove("is-typewriter-active-paragraph");
      activeTypewriterParagraphRef.current = null;
    }
  };

  const syncTypewriterParagraphFocus = (currentEditor: Editor) => {
    const proseMirror = draftEditorRef.current?.querySelector<HTMLElement>(".ProseMirror");
    if (!proseMirror) return;
    if (!typewriterDimmingEnabledRef.current) {
      clearTypewriterParagraphFocus();
      return;
    }

    proseMirror.classList.add("typewriter-dimming-active");
    const domAtSelection = currentEditor.view.domAtPos(currentEditor.state.selection.$head.pos);
    const baseNode = domAtSelection.node instanceof HTMLElement ? domAtSelection.node : domAtSelection.node.parentElement;
    const nextParagraph = baseNode?.closest("p") as HTMLParagraphElement | null;

    if (activeTypewriterParagraphRef.current && activeTypewriterParagraphRef.current !== nextParagraph) {
      activeTypewriterParagraphRef.current.classList.remove("is-typewriter-active-paragraph");
    }

    if (nextParagraph && proseMirror.contains(nextParagraph)) {
      nextParagraph.classList.add("is-typewriter-active-paragraph");
      activeTypewriterParagraphRef.current = nextParagraph;
      return;
    }

    activeTypewriterParagraphRef.current = null;
  };

  const sortedActions = useMemo(
    () => [...actions].sort((a, b) => b.id - a.id),
    [actions]
  );
  const activeChapter = useMemo(
    () => chapters.find((item) => item.id === activeChapterId) ?? null,
    [chapters, activeChapterId]
  );
  const maxChapterIndex = useMemo(
    () => Math.max(1, ...chapters.map((item) => Number(item.chapter_index || 0))),
    [chapters]
  );
  const activeChapterIndex = useMemo(
    () => Number(activeChapter?.chapter_index || 0),
    [activeChapter]
  );
  const activeVolume = useMemo(
    () => volumes.find((item) => item.id === activeVolumeId) ?? null,
    [volumes, activeVolumeId]
  );
  const activeChapterPos = useMemo(
    () => chapters.findIndex((item) => item.id === activeChapterId),
    [chapters, activeChapterId]
  );
  const canMoveChapterUp = activeChapterPos > 0;
  const canMoveChapterDown = activeChapterPos >= 0 && activeChapterPos < chapters.length - 1;
  const chapterOutlines = useMemo<ChapterOutlineEntry[]>(
    () => {
      let maxWords = 1;
      const baseEntries = chapters.map((chapter) => {
        const wordCount = (chapter.content || "").replace(/\s+/g, "").length;
        if (wordCount > maxWords) maxWords = wordCount;
        const lines = (chapter.content || "").split("\n");
        let preview = "（暂无正文）";
        for (let i = 0; i < lines.length; i++) {
          const trimmed = lines[i].trim();
          if (trimmed.length > 0) {
            preview = trimmed.slice(0, 42);
            break;
          }
        }
        return {
          id: chapter.id,
          chapterIndex: chapter.chapter_index,
          title: chapter.title,
          wordCount,
          preview,
          updatedAt: chapter.updated_at,
          progressPercent: 0,
        };
      });
      return baseEntries.map((item) => ({
        ...item,
        progressPercent: item.wordCount <= 0 ? 0 : Math.max(4, Math.round((item.wordCount / maxWords) * 100)),
      }));
    },
    [chapters]
  );
  const totalChapterWords = useMemo(
    () => chapterOutlines.reduce((sum, item) => sum + item.wordCount, 0),
    [chapterOutlines]
  );
  const todayAddedWords = useMemo(() => {
    const today = new Date();
    return chapterOutlines.reduce((sum, item) => {
      if (!isSameLocalDate(item.updatedAt, today)) return sum;
      return sum + item.wordCount;
    }, 0);
  }, [chapterOutlines]);
  const hasAppliedAction = useMemo(
    () => actions.some((item) => String(item.status || "").trim().toLowerCase() === "applied"),
    [actions]
  );
  const onboardingChecklist = useMemo(
    () => ({
      hasRoleCard: cards.length > 0,
      hasDraftParagraph: totalChapterWords >= 30,
      hasOpenedAssistant: hasOpenedAssistantOnce,
      hasAppliedSuggestion: hasAppliedAction,
    }),
    [cards.length, totalChapterWords, hasOpenedAssistantOnce, hasAppliedAction]
  );

  useEffect(() => {
    if (activeChapter?.volume_id && volumes.some((item) => item.id === activeChapter.volume_id)) {
      setActiveVolumeId(activeChapter.volume_id);
      return;
    }
    if (volumes.length > 0) {
      setActiveVolumeId(volumes[0].id);
      return;
    }
    setActiveVolumeId(null);
  }, [activeChapter?.volume_id, volumes]);

  useEffect(() => {
    setVolumeOutlineDraft(activeVolume?.outline ?? "");
  }, [activeVolume?.id, activeVolume?.outline]);

  const latestAssistantReply = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const message = messages[i];
      if (message.role === "assistant" && !message.streaming) {
        const content = message.content.trim();
        if (content) {
          return content;
        }
      }
    }
    return "";
  }, [messages]);
  const suggestionChapterGoal = useMemo(() => {
    const selectedBeat =
      sceneBeats.find((item) => item.id === activeSceneBeatId) ??
      sceneBeats.find((item) => item.status !== "done") ??
      sceneBeats[0] ??
      null;
    const beatGoal = String(selectedBeat?.content || "").trim();
    const chapterTitle = String(activeChapter?.title || "").trim();
    if (!beatGoal && !chapterTitle) return "";
    if (!beatGoal) return chapterTitle.slice(0, 160);
    if (!chapterTitle) return beatGoal.slice(0, 220);
    return `${chapterTitle}：${beatGoal}`.slice(0, 220);
  }, [sceneBeats, activeSceneBeatId, activeChapter?.title]);
  const suggestionActiveRoles = useMemo(() => {
    const roleHints = ["role", "character", "人物", "角色", "主角", "配角", "反派"];
    const picked: string[] = [];
    const seen = new Set<string>();
    const pushRole = (value: unknown) => {
      const text = String(value || "").trim();
      if (!text) return;
      const normalized = text.toLowerCase();
      if (seen.has(normalized)) return;
      seen.add(normalized);
      picked.push(text.slice(0, 24));
    };

    if (povMode === "character" && povAnchor.trim()) {
      pushRole(povAnchor.trim());
    }

    cards.forEach((card) => {
      const title = String(card.title || "").trim();
      if (!title) return;
      const contentObj = card.content && typeof card.content === "object" ? (card.content as Record<string, unknown>) : {};
      const typeValue = String(contentObj.type ?? contentObj.card_type ?? "").toLowerCase();
      const tags = Array.isArray(contentObj.tags)
        ? contentObj.tags.map((item) => String(item || "").toLowerCase())
        : [];
      const titleValue = title.toLowerCase();
      const isRoleCard = roleHints.some(
        (hint) => titleValue.includes(hint) || typeValue.includes(hint) || tags.some((tag) => tag.includes(hint))
      );
      if (isRoleCard) {
        pushRole(title);
      }
    });

    if (picked.length === 0) {
      cards.slice(0, 4).forEach((card) => pushRole(card.title));
    }
    return picked.slice(0, 8);
  }, [cards, povMode, povAnchor]);

  const draftWordCount = useMemo(() => {
    return draftText.replace(/\s+/g, "").length;
  }, [draftText]);

  const activePromptTemplate = useMemo(
    () => promptTemplates.find((item) => item.id === activePromptTemplateId) ?? null,
    [promptTemplates, activePromptTemplateId]
  );
  const isWritingWorkspace = !advancedPanelOpen;
  const debugPromptPanelReady = advancedPanelOpen;
  const proSnapshotPanelReady = advancedPanelOpen;
  const referenceProjectIds = useMemo(
    () => parseReferenceProjectIds(referenceProjectInput, projectId),
    [referenceProjectInput, projectId]
  );
  const selectedKnowledgeSettings = useMemo(
    () => (debugPromptPanelReady ? settings.filter((item) => templateKnowledgeSettingKeys.includes(item.key)) : []),
    [debugPromptPanelReady, settings, templateKnowledgeSettingKeys]
  );
  const selectedKnowledgeCards = useMemo(
    () => (debugPromptPanelReady ? cards.filter((item) => templateKnowledgeCardIds.includes(item.id)) : []),
    [debugPromptPanelReady, cards, templateKnowledgeCardIds]
  );
  const missingSettingKeys = useMemo(
    () =>
      debugPromptPanelReady
        ? templateKnowledgeSettingKeys.filter((key) => !settings.some((item) => item.key === key))
        : [],
    [debugPromptPanelReady, templateKnowledgeSettingKeys, settings]
  );
  const missingCardIds = useMemo(
    () =>
      debugPromptPanelReady
        ? templateKnowledgeCardIds.filter((id) => !cards.some((item) => item.id === id))
        : [],
    [debugPromptPanelReady, templateKnowledgeCardIds, cards]
  );
  const entityHighlightHints = useMemo(
    () => collectEntityHighlightHints(settings, cards),
    [settings, cards]
  );
  const estimatedPromptChars = useMemo(() => {
    if (!debugPromptPanelReady) return 0;
    const settingsChars = selectedKnowledgeSettings.reduce(
      (acc, item) => acc + JSON.stringify(item.value ?? {}).length,
      0
    );
    const cardsChars = selectedKnowledgeCards.reduce(
      (acc, item) => acc + JSON.stringify(item.content ?? {}).length,
      0
    );
    return templateSystemPrompt.length + templateUserPromptPrefix.length + settingsChars + cardsChars;
  }, [debugPromptPanelReady, selectedKnowledgeCards, selectedKnowledgeSettings, templateSystemPrompt, templateUserPromptPrefix]);
  const degradedReasons = useMemo(() => {
    const gate = evidence?.policy?.quality_gate;
    if (!gate || typeof gate !== "object" || !Array.isArray(gate.degrade_reasons)) {
      return [];
    }
    return gate.degrade_reasons
      .map((item) => String(item || "").trim())
      .filter((item) => item.length > 0);
  }, [evidence]);
  const retrievalDegraded = useMemo(() => {
    const gate = evidence?.policy?.quality_gate;
    if (!gate || typeof gate !== "object") return false;
    return Boolean(gate.degraded);
  }, [evidence]);
  const temperatureOverride = useMemo(() => {
    const raw = temperatureOverrideInput.trim();
    if (!raw) return null;
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return null;
    return Math.min(2, Math.max(0, parsed));
  }, [temperatureOverrideInput]);
  const awarenessTags = useMemo(
    () => collectAwarenessTags(evidence, { includeDebugSignals: advancedPanelOpen }),
    [advancedPanelOpen, evidence]
  );

  useEffect(() => {
    if (assistantDrawerOpen) {
      setHasOpenedAssistantOnce(true);
    }
  }, [assistantDrawerOpen]);

  useEffect(() => {
    if (!lastStreamMetrics) return;
    const completeMs = toFiniteNumber(lastStreamMetrics.completeMs);
    if (completeMs === null) return;
    setStreamLatencySamples((prev) => {
      const next = [...prev, { at: new Date().toISOString(), completeMs }];
      if (next.length <= PERFORMANCE_SAMPLE_LIMIT) return next;
      return next.slice(next.length - PERFORMANCE_SAMPLE_LIMIT);
    });
  }, [lastStreamMetrics]);

  useEffect(() => {
    if (!usage || typeof usage !== "object") return;
    const usageRecord = usage as Record<string, unknown>;
    const promptTokens = toFiniteNumber(usageRecord.prompt_tokens ?? usageRecord.input_tokens ?? usageRecord.promptTokens);
    const completionTokens = toFiniteNumber(
      usageRecord.completion_tokens ?? usageRecord.output_tokens ?? usageRecord.completionTokens
    );
    const directTotal = toFiniteNumber(usageRecord.total_tokens ?? usageRecord.totalTokens ?? usageRecord.total);
    const total =
      directTotal ??
      ((promptTokens ?? 0) + (completionTokens ?? 0) > 0 ? (promptTokens ?? 0) + (completionTokens ?? 0) : null);
    if (total === null || total <= 0) return;
    setTokenUsageSamples((prev) => {
      const next = [...prev, { at: new Date().toISOString(), total }];
      if (next.length <= PERFORMANCE_SAMPLE_LIMIT) return next;
      return next.slice(next.length - PERFORMANCE_SAMPLE_LIMIT);
    });
  }, [usage]);

  useEffect(() => {
    if (!evidence) return;
    const dsl = Math.max(0, Number(evidence.summary?.dsl ?? 0) || 0);
    const graph = Math.max(0, Number(evidence.summary?.graph ?? 0) || 0);
    const rag = Math.max(0, Number(evidence.summary?.rag ?? 0) || 0);
    setRetrievalHitSamples((prev) => {
      const next = [...prev, { at: new Date().toISOString(), dsl, graph, rag }];
      if (next.length <= PERFORMANCE_SAMPLE_LIMIT) return next;
      return next.slice(next.length - PERFORMANCE_SAMPLE_LIMIT);
    });
  }, [evidence]);

  useEffect(() => {
    setHasOpenedAssistantOnce(false);
    setStreamLatencySamples([]);
    setTokenUsageSamples([]);
    setRetrievalHitSamples([]);
  }, [projectId]);

  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        heading: false,
        blockquote: false,
        bulletList: false,
        orderedList: false,
        listItem: false,
        codeBlock: false,
        horizontalRule: false,
      }),
      Placeholder.configure({
        placeholder:
          "在这里写正文。你可以选中一段文字后让 AI 润色/扩写，也可以把助手回复一键写回正文。",
      }),
      EntityInlineHintExtension,
      SemanticDiffExtension,
    ],
    content: toEditorDoc(""),
    editable: false,
    onUpdate: ({ editor: currentEditor }) => {
      setDraftText(readEditorText(currentEditor));
      setSelectedDraftText(readSelectedText(currentEditor));
      scheduleTypewriterScroll(currentEditor);
      syncTypewriterParagraphFocus(currentEditor);
    },
    onSelectionUpdate: ({ editor: currentEditor }) => {
      setSelectedDraftText(readSelectedText(currentEditor));
      scheduleTypewriterScroll(currentEditor);
      syncTypewriterParagraphFocus(currentEditor);
    },
    onBlur: ({ editor: currentEditor }) => {
      setSelectedDraftText(readSelectedText(currentEditor));
    },
  });

  const resetTemplateDraft = () => {
    setTemplateDraftId(null);
    setTemplateName("默认模板");
    setTemplateSystemPrompt("");
    setTemplateUserPromptPrefix("");
    setTemplateKnowledgeSettingKeys([]);
    setTemplateKnowledgeCardIds([]);
  };

  const loadTemplateIntoDraft = (template: PromptTemplate | null) => {
    if (!template) {
      resetTemplateDraft();
      return;
    }
    setTemplateDraftId(template.id);
    setTemplateName(template.name);
    setTemplateSystemPrompt(template.system_prompt);
    setTemplateUserPromptPrefix(template.user_prompt_prefix);
    setTemplateKnowledgeSettingKeys(template.knowledge_setting_keys ?? []);
    setTemplateKnowledgeCardIds(template.knowledge_card_ids ?? []);
  };

  const resetModelProfileDraft = () => {
    setSelectedModelProfileId(null);
    setModelProfileDraftIdInput("");
    setModelProfileName("");
    setModelProfileProvider("openai_compatible");
    setModelProfileBaseUrl("");
    setModelProfileApiKey("");
    setModelProfileApiKeyMasked(null);
    setClearModelProfileApiKey(false);
    setModelProfileModel("");
  };

  const loadModelProfileIntoDraft = (profile: ModelProfile | null) => {
    if (!profile) {
      resetModelProfileDraft();
      return;
    }
    setSelectedModelProfileId(profile.profile_id);
    setModelProfileDraftIdInput(profile.profile_id);
    setModelProfileName((profile.name || "").trim());
    if (profile.provider === "deepseek" || profile.provider === "claude" || profile.provider === "gemini") {
      setModelProfileProvider(profile.provider);
    } else {
      setModelProfileProvider("openai_compatible");
    }
    setModelProfileBaseUrl((profile.base_url || "").trim());
    setModelProfileApiKey("");
    setModelProfileApiKeyMasked((profile.api_key_masked || "").trim() || null);
    setClearModelProfileApiKey(false);
    setModelProfileModel((profile.model || "").trim());
  };

  const loadChapterSnapshot = async (nextProjectId: number, chapterId: number) => {
    const requestSeq = chapterSnapshotRequestSeqRef.current + 1;
    chapterSnapshotRequestSeqRef.current = requestSeq;
    const cacheKey = `${nextProjectId}:${chapterId}`;

    let snapshotPromise = chapterSnapshotInFlightRef.current.get(cacheKey);
    if (!snapshotPromise) {
      snapshotPromise = (async (): Promise<ChapterSnapshotData> => {
        const [chapter, revisions, beats, overdueForeshadows] = await Promise.all([
          getProjectChapter(nextProjectId, chapterId),
          getProjectChapterRevisions(nextProjectId, chapterId, 20),
          getSceneBeats(nextProjectId, chapterId).catch(() => [] as SceneBeat[]),
          getForeshadowingCards(nextProjectId, {
            overdue_for_chapter_id: chapterId,
            chapter_gap: 50,
          }).catch(() => [] as ForeshadowingCard[]),
        ]);
        return { chapter, revisions, beats, overdueForeshadows };
      })();
      chapterSnapshotInFlightRef.current.set(cacheKey, snapshotPromise);
    }

    try {
      const snapshotData = await snapshotPromise;
      if (chapterSnapshotRequestSeqRef.current !== requestSeq) {
        return;
      }
      const { chapter, revisions, beats, overdueForeshadows } = snapshotData;
      const localSnapshot = readDraftRecoverySnapshot(nextProjectId, chapter.id);
      const shouldRestore = localSnapshot ? shouldRestoreDraftRecovery(localSnapshot, chapter) : false;
      const resolvedTitle = shouldRestore && localSnapshot ? localSnapshot.title : chapter.title;
      const resolvedContent = shouldRestore && localSnapshot ? localSnapshot.content : chapter.content;
      // Defensive: legacy local snapshots or unexpected API payloads can yield undefined.
      const nextTitle = typeof resolvedTitle === "string" ? resolvedTitle : "";
      const nextContent = typeof resolvedContent === "string" ? resolvedContent : "";

      setDraftTitle(nextTitle);
      setDraftText(nextContent);
      setDraftVersion(chapter.version);
      setDraftUpdatedAt(chapter.updated_at);
      setDraftRevisions(revisions);
      setSceneBeats(beats);
      const pendingBeat = beats.find((item) => item.status !== "done");
      setActiveSceneBeatId(pendingBeat?.id ?? beats[0]?.id ?? null);
      setOverdueForeshadowCards(overdueForeshadows);
      setSelectedDraftText("");
      lastSavedDraftRef.current = {
        chapterId: chapter.id,
        volumeId: chapter.volume_id ?? null,
        title: typeof chapter.title === "string" ? chapter.title : "",
        content: typeof chapter.content === "string" ? chapter.content : "",
      };
      if (shouldRestore && localSnapshot) {
        setAutoSaveState("pending");
        setAutoSaveAt(localSnapshot.saved_at);
        setLocalRecoveryNotice(`已恢复本地快照（${formatDateTime(localSnapshot.saved_at)}），请继续写作或手动保存。`);
      } else {
        setAutoSaveState("idle");
        setLocalRecoveryNotice(null);
        if (localSnapshot) {
          clearDraftRecoverySnapshot(nextProjectId, chapter.id);
        }
      }
    } finally {
      if (chapterSnapshotInFlightRef.current.get(cacheKey) === snapshotPromise) {
        chapterSnapshotInFlightRef.current.delete(cacheKey);
      }
    }
  };

  const refreshDraftSnapshot = async (nextProjectId: number, preferredChapterId?: number | null) => {
    const requestSeq = draftSnapshotRequestSeqRef.current + 1;
    draftSnapshotRequestSeqRef.current = requestSeq;
    const cacheKey = `${nextProjectId}`;

    let snapshotPromise = draftSnapshotInFlightRef.current.get(cacheKey);
    if (!snapshotPromise) {
      snapshotPromise = (async (): Promise<DraftSnapshotData> => {
        const [chapterList, volumeList, foreshadowList] = await Promise.all([
          getProjectChapters(nextProjectId),
          getProjectVolumes(nextProjectId).catch(() => [] as ProjectVolume[]),
          getForeshadowingCards(nextProjectId).catch(() => [] as ForeshadowingCard[]),
        ]);
        return { chapterList, volumeList, foreshadowList };
      })();
      draftSnapshotInFlightRef.current.set(cacheKey, snapshotPromise);
    }

    try {
      const snapshotData = await snapshotPromise;
      if (draftSnapshotRequestSeqRef.current !== requestSeq) {
        return;
      }

      const { chapterList, volumeList, foreshadowList } = snapshotData;
      setChapters(chapterList);
      setVolumes(volumeList);
      setForeshadowCards(foreshadowList);
      if (chapterList.length === 0) {
        setActiveChapterId(null);
        setDraftTitle("第1章");
        setDraftText("");
        setDraftVersion(0);
        setDraftUpdatedAt(null);
        setDraftRevisions([]);
        setSceneBeats([]);
        setActiveSceneBeatId(null);
        setOverdueForeshadowCards([]);
        setSelectedDraftText("");
        lastSavedDraftRef.current = {
          chapterId: null,
          volumeId: null,
          title: "第1章",
          content: "",
        };
        setAutoSaveState("idle");
        setLocalRecoveryNotice(null);
        return;
      }

      const preferred = preferredChapterId ?? activeChapterId;
      const resolved =
        preferred && chapterList.some((item) => item.id === preferred)
          ? preferred
          : chapterList[0].id;
      setActiveChapterId(resolved);
      if (draftSnapshotRequestSeqRef.current !== requestSeq) {
        return;
      }
      await loadChapterSnapshot(nextProjectId, resolved);
    } finally {
      if (draftSnapshotInFlightRef.current.get(cacheKey) === snapshotPromise) {
        draftSnapshotInFlightRef.current.delete(cacheKey);
      }
    }
  };

  const { isDraftDirty, persistDraftSnapshot, saveDraftSnapshot, switchChapter } = useDraftWorkspaceFlow({
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
  });

  const refreshPlanningData = async (nextProjectId: number, chapterId: number | null) => {
    const requestSeq = planningSnapshotRequestSeqRef.current + 1;
    planningSnapshotRequestSeqRef.current = requestSeq;
    const cacheKey = `${nextProjectId}:${chapterId ?? "none"}`;

    let snapshotPromise = planningSnapshotInFlightRef.current.get(cacheKey);
    if (!snapshotPromise) {
      snapshotPromise = (async (): Promise<PlanningSnapshotData> => {
        const [volumeList, foreshadowList] = await Promise.all([
          getProjectVolumes(nextProjectId).catch(() => [] as ProjectVolume[]),
          getForeshadowingCards(nextProjectId).catch(() => [] as ForeshadowingCard[]),
        ]);
        if (!chapterId) {
          return {
            volumeList,
            foreshadowList,
            beats: [],
            overdue: [],
          };
        }
        const [beats, overdue] = await Promise.all([
          getSceneBeats(nextProjectId, chapterId).catch(() => [] as SceneBeat[]),
          getForeshadowingCards(nextProjectId, {
            overdue_for_chapter_id: chapterId,
            chapter_gap: 50,
          }).catch(() => [] as ForeshadowingCard[]),
        ]);
        return { volumeList, foreshadowList, beats, overdue };
      })();
      planningSnapshotInFlightRef.current.set(cacheKey, snapshotPromise);
    }

    try {
      const snapshotData = await snapshotPromise;
      if (planningSnapshotRequestSeqRef.current !== requestSeq) {
        return;
      }
      setVolumes(snapshotData.volumeList);
      setForeshadowCards(snapshotData.foreshadowList);
      if (chapterId) {
        const pendingBeat = snapshotData.beats.find((item) => item.status !== "done");
        setSceneBeats(snapshotData.beats);
        setActiveSceneBeatId((prev) => {
          if (prev && snapshotData.beats.some((item) => item.id === prev)) return prev;
          return pendingBeat?.id ?? snapshotData.beats[0]?.id ?? null;
        });
        setOverdueForeshadowCards(snapshotData.overdue);
      } else {
        setSceneBeats([]);
        setActiveSceneBeatId(null);
        setOverdueForeshadowCards([]);
      }
    } finally {
      if (planningSnapshotInFlightRef.current.get(cacheKey) === snapshotPromise) {
        planningSnapshotInFlightRef.current.delete(cacheKey);
      }
    }
  };

  useEffect(() => {
    let cancelled = false;
    setDraftLoading(true);
    (async () => {
      try {
        await Promise.all([refreshDraftSnapshot(projectId), refreshProjectSnapshot(projectId)]);
        if (cancelled) return;
      } catch (loadError) {
        if (cancelled) return;
        const message = loadError instanceof Error ? loadError.message : "读取章节正文失败";
        setError(message);
      } finally {
        if (!cancelled) {
          setDraftLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId, setError]);

  useEffect(() => {
    if (!editor) return;
    editor.setEditable(!draftLoading && !!activeChapterId);
  }, [editor, draftLoading, activeChapterId]);

  useEffect(() => {
    if (!editor) return;
    editor.view.dispatch(editor.state.tr.setMeta(entityHintPluginKey, entityHighlightHints));
  }, [editor, entityHighlightHints]);

  useEffect(() => {
    if (!editor) return;
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<{ action?: string; id?: string }>).detail;
      if (!detail?.id) return;
      if (detail.action === "accept") {
        editor.commands.acceptDiffSuggestion(detail.id);
        setDraftText(readEditorText(editor));
        return;
      }
      if (detail.action === "reject") {
        editor.commands.rejectDiffSuggestion(detail.id);
      }
    };
    editor.view.dom.addEventListener("semantic-diff-action", handler as EventListener);
    return () => {
      editor.view.dom.removeEventListener("semantic-diff-action", handler as EventListener);
    };
  }, [editor]);

  useEffect(() => {
    typewriterModeEnabledRef.current = typewriterModeEnabled;
  }, [typewriterModeEnabled]);

  useEffect(() => {
    typewriterDimmingEnabledRef.current =
      isWritingWorkspace && typewriterModeEnabled && (draftFocusMode || zenMode);
    if (!editor) {
      clearTypewriterParagraphFocus();
      return;
    }
    syncTypewriterParagraphFocus(editor);
  }, [editor, isWritingWorkspace, typewriterModeEnabled, draftFocusMode, zenMode, activeChapterId]);

  useEffect(() => {
    return () => {
      clearTypewriterParagraphFocus();
    };
  }, []);

  useEffect(() => {
    if (!editor) return;
    const normalized = normalizeEditorText(draftText);
    if (readEditorText(editor) !== normalized) {
      editor.commands.clearDiffSuggestions();
      editor.commands.setContent(toEditorDoc(normalized), { emitUpdate: false });
      scheduleTypewriterScroll(editor);
      syncTypewriterParagraphFocus(editor);
    }
  }, [draftText, editor]);

  useEffect(() => {
    if (!editor || !typewriterModeEnabled) return;
    scheduleTypewriterScroll(editor);
  }, [editor, typewriterModeEnabled, activeChapterId]);

  useEffect(() => {
    if (isWritingWorkspace) return;
    if (!zenMode) return;
    setZenMode(false);
  }, [isWritingWorkspace, zenMode]);

  const openSettingsDialog = useCallback(() => {
    if (!settingsDialogOpen && typeof document !== "undefined") {
      const active = document.activeElement;
      settingsDialogReturnFocusRef.current = active instanceof HTMLElement ? active : null;
    }
    setSettingsDialogOpen(true);
  }, [settingsDialogOpen]);

  const closeSettingsDialog = useCallback(() => setSettingsDialogOpen(false), []);

  const setAssistantDrawerOpen = useCallback(
    (next: boolean | ((prev: boolean) => boolean)) => {
      const resolved = typeof next === "function" ? next(assistantDrawerOpen) : next;
      setAssistantDrawerOpenState(resolved);
    },
    [assistantDrawerOpen, setAssistantDrawerOpenState]
  );

  const toggleAdvancedPanel = useCallback(() => {
    const next = !advancedPanelOpen;
    setAdvancedPanelOpen(next);
    if (next) {
      setAssistantDrawerOpen(false);
    }
  }, [advancedPanelOpen, setAdvancedPanelOpen, setAssistantDrawerOpen]);

  const toggleZenMode = useCallback(() => {
    if (advancedPanelOpen) return;
    setZenMode((prev) => {
      const next = !prev;
      if (next) {
        setAssistantDrawerOpen(false);
        setSettingsDialogOpen(false);
      }
      return next;
    });
  }, [advancedPanelOpen, setAssistantDrawerOpen]);

  const toggleTypewriterMode = useCallback(() => {
    setTypewriterModeEnabled((prev) => !prev);
  }, []);

  const toggleDraftFocusMode = useCallback(() => {
    setDraftFocusMode((prev) => !prev);
  }, []);

  const bindActiveChapterToVolume = async (volumeId: number) => {
    if (!activeChapterId) return;
    if (draftSaving || draftLoading) return;
    setPlanningBusy(true);
    setError(null);
    try {
      await saveProjectChapter(projectId, activeChapterId, {
        title: draftTitle.trim() || "未命名章节",
        content: draftText,
        volume_id: volumeId,
        expected_version: null,
      });
      await refreshDraftSnapshot(projectId, activeChapterId);
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "绑定章节到卷失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const createVolume = async () => {
    if (planningBusy) return;
    setPlanningBusy(true);
    setError(null);
    try {
      const created = await createProjectVolume(projectId, {
        title: null,
        outline: "",
      });
      await refreshPlanningData(projectId, activeChapterId);
      setActiveVolumeId(created.id);
      setVolumeOutlineDraft(created.outline);
      if (activeChapterId) {
        await bindActiveChapterToVolume(created.id);
      }
    } catch (createError) {
      const message = createError instanceof Error ? createError.message : "创建卷失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const saveVolumeOutline = async () => {
    if (!activeVolumeId) {
      setError("请先选择卷");
      return;
    }
    if (planningBusy) return;
    const target = volumes.find((item) => item.id === activeVolumeId);
    if (!target) {
      setError("卷不存在，请刷新后重试");
      return;
    }
    setPlanningBusy(true);
    setError(null);
    try {
      await updateProjectVolume(projectId, activeVolumeId, {
        title: target.title,
        outline: volumeOutlineDraft,
      });
      await refreshPlanningData(projectId, activeChapterId);
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "保存卷纲失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const createBeatForActiveChapter = async () => {
    if (!activeChapterId) {
      setError("请先选择章节");
      return;
    }
    const content = newBeatContent.trim();
    if (!content) {
      setError("请先填写 Beat 内容");
      return;
    }
    if (planningBusy) return;
    setPlanningBusy(true);
    setError(null);
    try {
      const created = await createSceneBeat(projectId, activeChapterId, {
        content,
        status: "pending",
      });
      setNewBeatContent("");
      await refreshPlanningData(projectId, activeChapterId);
      setActiveSceneBeatId(created.id);
    } catch (createError) {
      const message = createError instanceof Error ? createError.message : "创建 Beat 失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const toggleBeatStatus = async (beatId: number, done: boolean) => {
    if (!activeChapterId || planningBusy) return;
    const target = sceneBeats.find((item) => item.id === beatId);
    if (!target) return;
    setPlanningBusy(true);
    setError(null);
    try {
      await updateSceneBeat(projectId, activeChapterId, beatId, {
        content: target.content,
        status: done ? "done" : "pending",
      });
      await refreshPlanningData(projectId, activeChapterId);
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "更新 Beat 失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const deleteBeat = async (beatId: number) => {
    if (!activeChapterId || planningBusy) return;
    if (!window.confirm("确认删除这个 Beat 吗？")) return;
    setPlanningBusy(true);
    setError(null);
    try {
      await deleteSceneBeat(projectId, activeChapterId, beatId);
      await refreshPlanningData(projectId, activeChapterId);
    } catch (deleteError) {
      const message = deleteError instanceof Error ? deleteError.message : "删除 Beat 失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const createForeshadow = async () => {
    const title = foreshadowDraftTitle.trim();
    if (!title) {
      setError("请先填写伏笔标题");
      return;
    }
    if (planningBusy) return;
    setPlanningBusy(true);
    setError(null);
    try {
      await createForeshadowingCard(projectId, {
        title,
        description: foreshadowDraftDescription,
        planted_in_chapter_id: activeChapterId,
      });
      setForeshadowDraftTitle("");
      setForeshadowDraftDescription("");
      await refreshPlanningData(projectId, activeChapterId);
    } catch (createError) {
      const message = createError instanceof Error ? createError.message : "创建伏笔卡失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const toggleForeshadowStatus = async (card: ForeshadowingCard, nextStatus: "open" | "resolved") => {
    if (planningBusy) return;
    setPlanningBusy(true);
    setError(null);
    try {
      await updateForeshadowingCard(projectId, card.id, {
        title: card.title,
        description: card.description,
        status: nextStatus,
        planted_in_chapter_id: card.planted_in_chapter_id,
        resolved_in_chapter_id: nextStatus === "resolved" ? (activeChapterId ?? card.resolved_in_chapter_id) : null,
      });
      await refreshPlanningData(projectId, activeChapterId);
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "更新伏笔卡失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const deleteForeshadow = async (cardId: number) => {
    if (planningBusy) return;
    if (!window.confirm("确认删除这个伏笔卡吗？")) return;
    setPlanningBusy(true);
    setError(null);
    try {
      await deleteForeshadowingCard(projectId, cardId);
      await refreshPlanningData(projectId, activeChapterId);
    } catch (deleteError) {
      const message = deleteError instanceof Error ? deleteError.message : "删除伏笔卡失败";
      setError(message);
    } finally {
      setPlanningBusy(false);
    }
  };

  const applyAssistantToDraft = (mode: "insert" | "replace") => {
    if (!editor) return;
    const assistantText = latestAssistantReply;
    if (!assistantText) {
      setError("暂无可写回的助手回复。先让助手生成内容。");      
      return;
    }

    setError(null);

    const { from, to, empty } = editor.state.selection;
    const shouldReplace = mode === "replace" && !empty;

    if (shouldReplace) {
      const originalText = editor.state.doc.textBetween(from, to, "\n", "\n");
      const suggestions = buildDiffSuggestions(from, to, originalText, assistantText);
      editor.commands.setDiffSuggestions(suggestions);
      if (suggestions.length === 0) {
        setError("助手回复与当前选区没有差异。");
      }
        return;
    }

    let insertion = assistantText;
    if (!shouldReplace) {
      const beforeChar = from > 1 ? editor.state.doc.textBetween(from - 1, from, "\n", "\n") : "";        
      const prefix = from > 1 && !/\s/.test(beforeChar) ? "\n\n" : "";
      insertion = `${prefix}${assistantText}`;
    }

    const insertionContent = toEditorDoc(insertion).content ?? [];
    editor.chain().focus().insertContentAt({ from, to }, insertionContent).run();
    setDraftText(readEditorText(editor));
    setSelectedDraftText("");
  };

  const fillPromptFromSelection = async (mode: "polish" | "expand") => {
    if (!selectedDraftText) {
      setError("请先在正文工作区选中一段文本。");
      return;
    }
    if (streaming) {
      setError("当前会话正在生成中，请稍后再试。");
      return;
    }

    setError(null);
    try {
      const result =
        mode === "expand"
          ? await expandSelection({
              project_id: projectId,
              chapter_id: activeChapterId,
              scene_beat_id: activeSceneBeatId,
              prompt_template_id: activePromptTemplateId,
              text: selectedDraftText,
              model: model.trim() ? model.trim() : null,
              model_profile_id: suggestionModelProfileId,
              temperature_profile: "chat",
              temperature_override: temperatureOverride,
            })
          : await polishSelection({
              project_id: projectId,
              chapter_id: activeChapterId,
              scene_beat_id: activeSceneBeatId,
              prompt_template_id: activePromptTemplateId,
              text: selectedDraftText,
              model: model.trim() ? model.trim() : null,
              model_profile_id: suggestionModelProfileId,
              temperature_profile: "chat",
              temperature_override: temperatureOverride,
            });
      const nextText = (result.suggestion || "").trim();
      if (!nextText) {
        setError("润色/扩写未返回可用内容");
        return;
      }
      if (!editor) {
        setError("编辑器不可用，暂无法应用结果");
        return;
      }
      const { from, to } = editor.state.selection;
      const suggestions = buildDiffSuggestions(from, to, selectedDraftText, nextText);
      editor.commands.setDiffSuggestions(suggestions);
      if (suggestions.length === 0) {
        setError("润色/扩写结果与原文没有差异");
      }
    } catch (rewriteError) {
      const message = rewriteError instanceof Error ? rewriteError.message : "润色/扩写失败";
      setError(message);
    }
  };

  const rollbackDraftToVersion = async (targetVersion: number) => {
    if (!activeChapterId) {
      setError("请先选择章节");
      return;
    }
    if (draftSaving) return;
    setDraftSaving(true);
    setError(null);
    try {
      await rollbackProjectChapter(projectId, activeChapterId, {
        target_version: targetVersion,
      });
      await refreshDraftSnapshot(projectId, activeChapterId);
    } catch (rollbackError) {
      const message = rollbackError instanceof Error ? rollbackError.message : "回滚正文失败";
      setError(message);
    } finally {
      setDraftSaving(false);
    }
  };

  const createChapterAndSwitch = async () => {
    if (draftSaving || draftLoading) return;
    setDraftSaving(true);
    setError(null);
    try {
      const created = await createProjectChapter(projectId, {
        title: null,
        volume_id: activeVolumeId,
      });
      await refreshDraftSnapshot(projectId, created.id);
    } catch (createError) {
      const message = createError instanceof Error ? createError.message : "创建章节失败";
      setError(message);
    } finally {
      setDraftSaving(false);
    }
  };

  const moveActiveChapter = async (direction: "up" | "down") => {
    if (!activeChapterId) {
      setError("请先选择章节");
      return;
    }
    if (draftSaving || draftLoading) return;
    setDraftSaving(true);
    setError(null);
    try {
      await moveProjectChapter(projectId, activeChapterId, {
        direction,
      });
      await refreshDraftSnapshot(projectId, activeChapterId);
    } catch (moveError) {
      const message = moveError instanceof Error ? moveError.message : "调整章节顺序失败";
      setError(message);
    } finally {
      setDraftSaving(false);
    }
  };

  const deleteActiveChapter = async () => {
    if (!activeChapterId) {
      setError("请先选择章节");
      return;
    }
    const label = activeChapter ? `${activeChapter.chapter_index}. ${activeChapter.title}` : `#${activeChapterId}`;
    if (!window.confirm(`确认删除章节「${label}」吗？`)) {
      return;
    }
    if (draftSaving || draftLoading) return;
    setDraftSaving(true);
    setError(null);
    try {
      const result = await deleteProjectChapter(projectId, activeChapterId);
      await refreshDraftSnapshot(projectId, result.active_chapter_id);
    } catch (deleteError) {
      const message = deleteError instanceof Error ? deleteError.message : "删除章节失败";
      setError(message);
    } finally {
      setDraftSaving(false);
    }
  };

  const handleOutlineDragStart = useCallback((chapterId: number) => {
    setDragChapterId(chapterId);
  }, []);

  const handleOutlineDragEnd = useCallback(() => {
    setDragChapterId(null);
  }, []);

  const reorderByDrag = async (targetChapterId: number) => {
    if (!dragChapterId || dragChapterId === targetChapterId) {
      setDragChapterId(null);
      return;
    }
    const fromIndex = chapters.findIndex((item) => item.id === dragChapterId);
    const toIndex = chapters.findIndex((item) => item.id === targetChapterId);
    if (fromIndex < 0 || toIndex < 0) {
      setDragChapterId(null);
      return;
    }

    const reordered = [...chapters];
    const [moved] = reordered.splice(fromIndex, 1);
    reordered.splice(toIndex, 0, moved);
    const orderedIds = reordered.map((item) => item.id);

    setDraftSaving(true);
    setError(null);
    try {
      await reorderProjectChapters(projectId, {
        ordered_ids: orderedIds,
      });
      await refreshDraftSnapshot(projectId, activeChapterId ?? dragChapterId);
    } catch (reorderError) {
      const message = reorderError instanceof Error ? reorderError.message : "拖拽排序失败";
      setError(message);
    } finally {
      setDraftSaving(false);
      setDragChapterId(null);
    }
  };

  const applyTemplatesSnapshot = (
    templates: PromptTemplate[],
    preferredActiveTemplateId?: number | null,
    preferredDraftTemplateId?: number | null
  ) => {
    setPromptTemplates(templates);
    if (templates.length === 0) {
      setActivePromptTemplateId(null);
      resetTemplateDraft();
      setTemplateRevisions([]);
      return;
    }

    const resolvedActive =
      preferredActiveTemplateId && templates.some((item) => item.id === preferredActiveTemplateId)
        ? preferredActiveTemplateId
        : templates[0].id;
    setActivePromptTemplateId(resolvedActive);

    const resolvedDraftId =
      preferredDraftTemplateId && templates.some((item) => item.id === preferredDraftTemplateId)
        ? preferredDraftTemplateId
        : resolvedActive;
    const draftTemplate = templates.find((item) => item.id === resolvedDraftId) ?? null;
    loadTemplateIntoDraft(draftTemplate);
  };

  const applyModelProfilesSnapshot = (
    profiles: ModelProfile[],
    preferredSelectedProfileId?: string | null
  ) => {
    setModelProfiles(profiles);
    const resolvedActive =
      profiles.find((item) => Boolean(item.is_active))?.profile_id ?? null;
    setActiveModelProfileId(resolvedActive);

    if (profiles.length === 0) {
      resetModelProfileDraft();
      return;
    }

    const selectedCandidate =
      (preferredSelectedProfileId && profiles.some((item) => item.profile_id === preferredSelectedProfileId)
        ? preferredSelectedProfileId
        : null) ??
      (selectedModelProfileId && profiles.some((item) => item.profile_id === selectedModelProfileId)
        ? selectedModelProfileId
        : null) ??
      (resolvedActive && profiles.some((item) => item.profile_id === resolvedActive) ? resolvedActive : null) ??
      profiles[0].profile_id;

    const draftProfile = profiles.find((item) => item.profile_id === selectedCandidate) ?? null;
    loadModelProfileIntoDraft(draftProfile);
  };

  const refreshTemplateRevisions = async (nextProjectId: number, templateId: number | null) => {
    const requestSeq = templateRevisionsRequestSeqRef.current + 1;
    templateRevisionsRequestSeqRef.current = requestSeq;

    if (!templateId) {
      setTemplateRevisions([]);
      setTemplateRevisionsLoading(false);
      return;
    }

    const cacheKey = `${nextProjectId}:${templateId}`;
    let revisionsPromise = templateRevisionsInFlightRef.current.get(cacheKey);
    if (!revisionsPromise) {
      revisionsPromise = getProjectPromptTemplateRevisions(nextProjectId, templateId, 20);
      templateRevisionsInFlightRef.current.set(cacheKey, revisionsPromise);
    }

    setTemplateRevisionsLoading(true);
    try {
      const revisions = await revisionsPromise;
      if (templateRevisionsRequestSeqRef.current !== requestSeq) {
        return;
      }
      setTemplateRevisions(revisions);
    } catch (revisionError) {
      if (templateRevisionsRequestSeqRef.current !== requestSeq) {
        return;
      }
      const message = revisionError instanceof Error ? revisionError.message : "读取模板历史失败";
      setError(message);
      setTemplateRevisions([]);
    } finally {
      if (templateRevisionsInFlightRef.current.get(cacheKey) === revisionsPromise) {
        templateRevisionsInFlightRef.current.delete(cacheKey);
      }
      if (templateRevisionsRequestSeqRef.current === requestSeq) {
        setTemplateRevisionsLoading(false);
      }
    }
  };

  const refreshGraphTimeline = async (requestedChapterIndex: number) => {
    const chapter = Number.isFinite(requestedChapterIndex) ? Math.max(0, Math.floor(requestedChapterIndex)) : 0;
    if (projectId <= 0 || chapter <= 0) {
      setGraphTimeline(null);
      return;
    }
    const requestSeq = graphTimelineRequestSeqRef.current + 1;
    graphTimelineRequestSeqRef.current = requestSeq;
    setGraphTimelineLoading(true);
    try {
      const snapshot = await getProjectGraphTimeline(projectId, chapter, 260);
      if (graphTimelineRequestSeqRef.current !== requestSeq) return;
      setGraphTimeline(snapshot);
    } catch {
      if (graphTimelineRequestSeqRef.current !== requestSeq) return;
      setGraphTimeline({
        project_id: projectId,
        chapter_index: chapter,
        nodes: [],
        edges: [],
        stats: { source: "fallback", nodes: 0, edges: 0 },
      });
    } finally {
      if (graphTimelineRequestSeqRef.current === requestSeq) {
        setGraphTimelineLoading(false);
      }
    }
  };

  useEffect(() => {
    if (activeChapterIndex > 0) {
      setGraphTimelineChapterIndex(activeChapterIndex);
    }
  }, [activeChapterIndex]);

  useEffect(() => {
    const targetChapter = graphTimelineChapterIndex > 0 ? graphTimelineChapterIndex : activeChapterIndex;
    if (projectId <= 0 || targetChapter <= 0) {
      setGraphTimeline(null);
      return;
    }
    void refreshGraphTimeline(targetChapter);
  }, [projectId, graphTimelineChapterIndex, activeChapterIndex]);

  const refreshProjectSnapshot = async (nextProjectId: number) => {
    const requestSeq = projectSnapshotRequestSeqRef.current + 1;
    projectSnapshotRequestSeqRef.current = requestSeq;
    const cacheKey = `${nextProjectId}`;

    let snapshotPromise = projectSnapshotInFlightRef.current.get(cacheKey);
    if (!snapshotPromise) {
      snapshotPromise = (async (): Promise<ProjectSnapshotData> => {
        const [
          settingsData,
          auditsData,
          modelProfilesData,
          cardsData,
          templatesData,
          volumesData,
          foreshadowData,
          sessionsData,
        ] = await Promise.all([
          getProjectSettings(nextProjectId),
          getProjectConsistencyAudits(nextProjectId, 8).catch(() => [] as ConsistencyAuditReport[]),
          getProjectModelProfiles(nextProjectId).catch(() => [] as ModelProfile[]),
          getProjectCards(nextProjectId),
          getProjectPromptTemplates(nextProjectId),
          getProjectVolumes(nextProjectId).catch(() => [] as ProjectVolume[]),
          getForeshadowingCards(nextProjectId).catch(() => [] as ForeshadowingCard[]),
          getProjectSessions(nextProjectId).catch(() => [] as ChatSessionSummary[]),
        ]);
        return {
          settingsData,
          auditsData,
          modelProfilesData,
          cardsData,
          templatesData,
          volumesData,
          foreshadowData,
          sessionsData,
        };
      })();
      projectSnapshotInFlightRef.current.set(cacheKey, snapshotPromise);
    }

    try {
      const snapshotData = await snapshotPromise;
      if (projectSnapshotRequestSeqRef.current !== requestSeq) {
        return;
      }
      setSettings(snapshotData.settingsData);
      setConsistencyAudits(snapshotData.auditsData);
      applyModelProfilesSnapshot(snapshotData.modelProfilesData, selectedModelProfileId);
      setCards(snapshotData.cardsData);
      setVolumes(snapshotData.volumesData);
      setForeshadowCards(snapshotData.foreshadowData);
      setProjectSessions(snapshotData.sessionsData);
      applyTemplatesSnapshot(snapshotData.templatesData, activePromptTemplateId, templateDraftId);
    } finally {
      if (projectSnapshotInFlightRef.current.get(cacheKey) === snapshotPromise) {
        projectSnapshotInFlightRef.current.delete(cacheKey);
      }
    }
  };

  const refreshSessionSnapshot = async (nextSessionId: number, nextProjectId: number) => {
    const requestSeq = fullSessionSnapshotRequestSeqRef.current + 1;
    fullSessionSnapshotRequestSeqRef.current = requestSeq;
    const cacheKey = `${nextProjectId}:${nextSessionId}`;

    let snapshotPromise = fullSessionSnapshotInFlightRef.current.get(cacheKey);
    if (!snapshotPromise) {
      snapshotPromise = (async (): Promise<FullSessionSnapshotData> => {
        const [
          messagesData,
          actionsData,
          settingsData,
          auditsData,
          modelProfilesData,
          cardsData,
          templatesData,
          volumesData,
          foreshadowData,
          sessionsData,
        ] =
          await Promise.all([
            getSessionMessages(nextSessionId),
            getSessionActions(nextSessionId),
            getProjectSettings(nextProjectId),
            getProjectConsistencyAudits(nextProjectId, 8).catch(() => [] as ConsistencyAuditReport[]),
            getProjectModelProfiles(nextProjectId).catch(() => [] as ModelProfile[]),
            getProjectCards(nextProjectId),
            getProjectPromptTemplates(nextProjectId),
            getProjectVolumes(nextProjectId).catch(() => [] as ProjectVolume[]),
            getForeshadowingCards(nextProjectId).catch(() => [] as ForeshadowingCard[]),
            getProjectSessions(nextProjectId).catch(() => [] as ChatSessionSummary[]),
          ]);
        return {
          messagesData,
          actionsData,
          settingsData,
          auditsData,
          modelProfilesData,
          cardsData,
          templatesData,
          volumesData,
          foreshadowData,
          sessionsData,
        };
      })();
      fullSessionSnapshotInFlightRef.current.set(cacheKey, snapshotPromise);
    }

    try {
      const snapshotData = await snapshotPromise;
      if (fullSessionSnapshotRequestSeqRef.current !== requestSeq) {
        return;
      }
      setMessages(snapshotData.messagesData.map(toUiMessage));
      setActions(snapshotData.actionsData);
      setSettings(snapshotData.settingsData);
      setConsistencyAudits(snapshotData.auditsData);
      applyModelProfilesSnapshot(snapshotData.modelProfilesData, selectedModelProfileId);
      setCards(snapshotData.cardsData);
      setVolumes(snapshotData.volumesData);
      setForeshadowCards(snapshotData.foreshadowData);
      setProjectSessions(snapshotData.sessionsData);
      applyTemplatesSnapshot(snapshotData.templatesData, activePromptTemplateId, templateDraftId);
    } finally {
      if (fullSessionSnapshotInFlightRef.current.get(cacheKey) === snapshotPromise) {
        fullSessionSnapshotInFlightRef.current.delete(cacheKey);
      }
    }
  };

  const refreshSessionPostChatSnapshot = async (nextSessionId: number, nextProjectId: number) => {
    const requestSeq = postChatSnapshotRequestSeqRef.current + 1;
    postChatSnapshotRequestSeqRef.current = requestSeq;
    const cacheKey = `${nextProjectId}:${nextSessionId}`;
    const cached = postChatSnapshotCacheRef.current.get(cacheKey);
    const now = Date.now();

    const applySnapshotData = (snapshotData: PostChatSnapshotData) => {
      if (postChatSnapshotRequestSeqRef.current !== requestSeq) {
        return;
      }
      setMessages(snapshotData.messagesData.map(toUiMessage));
      setActions(snapshotData.actionsData);
      setSettings(snapshotData.settingsData);
      setCards(snapshotData.cardsData);
      setProjectSessions(snapshotData.sessionsData);
    };

    if (cached && now - cached.at <= POST_CHAT_SNAPSHOT_TTL_MS) {
      applySnapshotData(cached.data);
      return;
    }

    let snapshotPromise = postChatSnapshotInFlightRef.current.get(cacheKey);
    if (!snapshotPromise) {
      snapshotPromise = (async (): Promise<PostChatSnapshotData> => {
        const [messagesData, actionsData, settingsData, cardsData, sessionsData] = await Promise.all([
          getSessionMessages(nextSessionId),
          getSessionActions(nextSessionId),
          getProjectSettings(nextProjectId),
          getProjectCards(nextProjectId),
          getProjectSessions(nextProjectId).catch(() => [] as ChatSessionSummary[]),
        ]);
        return { messagesData, actionsData, settingsData, cardsData, sessionsData };
      })();
      postChatSnapshotInFlightRef.current.set(cacheKey, snapshotPromise);
    }

    try {
      const snapshotData = await snapshotPromise;
      postChatSnapshotCacheRef.current.set(cacheKey, {
        at: Date.now(),
        data: snapshotData,
      });
      applySnapshotData(snapshotData);
    } finally {
      if (postChatSnapshotInFlightRef.current.get(cacheKey) === snapshotPromise) {
        postChatSnapshotInFlightRef.current.delete(cacheKey);
      }
    }
  };

  useEffect(() => {
    void refreshTemplateRevisions(projectId, templateDraftId);
  }, [projectId, templateDraftId]);

  useEffect(() => {
    if (!selectedModelProfileId) {
      if (modelProfiles.length === 0) {
        resetModelProfileDraft();
      } else {
        setModelProfileDraftIdInput("");
        setModelProfileName("");
        setModelProfileProvider("openai_compatible");
        setModelProfileBaseUrl("");
        setModelProfileApiKey("");
        setModelProfileApiKeyMasked(null);
        setClearModelProfileApiKey(false);
        setModelProfileModel("");
      }
      return;
    }
    const profile = modelProfiles.find((item) => item.profile_id === selectedModelProfileId) ?? null;
    if (!profile) return;
    loadModelProfileIntoDraft(profile);
  }, [modelProfiles, selectedModelProfileId]);

  const typewriterDimmingEnabled =
    isWritingWorkspace && typewriterModeEnabled && (draftFocusMode || zenMode);
  typewriterDimmingEnabledRef.current = typewriterDimmingEnabled;

  writingShortcutStateRef.current = {
    advancedPanelOpen,
    assistantDrawerOpen,
    settingsDialogOpen,
    draftLoading,
    draftSaving,
    activeChapterId,
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === "z") {
        if (advancedPanelOpen) return;
        event.preventDefault();
        toggleZenMode();
        return;
      }
      if (event.key === "F11") {
        if (advancedPanelOpen) return;
        event.preventDefault();
        toggleZenMode();
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === "a") {
        event.preventDefault();
        toggleAssistantDrawer();
      }
      if (event.key === "Escape") {
        closeAssistantDrawer();
        closeSettingsDialog();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [advancedPanelOpen]);

  useEffect(() => {
    if (typeof document === "undefined") return;
    const zenClass = "zen-mode-active";
    document.body.classList.toggle(zenClass, zenMode);
    return () => {
      document.body.classList.remove(zenClass);
    };
  }, [zenMode]);

  useLayoutEffect(() => {
    actionLogsCacheRef.current.clear();
    actionLogsInFlightRef.current.clear();
    actionLogsRequestSeqRef.current = 0;
    chapterSnapshotInFlightRef.current.clear();
    chapterSnapshotRequestSeqRef.current = 0;
    draftSnapshotInFlightRef.current.clear();
    draftSnapshotRequestSeqRef.current = 0;
    planningSnapshotInFlightRef.current.clear();
    planningSnapshotRequestSeqRef.current = 0;
    fullSessionSnapshotInFlightRef.current.clear();
    fullSessionSnapshotRequestSeqRef.current = 0;
    templateRevisionsInFlightRef.current.clear();
    templateRevisionsRequestSeqRef.current = 0;
    postChatSnapshotCacheRef.current.clear();
    postChatSnapshotInFlightRef.current.clear();
    postChatSnapshotRequestSeqRef.current = 0;
  }, [projectId, sessionId]);

  useLayoutEffect(() => {
    projectSnapshotInFlightRef.current.clear();
    projectSnapshotRequestSeqRef.current = 0;
  }, [projectId]);

  useEffect(() => {
    const wasOpen = previousSettingsDialogOpenRef.current;
    previousSettingsDialogOpenRef.current = settingsDialogOpen;
    if (!wasOpen || settingsDialogOpen) return;
    const target = settingsDialogReturnFocusRef.current;
    if (!target || typeof document === "undefined" || !document.contains(target)) return;
    window.setTimeout(() => target.focus(), 0);
  }, [settingsDialogOpen]);

  useEffect(() => {
    if (!assistantDrawerOpen || settingsDialogOpen) return;
    if (typeof document === "undefined") return;
    const container = assistantDrawerRef.current;
    if (!container) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const focusables = getFocusableElements(container);
      if (focusables.length === 0) return;

      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      const active = document.activeElement instanceof HTMLElement ? document.activeElement : null;

      if (event.shiftKey) {
        if (!active || active === first || !container.contains(active)) {
          event.preventDefault();
          last.focus();
        }
        return;
      }

      if (!active || active === last || !container.contains(active)) {
        event.preventDefault();
        first.focus();
      }
    };

    const onFocusIn = (event: FocusEvent) => {
      const target = event.target;
      if (target instanceof Node && container.contains(target)) return;
      const focusables = getFocusableElements(container);
      if (focusables.length > 0) {
        focusables[0].focus();
        return;
      }
      container.focus();
    };

    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("focusin", onFocusIn);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("focusin", onFocusIn);
    };
  }, [assistantDrawerOpen, settingsDialogOpen]);

  useEffect(() => {
    if (!settingsDialogOpen) return;
    if (typeof document === "undefined") return;
    const container = settingsDialogRef.current;
    if (!container) return;

    focusFirstDialogElement(container);

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const focusables = getFocusableElements(container);
      if (focusables.length === 0) return;

      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      const active = document.activeElement instanceof HTMLElement ? document.activeElement : null;

      if (event.shiftKey) {
        if (!active || active === first || !container.contains(active)) {
          event.preventDefault();
          last.focus();
        }
        return;
      }

      if (!active || active === last || !container.contains(active)) {
        event.preventDefault();
        first.focus();
      }
    };

    const onFocusIn = (event: FocusEvent) => {
      const target = event.target;
      if (target instanceof Node && container.contains(target)) return;
      focusFirstDialogElement(container);
    };

    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("focusin", onFocusIn);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("focusin", onFocusIn);
    };
  }, [settingsDialogOpen]);

  const {
    toggleAssistantDrawer,
    closeAssistantDrawer,
    openAssistantDrawer,
    focusAssistantComposer,
    startNewSession,
    switchSession,
    renameSession,
    deleteSession,
    handleSend,
  } = useAssistantSessionFlow({
    projectSessions,
    input,
    streaming,
    assistantDrawerOpen,
    setAssistantDrawerOpen,
    setAssistantSection,
    composerInputRef,
    assistantDrawerReturnFocusRef,
    previousAssistantDrawerOpenRef,
    streamOptions: {
      projectId,
      sessionId,
      activeChapterId,
      activeSceneBeatId,
      activePromptTemplateId,
      activeModelProfileId,
      model,
      povMode,
      povAnchor,
      ragMode,
      deterministicFirst,
      thinkingEnabled,
      referenceProjectIds,
      chatTemperatureProfile,
      temperatureOverride,
      contextWindowProfile,
    },
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
  });

  const loadLogs = async (actionId: number, options?: { force?: boolean }) => {
    const force = Boolean(options?.force);
    const requestSeq = actionLogsRequestSeqRef.current + 1;
    actionLogsRequestSeqRef.current = requestSeq;
    setSelectedActionId(actionId);
    if (!force) {
      const cached = actionLogsCacheRef.current.get(actionId);
      if (cached) {
        setActionLogs(cached);
        return;
      }
    }

    let logsPromise = actionLogsInFlightRef.current.get(actionId);
    if (!logsPromise || force) {
      logsPromise = getActionLogs(actionId);
      actionLogsInFlightRef.current.set(actionId, logsPromise);
    }

    try {
      const logs = await logsPromise;
      actionLogsCacheRef.current.set(actionId, logs);
      if (actionLogsRequestSeqRef.current !== requestSeq) {
        return;
      }
      setActionLogs(logs);
    } catch (loadError) {
      if (actionLogsRequestSeqRef.current !== requestSeq) {
        return;
      }
      const message =
        loadError instanceof Error ? loadError.message : "读取日志失败";
      setError(message);
      setActionLogs([]);
    } finally {
      if (actionLogsInFlightRef.current.get(actionId) === logsPromise) {
        actionLogsInFlightRef.current.delete(actionId);
      }
    }
  };

  const mutateAction = async (
    action: ChatAction,
    decision: "apply" | "reject" | "undo"
  ) => {
    if (mutatingActionId !== null) return;
    setMutatingActionId(action.id);
    setError(null);
    try {
      const eventPayload =
        decision === "apply" && isEntityMergeActionType(action.action_type)
          ? { manual_confirmed: true, review_surface: "action_drawer" }
          : {};
      const updatedAction = await decideAction(action.id, decision, eventPayload);
      const nextStatus = String(updatedAction.status || "").trim().toLowerCase();
      const canUseLightSessionRefresh =
        decision === "reject"
        || nextStatus === "rejected"
        || nextStatus === "failed"
        || (decision === "apply" && nextStatus !== "applied")
        || (decision === "undo" && nextStatus !== "undone");
      if (sessionId) {
        if (canUseLightSessionRefresh) {
          await refreshSessionPostChatSnapshot(sessionId, projectId);
        } else {
          await refreshSessionSnapshot(sessionId, projectId);
        }
      } else {
        await refreshProjectSnapshot(projectId);
      }
      if (selectedActionId === action.id) {
        actionLogsCacheRef.current.delete(action.id);
        await loadLogs(action.id, { force: true });
      }
    } catch (mutationError) {
      const message =
        mutationError instanceof Error ? mutationError.message : "动作执行失败";
      setError(message);
    } finally {
      setMutatingActionId(null);
    }
  };
  const handleRefresh = async () => {
    setError(null);
    try {
      void preheatContextPack(projectId).catch(() => undefined);
      if (sessionId) {
        await Promise.all([refreshSessionSnapshot(sessionId, projectId), refreshDraftSnapshot(projectId)]);
      } else {
        await Promise.all([refreshProjectSnapshot(projectId), refreshDraftSnapshot(projectId)]);
      }
    } catch (refreshError) {
      const message =
        refreshError instanceof Error ? refreshError.message : "刷新失败";
      setError(message);
    }
  };

  const handleActiveTemplateChange = (value: string) => {
    const nextId = value ? Number(value) : null;
    setActivePromptTemplateId(nextId);
    const template = promptTemplates.find((item) => item.id === nextId) ?? null;
    loadTemplateIntoDraft(template);
  };

  const startCreateTemplateDraft = () => {
    setError(null);
    resetTemplateDraft();
  };

  const saveTemplateDraft = async () => {
    const name = templateName.trim();
    if (!name) {
      setError("模板名称不能为空");
      return;
    }
    if (templateSaving) return;
    setTemplateSaving(true);
    setError(null);
    try {
      const payload = {
        name,
        system_prompt: templateSystemPrompt,
        user_prompt_prefix: templateUserPromptPrefix,
        knowledge_setting_keys: templateKnowledgeSettingKeys,
        knowledge_card_ids: templateKnowledgeCardIds,
      };
      const saved = templateDraftId
        ? await updateProjectPromptTemplate(projectId, templateDraftId, payload)
        : await createProjectPromptTemplate(projectId, payload);
      const templates = await getProjectPromptTemplates(projectId);
      const preferredActive = templateDraftId ? activePromptTemplateId : saved.id;
      applyTemplatesSnapshot(templates, preferredActive, saved.id);
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "保存模板失败";
      setError(message);
    } finally {
      setTemplateSaving(false);
    }
  };

  const deleteTemplateDraft = async () => {
    if (!templateDraftId) {
      setError("请先选择要删除的模板");
      return;
    }
    const target = promptTemplates.find((item) => item.id === templateDraftId);
    const label = target?.name ?? `#${templateDraftId}`;
    if (!window.confirm(`确认删除模板「${label}」吗？`)) return;
    if (templateSaving) return;
    setTemplateSaving(true);
    setError(null);
    try {
      await deleteProjectPromptTemplate(projectId, templateDraftId);
      const templates = await getProjectPromptTemplates(projectId);
      const preferredActive =
        activePromptTemplateId === templateDraftId ? null : activePromptTemplateId;
      applyTemplatesSnapshot(templates, preferredActive, null);
    } catch (deleteError) {
      const message = deleteError instanceof Error ? deleteError.message : "删除模板失败";
      setError(message);
    } finally {
      setTemplateSaving(false);
    }
  };

  const copyTemplateDraft = async () => {
    const name = templateName.trim();
    if (!name) {
      setError("请先填写模板名称");
      return;
    }
    if (templateSaving) return;
    setTemplateSaving(true);
    setError(null);
    try {
      const copied = await createProjectPromptTemplate(projectId, {
        name: `${name}-副本`.slice(0, 128),
        system_prompt: templateSystemPrompt,
        user_prompt_prefix: templateUserPromptPrefix,
        knowledge_setting_keys: templateKnowledgeSettingKeys,
        knowledge_card_ids: templateKnowledgeCardIds,
      });
      const templates = await getProjectPromptTemplates(projectId);
      applyTemplatesSnapshot(templates, copied.id, copied.id);
    } catch (copyError) {
      const message = copyError instanceof Error ? copyError.message : "复制模板失败";
      setError(message);
    } finally {
      setTemplateSaving(false);
    }
  };

  const rollbackTemplateToVersion = async (targetVersion: number) => {
    if (!templateDraftId) {
      setError("请先选择模板");
      return;
    }
    if (!window.confirm(`确认回滚模板到 v${targetVersion} 吗？`)) {
      return;
    }
    if (templateSaving) return;
    setTemplateSaving(true);
    setError(null);
    try {
      await rollbackProjectPromptTemplate(projectId, templateDraftId, {
        target_version: targetVersion,
      });
      const templates = await getProjectPromptTemplates(projectId);
      applyTemplatesSnapshot(templates, activePromptTemplateId, templateDraftId);
      await refreshTemplateRevisions(projectId, templateDraftId);
    } catch (rollbackError) {
      const message = rollbackError instanceof Error ? rollbackError.message : "模板回滚失败";
      setError(message);
    } finally {
      setTemplateSaving(false);
    }
  };

  const saveModelProfile = async () => {
    if (modelProfileSaving) return;
    setModelProfileSaving(true);
    setError(null);
    try {
      const payload: Record<string, string | null> = {
        name: modelProfileName.trim() || null,
        provider: modelProfileProvider,
        base_url: modelProfileBaseUrl.trim() || null,
        model: modelProfileModel.trim() || null,
      };
      const apiKeyInput = modelProfileApiKey.trim();
      if (selectedModelProfileId) {
        if (clearModelProfileApiKey) {
          payload.api_key = "";
        } else if (apiKeyInput) {
          payload.api_key = apiKeyInput;
        }
        const saved = await updateProjectModelProfile(projectId, selectedModelProfileId, payload);
        const profiles = await getProjectModelProfiles(projectId);
        applyModelProfilesSnapshot(profiles, saved.profile_id);
      } else {
        if (apiKeyInput) {
          payload.api_key = apiKeyInput;
        }
        const profileIdInput = modelProfileDraftIdInput.trim();
        if (profileIdInput) {
          payload.profile_id = profileIdInput;
        }
        const saved = await createProjectModelProfile(projectId, payload);
        const profiles = await getProjectModelProfiles(projectId);
        applyModelProfilesSnapshot(profiles, saved.profile_id);
      }
    } catch (profileError) {
      const message = profileError instanceof Error ? profileError.message : "保存模型 profile 失败";
      setError(message);
    } finally {
      setModelProfileSaving(false);
    }
  };

  const deleteModelProfile = async () => {
    if (!selectedModelProfileId) {
      setError("请先选择要删除的模型 profile");
      return;
    }
    if (!window.confirm(`确认删除模型 profile「${selectedModelProfileId}」吗？`)) return;
    if (modelProfileSaving) return;
    setModelProfileSaving(true);
    setError(null);
    try {
      await deleteProjectModelProfile(projectId, selectedModelProfileId);
      const profiles = await getProjectModelProfiles(projectId);
      applyModelProfilesSnapshot(profiles, null);
    } catch (profileError) {
      const message = profileError instanceof Error ? profileError.message : "删除模型 profile 失败";
      setError(message);
    } finally {
      setModelProfileSaving(false);
    }
  };

  const activateModelProfile = async () => {
    if (!selectedModelProfileId) {
      setError("请先选择要激活的模型 profile");
      return;
    }
    if (modelProfileSaving) return;
    setModelProfileSaving(true);
    setError(null);
    try {
      await activateProjectModelProfile(projectId, selectedModelProfileId);
      const profiles = await getProjectModelProfiles(projectId);
      applyModelProfilesSnapshot(profiles, selectedModelProfileId);
    } catch (profileError) {
      const message = profileError instanceof Error ? profileError.message : "激活模型 profile 失败";
      setError(message);
    } finally {
      setModelProfileSaving(false);
    }
  };

  const runConsistencyAudit = async () => {
    if (consistencyAuditRunning) return;
    setConsistencyAuditRunning(true);
    setError(null);
    try {
      const result = await runProjectConsistencyAudit(projectId, {
        run_mode: "sync",
        reason: "manual_ui",
        force: true,
        max_chapters: 3,
      });
      const report = result.report ?? null;
      if (report) {
        setConsistencyAudits((prev) => {
          const next = [report, ...prev.filter((item) => item.report_id !== report.report_id)];
          return next.slice(0, 8);
        });
      } else {
        const latest = await getProjectConsistencyAudits(projectId, 8);
        setConsistencyAudits(latest);
      }
    } catch (auditError) {
      const message = auditError instanceof Error ? auditError.message : "运行创作体检失败";
      setError(message);
    } finally {
      setConsistencyAuditRunning(false);
    }
  };

  return (
    <div
      className={`page-shell ${zenMode ? "zen-mode" : ""} ${advancedPanelOpen ? "advanced-panel-open" : ""}`}
      data-writing-theme={writingTheme}
      data-typewriter-focus={typewriterDimmingEnabled ? "on" : "off"}
    >
      <TopBar
        advancedPanelOpen={advancedPanelOpen}
        zenMode={zenMode}
        streaming={streaming}
        settingsDialogOpen={settingsDialogOpen}
        theme={theme}
        setTheme={setTheme}
        onToggleAdvancedPanel={toggleAdvancedPanel}
        onToggleZenMode={toggleZenMode}
        onOpenSettingsDialog={openSettingsDialog}
        onRefreshSnapshot={handleRefresh}
        onStartNewSession={startNewSession}
      />

      <ErrorBoundary
        fallbackTitle="写作区暂时不可用"
        fallbackDescription="编辑器出现异常，已进入降级模式。请刷新页面后继续写作。"
      >
        <DraftWorkspacePanel
          draftWordCount={draftWordCount}
          draftVersion={draftVersion}
          draftUpdatedAt={draftUpdatedAt}
          totalChapterWords={totalChapterWords}
          todayAddedWords={todayAddedWords}
          onboardingChecklist={onboardingChecklist}
          activeChapterId={activeChapterId}
          chapters={chapters}
          onSwitchChapter={switchChapter}
          draftLoading={draftLoading}
          draftSaving={draftSaving}
          draftTitle={draftTitle}
          setDraftTitle={setDraftTitle}
          onCreateChapterAndSwitch={createChapterAndSwitch}
          onMoveActiveChapter={moveActiveChapter}
          canMoveChapterUp={canMoveChapterUp}
          canMoveChapterDown={canMoveChapterDown}
          onDeleteActiveChapter={deleteActiveChapter}
          awarenessTags={awarenessTags}
          draftFocusMode={draftFocusMode}
          autoSaveState={autoSaveState}
          autoSaveAt={autoSaveAt}
          typewriterModeEnabled={typewriterModeEnabled}
          localRecoveryNotice={localRecoveryNotice}
          onToggleTypewriterMode={toggleTypewriterMode}
          onToggleDraftFocusMode={toggleDraftFocusMode}
          onToggleZenMode={toggleZenMode}
          zenMode={zenMode}
          canEnterZenMode={!advancedPanelOpen}
          draftEditorRef={draftEditorRef}
          editor={editor}
          onSaveDraftSnapshot={saveDraftSnapshot}
          onRefreshDraftSnapshot={refreshDraftSnapshot}
          projectId={projectId}
          onFillPromptFromSelection={fillPromptFromSelection}
          onApplyAssistantToDraft={applyAssistantToDraft}
          selectedDraftText={selectedDraftText}
          latestAssistantReply={latestAssistantReply}
          chapterOutlines={chapterOutlines}
          dragChapterId={dragChapterId}
          onOutlineDragStart={handleOutlineDragStart}
          onOutlineDragEnd={handleOutlineDragEnd}
          onReorderByDrag={reorderByDrag}
          draftRevisions={draftRevisions}
          onRollbackDraftToVersion={rollbackDraftToVersion}
        />
      </ErrorBoundary>

      {advancedPanelOpen ? (
        <section className="advanced-panel-stack" aria-label="进阶面板">
          <section className="workspace-bar">
            <div className="status-chip">
              <span>当前区域</span>
              <strong>写作优先壳层</strong>
            </div>
            <div className="status-chip">
              <span>会话 ID</span>
              <strong>{sessionId ?? "未创建"}</strong>
            </div>
            <div className="status-chip">
              <span>引用项目</span>
              <strong>{referenceProjectIds.length ? referenceProjectIds.join(", ") : "无"}</strong>
            </div>
            <div className={`status-chip ${retrievalDegraded ? "warn" : ""}`}>
              <span>检索状态</span>
              <strong>{retrievalDegraded ? "已降级（不中断写作）" : "正常"}</strong>
              {degradedReasons.length > 0 ? <small>{degradedReasons.slice(0, 2).join(" / ")}</small> : null}
            </div>
            <div className="status-chip">
              <span>Stream 指标</span>
              {lastStreamMetrics ? (
                <strong>
                  {`TTFB ${Math.round(lastStreamMetrics.ttfbMs)}ms / 首 token ${
                    lastStreamMetrics.firstTokenMs === null || lastStreamMetrics.firstTokenMs === undefined
                      ? "--"
                      : `${Math.round(lastStreamMetrics.firstTokenMs)}ms`
                  }`}
                </strong>
              ) : (
                <strong>未采样</strong>
              )}
            </div>
          </section>

          <section className="panel workbench-panel-bar">
            <div className="panel-title sub">
              <h3>进阶面板</h3>
              <small>默认折叠，只在需要规划与诊断时展开</small>
            </div>
            <div className="workbench-panel-toggles">
              {(Object.keys(WORKBENCH_PANEL_LABELS) as Array<keyof WorkbenchPanelVisibility>).map((panelKey) => (
                <label key={panelKey} className="workbench-panel-toggle">
                  <input
                    type="checkbox"
                    checked={workbenchPanelVisibility[panelKey]}
                    onChange={() =>
                      setWorkbenchPanelVisibility((prev) => ({
                        ...prev,
                        [panelKey]: !prev[panelKey],
                      }))
                    }
                  />
                  <span>{WORKBENCH_PANEL_LABELS[panelKey]}</span>
                </label>
              ))}
            </div>
          </section>

          {workbenchPanelVisibility.actions ? (
            <AssistantActionsPanel
              sortedActions={sortedActions}
              pendingActionIds={pendingActionIds}
              mutatingActionId={mutatingActionId}
              streamLatencySamples={streamLatencySamples}
              tokenUsageSamples={tokenUsageSamples}
              retrievalHitSamples={retrievalHitSamples}
              consistencyAudits={consistencyAudits}
              consistencyAuditRunning={consistencyAuditRunning}
              traceEvents={traceEvents}
              graphTimeline={graphTimeline}
              graphTimelineLoading={graphTimelineLoading}
              graphTimelineChapterIndex={graphTimelineChapterIndex}
              maxChapterIndex={maxChapterIndex}
              setGraphTimelineChapterIndex={setGraphTimelineChapterIndex}
              selectedActionId={selectedActionId}
              actionLogs={actionLogs}
              onLoadLogs={loadLogs}
              onMutateAction={mutateAction}
              onRunConsistencyAudit={runConsistencyAudit}
            />
          ) : null}

          {workbenchPanelVisibility.prompt && debugPromptPanelReady ? (
            <PromptWorkshopPanel
              activePromptTemplate={activePromptTemplate}
              activePromptTemplateId={activePromptTemplateId}
              templateSaving={templateSaving}
              promptTemplates={promptTemplates}
              onHandleActiveTemplateChange={handleActiveTemplateChange}
              onStartCreateTemplateDraft={startCreateTemplateDraft}
              onCopyTemplateDraft={copyTemplateDraft}
              templateName={templateName}
              setTemplateName={setTemplateName}
              templateSystemPrompt={templateSystemPrompt}
              setTemplateSystemPrompt={setTemplateSystemPrompt}
              templateUserPromptPrefix={templateUserPromptPrefix}
              setTemplateUserPromptPrefix={setTemplateUserPromptPrefix}
              settings={settings}
              templateKnowledgeSettingKeys={templateKnowledgeSettingKeys}
              setTemplateKnowledgeSettingKeys={setTemplateKnowledgeSettingKeys}
              cards={cards}
              templateKnowledgeCardIds={templateKnowledgeCardIds}
              setTemplateKnowledgeCardIds={setTemplateKnowledgeCardIds}
              onSaveTemplateDraft={saveTemplateDraft}
              templateDraftId={templateDraftId}
              onDeleteTemplateDraft={deleteTemplateDraft}
              onRefreshProjectSnapshot={refreshProjectSnapshot}
              projectId={projectId}
              selectedKnowledgeSettings={selectedKnowledgeSettings}
              selectedKnowledgeCards={selectedKnowledgeCards}
              estimatedPromptChars={estimatedPromptChars}
              missingSettingKeys={missingSettingKeys}
              missingCardIds={missingCardIds}
              templateRevisions={templateRevisions}
              templateRevisionsLoading={templateRevisionsLoading}
              onRollbackTemplateToVersion={rollbackTemplateToVersion}
            />
          ) : null}

          {workbenchPanelVisibility.snapshot && proSnapshotPanelReady ? (
            <DebugSnapshotGrid evidence={evidence} settings={settings} cards={cards} />
          ) : null}
        </section>
      ) : null}

      <ErrorBoundary
        fallbackTitle="写作助手暂时不可用"
        fallbackDescription="助手面板发生异常，已自动保护主编辑区。可刷新页面后继续。"
      >
        <AssistantDrawer
          projectId={projectId}
          assistantDrawerOpen={assistantDrawerOpen}
          assistantSection={assistantSection}
          onOpenAssistantDrawer={openAssistantDrawer}
          onCloseAssistantDrawer={closeAssistantDrawer}
          onSelectAssistantSection={setAssistantSection}
          onFocusAssistantComposer={focusAssistantComposer}
          onStartNewSession={startNewSession}
          onSwitchSession={switchSession}
          onRenameSession={renameSession}
          onDeleteSession={deleteSession}
          assistantDrawerRef={assistantDrawerRef}
          sessionId={sessionId}
          projectSessions={projectSessions}
          usage={usage}
          messages={messages}
          settings={settings}
          cards={cards}
          input={input}
          streaming={streaming}
          composerInputRef={composerInputRef}
          setInput={setInput}
          onSend={handleSend}
          sortedActions={sortedActions}
          pendingActionIds={pendingActionIds}
          mutatingActionId={mutatingActionId}
          streamLatencySamples={streamLatencySamples}
          tokenUsageSamples={tokenUsageSamples}
          retrievalHitSamples={retrievalHitSamples}
          consistencyAudits={consistencyAudits}
          consistencyAuditRunning={consistencyAuditRunning}
          traceEvents={traceEvents}
          graphTimeline={graphTimeline}
          graphTimelineLoading={graphTimelineLoading}
          graphTimelineChapterIndex={graphTimelineChapterIndex}
          maxChapterIndex={maxChapterIndex}
          setGraphTimelineChapterIndex={setGraphTimelineChapterIndex}
          selectedActionId={selectedActionId}
          actionLogs={actionLogs}
          onLoadLogs={loadLogs}
          onMutateAction={mutateAction}
          onRunConsistencyAudit={runConsistencyAudit}
          planningPanelNode={
            <StoryPlanningPanel
              activeChapterId={activeChapterId}
              volumes={volumes}
              activeVolumeId={activeVolumeId}
              onSelectVolume={setActiveVolumeId}
              onCreateVolume={createVolume}
              onBindChapterToVolume={bindActiveChapterToVolume}
              volumeOutlineDraft={volumeOutlineDraft}
              setVolumeOutlineDraft={setVolumeOutlineDraft}
              onSaveVolumeOutline={saveVolumeOutline}
              sceneBeats={sceneBeats}
              activeSceneBeatId={activeSceneBeatId}
              onSelectSceneBeat={setActiveSceneBeatId}
              newBeatContent={newBeatContent}
              setNewBeatContent={setNewBeatContent}
              onCreateSceneBeat={createBeatForActiveChapter}
              onToggleSceneBeatStatus={toggleBeatStatus}
              onDeleteSceneBeat={deleteBeat}
              foreshadowCards={foreshadowCards}
              overdueForeshadowCards={overdueForeshadowCards}
              foreshadowDraftTitle={foreshadowDraftTitle}
              setForeshadowDraftTitle={setForeshadowDraftTitle}
              foreshadowDraftDescription={foreshadowDraftDescription}
              setForeshadowDraftDescription={setForeshadowDraftDescription}
              onCreateForeshadowCard={createForeshadow}
              onToggleForeshadowStatus={toggleForeshadowStatus}
              onDeleteForeshadowCard={deleteForeshadow}
              busy={planningBusy}
            />
          }
        />
      </ErrorBoundary>

      <SettingsDialog
        settingsDialogOpen={settingsDialogOpen}
        onCloseSettingsDialog={closeSettingsDialog}
        settingsDialogRef={settingsDialogRef}
        projectId={projectId}
        setProjectId={setProjectId}
        model={model}
        setModel={setModel}
        modelProfiles={modelProfiles}
        suggestionModelProfileId={suggestionModelProfileId}
        setSuggestionModelProfileId={setSuggestionModelProfileId}
        selectedModelProfileId={selectedModelProfileId}
        setSelectedModelProfileId={setSelectedModelProfileId}
        modelProfileDraftIdInput={modelProfileDraftIdInput}
        setModelProfileDraftIdInput={setModelProfileDraftIdInput}
        modelProfileName={modelProfileName}
        setModelProfileName={setModelProfileName}
        modelProfileProvider={modelProfileProvider}
        setModelProfileProvider={setModelProfileProvider}
        modelProfileBaseUrl={modelProfileBaseUrl}
        setModelProfileBaseUrl={setModelProfileBaseUrl}
        modelProfileApiKey={modelProfileApiKey}
        setModelProfileApiKey={setModelProfileApiKey}
        modelProfileApiKeyMasked={modelProfileApiKeyMasked}
        clearModelProfileApiKey={clearModelProfileApiKey}
        setClearModelProfileApiKey={setClearModelProfileApiKey}
        modelProfileModel={modelProfileModel}
        setModelProfileModel={setModelProfileModel}
        modelProfileSaving={modelProfileSaving}
        onSaveModelProfile={saveModelProfile}
        onDeleteModelProfile={deleteModelProfile}
        onActivateModelProfile={activateModelProfile}
        onResetModelProfileDraft={resetModelProfileDraft}
        chatTemperatureProfile={chatTemperatureProfile}
        setChatTemperatureProfile={setChatTemperatureProfile}
        suggestionTemperatureProfile={suggestionTemperatureProfile}
        setSuggestionTemperatureProfile={setSuggestionTemperatureProfile}
        temperatureOverrideInput={temperatureOverrideInput}
        setTemperatureOverrideInput={setTemperatureOverrideInput}
        contextWindowProfile={contextWindowProfile}
        setContextWindowProfile={setContextWindowProfile}
        povMode={povMode}
        setPovMode={setPovMode}
        povAnchor={povAnchor}
        setPovAnchor={setPovAnchor}
        ragMode={ragMode}
        setRagMode={setRagMode}
        deterministicFirst={deterministicFirst}
        setDeterministicFirst={setDeterministicFirst}
        thinkingEnabled={thinkingEnabled}
        setThinkingEnabled={setThinkingEnabled}
        referenceProjectInput={referenceProjectInput}
        setReferenceProjectInput={setReferenceProjectInput}
        typewriterModeEnabled={typewriterModeEnabled}
        setTypewriterModeEnabled={setTypewriterModeEnabled}
        writingTheme={writingTheme}
        setWritingTheme={setWritingTheme}
        streaming={streaming}
      />

      {error ? <div className="error-banner">错误：{error}</div> : null}
    </div>
  );
}



