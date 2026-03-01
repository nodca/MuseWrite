import { create } from "zustand";
import type {
  ActionAuditLog,
  ChatAction,
  EvidencePayload,
  SettingEntry,
  StoryCard,
  UiMessage,
} from "../types";

interface ChatStoreState {
  uiMode: "writing" | "pro";
  projectId: number;
  model: string;
  povMode: "global" | "character";
  povAnchor: string;
  ragMode: "local" | "global" | "hybrid" | "mix";
  deterministicFirst: boolean;
  thinkingEnabled: boolean;
  sessionId: number | null;
  streaming: boolean;
  error: string | null;
  usage: Record<string, unknown> | null;

  messages: UiMessage[];
  actions: ChatAction[];
  pendingActionIds: number[];
  settings: SettingEntry[];
  cards: StoryCard[];
  selectedActionId: number | null;
  actionLogs: ActionAuditLog[];
  evidence: EvidencePayload | null;

  setUiMode: (uiMode: "writing" | "pro") => void;
  setProjectId: (projectId: number) => void;
  setModel: (model: string) => void;
  setPovMode: (povMode: "global" | "character") => void;
  setPovAnchor: (povAnchor: string) => void;
  setRagMode: (ragMode: "local" | "global" | "hybrid" | "mix") => void;
  setDeterministicFirst: (deterministicFirst: boolean) => void;
  setThinkingEnabled: (thinkingEnabled: boolean) => void;
  setSessionId: (sessionId: number | null) => void;
  setStreaming: (streaming: boolean) => void;
  setError: (error: string | null) => void;
  setUsage: (usage: Record<string, unknown> | null) => void;

  setMessages: (messages: UiMessage[]) => void;
  appendMessage: (message: UiMessage) => void;
  appendMessageDelta: (messageId: string, delta: string) => void;
  updateMessage: (messageId: string, updater: (message: UiMessage) => UiMessage) => void;

  setActions: (actions: ChatAction[]) => void;
  setPendingActionIds: (actionIds: number[]) => void;
  setSettings: (settings: SettingEntry[]) => void;
  setCards: (cards: StoryCard[]) => void;
  setSelectedActionId: (actionId: number | null) => void;
  setActionLogs: (logs: ActionAuditLog[]) => void;
  setEvidence: (evidence: EvidencePayload | null) => void;
  resetSessionState: () => void;
}

export const useChatStore = create<ChatStoreState>((set) => ({
  uiMode: "writing",
  projectId: 1,
  model: "",
  povMode: "global",
  povAnchor: "",
  ragMode: "mix",
  deterministicFirst: false,
  thinkingEnabled: false,
  sessionId: null,
  streaming: false,
  error: null,
  usage: null,
  messages: [],
  actions: [],
  pendingActionIds: [],
  settings: [],
  cards: [],
  selectedActionId: null,
  actionLogs: [],
  evidence: null,

  setUiMode: (uiMode) => set((state) => (state.uiMode === uiMode ? state : { uiMode })),
  setProjectId: (projectId) => set((state) => (state.projectId === projectId ? state : { projectId })),
  setModel: (model) => set((state) => (state.model === model ? state : { model })),
  setPovMode: (povMode) => set((state) => (state.povMode === povMode ? state : { povMode })),
  setPovAnchor: (povAnchor) => set((state) => (state.povAnchor === povAnchor ? state : { povAnchor })),
  setRagMode: (ragMode) => set((state) => (state.ragMode === ragMode ? state : { ragMode })),
  setDeterministicFirst: (deterministicFirst) =>
    set((state) => (state.deterministicFirst === deterministicFirst ? state : { deterministicFirst })),
  setThinkingEnabled: (thinkingEnabled) =>
    set((state) => (state.thinkingEnabled === thinkingEnabled ? state : { thinkingEnabled })),
  setSessionId: (sessionId) => set((state) => (state.sessionId === sessionId ? state : { sessionId })),
  setStreaming: (streaming) => set((state) => (state.streaming === streaming ? state : { streaming })),
  setError: (error) => set((state) => (state.error === error ? state : { error })),
  setUsage: (usage) => set((state) => (state.usage === usage ? state : { usage })),

  setMessages: (messages) => set((state) => (state.messages === messages ? state : { messages })),
  appendMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),
  appendMessageDelta: (messageId, delta) =>
    set((state) => {
      if (!delta) return state;

      const total = state.messages.length;
      if (total === 0) return state;

      const lastIndex = total - 1;
      const applyDeltaAt = (index: number) => {
        const nextMessages = [...state.messages];
        const target = nextMessages[index];
        nextMessages[index] = { ...target, content: target.content + delta };
        return { messages: nextMessages };
      };

      if (state.messages[lastIndex].id === messageId) {
        return applyDeltaAt(lastIndex);
      }

      const index = state.messages.findIndex((message) => message.id === messageId);
      if (index < 0) return state;
      return applyDeltaAt(index);
    }),
  updateMessage: (messageId, updater) =>
    set((state) => {
      const total = state.messages.length;
      if (total === 0) return state;

      const applyUpdateAt = (index: number) => {
        const original = state.messages[index];
        const updated = updater(original);
        if (updated === original) return state;

        const nextMessages = [...state.messages];
        nextMessages[index] = updated;
        return { messages: nextMessages };
      };

      const lastIndex = total - 1;
      if (state.messages[lastIndex].id === messageId) {
        return applyUpdateAt(lastIndex);
      }

      const index = state.messages.findIndex((message) => message.id === messageId);
      if (index < 0) return state;
      return applyUpdateAt(index);
    }),

  setActions: (actions) => set((state) => (state.actions === actions ? state : { actions })),
  setPendingActionIds: (pendingActionIds) =>
    set((state) => (state.pendingActionIds === pendingActionIds ? state : { pendingActionIds })),
  setSettings: (settings) => set((state) => (state.settings === settings ? state : { settings })),
  setCards: (cards) => set((state) => (state.cards === cards ? state : { cards })),
  setSelectedActionId: (selectedActionId) =>
    set((state) => (state.selectedActionId === selectedActionId ? state : { selectedActionId })),
  setActionLogs: (actionLogs) => set((state) => (state.actionLogs === actionLogs ? state : { actionLogs })),
  setEvidence: (evidence) => set((state) => (state.evidence === evidence ? state : { evidence })),

  resetSessionState: () =>
    set((state) => {
      if (
        state.sessionId === null &&
        state.messages.length === 0 &&
        state.actions.length === 0 &&
        state.pendingActionIds.length === 0 &&
        state.selectedActionId === null &&
        state.actionLogs.length === 0 &&
        state.evidence === null &&
        state.usage === null &&
        state.error === null
      ) {
        return state;
      }
      return {
        sessionId: null,
        messages: [],
        actions: [],
        pendingActionIds: [],
        selectedActionId: null,
        actionLogs: [],
        evidence: null,
        usage: null,
        error: null,
      };
    }),
}));
