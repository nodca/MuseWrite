import { memo, useEffect, useState } from "react";
import type { ReactNode } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import * as Tabs from "@radix-ui/react-tabs";
import { motion, AnimatePresence } from "framer-motion";
import { MessageCircle, Plus, Pencil, Trash2, X } from "lucide-react";
import { formatDateTime } from "../utils/formatting";
import { AssistantChatPanel } from "./AssistantChatPanel";
import { AssistantActionsPanel } from "./AssistantActionsPanel";
import type { StreamLatencySample, TokenUsageSample, RetrievalHitSample } from "./AssistantActionsPanel";
import { GraphCandidateReviewPanel } from "../debugPanels";
import type {
  ActionAuditLog,
  ChatAction,
  ChatSessionSummary,
  ChatStreamTraceEvent,
  ConsistencyAuditReport,
  GraphTimelineSnapshot,
  SettingEntry,
  StoryCard,
  UiMessage,
} from "../types";

export type AssistantDrawerProps = {
  projectId: number;
  assistantDrawerOpen: boolean;
  assistantSection: "planning" | "chat";
  onOpenAssistantDrawer: () => void;
  onCloseAssistantDrawer: () => void;
  onSelectAssistantSection: (section: "planning" | "chat") => void;
  onFocusAssistantComposer: () => void;
  onStartNewSession: () => void;
  onSwitchSession: (sessionId: number) => Promise<void>;
  onRenameSession: () => Promise<void>;
  onDeleteSession: () => Promise<void>;
  assistantDrawerRef: { current: HTMLElement | null };
  sessionId: number | null;
  projectSessions: ChatSessionSummary[];
  usage: Record<string, unknown> | null;
  messages: UiMessage[];
  settings: SettingEntry[];
  cards: StoryCard[];
  input: string;
  streaming: boolean;
  composerInputRef: { current: HTMLTextAreaElement | null };
  setInput: (value: string) => void;
  onSend: () => Promise<void>;
  sortedActions: ChatAction[];
  pendingActionIds: number[];
  mutatingActionId: number | null;
  streamLatencySamples: StreamLatencySample[];
  tokenUsageSamples: TokenUsageSample[];
  retrievalHitSamples: RetrievalHitSample[];
  consistencyAudits: ConsistencyAuditReport[];
  consistencyAuditRunning: boolean;
  traceEvents: ChatStreamTraceEvent[];
  graphTimeline: GraphTimelineSnapshot | null;
  graphTimelineLoading: boolean;
  graphTimelineChapterIndex: number;
  maxChapterIndex: number;
  setGraphTimelineChapterIndex: (chapterIndex: number) => void;
  selectedActionId: number | null;
  actionLogs: ActionAuditLog[];
  onLoadLogs: (actionId: number) => Promise<void>;
  onMutateAction: (action: ChatAction, decision: "apply" | "reject" | "undo") => Promise<void>;
  onRunConsistencyAudit: () => Promise<void>;
  planningPanelNode: ReactNode;
};

export const AssistantDrawer = memo(function AssistantDrawer({
  projectId,
  assistantDrawerOpen,
  assistantSection,
  onOpenAssistantDrawer,
  onCloseAssistantDrawer,
  onSelectAssistantSection,
  onFocusAssistantComposer,
  onStartNewSession,
  onSwitchSession,
  onRenameSession,
  onDeleteSession,
  assistantDrawerRef,
  sessionId,
  projectSessions,
  usage,
  messages,
  settings,
  cards,
  input,
  streaming,
  composerInputRef,
  setInput,
  onSend,
  sortedActions,
  pendingActionIds,
  mutatingActionId,
  streamLatencySamples,
  tokenUsageSamples,
  retrievalHitSamples,
  consistencyAudits,
  consistencyAuditRunning,
  traceEvents,
  graphTimeline,
  graphTimelineLoading,
  graphTimelineChapterIndex,
  maxChapterIndex,
  setGraphTimelineChapterIndex,
  selectedActionId,
  actionLogs,
  onLoadLogs,
  onMutateAction,
  onRunConsistencyAudit,
  planningPanelNode,
}: AssistantDrawerProps) {
  const [sideTab, setSideTab] = useState<"actions" | "candidates">("actions");

  useEffect(() => {
    if (!assistantDrawerOpen) {
      setSideTab("actions");
    }
  }, [assistantDrawerOpen]);

  const handleSessionChange = (value: string) => {
    const raw = value.trim();
    if (!raw) {
      onStartNewSession();
      return;
    }
    const nextSessionId = Number(raw);
    if (!Number.isFinite(nextSessionId) || nextSessionId <= 0) return;
    if (sessionId === nextSessionId) return;
    void onSwitchSession(nextSessionId);
  };

  return (
    <>
      {/* FAB button — keep .assistant-fab class for E2E */}
      <button
        type="button"
        className="assistant-fab fixed bottom-6 right-6 z-30 flex items-center gap-2 rounded-full bg-accent-primary px-4 py-3 text-white shadow-lg hover:bg-accent-primary-hover transition-colors"
        onClick={onOpenAssistantDrawer}
        aria-haspopup="dialog"
        aria-expanded={assistantDrawerOpen}
        aria-controls="assistant-drawer"
      >
        <MessageCircle size={18} />
        写作助手
      </button>

      <Dialog.Root
        open={assistantDrawerOpen}
        onOpenChange={(open) => {
          if (!open) onCloseAssistantDrawer();
        }}
      >
        <Dialog.Portal forceMount>
          <AnimatePresence>
            {assistantDrawerOpen && (
              <>
              <Dialog.Overlay asChild forceMount>
                <motion.div
                  className="fixed inset-0 bg-overlay-bg z-40"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                />
              </Dialog.Overlay>
              <Dialog.Content asChild forceMount>
                <motion.aside
                  id="assistant-drawer"
                  ref={assistantDrawerRef as React.Ref<HTMLElement>}
                  aria-hidden={!assistantDrawerOpen}
                  initial={{ x: "100%" }}
                  animate={{ x: 0 }}
                  exit={{ x: "100%" }}
                  transition={{ type: "spring", damping: 30, stiffness: 300, duration: 0.3 }}
                  className="fixed top-0 right-0 bottom-0 z-50 w-full max-w-[860px] bg-surface-primary border-l border-border-default shadow-lg flex flex-col overflow-hidden"
                >
                  {/* ── Top bar ── */}
                  <div className="flex items-center justify-between px-5 py-3 border-b border-border-default">
                    <div className="flex items-center gap-2">
                      <MessageCircle size={18} className="text-accent-primary" />
                      <Dialog.Title className="text-base font-semibold text-text-primary">
                        写作助手
                      </Dialog.Title>
                    </div>
                    <Dialog.Description className="sr-only">
                      默认折叠，按 Ctrl/Cmd + Shift + A 快速呼出
                    </Dialog.Description>
                    <Dialog.Close asChild>
                      <button
                        type="button"
                        className="rounded-lg p-1.5 text-text-secondary hover:text-text-primary hover:bg-surface-elevated transition-colors"
                        aria-label="关闭写作助手"
                      >
                        <X size={18} />
                      </button>
                    </Dialog.Close>
                  </div>

                  {/* ── Session switcher ── */}
                  <div className="flex items-center gap-2 px-5 py-2 border-b border-border-default bg-surface-base/50">
                    <label htmlFor="assistant-session-select" className="text-xs text-text-secondary shrink-0">
                      会话
                    </label>
                    <select
                      id="assistant-session-select"
                      className="flex-1 min-w-0 text-sm bg-transparent border border-border-default rounded-md px-2 py-1 text-text-primary focus:outline-none focus:ring-2 focus:ring-accent-primary/30"
                      value={sessionId ?? ""}
                      onChange={(event) => handleSessionChange(event.target.value)}
                      disabled={streaming}
                    >
                      <option value="">新会话（未创建）</option>
                      {projectSessions.map((item) => (
                        <option key={item.id} value={item.id}>
                          {`${item.title || `会话 #${item.id}`} · ${formatDateTime(item.updated_at)}`}
                        </option>
                      ))}
                    </select>
                    <button
                      type="button"
                      className="rounded-md p-1.5 text-text-secondary hover:text-text-primary hover:bg-surface-elevated transition-colors"
                      onClick={onStartNewSession}
                      disabled={streaming}
                      aria-label="新会话"
                      title="新会话"
                    >
                      <Plus size={15} />
                    </button>
                    {sessionId ? (
                      <>
                        <button
                          type="button"
                          className="rounded-md p-1.5 text-text-secondary hover:text-text-primary hover:bg-surface-elevated transition-colors"
                          onClick={() => void onRenameSession()}
                          disabled={streaming}
                          aria-label="重命名"
                          title="重命名"
                        >
                          <Pencil size={15} />
                        </button>
                        <button
                          type="button"
                          className="rounded-md p-1.5 text-text-secondary hover:text-danger hover:bg-danger-bg transition-colors"
                          onClick={() => void onDeleteSession()}
                          disabled={streaming}
                          aria-label="删除会话"
                          title="删除会话"
                        >
                          <Trash2 size={15} />
                        </button>
                      </>
                    ) : null}
                  </div>

                  {/* ── Tabs ── */}
                  <Tabs.Root
                    value={assistantSection}
                    onValueChange={(v) => {
                      const section = v as "planning" | "chat";
                      onSelectAssistantSection(section);
                      if (section === "chat") onFocusAssistantComposer();
                    }}
                    className="flex flex-col flex-1 min-h-0"
                  >
                    <Tabs.List className="flex border-b border-border-default px-5 gap-1 shrink-0">
                      <Tabs.Trigger
                        value="planning"
                        className="px-3 py-2.5 text-sm text-text-secondary hover:text-text-primary transition-colors -mb-px data-[state=active]:text-accent-primary data-[state=active]:border-b-2 data-[state=active]:border-accent-primary"
                      >
                        规划
                      </Tabs.Trigger>
                      <Tabs.Trigger
                        value="chat"
                        className="px-3 py-2.5 text-sm text-text-secondary hover:text-text-primary transition-colors -mb-px data-[state=active]:text-accent-primary data-[state=active]:border-b-2 data-[state=active]:border-accent-primary"
                      >
                        对话
                      </Tabs.Trigger>
                    </Tabs.List>

                    <Tabs.Content value="planning" className="flex-1 overflow-y-auto p-5">
                      {planningPanelNode}
                    </Tabs.Content>

                    <Tabs.Content value="chat" className="flex-1 overflow-hidden">
                      {/* Two-column layout: chat + actions */}
                      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] h-full divide-x divide-border-default">
                        <AssistantChatPanel
                          usage={usage}
                          messages={messages}
                          settings={settings}
                          cards={cards}
                          input={input}
                          streaming={streaming}
                          composerInputRef={composerInputRef}
                          setInput={setInput}
                          onSend={onSend}
                        />

                        <div className="flex flex-col min-h-0">
                          {/* Sub-tabs: 动作提议 / 候选审核 */}
                          <div className="flex gap-1 px-4 py-2 border-b border-border-default shrink-0">
                            <button
                              type="button"
                              className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
                                sideTab === "actions"
                                  ? "bg-accent-primary text-white"
                                  : "text-text-secondary hover:text-text-primary hover:bg-surface-elevated"
                              }`}
                              onClick={() => setSideTab("actions")}
                            >
                              动作提议
                            </button>
                            <button
                              type="button"
                              className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
                                sideTab === "candidates"
                                  ? "bg-accent-primary text-white"
                                  : "text-text-secondary hover:text-text-primary hover:bg-surface-elevated"
                              }`}
                              onClick={() => setSideTab("candidates")}
                            >
                              候选审核
                            </button>
                          </div>

                          <div className="flex-1 overflow-y-auto">
                            {sideTab === "actions" ? (
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
                                onLoadLogs={onLoadLogs}
                                onMutateAction={onMutateAction}
                                onRunConsistencyAudit={onRunConsistencyAudit}
                              />
                            ) : (
                              <GraphCandidateReviewPanel projectId={projectId} />
                            )}
                          </div>
                        </div>
                      </div>
                    </Tabs.Content>
                  </Tabs.Root>
                </motion.aside>
              </Dialog.Content>
              </>
            )}
          </AnimatePresence>
        </Dialog.Portal>
      </Dialog.Root>
    </>
  );
});
