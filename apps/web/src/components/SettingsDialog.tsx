import { memo } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import * as Tabs from "@radix-ui/react-tabs";
import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";
import type { ModelProfile } from "../types";
import { WritingTab } from "./settings/WritingTab";
import { AIModelTab } from "./settings/AIModelTab";
import { ContextTab } from "./settings/ContextTab";
import { BehaviorTab } from "./settings/BehaviorTab";

export type WritingTheme = "paper" | "wenkai" | "modern" | "contrast";

export type SettingsDialogProps = {
  settingsDialogOpen: boolean;
  onCloseSettingsDialog: () => void;
  settingsDialogRef: { current: HTMLElement | null };
  projectId: number;
  setProjectId: (projectId: number) => void;
  model: string;
  setModel: (value: string) => void;
  modelProfiles: ModelProfile[];
  suggestionModelProfileId: string | null;
  setSuggestionModelProfileId: (value: string | null) => void;
  selectedModelProfileId: string | null;
  setSelectedModelProfileId: (value: string | null) => void;
  modelProfileDraftIdInput: string;
  setModelProfileDraftIdInput: (value: string) => void;
  modelProfileName: string;
  setModelProfileName: (value: string) => void;
  modelProfileProvider: "openai_compatible" | "deepseek" | "claude" | "gemini";
  setModelProfileProvider: (value: "openai_compatible" | "deepseek" | "claude" | "gemini") => void;
  modelProfileBaseUrl: string;
  setModelProfileBaseUrl: (value: string) => void;
  modelProfileApiKey: string;
  setModelProfileApiKey: (value: string) => void;
  modelProfileApiKeyMasked: string | null;
  clearModelProfileApiKey: boolean;
  setClearModelProfileApiKey: (value: boolean) => void;
  modelProfileModel: string;
  setModelProfileModel: (value: string) => void;
  modelProfileSaving: boolean;
  onSaveModelProfile: () => Promise<void>;
  onDeleteModelProfile: () => Promise<void>;
  onActivateModelProfile: () => Promise<void>;
  onResetModelProfileDraft: () => void;
  chatTemperatureProfile: "action" | "chat" | "brainstorm";
  setChatTemperatureProfile: (value: "action" | "chat" | "brainstorm") => void;
  suggestionTemperatureProfile: "suggestion" | "chat" | "action" | "brainstorm";
  setSuggestionTemperatureProfile: (value: "suggestion" | "chat" | "action" | "brainstorm") => void;
  temperatureOverrideInput: string;
  setTemperatureOverrideInput: (value: string) => void;
  contextWindowProfile: "balanced" | "chapter_focus" | "world_focus" | "minimal";
  setContextWindowProfile: (value: "balanced" | "chapter_focus" | "world_focus" | "minimal") => void;
  povMode: "global" | "character";
  setPovMode: (value: "global" | "character") => void;
  povAnchor: string;
  setPovAnchor: (value: string) => void;
  ragMode: "local" | "global" | "hybrid" | "mix";
  setRagMode: (value: "local" | "global" | "hybrid" | "mix") => void;
  deterministicFirst: boolean;
  setDeterministicFirst: (value: boolean) => void;
  thinkingEnabled: boolean;
  setThinkingEnabled: (value: boolean) => void;
  referenceProjectInput: string;
  setReferenceProjectInput: (value: string) => void;
  typewriterModeEnabled: boolean;
  setTypewriterModeEnabled: (value: boolean) => void;
  writingTheme: WritingTheme;
  setWritingTheme: (value: WritingTheme) => void;
  streaming: boolean;
};

const tabTriggerClass =
  "px-3 py-2.5 text-sm text-text-secondary hover:text-text-primary data-[state=active]:text-accent-primary data-[state=active]:border-b-2 data-[state=active]:border-accent-primary -mb-px transition-colors cursor-pointer";

export const SettingsDialog = memo(function SettingsDialog({
  settingsDialogOpen,
  onCloseSettingsDialog,
  settingsDialogRef,
  projectId,
  setProjectId,
  model,
  setModel,
  modelProfiles,
  suggestionModelProfileId,
  setSuggestionModelProfileId,
  selectedModelProfileId,
  setSelectedModelProfileId,
  modelProfileDraftIdInput,
  setModelProfileDraftIdInput,
  modelProfileName,
  setModelProfileName,
  modelProfileProvider,
  setModelProfileProvider,
  modelProfileBaseUrl,
  setModelProfileBaseUrl,
  modelProfileApiKey,
  setModelProfileApiKey,
  modelProfileApiKeyMasked,
  clearModelProfileApiKey,
  setClearModelProfileApiKey,
  modelProfileModel,
  setModelProfileModel,
  modelProfileSaving,
  onSaveModelProfile,
  onDeleteModelProfile,
  onActivateModelProfile,
  onResetModelProfileDraft,
  chatTemperatureProfile,
  setChatTemperatureProfile,
  suggestionTemperatureProfile,
  setSuggestionTemperatureProfile,
  temperatureOverrideInput,
  setTemperatureOverrideInput,
  contextWindowProfile,
  setContextWindowProfile,
  povMode,
  setPovMode,
  povAnchor,
  setPovAnchor,
  ragMode,
  setRagMode,
  deterministicFirst,
  setDeterministicFirst,
  thinkingEnabled,
  setThinkingEnabled,
  referenceProjectInput,
  setReferenceProjectInput,
  typewriterModeEnabled,
  setTypewriterModeEnabled,
  writingTheme,
  setWritingTheme,
  streaming,
}: SettingsDialogProps) {
  return (
    <Dialog.Root
      open={settingsDialogOpen}
      onOpenChange={(open) => {
        if (!open) onCloseSettingsDialog();
      }}
    >
      <Dialog.Portal forceMount>
        <AnimatePresence>
          {settingsDialogOpen && (
            <>
              <Dialog.Overlay asChild forceMount>
                <motion.div
                  className="fixed inset-0 bg-overlay-bg z-40"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.15 }}
                />
              </Dialog.Overlay>
              <Dialog.Content asChild forceMount>
                <motion.div
                  id="settings-dialog"
                  ref={settingsDialogRef as React.RefObject<HTMLDivElement>}
                  className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-2xl max-h-[85vh] overflow-y-auto rounded-xl bg-surface-primary border border-border-default shadow-lg p-0"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                >
          {/* Header */}
          <div className="flex items-center justify-between px-5 py-4 border-b border-border-default">
            <Dialog.Title className="text-lg font-semibold text-text-primary">写作设置</Dialog.Title>
            <Dialog.Close asChild>
              <button
                type="button"
                className="rounded-lg p-1.5 text-text-secondary hover:text-text-primary hover:bg-surface-elevated transition-colors"
                aria-label="关闭写作设置"
              >
                <X size={18} />
              </button>
            </Dialog.Close>
          </div>

          {/* Tabs */}
          <Tabs.Root defaultValue="writing" className="flex flex-col">
            <Tabs.List className="flex border-b border-border-default px-5 gap-1">
              <Tabs.Trigger value="writing" className={tabTriggerClass}>
                写作
              </Tabs.Trigger>
              <Tabs.Trigger value="ai-model" className={tabTriggerClass}>
                AI 模型
              </Tabs.Trigger>
              <Tabs.Trigger value="context" className={tabTriggerClass}>
                上下文
              </Tabs.Trigger>
              <Tabs.Trigger value="behavior" className={tabTriggerClass}>
                行为
              </Tabs.Trigger>
            </Tabs.List>

            <div className="p-5">
              <Tabs.Content value="writing">
                <WritingTab
                  writingTheme={writingTheme}
                  setWritingTheme={setWritingTheme}
                  typewriterModeEnabled={typewriterModeEnabled}
                  setTypewriterModeEnabled={setTypewriterModeEnabled}
                  projectId={projectId}
                  setProjectId={setProjectId}
                  streaming={streaming}
                />
              </Tabs.Content>

              <Tabs.Content value="ai-model">
                <AIModelTab
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
                  onSaveModelProfile={onSaveModelProfile}
                  onDeleteModelProfile={onDeleteModelProfile}
                  onActivateModelProfile={onActivateModelProfile}
                  onResetModelProfileDraft={onResetModelProfileDraft}
                  streaming={streaming}
                />
              </Tabs.Content>

              <Tabs.Content value="context">
                <ContextTab
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
                  referenceProjectInput={referenceProjectInput}
                  setReferenceProjectInput={setReferenceProjectInput}
                  streaming={streaming}
                />
              </Tabs.Content>

              <Tabs.Content value="behavior">
                <BehaviorTab
                  chatTemperatureProfile={chatTemperatureProfile}
                  setChatTemperatureProfile={setChatTemperatureProfile}
                  suggestionTemperatureProfile={suggestionTemperatureProfile}
                  setSuggestionTemperatureProfile={setSuggestionTemperatureProfile}
                  thinkingEnabled={thinkingEnabled}
                  setThinkingEnabled={setThinkingEnabled}
                  temperatureOverrideInput={temperatureOverrideInput}
                  setTemperatureOverrideInput={setTemperatureOverrideInput}
                  streaming={streaming}
                />
              </Tabs.Content>
            </div>
          </Tabs.Root>
                </motion.div>
              </Dialog.Content>
            </>
          )}
        </AnimatePresence>
      </Dialog.Portal>
    </Dialog.Root>
  );
});
