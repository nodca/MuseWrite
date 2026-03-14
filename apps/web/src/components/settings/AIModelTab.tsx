import { memo, useMemo } from "react";
import type { ModelProfile } from "../../types";

export type AIModelTabProps = {
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
  streaming: boolean;
};

const selectClass =
  "w-full rounded-md border border-border-default bg-surface-primary px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent-primary/30";
const inputClass =
  "w-full rounded-md border border-border-default bg-surface-primary px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent-primary/30";
const btnPrimary =
  "rounded-md px-3 py-1.5 text-sm font-medium transition-colors bg-accent-primary text-white hover:bg-accent-primary-hover disabled:opacity-50 disabled:cursor-not-allowed";
const btnGhost =
  "rounded-md px-3 py-1.5 text-sm font-medium transition-colors text-text-secondary hover:text-text-primary hover:bg-surface-elevated disabled:opacity-50 disabled:cursor-not-allowed";

export const AIModelTab = memo(function AIModelTab({
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
  streaming,
}: AIModelTabProps) {
  const activeProfile = useMemo(() => {
    return modelProfiles.find((profile) => Boolean(profile.is_active)) ?? null;
  }, [modelProfiles]);

  const resolvedWritingProfile = useMemo(() => {
    const resolvedId = suggestionModelProfileId ?? activeProfile?.profile_id ?? null;
    if (!resolvedId) return null;
    return modelProfiles.find((profile) => profile.profile_id === resolvedId) ?? null;
  }, [activeProfile?.profile_id, suggestionModelProfileId, modelProfiles]);

  const writingProfileHint = useMemo(() => {
    if (resolvedWritingProfile) {
      const label = `${resolvedWritingProfile.name || resolvedWritingProfile.profile_id} (${resolvedWritingProfile.provider})`;
      return suggestionModelProfileId ? `当前生效：固定 → ${label}` : `当前生效：跟随 active → ${label}`;
    }
    if (suggestionModelProfileId) {
      return "当前生效：固定 Profile 未命中（将回退后端默认 LLM）";
    }
    if (activeProfile) {
      return "当前生效：active Profile 未命中（将回退后端默认 LLM）";
    }
    return "当前生效：未配置（将回退后端默认 LLM）";
  }, [activeProfile, suggestionModelProfileId, resolvedWritingProfile]);

  const profileOptionLabel = (profile: ModelProfile) =>
    `${profile.is_active ? "★ " : ""}${profile.name || profile.profile_id} (${profile.provider})`;

  return (
    <div className="space-y-6">
      {/* Model Override */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-text-primary">模型覆盖（可空）</label>
        <p className="text-xs text-text-secondary">临时指定本次写作用哪个模型；留空时使用当前激活 Profile。</p>
        <input
          type="text"
          className={inputClass}
          placeholder="gpt-4o-mini"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          disabled={streaming}
        />
      </div>

      {/* Suggestion Profile Selector */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-text-primary">编辑辅助 Profile（可空）</label>
        <p className="text-xs text-text-secondary">影响润色、扩写与自动补全建议；留空时跟随当前激活 Profile。</p>
        <select
          className={selectClass}
          value={suggestionModelProfileId ?? ""}
          onChange={(e) => setSuggestionModelProfileId(e.target.value || null)}
          disabled={streaming || modelProfileSaving}
        >
          <option value="">跟随当前激活 Profile</option>
          {modelProfiles.map((profile) => (
            <option key={profile.profile_id} value={profile.profile_id}>
              {profileOptionLabel(profile)}
            </option>
          ))}
        </select>
        <p className="text-xs text-text-tertiary">{writingProfileHint}</p>
      </div>

      {/* Collapsible: Profile CRUD */}
      <details className="group rounded-lg border border-border-default">
        <summary className="flex cursor-pointer items-center gap-2 px-4 py-3 text-sm font-medium text-text-secondary hover:text-text-primary select-none">
          <span className="transition-transform group-open:rotate-90">▸</span>
          <span>Profile 管理中心</span>
          <span className="ml-auto text-xs text-text-tertiary">新建、编辑、删除模型配置</span>
        </summary>
        <div className="space-y-4 border-t border-border-default px-4 py-4">
          {/* Profile Selector */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-text-primary">模型配置中心（Profile）</label>
            <p className="text-xs text-text-secondary">切换你预设好的模型连接配置，适合在不同服务间快速切换。</p>
            <select
              className={selectClass}
              value={selectedModelProfileId ?? ""}
              onChange={(e) => setSelectedModelProfileId(e.target.value || null)}
              disabled={streaming || modelProfileSaving}
            >
              <option value="">新建 profile</option>
              {modelProfiles.map((profile) => (
                <option key={profile.profile_id} value={profile.profile_id}>
                  {profileOptionLabel(profile)}
                </option>
              ))}
            </select>
          </div>

          {/* Profile ID */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-text-primary">Profile ID（新建可填）</label>
            <p className="text-xs text-text-secondary">给这套配置起一个机器可识别的代号，后续便于复用。</p>
            <input
              type="text"
              className={inputClass}
              placeholder="relay-main"
              value={modelProfileDraftIdInput}
              onChange={(e) => setModelProfileDraftIdInput(e.target.value)}
              disabled={streaming || modelProfileSaving || Boolean(selectedModelProfileId)}
            />
          </div>

          {/* Profile Name */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-text-primary">Profile 名称</label>
            <p className="text-xs text-text-secondary">给自己看的名称，建议写成"用途 + 场景"，便于识别。</p>
            <input
              type="text"
              className={inputClass}
              placeholder="主中转"
              value={modelProfileName}
              onChange={(e) => setModelProfileName(e.target.value)}
              disabled={streaming || modelProfileSaving}
            />
          </div>

          {/* Provider */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-text-primary">Provider</label>
            <p className="text-xs text-text-secondary">模型服务类型；不知道怎么选时，优先保持默认即可。</p>
            <select
              className={selectClass}
              value={modelProfileProvider}
              onChange={(e) =>
                setModelProfileProvider(e.target.value as "openai_compatible" | "deepseek" | "claude" | "gemini")
              }
              disabled={streaming || modelProfileSaving}
            >
              <option value="openai_compatible">openai_compatible（推荐中转）</option>
              <option value="deepseek">deepseek</option>
              <option value="claude">claude</option>
              <option value="gemini">gemini</option>
            </select>
          </div>

          {/* Base URL */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-text-primary">Base URL</label>
            <p className="text-xs text-text-secondary">模型服务入口地址，通常由服务商或中转服务提供。</p>
            <input
              type="text"
              className={inputClass}
              placeholder="https://api.example.com/v1"
              value={modelProfileBaseUrl}
              onChange={(e) => setModelProfileBaseUrl(e.target.value)}
              disabled={streaming || modelProfileSaving}
            />
          </div>

          {/* API Key */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-text-primary">API Key</label>
            <p className="text-xs text-text-secondary">访问模型服务的密钥，平台仅在调用时使用，不会展示明文。</p>
            <input
              type="password"
              className={inputClass}
              placeholder={selectedModelProfileId && modelProfileApiKeyMasked ? modelProfileApiKeyMasked : "sk-..."}
              value={modelProfileApiKey}
              onChange={(e) => {
                setClearModelProfileApiKey(false);
                setModelProfileApiKey(e.target.value);
              }}
              disabled={streaming || modelProfileSaving}
            />
          </div>

          {/* Model Name */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-text-primary">模型名</label>
            <p className="text-xs text-text-secondary">服务端可用的模型标识，不确定时使用服务商推荐值。</p>
            <input
              type="text"
              className={inputClass}
              placeholder="gpt-5-mini"
              value={modelProfileModel}
              onChange={(e) => setModelProfileModel(e.target.value)}
              disabled={streaming || modelProfileSaving}
            />
          </div>

          {/* Key Operation */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-text-primary">Key 操作</label>
            <p className="text-xs text-text-secondary">更新 Profile 时，选择保留现有密钥还是清空重设。</p>
            <select
              className={selectClass}
              value={clearModelProfileApiKey ? "clear" : "keep"}
              onChange={(e) => setClearModelProfileApiKey(e.target.value === "clear")}
              disabled={streaming || modelProfileSaving || !selectedModelProfileId}
            >
              <option value="keep">保持现有 Key</option>
              <option value="clear">清空现有 Key</option>
            </select>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap items-center gap-2 pt-2">
            <button
              type="button"
              className={btnGhost}
              onClick={onResetModelProfileDraft}
              disabled={streaming || modelProfileSaving}
            >
              新建草稿
            </button>
            <button
              type="button"
              className={btnPrimary}
              onClick={() => void onSaveModelProfile()}
              disabled={streaming || modelProfileSaving}
            >
              {modelProfileSaving ? "保存中..." : selectedModelProfileId ? "更新 Profile" : "创建 Profile"}
            </button>
            <button
              type="button"
              className={btnGhost}
              onClick={() => void onActivateModelProfile()}
              disabled={streaming || modelProfileSaving || !selectedModelProfileId}
            >
              设为激活
            </button>
            <button
              type="button"
              className={`${btnGhost} hover:!text-danger`}
              onClick={() => void onDeleteModelProfile()}
              disabled={streaming || modelProfileSaving || !selectedModelProfileId}
            >
              删除
            </button>
          </div>
          <p className="text-xs text-text-tertiary">保存后即可生效；"设为激活"会作为默认模型配置。</p>
        </div>
      </details>
    </div>
  );
});
