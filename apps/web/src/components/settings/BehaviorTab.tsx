import { memo } from "react";

export type BehaviorTabProps = {
  chatTemperatureProfile: "action" | "chat" | "brainstorm";
  setChatTemperatureProfile: (value: "action" | "chat" | "brainstorm") => void;
  suggestionTemperatureProfile: "suggestion" | "chat" | "action" | "brainstorm";
  setSuggestionTemperatureProfile: (value: "suggestion" | "chat" | "action" | "brainstorm") => void;
  thinkingEnabled: boolean;
  setThinkingEnabled: (value: boolean) => void;
  temperatureOverrideInput: string;
  setTemperatureOverrideInput: (value: string) => void;
  streaming: boolean;
};

const selectClass =
  "w-full rounded-md border border-border-default bg-surface-primary px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent-primary/30";
const inputClass =
  "w-full rounded-md border border-border-default bg-surface-primary px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent-primary/30";

export const BehaviorTab = memo(function BehaviorTab({
  chatTemperatureProfile,
  setChatTemperatureProfile,
  suggestionTemperatureProfile,
  setSuggestionTemperatureProfile,
  thinkingEnabled,
  setThinkingEnabled,
  temperatureOverrideInput,
  setTemperatureOverrideInput,
  streaming,
}: BehaviorTabProps) {
  return (
    <div className="space-y-6">
      {/* Chat Temperature Profile */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-text-primary">聊天温度策略</label>
        <p className="text-xs text-text-secondary">影响助手回答风格：越保守越稳定，越发散越有新意。</p>
        <select
          className={selectClass}
          value={chatTemperatureProfile}
          onChange={(e) => setChatTemperatureProfile(e.target.value as "action" | "chat" | "brainstorm")}
          disabled={streaming}
        >
          <option value="action">action（稳健提案）</option>
          <option value="chat">chat（常规写作）</option>
          <option value="brainstorm">brainstorm（发散灵感）</option>
        </select>
      </div>

      {/* Suggestion Temperature Profile */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-text-primary">补全建议温度策略</label>
        <p className="text-xs text-text-secondary">控制自动补全建议的创造程度，建议与当前写作阶段匹配。</p>
        <select
          className={selectClass}
          value={suggestionTemperatureProfile}
          onChange={(e) =>
            setSuggestionTemperatureProfile(e.target.value as "suggestion" | "chat" | "action" | "brainstorm")
          }
          disabled={streaming}
        >
          <option value="suggestion">suggestion（自动补全）</option>
          <option value="chat">chat（常规）</option>
          <option value="action">action（保守）</option>
          <option value="brainstorm">brainstorm（发散）</option>
        </select>
      </div>

      {/* Thinking Toggle */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-text-primary">Thinking</label>
        <p className="text-xs text-text-secondary">开启后会更谨慎地组织答案，通常更稳但速度略慢。</p>
        <select
          className={selectClass}
          value={thinkingEnabled ? "on" : "off"}
          onChange={(e) => setThinkingEnabled(e.target.value === "on")}
          disabled={streaming}
        >
          <option value="off">关闭</option>
          <option value="on">开启（更稳健）</option>
        </select>
      </div>

      {/* Collapsible: Temperature Override */}
      <details className="group rounded-lg border border-border-default">
        <summary className="flex cursor-pointer items-center gap-2 px-4 py-3 text-sm font-medium text-text-secondary hover:text-text-primary select-none">
          <span className="transition-transform group-open:rotate-90">▸</span>
          <span>专家设置</span>
          <span className="ml-auto text-xs text-text-tertiary">仅在需要手动覆盖默认策略时展开</span>
        </summary>
        <div className="border-t border-border-default px-4 py-4">
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-text-primary">温度覆盖（可空 0~2）</label>
            <p className="text-xs text-text-secondary">手动覆盖所有温度策略；留空时按上面的策略自动决定。</p>
            <input
              type="number"
              className={inputClass}
              min={0}
              max={2}
              step={0.05}
              value={temperatureOverrideInput}
              onChange={(e) => setTemperatureOverrideInput(e.target.value)}
              disabled={streaming}
            />
          </div>
        </div>
      </details>
    </div>
  );
});
