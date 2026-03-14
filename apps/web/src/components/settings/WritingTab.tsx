import { memo } from "react";
import type { WritingTheme } from "../SettingsDialog";

export type WritingTabProps = {
  writingTheme: WritingTheme;
  setWritingTheme: (value: WritingTheme) => void;
  typewriterModeEnabled: boolean;
  setTypewriterModeEnabled: (value: boolean) => void;
  projectId: number;
  setProjectId: (projectId: number) => void;
  streaming: boolean;
};

const selectClass =
  "w-full rounded-md border border-border-default bg-surface-primary px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent-primary/30";
const inputClass =
  "w-full rounded-md border border-border-default bg-surface-primary px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent-primary/30";

export const WritingTab = memo(function WritingTab({
  writingTheme,
  setWritingTheme,
  typewriterModeEnabled,
  setTypewriterModeEnabled,
  projectId,
  setProjectId,
  streaming,
}: WritingTabProps) {
  return (
    <div className="space-y-6">
      {/* Writing Theme */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-text-primary">写作主题</label>
        <p className="text-xs text-text-secondary">选择编辑器的视觉主题风格。</p>
        <select
          data-autofocus
          className={selectClass}
          value={writingTheme}
          onChange={(e) => setWritingTheme(e.target.value as WritingTheme)}
        >
          <option value="paper">paper（纸感衬线）</option>
          <option value="wenkai">wenkai（文楷复古）</option>
          <option value="modern">modern（现代简洁）</option>
          <option value="contrast">contrast（高对比）</option>
        </select>
      </div>

      {/* Typewriter mode */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-text-primary">打字机滚动</label>
        <p className="text-xs text-text-secondary">开启后，光标所在行始终保持在编辑器中央。</p>
        <select
          className={selectClass}
          value={typewriterModeEnabled ? "on" : "off"}
          onChange={(e) => setTypewriterModeEnabled(e.target.value === "on")}
        >
          <option value="on">开启（光标行居中）</option>
          <option value="off">关闭</option>
        </select>
      </div>

      {/* Collapsible: Project & Debug */}
      <details className="group rounded-lg border border-border-default">
        <summary className="flex cursor-pointer items-center gap-2 px-4 py-3 text-sm font-medium text-text-secondary hover:text-text-primary select-none">
          <span className="transition-transform group-open:rotate-90">▸</span>
          <span>项目与调试</span>
          <span className="ml-auto text-xs text-text-tertiary">仅在切换项目或排查问题时使用</span>
        </summary>
        <div className="border-t border-border-default px-4 py-4">
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-text-primary">项目 ID</label>
            <p className="text-xs text-text-secondary">当前写作项目的标识符。</p>
            <input
              type="number"
              className={inputClass}
              value={projectId}
              min={1}
              onChange={(e) => setProjectId(Number(e.target.value || 1))}
              disabled={streaming}
            />
          </div>
        </div>
      </details>
    </div>
  );
});
