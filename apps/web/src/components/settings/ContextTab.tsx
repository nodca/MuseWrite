import { memo } from "react";

export type ContextTabProps = {
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
  referenceProjectInput: string;
  setReferenceProjectInput: (value: string) => void;
  streaming: boolean;
};

const selectClass =
  "w-full rounded-md border border-border-default bg-surface-primary px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent-primary/30";
const inputClass =
  "w-full rounded-md border border-border-default bg-surface-primary px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent-primary/30";

export const ContextTab = memo(function ContextTab({
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
  referenceProjectInput,
  setReferenceProjectInput,
  streaming,
}: ContextTabProps) {
  return (
    <div className="space-y-6">
      {/* Context Window Strategy */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-text-primary">上下文滑窗策略</label>
        <p className="text-xs text-text-secondary">控制助手优先关注"当前章节"还是"全局世界观"。</p>
        <select
          className={selectClass}
          value={contextWindowProfile}
          onChange={(e) =>
            setContextWindowProfile(e.target.value as "balanced" | "chapter_focus" | "world_focus" | "minimal")
          }
          disabled={streaming}
        >
          <option value="balanced">balanced（默认均衡）</option>
          <option value="chapter_focus">chapter_focus（章节优先）</option>
          <option value="world_focus">world_focus（世界观优先）</option>
          <option value="minimal">minimal（最小上下文）</option>
        </select>
      </div>

      {/* POV Mode */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-text-primary">POV 模式</label>
        <p className="text-xs text-text-secondary">决定叙事视角是全局叙述，还是围绕某个角色展开。</p>
        <select
          className={selectClass}
          value={povMode}
          onChange={(e) => setPovMode(e.target.value as "global" | "character")}
          disabled={streaming}
        >
          <option value="global">全局视角</option>
          <option value="character">角色沙箱</option>
        </select>
      </div>

      {/* POV Anchor */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-text-primary">POV 锚点（角色名）</label>
        <p className="text-xs text-text-secondary">仅在角色视角时填写，用来指定"围绕谁来写"。</p>
        <input
          type="text"
          className={inputClass}
          placeholder={povMode === "character" ? "例如：林澈" : "全局模式可留空"}
          value={povAnchor}
          onChange={(e) => setPovAnchor(e.target.value)}
          disabled={streaming}
        />
      </div>

      {/* RAG Routing */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-text-primary">RAG 路由</label>
        <p className="text-xs text-text-secondary">决定检索资料来自当前项目、全局知识，或两者混合。</p>
        <select
          className={selectClass}
          value={ragMode}
          onChange={(e) => setRagMode(e.target.value as "local" | "global" | "hybrid" | "mix")}
          disabled={streaming}
        >
          <option value="local">local</option>
          <option value="global">global</option>
          <option value="hybrid">hybrid</option>
          <option value="mix">mix</option>
        </select>
      </div>

      {/* Deterministic First */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-text-primary">事实短路</label>
        <p className="text-xs text-text-secondary">开启后优先采用已确认事实，减少跑偏但灵活度会降低。</p>
        <select
          className={selectClass}
          value={deterministicFirst ? "on" : "off"}
          onChange={(e) => setDeterministicFirst(e.target.value === "on")}
          disabled={streaming}
        >
          <option value="off">关闭</option>
          <option value="on">开启（DSL+GRAPH优先）</option>
        </select>
      </div>

      {/* Cross-project References */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-text-primary">跨项目引用（逗号分隔）</label>
        <p className="text-xs text-text-secondary">把其他项目当作参考资料源，适合系列文共享设定。</p>
        <input
          type="text"
          className={inputClass}
          placeholder="例如：2,3"
          value={referenceProjectInput}
          onChange={(e) => setReferenceProjectInput(e.target.value)}
          disabled={streaming}
        />
      </div>
    </div>
  );
});
