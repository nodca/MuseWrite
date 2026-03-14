import { memo } from "react";
import clsx from "clsx";

import type { ForeshadowingCard, ProjectVolume, SceneBeat } from "../types";

export type StoryPlanningPanelProps = {
  activeChapterId: number | null;
  volumes: ProjectVolume[];
  activeVolumeId: number | null;
  onSelectVolume: (volumeId: number) => void;
  onCreateVolume: () => Promise<void>;
  onBindChapterToVolume: (volumeId: number) => Promise<void>;
  volumeOutlineDraft: string;
  setVolumeOutlineDraft: (value: string) => void;
  onSaveVolumeOutline: () => Promise<void>;
  sceneBeats: SceneBeat[];
  activeSceneBeatId: number | null;
  onSelectSceneBeat: (beatId: number | null) => void;
  newBeatContent: string;
  setNewBeatContent: (value: string) => void;
  onCreateSceneBeat: () => Promise<void>;
  onToggleSceneBeatStatus: (beatId: number, done: boolean) => Promise<void>;
  onDeleteSceneBeat: (beatId: number) => Promise<void>;
  foreshadowCards: ForeshadowingCard[];
  overdueForeshadowCards: ForeshadowingCard[];
  foreshadowDraftTitle: string;
  setForeshadowDraftTitle: (value: string) => void;
  foreshadowDraftDescription: string;
  setForeshadowDraftDescription: (value: string) => void;
  onCreateForeshadowCard: () => Promise<void>;
  onToggleForeshadowStatus: (card: ForeshadowingCard, nextStatus: "open" | "resolved") => Promise<void>;
  onDeleteForeshadowCard: (cardId: number) => Promise<void>;
  busy: boolean;
};

const inputClass =
  "w-full rounded-md border border-border-default bg-surface-primary px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent-primary/30 disabled:opacity-40";

const btnGhostTiny =
  "rounded-md border border-border-default px-3 py-1.5 text-xs text-text-secondary hover:text-text-primary hover:bg-surface-elevated disabled:opacity-40 transition-colors";

export const StoryPlanningPanel = memo(function StoryPlanningPanel({
  activeChapterId,
  volumes,
  activeVolumeId,
  onSelectVolume,
  onCreateVolume,
  onBindChapterToVolume,
  volumeOutlineDraft,
  setVolumeOutlineDraft,
  onSaveVolumeOutline,
  sceneBeats,
  activeSceneBeatId,
  onSelectSceneBeat,
  newBeatContent,
  setNewBeatContent,
  onCreateSceneBeat,
  onToggleSceneBeatStatus,
  onDeleteSceneBeat,
  foreshadowCards,
  overdueForeshadowCards,
  foreshadowDraftTitle,
  setForeshadowDraftTitle,
  foreshadowDraftDescription,
  setForeshadowDraftDescription,
  onCreateForeshadowCard,
  onToggleForeshadowStatus,
  onDeleteForeshadowCard,
  busy,
}: StoryPlanningPanelProps) {
  return (
    <section className="space-y-4 p-4">
      <div className="flex items-center justify-between mb-3">
        <h2>结构化大纲与伏笔</h2>
        <small>Volume / Scene Beat / Foreshadow</small>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <article className="rounded-lg border border-border-default bg-surface-primary p-4 space-y-3">
          <div className="flex items-center justify-between mb-2">
            <h3>卷纲</h3>
            <small>{activeVolumeId ? `卷 #${activeVolumeId}` : "未绑定"}</small>
          </div>
          <div className="flex items-center gap-2">
            <select
              className={inputClass}
              value={activeVolumeId ?? ""}
              onChange={(event) => {
                const nextId = Number(event.target.value || 0);
                if (!nextId) return;
                onSelectVolume(nextId);
                if (activeChapterId) {
                  void onBindChapterToVolume(nextId);
                }
              }}
              disabled={busy}
            >
              <option value="">选择卷</option>
              {volumes.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.volume_index}. {item.title}
                </option>
              ))}
            </select>
            <button className={btnGhostTiny} onClick={() => void onCreateVolume()} disabled={busy}>
              新建卷
            </button>
          </div>
          <textarea
            className={inputClass}
            rows={4}
            value={volumeOutlineDraft}
            onChange={(event) => setVolumeOutlineDraft(event.target.value)}
            placeholder="卷纲：本卷核心冲突、推进目标与收束点。"
            disabled={busy || !activeVolumeId}
          />
          <div className="flex items-center gap-2">
            <button className={btnGhostTiny} onClick={() => void onSaveVolumeOutline()} disabled={busy || !activeVolumeId}>
              保存卷纲
            </button>
          </div>
        </article>

        <article className="rounded-lg border border-border-default bg-surface-primary p-4 space-y-3">
          <div className="flex items-center justify-between mb-2">
            <h3>Scene Beats</h3>
            <small>{sceneBeats.length} 条</small>
          </div>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {sceneBeats.length === 0 ? <p className="text-sm text-text-tertiary italic">当前章节还没有 Beat</p> : null}
            {sceneBeats.map((beat) => (
              <article
                key={beat.id}
                className={clsx(
                  "rounded-md border border-border-default p-3 cursor-pointer hover:bg-surface-elevated transition-colors",
                  beat.id === activeSceneBeatId && "ring-2 ring-accent-primary/30",
                )}
                role="button"
                tabIndex={0}
                aria-pressed={beat.id === activeSceneBeatId}
                onClick={() => onSelectSceneBeat(beat.id)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelectSceneBeat(beat.id);
                  }
                }}
              >
                <strong>
                  Beat {beat.beat_index} · {beat.status === "done" ? "已完成" : "进行中"}
                </strong>
                <p>{beat.content || "（空）"}</p>
                <div className="flex items-center gap-2 mt-2">
                  <button
                    className={btnGhostTiny}
                    onClick={(event) => {
                      event.stopPropagation();
                      void onToggleSceneBeatStatus(beat.id, beat.status !== "done");
                    }}
                    disabled={busy}
                  >
                    {beat.status === "done" ? "标记进行中" : "标记已完成"}
                  </button>
                  <button
                    className={btnGhostTiny}
                    onClick={(event) => {
                      event.stopPropagation();
                      void onDeleteSceneBeat(beat.id);
                    }}
                    disabled={busy}
                  >
                    删除
                  </button>
                </div>
              </article>
            ))}
          </div>
          <textarea
            className={inputClass}
            rows={3}
            value={newBeatContent}
            onChange={(event) => setNewBeatContent(event.target.value)}
            placeholder="新增 Beat：例如「男主发现破绽并留下悬念」"
            disabled={busy || !activeChapterId}
          />
          <div className="flex items-center gap-2">
            <button className={btnGhostTiny} onClick={() => void onCreateSceneBeat()} disabled={busy || !activeChapterId}>
              添加 Beat
            </button>
            <button className={btnGhostTiny} onClick={() => onSelectSceneBeat(null)} disabled={busy}>
              不使用 Beat 约束
            </button>
          </div>
        </article>

        <article className="rounded-lg border border-border-default bg-surface-primary p-4 space-y-3">
          <div className="flex items-center justify-between mb-2">
            <h3>伏笔追踪</h3>
            <small>{foreshadowCards.length} 条</small>
          </div>
          {overdueForeshadowCards.length > 0 ? (
            <p className="text-sm text-warning font-medium">提醒：有 {overdueForeshadowCards.length} 条伏笔已超 50 章未收束。</p>
          ) : null}
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {foreshadowCards.length === 0 ? <p className="text-sm text-text-tertiary italic">暂无伏笔卡</p> : null}
            {foreshadowCards.map((item) => (
              <article key={item.id} className="rounded-md border border-border-default p-3">
                <strong>
                  {item.title} · {item.status === "resolved" ? "已收束" : "未收束"}
                </strong>
                <p>{item.description || "（无描述）"}</p>
                <div className="flex items-center gap-2 mt-2">
                  <button
                    className={btnGhostTiny}
                    onClick={() =>
                      void onToggleForeshadowStatus(item, item.status === "resolved" ? "open" : "resolved")
                    }
                    disabled={busy}
                  >
                    {item.status === "resolved" ? "改为未收束" : "标记已收束"}
                  </button>
                  <button className={btnGhostTiny} onClick={() => void onDeleteForeshadowCard(item.id)} disabled={busy}>
                    删除
                  </button>
                </div>
              </article>
            ))}
          </div>
          <input
            className={inputClass}
            type="text"
            value={foreshadowDraftTitle}
            onChange={(event) => setForeshadowDraftTitle(event.target.value)}
            placeholder="伏笔标题：如「半块玉佩」"
            disabled={busy}
          />
          <textarea
            className={inputClass}
            rows={3}
            value={foreshadowDraftDescription}
            onChange={(event) => setForeshadowDraftDescription(event.target.value)}
            placeholder="伏笔描述：埋入信息、预期收束方向。"
            disabled={busy}
          />
          <div className="flex items-center gap-2">
            <button className={btnGhostTiny} onClick={() => void onCreateForeshadowCard()} disabled={busy}>
              新建伏笔卡
            </button>
          </div>
        </article>
      </div>
    </section>
  );
});
