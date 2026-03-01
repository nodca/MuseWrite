import { memo } from "react";

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
    <section className="panel planning-panel">
      <div className="panel-title">
        <h2>结构化大纲与伏笔</h2>
        <small>Volume / Scene Beat / Foreshadow</small>
      </div>
      <div className="planning-grid">
        <article className="planning-card">
          <div className="panel-title sub">
            <h3>卷纲</h3>
            <small>{activeVolumeId ? `卷 #${activeVolumeId}` : "未绑定"}</small>
          </div>
          <div className="planning-row">
            <select
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
            <button className="btn ghost tiny" onClick={() => void onCreateVolume()} disabled={busy}>
              新建卷
            </button>
          </div>
          <textarea
            rows={4}
            value={volumeOutlineDraft}
            onChange={(event) => setVolumeOutlineDraft(event.target.value)}
            placeholder="卷纲：本卷核心冲突、推进目标与收束点。"
            disabled={busy || !activeVolumeId}
          />
          <div className="planning-row">
            <button className="btn ghost tiny" onClick={() => void onSaveVolumeOutline()} disabled={busy || !activeVolumeId}>
              保存卷纲
            </button>
          </div>
        </article>

        <article className="planning-card">
          <div className="panel-title sub">
            <h3>Scene Beats</h3>
            <small>{sceneBeats.length} 条</small>
          </div>
          <div className="scene-beat-list">
            {sceneBeats.length === 0 ? <p className="empty">当前章节还没有 Beat</p> : null}
            {sceneBeats.map((beat) => (
              <article
                key={beat.id}
                className={`scene-beat-item ${beat.id === activeSceneBeatId ? "active" : ""}`}
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
                <div className="action-ops">
                  <button
                    className="btn ghost tiny"
                    onClick={(event) => {
                      event.stopPropagation();
                      void onToggleSceneBeatStatus(beat.id, beat.status !== "done");
                    }}
                    disabled={busy}
                  >
                    {beat.status === "done" ? "标记进行中" : "标记已完成"}
                  </button>
                  <button
                    className="btn ghost tiny"
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
            rows={3}
            value={newBeatContent}
            onChange={(event) => setNewBeatContent(event.target.value)}
            placeholder="新增 Beat：例如「男主发现破绽并留下悬念」"
            disabled={busy || !activeChapterId}
          />
          <div className="planning-row">
            <button className="btn ghost tiny" onClick={() => void onCreateSceneBeat()} disabled={busy || !activeChapterId}>
              添加 Beat
            </button>
            <button className="btn ghost tiny" onClick={() => onSelectSceneBeat(null)} disabled={busy}>
              不使用 Beat 约束
            </button>
          </div>
        </article>

        <article className="planning-card">
          <div className="panel-title sub">
            <h3>伏笔追踪</h3>
            <small>{foreshadowCards.length} 条</small>
          </div>
          {overdueForeshadowCards.length > 0 ? (
            <p className="draft-hint warning">提醒：有 {overdueForeshadowCards.length} 条伏笔已超 50 章未收束。</p>
          ) : null}
          <div className="foreshadow-list">
            {foreshadowCards.length === 0 ? <p className="empty">暂无伏笔卡</p> : null}
            {foreshadowCards.map((item) => (
              <article key={item.id} className="foreshadow-item">
                <strong>
                  {item.title} · {item.status === "resolved" ? "已收束" : "未收束"}
                </strong>
                <p>{item.description || "（无描述）"}</p>
                <div className="action-ops">
                  <button
                    className="btn ghost tiny"
                    onClick={() =>
                      void onToggleForeshadowStatus(item, item.status === "resolved" ? "open" : "resolved")
                    }
                    disabled={busy}
                  >
                    {item.status === "resolved" ? "改为未收束" : "标记已收束"}
                  </button>
                  <button className="btn ghost tiny" onClick={() => void onDeleteForeshadowCard(item.id)} disabled={busy}>
                    删除
                  </button>
                </div>
              </article>
            ))}
          </div>
          <input
            type="text"
            value={foreshadowDraftTitle}
            onChange={(event) => setForeshadowDraftTitle(event.target.value)}
            placeholder="伏笔标题：如「半块玉佩」"
            disabled={busy}
          />
          <textarea
            rows={3}
            value={foreshadowDraftDescription}
            onChange={(event) => setForeshadowDraftDescription(event.target.value)}
            placeholder="伏笔描述：埋入信息、预期收束方向。"
            disabled={busy}
          />
          <div className="planning-row">
            <button className="btn ghost tiny" onClick={() => void onCreateForeshadowCard()} disabled={busy}>
              新建伏笔卡
            </button>
          </div>
        </article>
      </div>
    </section>
  );
});
