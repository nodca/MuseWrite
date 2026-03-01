import { memo, useEffect, useMemo, useState, type FormEvent } from "react";
import { getProjectGraphCandidates, reviewProjectGraphCandidates } from "./api/chatApi";
import type {
  EvidencePayload,
  GraphCandidateFact,
  PromptTemplate,
  PromptTemplateRevision,
  SettingEntry,
  StoryCard,
} from "./types";

function formatJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function formatDateTime(value?: string | null): string {
  if (!value) return "--";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

export type DebugSnapshotGridProps = {
  evidence: EvidencePayload | null;
  settings: SettingEntry[];
  cards: StoryCard[];
};

export const DebugSnapshotGrid = memo(function DebugSnapshotGrid({
  evidence,
  settings,
  cards,
}: DebugSnapshotGridProps) {
  return (
    <section className="snapshot-grid">
      <article className="panel">
        <div className="panel-title sub">
          <h3>检索证据</h3>
          <small>{evidence ? evidence.policy.resolver_order : "未收到 evidence"}</small>
        </div>
        <div className="snapshot-list">
          {!evidence ? <p className="empty">发送消息后会显示 DSL / GRAPH / RAG 命中。</p> : null}
          {evidence ? (
            <article className="snapshot-card">
              <strong>
                POV: {evidence.policy.mode}
                {evidence.policy.anchor ? ` (${evidence.policy.anchor})` : ""}
              </strong>
              <pre>
{formatJson({
  summary: evidence.summary,
  providers: evidence.policy.providers,
  ragRoute: evidence.policy.rag_route,
  ragShortCircuit: evidence.policy.rag_short_circuit,
  contextPack: evidence.policy.context_pack,
  chapterContext: evidence.policy.chapter_context,
  ranking: evidence.policy.ranking_dimensions,
  referenceProjects: evidence.policy.reference_projects,
  runtimeOptions: evidence.policy.runtime_options,
  qualityGate: evidence.policy.quality_gate,
  notes: evidence.policy.notes,
})}
              </pre>
            </article>
          ) : null}
          {evidence?.sources.dsl?.map((item, idx) => (
            <article key={`dsl-${item.id}-${idx}`} className="snapshot-card">
              <strong>[DSL] {item.title}</strong>
              <pre>{item.snippet || ""}</pre>
            </article>
          ))}
          {evidence?.sources.graph?.map((item, idx) => (
            <article key={`graph-${item.id}-${idx}`} className="snapshot-card">
              <strong>[GRAPH] {item.title}</strong>
              <pre>{item.fact || item.snippet || ""}</pre>
            </article>
          ))}
          {evidence?.sources.rag?.map((item, idx) => (
            <article key={`rag-${item.id}-${idx}`} className="snapshot-card">
              <strong>
                [RAG] {item.title}
                {typeof item.score === "number" ? ` (score=${item.score})` : ""}
              </strong>
              {item.citation ? (
                <small>
                  citation: {item.citation.source || "unknown"}
                  {item.citation.chunk ? `#${item.citation.chunk}` : ""}
                </small>
              ) : null}
              <pre>{item.snippet || ""}</pre>
            </article>
          ))}
        </div>
      </article>

      <article className="panel">
        <div className="panel-title sub">
          <h3>设定快照</h3>
          <small>{settings.length} 条</small>
        </div>
        <div className="snapshot-list">
          {settings.length === 0 ? <p className="empty">暂无设定</p> : null}
          {settings.map((item) => (
            <article key={item.id} className="snapshot-card">
              <strong>{item.key}</strong>
              <pre>{formatJson(item.value)}</pre>
            </article>
          ))}
        </div>
      </article>

      <article className="panel">
        <div className="panel-title sub">
          <h3>卡片快照</h3>
          <small>{cards.length} 张</small>
        </div>
        <div className="snapshot-list">
          {cards.length === 0 ? <p className="empty">暂无卡片</p> : null}
          {cards.map((card) => (
            <article key={card.id} className="snapshot-card">
              <strong>{card.title}</strong>
              <pre>{formatJson(card.content)}</pre>
            </article>
          ))}
        </div>
      </article>
    </section>
  );
});

export type PromptWorkshopPanelProps = {
  activePromptTemplate: PromptTemplate | null;
  activePromptTemplateId: number | null;
  templateSaving: boolean;
  promptTemplates: PromptTemplate[];
  onHandleActiveTemplateChange: (value: string) => void;
  onStartCreateTemplateDraft: () => void;
  onCopyTemplateDraft: () => Promise<void>;
  templateName: string;
  setTemplateName: (value: string) => void;
  templateSystemPrompt: string;
  setTemplateSystemPrompt: (value: string) => void;
  templateUserPromptPrefix: string;
  setTemplateUserPromptPrefix: (value: string) => void;
  settings: SettingEntry[];
  templateKnowledgeSettingKeys: string[];
  setTemplateKnowledgeSettingKeys: (value: string[]) => void;
  cards: StoryCard[];
  templateKnowledgeCardIds: number[];
  setTemplateKnowledgeCardIds: (value: number[]) => void;
  onSaveTemplateDraft: () => Promise<void>;
  templateDraftId: number | null;
  onDeleteTemplateDraft: () => Promise<void>;
  onRefreshProjectSnapshot: (projectId: number) => Promise<void>;
  projectId: number;
  selectedKnowledgeSettings: SettingEntry[];
  selectedKnowledgeCards: StoryCard[];
  estimatedPromptChars: number;
  missingSettingKeys: string[];
  missingCardIds: number[];
  templateRevisions: PromptTemplateRevision[];
  templateRevisionsLoading: boolean;
  onRollbackTemplateToVersion: (targetVersion: number) => Promise<void>;
};

export const PromptWorkshopPanel = memo(function PromptWorkshopPanel({
  activePromptTemplate,
  activePromptTemplateId,
  templateSaving,
  promptTemplates,
  onHandleActiveTemplateChange,
  onStartCreateTemplateDraft,
  onCopyTemplateDraft,
  templateName,
  setTemplateName,
  templateSystemPrompt,
  setTemplateSystemPrompt,
  templateUserPromptPrefix,
  setTemplateUserPromptPrefix,
  settings,
  templateKnowledgeSettingKeys,
  setTemplateKnowledgeSettingKeys,
  cards,
  templateKnowledgeCardIds,
  setTemplateKnowledgeCardIds,
  onSaveTemplateDraft,
  templateDraftId,
  onDeleteTemplateDraft,
  onRefreshProjectSnapshot,
  projectId,
  selectedKnowledgeSettings,
  selectedKnowledgeCards,
  estimatedPromptChars,
  missingSettingKeys,
  missingCardIds,
  templateRevisions,
  templateRevisionsLoading,
  onRollbackTemplateToVersion,
}: PromptWorkshopPanelProps) {
  return (
    <section className="panel prompt-panel">
      <div className="panel-title">
        <h2>Prompt + 知识库面板</h2>
        <small>
          当前生效模板：
          {activePromptTemplate ? `${activePromptTemplate.name} (#${activePromptTemplate.id})` : "无"}
        </small>
      </div>
      <div className="prompt-toolbar">
        <label>
          会话模板
          <select
            value={activePromptTemplateId ?? ""}
            onChange={(event) => onHandleActiveTemplateChange(event.target.value)}
            disabled={templateSaving}
          >
            <option value="">不使用模板</option>
            {promptTemplates.map((template) => (
              <option key={template.id} value={template.id}>
                {template.name} (#{template.id})
              </option>
            ))}
          </select>
        </label>
        <button className="btn ghost tiny" onClick={onStartCreateTemplateDraft} disabled={templateSaving}>
          新建模板草稿
        </button>
        <button className="btn ghost tiny" onClick={() => void onCopyTemplateDraft()} disabled={templateSaving}>
          复制当前模板
        </button>
      </div>

      <div className="prompt-form-grid">
        <label>
          模板名称
          <input
            type="text"
            value={templateName}
            onChange={(event) => setTemplateName(event.target.value)}
            disabled={templateSaving}
          />
        </label>

        <label>
          System Prompt
          <textarea
            rows={5}
            value={templateSystemPrompt}
            onChange={(event) => setTemplateSystemPrompt(event.target.value)}
            disabled={templateSaving}
          />
        </label>

        <label>
          User Prompt Prefix
          <textarea
            rows={3}
            value={templateUserPromptPrefix}
            onChange={(event) => setTemplateUserPromptPrefix(event.target.value)}
            disabled={templateSaving}
          />
        </label>

        <label>
          注入设定（多选）
          <select
            multiple
            size={Math.min(Math.max(settings.length, 4), 10)}
            value={templateKnowledgeSettingKeys}
            onChange={(event) => {
              const selected = Array.from(event.currentTarget.selectedOptions).map((item) => item.value);
              setTemplateKnowledgeSettingKeys(selected);
            }}
            disabled={templateSaving}
          >
            {settings.map((item) => (
              <option key={item.id} value={item.key}>
                {item.key}
              </option>
            ))}
          </select>
        </label>

        <label>
          注入卡片（多选）
          <select
            multiple
            size={Math.min(Math.max(cards.length, 4), 10)}
            value={templateKnowledgeCardIds.map((item) => String(item))}
            onChange={(event) => {
              const selected = Array.from(event.currentTarget.selectedOptions)
                .map((item) => Number(item.value))
                .filter((item) => Number.isFinite(item) && item > 0);
              setTemplateKnowledgeCardIds(selected);
            }}
            disabled={templateSaving}
          >
            {cards.map((card) => (
              <option key={card.id} value={card.id}>
                #{card.id} {card.title}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="draft-actions">
        <button className="btn primary tiny" onClick={() => void onSaveTemplateDraft()} disabled={templateSaving}>
          {templateSaving ? "保存中..." : templateDraftId ? "更新模板" : "创建模板"}
        </button>
        <button
          className="btn ghost tiny"
          onClick={() => void onDeleteTemplateDraft()}
          disabled={templateSaving || !templateDraftId}
        >
          删除当前草稿模板
        </button>
        <button
          className="btn ghost tiny"
          onClick={() => void onRefreshProjectSnapshot(projectId)}
          disabled={templateSaving}
        >
          刷新模板列表
        </button>
      </div>
      <div className="prompt-preview">
        <p className="draft-hint">
          注入预览：设定 {selectedKnowledgeSettings.length} 条，卡片 {selectedKnowledgeCards.length} 条，预计上下文约{" "}
          {estimatedPromptChars} 字符。
        </p>
        {missingSettingKeys.length > 0 ? (
          <p className="draft-hint warning">冲突提示：有 {missingSettingKeys.length} 个设定 key 已不存在。</p>
        ) : null}
        {missingCardIds.length > 0 ? (
          <p className="draft-hint warning">冲突提示：有 {missingCardIds.length} 个卡片 ID 已不存在。</p>
        ) : null}
        {estimatedPromptChars > 12000 ? (
          <p className="draft-hint warning">提示：注入上下文较大，可能增加延迟与成本。</p>
        ) : null}
      </div>
      <details className="prompt-revision-history">
        <summary>
          模板版本历史（最近 {templateRevisions.length} 条）
          {templateRevisionsLoading ? " · 加载中..." : ""}
        </summary>
        <div className="draft-revision-list">
          {templateRevisions.length === 0 ? <p className="empty">暂无模板历史</p> : null}
          {templateRevisions.map((revision) => (
            <article key={revision.id} className="draft-revision-card">
              <div className="msg-head">
                <span>
                  v{revision.version} · {revision.source}
                </span>
                <small>{formatDateTime(revision.created_at)}</small>
              </div>
              <pre>{`${revision.name}\n${revision.user_prompt_prefix.slice(0, 180)}`}</pre>
              <div className="action-ops">
                <button
                  className="btn ghost tiny"
                  onClick={() => void onRollbackTemplateToVersion(revision.version)}
                  disabled={templateSaving || !templateDraftId}
                >
                  回滚到此版本
                </button>
              </div>
            </article>
          ))}
        </div>
      </details>
      <p className="draft-hint">
        聊天请求会携带 `prompt_template_id`。证据面板的 `promptWorkshop` 会显示本次注入与命中信息。
      </p>
    </section>
  );
});

function formatConfidence(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "--";
  return value.toFixed(3);
}

function shortenText(value: string, maxChars = 72): string {
  const text = String(value || "").trim();
  if (!text) return "--";
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars).trimEnd()}...`;
}

type GraphCandidateReviewPanelProps = {
  projectId: number;
};

export const GraphCandidateReviewPanel = memo(function GraphCandidateReviewPanel({
  projectId,
}: GraphCandidateReviewPanelProps) {
  const [items, setItems] = useState<GraphCandidateFact[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [keywordInput, setKeywordInput] = useState("");
  const [sourceRefInput, setSourceRefInput] = useState("");
  const [minConfidenceInput, setMinConfidenceInput] = useState("");
  const [chapterIndexInput, setChapterIndexInput] = useState("");
  const [keyword, setKeyword] = useState("");
  const [sourceRef, setSourceRef] = useState("");
  const [minConfidence, setMinConfidence] = useState<number | null>(null);
  const [chapterIndex, setChapterIndex] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [reviewing, setReviewing] = useState<"confirm" | "reject" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);
  const [selectedFactKeys, setSelectedFactKeys] = useState<string[]>([]);

  const totalPages = useMemo(() => Math.max(Math.ceil(total / Math.max(pageSize, 1)), 1), [total, pageSize]);
  const selectedFactKeySet = useMemo(() => new Set(selectedFactKeys), [selectedFactKeys]);
  const allPageSelected = useMemo(
    () => items.length > 0 && items.every((item) => selectedFactKeySet.has(item.fact_key)),
    [items, selectedFactKeySet]
  );

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await getProjectGraphCandidates(projectId, {
          page,
          page_size: pageSize,
          keyword: keyword || null,
          source_ref: sourceRef || null,
          min_confidence: minConfidence,
          chapter_index: chapterIndex,
        });
        if (cancelled) return;
        setItems(response.items ?? []);
        setTotal(Number(response.total ?? 0) || 0);
        setSelectedFactKeys((prev) => prev.filter((key) => (response.items ?? []).some((item) => item.fact_key === key)));
        if ((response.items ?? []).length === 0 && page > 1 && Number(response.total ?? 0) > 0) {
          const fallbackPage = Math.max(Math.ceil(Number(response.total ?? 0) / Math.max(pageSize, 1)), 1);
          if (fallbackPage !== page) {
            setPage(fallbackPage);
          }
        }
      } catch (loadError) {
        if (cancelled) return;
        const message = loadError instanceof Error ? loadError.message : "读取候选列表失败";
        setError(message);
        setItems([]);
        setTotal(0);
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [projectId, page, pageSize, keyword, sourceRef, minConfidence, chapterIndex, refreshToken]);

  const handleQuery = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const nextKeyword = keywordInput.trim();
    const nextSourceRef = sourceRefInput.trim();
    const minRaw = minConfidenceInput.trim();
    const chapterRaw = chapterIndexInput.trim();
    const parsedConfidence = minRaw ? Number(minRaw) : Number.NaN;
    const parsedChapter = chapterRaw ? Number(chapterRaw) : Number.NaN;

    setKeyword(nextKeyword);
    setSourceRef(nextSourceRef);
    setMinConfidence(
      minRaw && Number.isFinite(parsedConfidence)
        ? Math.max(0, Math.min(1, parsedConfidence))
        : null
    );
    setChapterIndex(
      chapterRaw && Number.isFinite(parsedChapter) && parsedChapter > 0
        ? Math.floor(parsedChapter)
        : null
    );
    setPage(1);
    setNotice(null);
  };

  const resetQuery = () => {
    setKeywordInput("");
    setSourceRefInput("");
    setMinConfidenceInput("");
    setChapterIndexInput("");
    setKeyword("");
    setSourceRef("");
    setMinConfidence(null);
    setChapterIndex(null);
    setPage(1);
    setNotice(null);
  };

  const toggleFactKey = (factKey: string) => {
    setSelectedFactKeys((prev) => (prev.includes(factKey) ? prev.filter((item) => item !== factKey) : [...prev, factKey]));
  };

  const toggleSelectPage = () => {
    const pageFactKeys = items.map((item) => item.fact_key);
    if (allPageSelected) {
      setSelectedFactKeys((prev) => prev.filter((key) => !pageFactKeys.includes(key)));
      return;
    }
    setSelectedFactKeys((prev) => {
      const merged = [...prev];
      for (const factKey of pageFactKeys) {
        if (!merged.includes(factKey)) {
          merged.push(factKey);
        }
      }
      return merged;
    });
  };

  const handleBatchReview = async (decision: "confirm" | "reject") => {
    if (selectedFactKeys.length === 0) return;
    const actionLabel = decision === "confirm" ? "确认" : "驳回";
    if (!window.confirm(`确认${actionLabel}已选中的 ${selectedFactKeys.length} 条候选事实吗？`)) {
      return;
    }
    setReviewing(decision);
    setError(null);
    setNotice(null);
    try {
      const response = await reviewProjectGraphCandidates(projectId, {
        decision,
        fact_keys: selectedFactKeys,
        manual_confirmed: true,
        chapter_index: chapterIndex,
      });
      setNotice(`${actionLabel}完成：${response.reviewed_count}/${response.requested_count} 条`);
      setSelectedFactKeys([]);
      setRefreshToken((prev) => prev + 1);
    } catch (reviewError) {
      const message = reviewError instanceof Error ? reviewError.message : "批量审核失败";
      setError(message);
    } finally {
      setReviewing(null);
    }
  };

  return (
    <section className="panel candidate-review-panel">
      <div className="panel-title">
        <h2>候选列表查询 / 批量审核</h2>
        <small>{`总计 ${total} 条，已选 ${selectedFactKeys.length} 条`}</small>
      </div>

      <form className="candidate-query-toolbar" onSubmit={handleQuery}>
        <label>
          关键词
          <input
            type="text"
            value={keywordInput}
            onChange={(event) => setKeywordInput(event.target.value)}
            placeholder="fact_key / 实体 / 关系 / evidence"
          />
        </label>
        <label>
          source_ref
          <input
            type="text"
            value={sourceRefInput}
            onChange={(event) => setSourceRefInput(event.target.value)}
            placeholder="chat.action.*"
          />
        </label>
        <label>
          min_confidence
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={minConfidenceInput}
            onChange={(event) => setMinConfidenceInput(event.target.value)}
            placeholder="0.00 ~ 1.00"
          />
        </label>
        <label>
          chapter_index
          <input
            type="number"
            min={1}
            step={1}
            value={chapterIndexInput}
            onChange={(event) => setChapterIndexInput(event.target.value)}
            placeholder="可选"
          />
        </label>
        <label>
          page_size
          <select
            value={pageSize}
            onChange={(event) => {
              const next = Number(event.target.value);
              setPageSize(Number.isFinite(next) && next > 0 ? next : 20);
              setPage(1);
            }}
          >
            <option value={20}>20</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
        </label>
        <div className="candidate-toolbar-actions">
          <button className="btn primary tiny" type="submit" disabled={loading || reviewing !== null}>
            {loading ? "查询中..." : "查询"}
          </button>
          <button
            className="btn ghost tiny"
            type="button"
            onClick={resetQuery}
            disabled={loading || reviewing !== null}
          >
            重置
          </button>
        </div>
      </form>

      <div className="candidate-batch-actions">
        <button
          className="btn primary tiny"
          type="button"
          onClick={() => void handleBatchReview("confirm")}
          disabled={selectedFactKeys.length === 0 || loading || reviewing !== null}
        >
          {reviewing === "confirm" ? "确认中..." : "批量确认"}
        </button>
        <button
          className="btn ghost tiny"
          type="button"
          onClick={() => void handleBatchReview("reject")}
          disabled={selectedFactKeys.length === 0 || loading || reviewing !== null}
        >
          {reviewing === "reject" ? "驳回中..." : "批量驳回"}
        </button>
        <button
          className="btn ghost tiny"
          type="button"
          onClick={toggleSelectPage}
          disabled={items.length === 0 || loading || reviewing !== null}
        >
          {allPageSelected ? "取消勾选本页" : "勾选本页"}
        </button>
        <small>{`第 ${page}/${totalPages} 页`}</small>
      </div>

      {notice ? <p className="candidate-notice">{notice}</p> : null}
      {error ? <p className="candidate-error">{error}</p> : null}

      <div className="candidate-table-wrap">
        <table className="candidate-table">
          <thead>
            <tr>
              <th>选择</th>
              <th>source</th>
              <th>relation</th>
              <th>target</th>
              <th>confidence</th>
              <th>source_ref</th>
              <th>evidence</th>
              <th>updated_at</th>
            </tr>
          </thead>
          <tbody>
            {items.length === 0 ? (
              <tr>
                <td colSpan={8} className="candidate-empty-cell">
                  {loading ? "加载中..." : "暂无候选数据"}
                </td>
              </tr>
            ) : (
              items.map((item) => (
                <tr key={item.fact_key}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedFactKeySet.has(item.fact_key)}
                      onChange={() => toggleFactKey(item.fact_key)}
                    />
                  </td>
                  <td>{item.source_entity}</td>
                  <td>{item.relation}</td>
                  <td>{item.target_entity}</td>
                  <td>{formatConfidence(item.confidence)}</td>
                  <td title={item.source_ref || ""}>{shortenText(item.source_ref || "", 28)}</td>
                  <td title={item.evidence || ""}>{shortenText(item.evidence || "", 82)}</td>
                  <td>{formatDateTime(item.updated_at)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <div className="candidate-pagination">
        <button
          className="btn ghost tiny"
          type="button"
          onClick={() => setPage((prev) => Math.max(1, prev - 1))}
          disabled={page <= 1 || loading}
        >
          上一页
        </button>
        <button
          className="btn ghost tiny"
          type="button"
          onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
          disabled={page >= totalPages || loading}
        >
          下一页
        </button>
      </div>
    </section>
  );
});
