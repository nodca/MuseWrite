import type {
  ActionAuditLog,
  ChatAction,
  ChatMessageDto,
  ChatSessionDeleteResult,
  ChatSessionSummary,
  ConsistencyAuditReport,
  ConsistencyAuditRunRequest,
  ConsistencyAuditRunResponse,
  ForeshadowingCard,
  ForeshadowingCardDeleteResult,
  GraphCandidateBatchReviewRequest,
  GraphCandidateBatchReviewResponse,
  GraphCandidateListQuery,
  GraphCandidateListResponse,
  GraphTimelineSnapshot,
  GhostTextRequest,
  GhostTextResponse,
  ModelProfile,
  ModelProfileDeleteResult,
  ModelProfileUpsertPayload,
  PromptTemplate,
  PromptTemplateDeleteResult,
  PromptTemplateRevision,
  ProjectChapter,
  ProjectChapterDeleteResult,
  ProjectChapterRevision,
  ProjectVolume,
  ProjectVolumeDeleteResult,
  SceneBeat,
  SceneBeatDeleteResult,
  ChatStreamEvent,
  ChatStreamRequest,
  SettingEntry,
  StoryCard,
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";
const API_TOKEN = (import.meta.env.VITE_API_TOKEN ?? "").trim();
const inFlightGetJsonRequests = new Map<string, Promise<unknown>>();
const swrResponseCache = new Map<string, { at: number; data: unknown; ttlMs: number }>();
const swrRevalidateInFlight = new Map<string, Promise<unknown>>();
const SWR_CACHE_MAX_ENTRIES = 128;
const SWR_CACHE_RULES: Array<{ pattern: RegExp; ttlMs: number }> = [
  { pattern: /^\/api\/chat\/projects\/\d+\/settings(?:\?.*)?$/i, ttlMs: 2200 },
  { pattern: /^\/api\/chat\/projects\/\d+\/cards(?:\?.*)?$/i, ttlMs: 2200 },
  { pattern: /^\/api\/chat\/projects\/\d+\/consistency-audits(?:\?.*)?$/i, ttlMs: 2000 },
  { pattern: /^\/api\/chat\/projects\/\d+\/graph-candidates(?:\?.*)?$/i, ttlMs: 1200 },
  { pattern: /^\/api\/chat\/projects\/\d+\/graph-timeline(?:\?.*)?$/i, ttlMs: 1200 },
  { pattern: /^\/api\/chat\/projects\/\d+\/chapters(?:\/\d+(?:\/revisions)?)?(?:\?.*)?$/i, ttlMs: 2200 },
  { pattern: /^\/api\/chat\/projects\/\d+\/draft(?:\/revisions)?(?:\?.*)?$/i, ttlMs: 2200 },
];

function authHeaders(): Record<string, string> {
  if (!API_TOKEN) return {};
  return {
    Authorization: `Bearer ${API_TOKEN}`,
  };
}

function buildHeaders(
  initHeaders?: HeadersInit,
  options?: {
    includeJsonContentType?: boolean;
  }
): Headers {
  const headers = new Headers(initHeaders);
  Object.entries(authHeaders()).forEach(([key, value]) => {
    headers.set(key, value);
  });
  if (options?.includeJsonContentType && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  return headers;
}

async function parseError(resp: Response): Promise<string> {
  const contentType = resp.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    const payload = (await resp.json()) as { detail?: string; error?: string };
    return payload.detail ?? payload.error ?? `Request failed: ${resp.status}`;
  }
  return (await resp.text()) || `Request failed: ${resp.status}`;
}

function serializeHeaders(headers: Headers): string {
  return Array.from(headers.entries())
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => `${key}:${value}`)
    .join("|");
}

function nowMs(): number {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return performance.now();
  }
  return Date.now();
}

function resolveSWRCacheTtlMs(path: string): number | null {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  for (const rule of SWR_CACHE_RULES) {
    if (rule.pattern.test(normalized)) {
      return rule.ttlMs;
    }
  }
  return null;
}

function setSWRCacheEntry(key: string, data: unknown, ttlMs: number): void {
  swrResponseCache.set(key, {
    at: nowMs(),
    data,
    ttlMs,
  });
  if (swrResponseCache.size <= SWR_CACHE_MAX_ENTRIES) {
    return;
  }
  const oldest = swrResponseCache.keys().next().value;
  if (typeof oldest === "string") {
    swrResponseCache.delete(oldest);
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const includeJsonContentType = init?.body !== undefined && init?.body !== null;
  const headers = buildHeaders(init?.headers, {
    includeJsonContentType,
  });
  const method = (init?.method ?? "GET").toUpperCase();
  const canReuseInFlightGet =
    method === "GET" &&
    !includeJsonContentType &&
    init?.signal === undefined;
  const swrTtlMs = canReuseInFlightGet ? resolveSWRCacheTtlMs(path) : null;
  const requestKey = `${url}::${serializeHeaders(headers)}`;

  const fetchAndParse = async (): Promise<T> => {
    const resp = await fetch(url, {
      ...init,
      method,
      headers,
    });
    if (!resp.ok) {
      throw new Error(await parseError(resp));
    }
    const data = (await resp.json()) as T;
    if (swrTtlMs !== null) {
      setSWRCacheEntry(requestKey, data, swrTtlMs);
    }
    return data;
  };

  if (!canReuseInFlightGet) {
    const data = await fetchAndParse();
    if (method !== "GET") {
      swrResponseCache.clear();
      swrRevalidateInFlight.clear();
    }
    return data;
  }

  if (swrTtlMs !== null) {
    const cached = swrResponseCache.get(requestKey);
    if (cached) {
      const ageMs = nowMs() - cached.at;
      if (ageMs <= cached.ttlMs) {
        return cached.data as T;
      }
      if (!swrRevalidateInFlight.has(requestKey) && !inFlightGetJsonRequests.has(requestKey)) {
        const refreshPromise = fetchAndParse()
          .catch(() => undefined)
          .finally(() => {
            if (swrRevalidateInFlight.get(requestKey) === refreshPromise) {
              swrRevalidateInFlight.delete(requestKey);
            }
          });
        swrRevalidateInFlight.set(requestKey, refreshPromise);
      }
      return cached.data as T;
    }
  }

  let inFlightRequest = inFlightGetJsonRequests.get(requestKey) as Promise<T> | undefined;
  if (!inFlightRequest) {
    inFlightRequest = fetchAndParse();
    inFlightGetJsonRequests.set(requestKey, inFlightRequest);
  }

  try {
    return await inFlightRequest;
  } finally {
    if (inFlightGetJsonRequests.get(requestKey) === inFlightRequest) {
      inFlightGetJsonRequests.delete(requestKey);
    }
  }
}

export type ChatStreamTimingMetrics = {
  ttfbMs: number;
  firstEventMs: number | null;
  firstTokenMs: number | null;
  completeMs: number;
};

function parseSseEventBlock(block: string): ChatStreamEvent | null {
  let offset = 0;
  while (offset <= block.length) {
    const nextLineBreak = block.indexOf("\n", offset);
    const end = nextLineBreak === -1 ? block.length : nextLineBreak;
    const line = block.slice(offset, end).trim();
    if (line.startsWith("data:")) {
      const payload = line.slice(5).trimStart();
      if (!payload) return null;
      return JSON.parse(payload) as ChatStreamEvent;
    }
    if (nextLineBreak === -1) break;
    offset = nextLineBreak + 1;
  }
  return null;
}

export async function streamChat(
  req: ChatStreamRequest,
  onEvent: (event: ChatStreamEvent) => void,
  options?: {
    signal?: AbortSignal;
    onMetrics?: (metrics: ChatStreamTimingMetrics) => void;
  }
): Promise<void> {
  const startedAt = nowMs();
  const resp = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: buildHeaders(undefined, { includeJsonContentType: true }),
    body: JSON.stringify(req),
    signal: options?.signal,
  });
  const ttfbMs = nowMs() - startedAt;

  if (!resp.ok) {
    throw new Error(await parseError(resp));
  }

  const reader = resp.body?.getReader();
  if (!reader) {
    throw new Error("Stream reader is unavailable");
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let firstEventMs: number | null = null;
  let firstTokenMs: number | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundaryIndex = buffer.indexOf("\n\n");
    while (boundaryIndex !== -1) {
      const block = buffer.slice(0, boundaryIndex);
      buffer = buffer.slice(boundaryIndex + 2);
      const parsed = parseSseEventBlock(block);
      if (parsed) {
        if (firstEventMs === null) {
          firstEventMs = nowMs() - startedAt;
        }
        if (firstTokenMs === null && parsed.type === "delta" && parsed.text.trim().length > 0) {
          firstTokenMs = nowMs() - startedAt;
        }
        onEvent(parsed);
      }
      boundaryIndex = buffer.indexOf("\n\n");
    }
  }

  if (buffer.trim()) {
    const parsed = parseSseEventBlock(buffer);
    if (parsed) {
      if (firstEventMs === null) {
        firstEventMs = nowMs() - startedAt;
      }
      if (firstTokenMs === null && parsed.type === "delta" && parsed.text.trim().length > 0) {
        firstTokenMs = nowMs() - startedAt;
      }
      onEvent(parsed);
    }
  }

  options?.onMetrics?.({
    ttfbMs: Number(ttfbMs.toFixed(2)),
    firstEventMs: firstEventMs === null ? null : Number(firstEventMs.toFixed(2)),
    firstTokenMs: firstTokenMs === null ? null : Number(firstTokenMs.toFixed(2)),
    completeMs: Number((nowMs() - startedAt).toFixed(2)),
  });
}

export function generateGhostText(payload: GhostTextRequest): Promise<GhostTextResponse> {
  return requestJson<GhostTextResponse>("/api/chat/ghost-text", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getSessionMessages(sessionId: number): Promise<ChatMessageDto[]> {
  return requestJson<ChatMessageDto[]>(`/api/chat/sessions/${sessionId}/messages`);
}

export function getProjectSessions(projectId: number, limit = 24): Promise<ChatSessionSummary[]> {
  return requestJson<ChatSessionSummary[]>(`/api/chat/projects/${projectId}/sessions?limit=${limit}`);
}

export function updateProjectSession(
  projectId: number,
  sessionId: number,
  payload: {
    title: string;
  }
): Promise<ChatSessionSummary> {
  return requestJson<ChatSessionSummary>(`/api/chat/projects/${projectId}/sessions/${sessionId}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function deleteProjectSession(
  projectId: number,
  sessionId: number
): Promise<ChatSessionDeleteResult> {
  return requestJson<ChatSessionDeleteResult>(`/api/chat/projects/${projectId}/sessions/${sessionId}`, {
    method: "DELETE",
  });
}

export function getSessionActions(sessionId: number): Promise<ChatAction[]> {
  return requestJson<ChatAction[]>(`/api/chat/sessions/${sessionId}/actions`);
}

export function getActionLogs(actionId: number): Promise<ActionAuditLog[]> {
  return requestJson<ActionAuditLog[]>(`/api/chat/actions/${actionId}/logs`);
}

export function decideAction(
  actionId: number,
  decision: "apply" | "reject" | "undo",
  eventPayload: Record<string, unknown> = {}
): Promise<ChatAction> {
  return requestJson<ChatAction>(`/api/chat/actions/${actionId}/${decision}`, {
    method: "POST",
    body: JSON.stringify({
      event_payload: {
        source: "web-ui",
        ...eventPayload,
      },
    }),
  });
}

export function getProjectSettings(projectId: number): Promise<SettingEntry[]> {
  return requestJson<SettingEntry[]>(`/api/chat/projects/${projectId}/settings`);
}

export function getProjectConsistencyAudits(
  projectId: number,
  limit = 20
): Promise<ConsistencyAuditReport[]> {
  return requestJson<ConsistencyAuditReport[]>(
    `/api/chat/projects/${projectId}/consistency-audits?limit=${limit}`
  );
}

export function runProjectConsistencyAudit(
  projectId: number,
  payload: ConsistencyAuditRunRequest
): Promise<ConsistencyAuditRunResponse> {
  return requestJson<ConsistencyAuditRunResponse>(
    `/api/chat/projects/${projectId}/consistency-audits/run`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}

export function getProjectGraphTimeline(
  projectId: number,
  chapterIndex: number,
  limit = 240
): Promise<GraphTimelineSnapshot> {
  const normalizedChapter = Number.isFinite(chapterIndex) ? Math.max(0, Math.floor(chapterIndex)) : 0;
  const normalizedLimit = Number.isFinite(limit) ? Math.min(1200, Math.max(20, Math.floor(limit))) : 240;
  return requestJson<GraphTimelineSnapshot>(
    `/api/chat/projects/${projectId}/graph-timeline?chapter_index=${normalizedChapter}&limit=${normalizedLimit}`
  );
}

export function getProjectModelProfiles(projectId: number): Promise<ModelProfile[]> {
  return requestJson<ModelProfile[]>(`/api/chat/projects/${projectId}/model-profiles`);
}

export function createProjectModelProfile(
  projectId: number,
  payload: ModelProfileUpsertPayload
): Promise<ModelProfile> {
  return requestJson<ModelProfile>(`/api/chat/projects/${projectId}/model-profiles`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateProjectModelProfile(
  projectId: number,
  profileId: string,
  payload: ModelProfileUpsertPayload
): Promise<ModelProfile> {
  return requestJson<ModelProfile>(`/api/chat/projects/${projectId}/model-profiles/${encodeURIComponent(profileId)}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function activateProjectModelProfile(projectId: number, profileId: string): Promise<ModelProfile> {
  return requestJson<ModelProfile>(
    `/api/chat/projects/${projectId}/model-profiles/${encodeURIComponent(profileId)}/activate`,
    {
      method: "POST",
    }
  );
}

export function deleteProjectModelProfile(
  projectId: number,
  profileId: string
): Promise<ModelProfileDeleteResult> {
  return requestJson<ModelProfileDeleteResult>(
    `/api/chat/projects/${projectId}/model-profiles/${encodeURIComponent(profileId)}`,
    {
      method: "DELETE",
    }
  );
}

export function getProjectCards(projectId: number): Promise<StoryCard[]> {
  return requestJson<StoryCard[]>(`/api/chat/projects/${projectId}/cards`);
}

export function getProjectGraphCandidates(
  projectId: number,
  query: GraphCandidateListQuery = {}
): Promise<GraphCandidateListResponse> {
  const params = new URLSearchParams();
  if (query.page && query.page > 0) params.set("page", String(Math.floor(query.page)));
  if (query.page_size && query.page_size > 0) params.set("page_size", String(Math.floor(query.page_size)));
  if (typeof query.keyword === "string" && query.keyword.trim()) params.set("keyword", query.keyword.trim());
  if (typeof query.source_ref === "string" && query.source_ref.trim()) params.set("source_ref", query.source_ref.trim());
  if (typeof query.min_confidence === "number" && Number.isFinite(query.min_confidence)) {
    params.set("min_confidence", String(query.min_confidence));
  }
  if (typeof query.chapter_index === "number" && Number.isFinite(query.chapter_index) && query.chapter_index > 0) {
    params.set("chapter_index", String(Math.floor(query.chapter_index)));
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  return requestJson<GraphCandidateListResponse>(`/api/chat/projects/${projectId}/graph-candidates${suffix}`);
}

export function reviewProjectGraphCandidates(
  projectId: number,
  payload: GraphCandidateBatchReviewRequest
): Promise<GraphCandidateBatchReviewResponse> {
  return requestJson<GraphCandidateBatchReviewResponse>(`/api/chat/projects/${projectId}/graph-candidates/review`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getProjectPromptTemplates(projectId: number): Promise<PromptTemplate[]> {
  return requestJson<PromptTemplate[]>(`/api/chat/projects/${projectId}/prompt-templates`);
}

export function createProjectPromptTemplate(
  projectId: number,
  payload: {
    name: string;
    system_prompt: string;
    user_prompt_prefix: string;
    knowledge_setting_keys: string[];
    knowledge_card_ids: number[];
  }
): Promise<PromptTemplate> {
  return requestJson<PromptTemplate>(`/api/chat/projects/${projectId}/prompt-templates`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateProjectPromptTemplate(
  projectId: number,
  templateId: number,
  payload: {
    name: string;
    system_prompt: string;
    user_prompt_prefix: string;
    knowledge_setting_keys: string[];
    knowledge_card_ids: number[];
  }
): Promise<PromptTemplate> {
  return requestJson<PromptTemplate>(`/api/chat/projects/${projectId}/prompt-templates/${templateId}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function deleteProjectPromptTemplate(
  projectId: number,
  templateId: number
): Promise<PromptTemplateDeleteResult> {
  return requestJson<PromptTemplateDeleteResult>(
    `/api/chat/projects/${projectId}/prompt-templates/${templateId}`,
    {
      method: "DELETE",
    }
  );
}

export function getProjectPromptTemplateRevisions(
  projectId: number,
  templateId: number,
  limit = 20
): Promise<PromptTemplateRevision[]> {
  return requestJson<PromptTemplateRevision[]>(
    `/api/chat/projects/${projectId}/prompt-templates/${templateId}/revisions?limit=${limit}`
  );
}

export function rollbackProjectPromptTemplate(
  projectId: number,
  templateId: number,
  payload: {
    target_version: number;
  }
): Promise<PromptTemplate> {
  return requestJson<PromptTemplate>(
    `/api/chat/projects/${projectId}/prompt-templates/${templateId}/rollback`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}

export function preheatContextPack(projectId: number): Promise<{
  project_id: number;
  settings_count: number;
  cards_count: number;
  ttl_seconds: number;
}> {
  return requestJson(`/api/chat/projects/${projectId}/context-pack/preheat`, {
    method: "POST",
  });
}

export function getProjectVolumes(projectId: number): Promise<ProjectVolume[]> {
  return requestJson<ProjectVolume[]>(`/api/chat/projects/${projectId}/volumes`);
}

export function createProjectVolume(
  projectId: number,
  payload: {
    title: string | null;
    outline: string;
  }
): Promise<ProjectVolume> {
  return requestJson<ProjectVolume>(`/api/chat/projects/${projectId}/volumes`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateProjectVolume(
  projectId: number,
  volumeId: number,
  payload: {
    title: string;
    outline: string;
  }
): Promise<ProjectVolume> {
  return requestJson<ProjectVolume>(`/api/chat/projects/${projectId}/volumes/${volumeId}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function deleteProjectVolume(
  projectId: number,
  volumeId: number
): Promise<ProjectVolumeDeleteResult> {
  return requestJson<ProjectVolumeDeleteResult>(`/api/chat/projects/${projectId}/volumes/${volumeId}`, {
    method: "DELETE",
  });
}

export function getProjectChapters(projectId: number): Promise<ProjectChapter[]> {
  return requestJson<ProjectChapter[]>(`/api/chat/projects/${projectId}/chapters`);
}

export function reorderProjectChapters(
  projectId: number,
  payload: {
    ordered_ids: number[];
  }
): Promise<ProjectChapter[]> {
  return requestJson<ProjectChapter[]>(`/api/chat/projects/${projectId}/chapters/reorder`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function createProjectChapter(
  projectId: number,
  payload: {
    title: string | null;
    volume_id?: number | null;
  }
): Promise<ProjectChapter> {
  return requestJson<ProjectChapter>(`/api/chat/projects/${projectId}/chapters`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getProjectChapter(projectId: number, chapterId: number): Promise<ProjectChapter> {
  return requestJson<ProjectChapter>(`/api/chat/projects/${projectId}/chapters/${chapterId}`);
}

export function saveProjectChapter(
  projectId: number,
  chapterId: number,
  payload: {
    title: string;
    content: string;
    volume_id?: number | null;
    expected_version: number | null;
  }
): Promise<ProjectChapter> {
  return requestJson<ProjectChapter>(`/api/chat/projects/${projectId}/chapters/${chapterId}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function getProjectChapterRevisions(
  projectId: number,
  chapterId: number,
  limit = 20
): Promise<ProjectChapterRevision[]> {
  return requestJson<ProjectChapterRevision[]>(
    `/api/chat/projects/${projectId}/chapters/${chapterId}/revisions?limit=${limit}`
  );
}

export function rollbackProjectChapter(
  projectId: number,
  chapterId: number,
  payload: {
    target_version: number;
  }
): Promise<ProjectChapter> {
  return requestJson<ProjectChapter>(`/api/chat/projects/${projectId}/chapters/${chapterId}/rollback`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function moveProjectChapter(
  projectId: number,
  chapterId: number,
  payload: {
    direction: "up" | "down";
  }
): Promise<ProjectChapter> {
  return requestJson<ProjectChapter>(`/api/chat/projects/${projectId}/chapters/${chapterId}/move`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function deleteProjectChapter(
  projectId: number,
  chapterId: number
): Promise<ProjectChapterDeleteResult> {
  return requestJson<ProjectChapterDeleteResult>(`/api/chat/projects/${projectId}/chapters/${chapterId}`, {
    method: "DELETE",
  });
}

export function getSceneBeats(projectId: number, chapterId: number): Promise<SceneBeat[]> {
  return requestJson<SceneBeat[]>(`/api/chat/projects/${projectId}/chapters/${chapterId}/scene-beats`);
}

export function createSceneBeat(
  projectId: number,
  chapterId: number,
  payload: {
    content: string;
    status: "pending" | "done";
  }
): Promise<SceneBeat> {
  return requestJson<SceneBeat>(`/api/chat/projects/${projectId}/chapters/${chapterId}/scene-beats`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateSceneBeat(
  projectId: number,
  chapterId: number,
  beatId: number,
  payload: {
    content: string;
    status: "pending" | "done";
  }
): Promise<SceneBeat> {
  return requestJson<SceneBeat>(`/api/chat/projects/${projectId}/chapters/${chapterId}/scene-beats/${beatId}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function deleteSceneBeat(
  projectId: number,
  chapterId: number,
  beatId: number
): Promise<SceneBeatDeleteResult> {
  return requestJson<SceneBeatDeleteResult>(
    `/api/chat/projects/${projectId}/chapters/${chapterId}/scene-beats/${beatId}`,
    {
      method: "DELETE",
    }
  );
}

export function getForeshadowingCards(
  projectId: number,
  options?: {
    status?: "open" | "resolved";
    overdue_for_chapter_id?: number | null;
    chapter_gap?: number;
  }
): Promise<ForeshadowingCard[]> {
  const query = new URLSearchParams();
  if (options?.status) query.set("status", options.status);
  if (options?.overdue_for_chapter_id) query.set("overdue_for_chapter_id", String(options.overdue_for_chapter_id));
  if (options?.chapter_gap) query.set("chapter_gap", String(options.chapter_gap));
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return requestJson<ForeshadowingCard[]>(`/api/chat/projects/${projectId}/foreshadowing-cards${suffix}`);
}

export function createForeshadowingCard(
  projectId: number,
  payload: {
    title: string;
    description: string;
    planted_in_chapter_id?: number | null;
    source_action_id?: number | null;
  }
): Promise<ForeshadowingCard> {
  return requestJson<ForeshadowingCard>(`/api/chat/projects/${projectId}/foreshadowing-cards`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateForeshadowingCard(
  projectId: number,
  cardId: number,
  payload: {
    title: string;
    description: string;
    status: "open" | "resolved";
    planted_in_chapter_id?: number | null;
    resolved_in_chapter_id?: number | null;
  }
): Promise<ForeshadowingCard> {
  return requestJson<ForeshadowingCard>(`/api/chat/projects/${projectId}/foreshadowing-cards/${cardId}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function deleteForeshadowingCard(
  projectId: number,
  cardId: number
): Promise<ForeshadowingCardDeleteResult> {
  return requestJson<ForeshadowingCardDeleteResult>(`/api/chat/projects/${projectId}/foreshadowing-cards/${cardId}`, {
    method: "DELETE",
  });
}
