export type ChatRole = "user" | "assistant" | "system";

export interface UiMessage {
  id: string;
  role: ChatRole;
  content: string;
  streaming?: boolean;
}

export type DraftAutoSaveState = "idle" | "pending" | "saving" | "saved" | "error";

export interface ChatStreamRequest {
  project_id: number;
  content: string;
  session_id: number | null;
  chapter_id?: number | null;
  scene_beat_id?: number | null;
  prompt_template_id?: number | null;
  model: string | null;
  pov_mode: "global" | "character";
  pov_anchor: string | null;
  rag_mode?: "local" | "global" | "hybrid" | "mix" | null;
  deterministic_first?: boolean;
  thinking_enabled?: boolean;
  reference_project_ids?: number[];
  temperature_profile?: "action" | "chat" | "ghost" | "brainstorm" | null;
  temperature_override?: number | null;
  context_window_profile?: "balanced" | "chapter_focus" | "world_focus" | "minimal" | null;
  model_profile_id?: string | null;
}

export interface GhostTextRequest {
  project_id: number;
  chapter_id?: number | null;
  scene_beat_id?: number | null;
  prompt_template_id?: number | null;
  prefix_text: string;
  model: string | null;
  style_guard?: boolean;
  temperature_profile?: "action" | "chat" | "ghost" | "brainstorm" | null;
  temperature_override?: number | null;
  model_profile_id?: string | null;
}

export interface GhostTextResponse {
  suggestion: string;
  usage: Record<string, unknown>;
  evidence_policy: Record<string, unknown>;
}

export interface ChatStreamMetaEvent {
  type: "meta";
  session_id: number;
  assistant_message_id: number;
  proposed_action_ids: number[];
}

export interface EvidenceItem {
  kind: string;
  id: number;
  project_id?: number;
  title: string;
  score?: number;
  snippet?: string;
  fact?: string;
  confidence?: number | null;
  freshness_days?: number | null;
  citation?: {
    source?: string;
    chunk?: string;
  };
}

export interface EvidencePayload {
  type: "evidence";
  policy: {
    mode: "global" | "character";
    anchor: string | null;
    notes: string[];
    resolver_order: string;
    ranking_dimensions?: string;
    providers?: {
      dsl: string;
      graph: string;
      rag: string;
    };
    rag_route?: {
      mode: string;
      reason: string;
      source?: string;
    };
    rag_short_circuit?: {
      enabled: boolean;
      reason: string;
    };
    quality_gate?: {
      degraded: boolean;
      degrade_reasons: string[];
      citation_required: boolean;
      citation_count: number;
      reranker_expected: boolean;
      reranker_effective: boolean;
    };
    context_pack?: {
      enabled: boolean;
      source: string;
      age_ms: number;
    };
    reference_projects?: {
      requested: number[];
      resolved: Array<{
        project_id: number;
        settings: number;
        cards: number;
      }>;
      settings_count: number;
      cards_count: number;
    };
    runtime_options?: {
      thinking_enabled: boolean;
      context_window_profile?: string;
    };
    prompt_workshop?: {
      enabled: boolean;
      reason: string;
      requested_template_id: number | null;
      template_id?: number;
      template_name?: string;
      injected_settings: number;
      injected_cards: number;
    };
    chapter_context?: {
      enabled: boolean;
      reason: string;
      requested_chapter_id: number | null;
      chapter_id?: number;
      chapter_index?: number;
      chapter_version?: number;
      updated_at?: string | null;
      total_chars?: number;
    };
    outline_context?: {
      enabled: boolean;
      reason: string;
      requested_scene_beat_id?: number | null;
      selected_scene_beat_id?: number | null;
    };
  };
  summary: {
    dsl: number;
    graph: number;
    rag: number;
  };
  sources: {
    dsl: EvidenceItem[];
    graph: EvidenceItem[];
    rag: EvidenceItem[];
  };
}

export interface ChatStreamDeltaEvent {
  type: "delta";
  text: string;
}

export interface ChatStreamDoneEvent {
  type: "done";
  assistant_message_id: number;
  usage: Record<string, unknown>;
}

export interface ChatStreamErrorEvent {
  type: "error";
  message: string;
}

export interface ChatStreamTraceEvent {
  type: "trace";
  seq: number;
  scope: "pipeline" | "retrieval" | "tot" | string;
  stage: string;
  status: "info" | "running" | "ok" | "warning" | "error" | "skip" | string;
  message: string;
  step?: number;
  total?: number;
  meta?: Record<string, unknown>;
}

export type ChatStreamEvent =
  | ChatStreamMetaEvent
  | EvidencePayload
  | ChatStreamTraceEvent
  | ChatStreamDeltaEvent
  | ChatStreamDoneEvent
  | ChatStreamErrorEvent;

export interface ChatMessageDto {
  id: number;
  session_id: number;
  role: ChatRole;
  content: string;
  model?: string | null;
  created_at: string;
}

export interface ChatSessionSummary {
  id: number;
  project_id: number;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatSessionDeleteResult {
  deleted_session_id: number;
}

export interface ChatAction {
  id: number;
  session_id: number;
  action_type: string;
  status: string;
  payload: Record<string, unknown>;
  apply_result: Record<string, unknown>;
  undo_payload: Record<string, unknown>;
  idempotency_key: string;
  operator_id: string;
  created_at: string;
  applied_at?: string | null;
  undone_at?: string | null;
}

export interface ActionAuditLog {
  id: number;
  action_id: number;
  event_type: string;
  event_payload: Record<string, unknown>;
  operator_id: string;
  created_at: string;
}

export interface SettingEntry {
  id: number;
  project_id: number;
  key: string;
  value: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface StoryCard {
  id: number;
  project_id: number;
  title: string;
  content: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ModelProfile {
  profile_id: string;
  name?: string | null;
  provider: "openai_compatible" | "deepseek" | "claude" | "gemini" | "stub" | string;
  base_url?: string | null;
  model?: string | null;
  has_api_key: boolean;
  api_key_masked?: string | null;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface ModelProfileUpsertPayload {
  profile_id?: string | null;
  name?: string | null;
  provider?: string | null;
  base_url?: string | null;
  api_key?: string | null;
  model?: string | null;
}

export interface ModelProfileDeleteResult {
  deleted_profile_id: string;
}

export interface ConsistencyAuditReport {
  report_id: string;
  project_id: number;
  reason: string;
  trigger_source: string;
  status: "ok" | "warning" | string;
  summary: Record<string, unknown>;
  items: Array<Record<string, unknown>>;
  generated_at: string;
  generated_by: string;
  stored_key: string;
}

export interface ConsistencyAuditRunRequest {
  reason?: string;
  run_mode?: "async" | "sync";
  force?: boolean;
  max_chapters?: number | null;
}

export interface ConsistencyAuditRunResponse {
  project_id: number;
  queued: boolean;
  run_mode: "async" | "sync";
  reason: string;
  trigger_source: string;
  idempotency_key?: string | null;
  report?: ConsistencyAuditReport | null;
}

export interface GraphTimelineNode {
  id: string;
  label: string;
  kind: string;
  degree: number;
}

export interface GraphTimelineEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  confidence?: number | null;
  valid_from_chapter?: number | null;
  valid_to_chapter?: number | null;
  freshness_days?: number | null;
}

export interface GraphTimelineSnapshot {
  project_id: number;
  chapter_index: number;
  nodes: GraphTimelineNode[];
  edges: GraphTimelineEdge[];
  stats: Record<string, unknown>;
}

export interface PromptTemplate {
  id: number;
  project_id: number;
  name: string;
  system_prompt: string;
  user_prompt_prefix: string;
  knowledge_setting_keys: string[];
  knowledge_card_ids: number[];
  created_at: string;
  updated_at: string;
}

export interface PromptTemplateDeleteResult {
  deleted_template_id: number;
}

export interface PromptTemplateRevision {
  id: number;
  template_id: number;
  project_id: number;
  version: number;
  name: string;
  system_prompt: string;
  user_prompt_prefix: string;
  knowledge_setting_keys: string[];
  knowledge_card_ids: number[];
  operator_id: string;
  source: string;
  created_at: string;
}

export interface ProjectChapter {
  id: number;
  project_id: number;
  volume_id: number | null;
  chapter_index: number;
  title: string;
  content: string;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface ProjectChapterRevision {
  id: number;
  chapter_id: number;
  project_id: number;
  version: number;
  title: string;
  content: string;
  operator_id: string;
  source: string;
  semantic_summary: string[];
  created_at: string;
}

export interface ProjectChapterDeleteResult {
  deleted_chapter_id: number;
  active_chapter_id: number | null;
}

export interface ProjectVolume {
  id: number;
  project_id: number;
  volume_index: number;
  title: string;
  outline: string;
  created_at: string;
  updated_at: string;
}

export interface ProjectVolumeDeleteResult {
  deleted_volume_id: number;
  fallback_volume_id: number;
}

export interface SceneBeat {
  id: number;
  project_id: number;
  chapter_id: number;
  beat_index: number;
  content: string;
  status: "pending" | "done";
  created_at: string;
  updated_at: string;
}

export interface SceneBeatDeleteResult {
  deleted_beat_id: number;
}

export interface ForeshadowingCard {
  id: number;
  project_id: number;
  title: string;
  description: string;
  status: "open" | "resolved";
  planted_in_chapter_id: number | null;
  resolved_in_chapter_id: number | null;
  source_action_id: number | null;
  created_at: string;
  updated_at: string;
  resolved_at: string | null;
}

export interface ForeshadowingCardDeleteResult {
  deleted_foreshadow_id: number;
}

export interface GraphCandidateFact {
  fact_key: string;
  source_entity: string;
  relation: string;
  target_entity: string;
  confidence: number | null;
  source_ref: string;
  origin: string;
  evidence: string;
  state: string;
  valid_from_chapter: number | null;
  valid_to_chapter: number | null;
  updated_at: string | null;
}

export interface GraphCandidateListQuery {
  page?: number;
  page_size?: number;
  keyword?: string | null;
  source_ref?: string | null;
  min_confidence?: number | null;
  chapter_index?: number | null;
}

export interface GraphCandidateListResponse {
  project_id: number;
  page: number;
  page_size: number;
  total: number;
  items: GraphCandidateFact[];
}

export interface GraphCandidateBatchReviewRequest {
  decision: "confirm" | "reject";
  fact_keys: string[];
  manual_confirmed: boolean;
  chapter_index?: number | null;
}

export interface GraphCandidateBatchReviewResponse {
  project_id: number;
  decision: "confirm" | "reject" | string;
  requested_count: number;
  reviewed_count: number;
  fact_keys: string[];
}
