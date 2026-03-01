from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    project_id: int
    content: str = Field(min_length=1, max_length=20000)
    session_id: Optional[int] = None
    chapter_id: Optional[int] = Field(default=None, ge=1)
    scene_beat_id: Optional[int] = Field(default=None, ge=1)
    prompt_template_id: Optional[int] = Field(default=None, ge=1)
    model: Optional[str] = Field(default=None, max_length=128)
    pov_mode: str = Field(default="global", max_length=32)
    pov_anchor: Optional[str] = Field(default=None, max_length=128)
    rag_mode: Optional[str] = Field(default=None, max_length=16)
    deterministic_first: bool = False
    thinking_enabled: bool = False
    reference_project_ids: list[int] = Field(default_factory=list, max_length=8)
    temperature_profile: Optional[str] = Field(default=None, max_length=32)
    temperature_override: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    context_window_profile: Optional[str] = Field(default=None, max_length=32)
    budget_mode: Optional[str] = Field(default=None, max_length=32)
    enable_tot: bool = False
    current_location: Optional[str] = Field(default=None, max_length=128)
    model_profile_id: Optional[str] = Field(default=None, max_length=64)


class GhostTextRequest(BaseModel):
    project_id: int
    chapter_id: Optional[int] = Field(default=None, ge=1)
    scene_beat_id: Optional[int] = Field(default=None, ge=1)
    prompt_template_id: Optional[int] = Field(default=None, ge=1)
    prefix_text: str = Field(default="", max_length=8000)
    model: Optional[str] = Field(default=None, max_length=128)
    style_guard: bool = True
    temperature_profile: Optional[str] = Field(default=None, max_length=32)
    temperature_override: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    model_profile_id: Optional[str] = Field(default=None, max_length=64)


class GhostTextResponse(BaseModel):
    suggestion: str
    usage: dict
    evidence_policy: dict


class ChatMessageRead(BaseModel):
    id: int
    session_id: int
    role: str
    content: str
    model: Optional[str]
    created_at: datetime


class ChatSessionRead(BaseModel):
    id: int
    project_id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ChatSessionUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class ChatSessionDeleteResult(BaseModel):
    deleted_session_id: int


class ChatActionRead(BaseModel):
    id: int
    session_id: int
    action_type: str
    status: str
    payload: dict
    apply_result: dict
    undo_payload: dict
    idempotency_key: str
    operator_id: str
    created_at: datetime
    applied_at: Optional[datetime]
    undone_at: Optional[datetime]


class ChatActionDecisionRequest(BaseModel):
    event_payload: dict = Field(default_factory=dict)


class ChatActionCreateRequest(BaseModel):
    action_type: str = Field(min_length=1, max_length=64)
    payload: dict = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=1, max_length=128)


class SettingEntryRead(BaseModel):
    id: int
    project_id: int
    key: str
    value: dict
    aliases: list[str]
    created_at: datetime
    updated_at: datetime


class ModelProfileRead(BaseModel):
    profile_id: str
    name: Optional[str] = None
    provider: str
    base_url: Optional[str] = None
    model: Optional[str] = None
    has_api_key: bool
    api_key_masked: Optional[str] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class ModelProfileUpsertRequest(BaseModel):
    profile_id: Optional[str] = Field(default=None, max_length=64)
    name: Optional[str] = Field(default=None, max_length=128)
    provider: Optional[str] = Field(default=None, max_length=32)
    base_url: Optional[str] = Field(default=None, max_length=512)
    api_key: Optional[str] = Field(default=None, max_length=512)
    model: Optional[str] = Field(default=None, max_length=128)


class ModelProfileDeleteResult(BaseModel):
    deleted_profile_id: str


class ConsistencyAuditReportRead(BaseModel):
    report_id: str
    project_id: int
    reason: str
    trigger_source: str
    status: str
    summary: dict = Field(default_factory=dict)
    items: list[dict] = Field(default_factory=list)
    generated_at: datetime
    generated_by: str
    stored_key: str


class ConsistencyAuditRunRequest(BaseModel):
    reason: str = Field(default="manual", max_length=64)
    run_mode: str = Field(default="async", pattern="^(async|sync)$")
    force: bool = False
    max_chapters: int | None = Field(default=None, ge=1, le=20)


class ConsistencyAuditRunResponse(BaseModel):
    project_id: int
    queued: bool
    run_mode: str
    reason: str
    trigger_source: str
    idempotency_key: str | None = None
    report: ConsistencyAuditReportRead | None = None


class GraphTimelineNodeRead(BaseModel):
    id: str
    label: str
    kind: str = "entity"
    degree: int = 0


class GraphTimelineEdgeRead(BaseModel):
    id: str
    source: str
    target: str
    relation: str
    confidence: float | None = None
    valid_from_chapter: int | None = None
    valid_to_chapter: int | None = None
    freshness_days: int | None = None


class GraphTimelineSnapshotRead(BaseModel):
    project_id: int
    chapter_index: int
    nodes: list[GraphTimelineNodeRead] = Field(default_factory=list)
    edges: list[GraphTimelineEdgeRead] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)


class StoryCardRead(BaseModel):
    id: int
    project_id: int
    title: str
    content: dict
    aliases: list[str]
    created_at: datetime
    updated_at: datetime


class PromptTemplateRead(BaseModel):
    id: int
    project_id: int
    name: str
    system_prompt: str
    user_prompt_prefix: str
    knowledge_setting_keys: list[str]
    knowledge_card_ids: list[int]
    created_at: datetime
    updated_at: datetime


class PromptTemplateUpsertRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    system_prompt: str = Field(default="", max_length=40000)
    user_prompt_prefix: str = Field(default="", max_length=20000)
    knowledge_setting_keys: list[str] = Field(default_factory=list, max_length=200)
    knowledge_card_ids: list[int] = Field(default_factory=list, max_length=200)


class PromptTemplateRevisionRead(BaseModel):
    id: int
    template_id: int
    project_id: int
    version: int
    name: str
    system_prompt: str
    user_prompt_prefix: str
    knowledge_setting_keys: list[str]
    knowledge_card_ids: list[int]
    operator_id: str
    source: str
    created_at: datetime


class PromptTemplateRollbackRequest(BaseModel):
    target_version: int = Field(ge=1)


class VolumeMemoryConsolidationRequest(BaseModel):
    force: bool = False


class VolumeMemoryConsolidationResponse(BaseModel):
    project_id: int
    volume_id: int
    volume_index: int
    chapters_count: int
    stored_key: str
    fact_count: int
    source: str


class ProjectChapterRead(BaseModel):
    id: int
    project_id: int
    volume_id: int | None = None
    chapter_index: int
    title: str
    content: str
    version: int
    created_at: datetime
    updated_at: datetime


class ProjectChapterCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    volume_id: int | None = Field(default=None, ge=1)


class ProjectChapterSaveRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(default="", max_length=1000000)
    volume_id: int | None = Field(default=None, ge=1)
    expected_version: int | None = Field(default=None, ge=1)


class ProjectChapterRevisionRead(BaseModel):
    id: int
    chapter_id: int
    project_id: int
    version: int
    title: str
    content: str
    operator_id: str
    source: str
    semantic_summary: list[str] = Field(default_factory=list)
    created_at: datetime


class ProjectVolumeRead(BaseModel):
    id: int
    project_id: int
    volume_index: int
    title: str
    outline: str
    created_at: datetime
    updated_at: datetime


class ProjectVolumeCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    outline: str = Field(default="", max_length=200000)


class ProjectVolumeUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    outline: str = Field(default="", max_length=200000)


class ProjectVolumeDeleteResult(BaseModel):
    deleted_volume_id: int
    fallback_volume_id: int


class SceneBeatRead(BaseModel):
    id: int
    project_id: int
    chapter_id: int
    beat_index: int
    content: str
    status: str
    created_at: datetime
    updated_at: datetime


class SceneBeatCreateRequest(BaseModel):
    content: str = Field(default="", max_length=20000)
    status: str = Field(default="pending", pattern="^(pending|done)$")


class SceneBeatUpdateRequest(BaseModel):
    content: str = Field(default="", max_length=20000)
    status: str = Field(default="pending", pattern="^(pending|done)$")


class SceneBeatDeleteResult(BaseModel):
    deleted_beat_id: int


class ForeshadowingCardRead(BaseModel):
    id: int
    project_id: int
    title: str
    description: str
    status: str
    planted_in_chapter_id: int | None = None
    resolved_in_chapter_id: int | None = None
    source_action_id: int | None = None
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None = None


class ForeshadowingCardCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=50000)
    planted_in_chapter_id: int | None = Field(default=None, ge=1)
    source_action_id: int | None = Field(default=None, ge=1)


class ForeshadowingCardUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=50000)
    status: str = Field(default="open", pattern="^(open|resolved)$")
    planted_in_chapter_id: int | None = Field(default=None, ge=1)
    resolved_in_chapter_id: int | None = Field(default=None, ge=1)


class ForeshadowingCardDeleteResult(BaseModel):
    deleted_foreshadow_id: int


class ProjectChapterRollbackRequest(BaseModel):
    target_version: int = Field(ge=1)


class ProjectChapterMoveRequest(BaseModel):
    direction: str = Field(default="up", pattern="^(up|down)$")


class ProjectChapterDeleteRequest(BaseModel):
    pass


class ProjectChapterDeleteResult(BaseModel):
    deleted_chapter_id: int
    active_chapter_id: int | None = None


class ProjectChapterReorderRequest(BaseModel):
    ordered_ids: list[int] = Field(min_length=1, max_length=1000)


class ActionAuditLogRead(BaseModel):
    id: int
    action_id: int
    event_type: str
    event_payload: dict
    operator_id: str
    created_at: datetime


class IndexLifecycleDeadLetterRead(BaseModel):
    project_id: int
    operator_id: str
    reason: str
    action_id: int
    mutation_id: str
    expected_version: int = 0
    idempotency_key: str = ""
    lifecycle_slot: str = "default"
    attempt: int = 0
    queued_at: int | None = None
    dead_letter_at: int | None = None
    error: str = ""


class LightRAGInsertTextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200000)
    file_source: str | None = Field(default=None, max_length=512)


class LightRAGListDocumentsRequest(BaseModel):
    page: int = Field(default=1, ge=1, le=100000)
    page_size: int = Field(default=50, ge=1, le=500)
    status_filter: str | None = Field(default=None, max_length=64)
    sort_field: str = Field(default="updated_at", max_length=64)
    sort_direction: str = Field(default="desc", max_length=8)


class LightRAGDeleteDocumentsRequest(BaseModel):
    doc_ids: list[str] = Field(default_factory=list, min_length=1, max_length=200)
    delete_file: bool = False
    delete_llm_cache: bool = False


class IndexLifecycleReplayRequest(BaseModel):
    project_id: int | None = None
    limit: int = Field(default=20, ge=1, le=200)


class IndexLifecycleReplayResult(BaseModel):
    requested: int
    project_id: int | None = None
    replayed: int
    requeue_failed: int
    skipped_invalid: int
    replay_request_id: str


class EntityMergeScanRequest(BaseModel):
    run_mode: str = Field(default="sync", pattern="^(sync|async)$")
    max_proposals: int = Field(default=3, ge=1, le=12)


class EntityMergeScanResult(BaseModel):
    project_id: int
    run_mode: str
    queued: bool = False
    result: dict = Field(default_factory=dict)


class GraphCandidateFactRead(BaseModel):
    fact_key: str
    source_entity: str
    relation: str
    target_entity: str
    confidence: float | None = None
    source_ref: str = ""
    origin: str = ""
    evidence: str = ""
    state: str = "candidate"
    valid_from_chapter: int | None = None
    valid_to_chapter: int | None = None
    updated_at: str | None = None


class GraphCandidateListResponse(BaseModel):
    project_id: int
    page: int
    page_size: int
    total: int
    items: list[GraphCandidateFactRead] = Field(default_factory=list)


class GraphCandidateBatchReviewRequest(BaseModel):
    decision: str = Field(default="confirm", pattern="^(confirm|reject)$")
    fact_keys: list[str] = Field(default_factory=list, min_length=1, max_length=1000)
    manual_confirmed: bool = False
    chapter_index: int | None = Field(default=None, ge=1)


class GraphCandidateBatchReviewResponse(BaseModel):
    project_id: int
    decision: str
    requested_count: int
    reviewed_count: int
    fact_keys: list[str] = Field(default_factory=list)
