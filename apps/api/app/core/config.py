import os

from app.core.settings import CoreSettings, PolicySettings, RuntimeSettings

def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _stringify_env_default(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


PROFILE_ALIASES = {
    "dev": "local-dev",
    "local": "local-dev",
    "default": "local-dev",
    "quality": "quality-first",
    "quality_first": "quality-first",
}


STRATEGY_DEFAULTS = {
    "local-dev": {
        "RAG_ROUTE_POLICY": "mix",
        "QUALITY_GATE_ENFORCE": False,
        "CITATION_POLICY": "off",
        "CITATION_MIN_COUNT": 1,
        "CONTEXT_WINDOW_PROFILE": "balanced",
        "CONTEXT_COMPRESSION_MODE": "rerank",
    },
    "quality-first": {
        "RAG_ROUTE_POLICY": "graph_first",
        "QUALITY_GATE_ENFORCE": True,
        "CITATION_POLICY": "inline",
        "CITATION_MIN_COUNT": 2,
        "CONTEXT_WINDOW_PROFILE": "quality",
        "CONTEXT_COMPRESSION_MODE": "task_aware",
    },
}


IMPLEMENTATION_DEFAULTS = {
    "local-dev": {
        "LLM_TIMEOUT_SECONDS": 45,
        "LLM_MAX_OUTPUT_TOKENS": 1600,
        "CONTEXT_CACHE_TTL_SECONDS": 1800,
        "CONTEXT_CACHE_MAX_ENTRIES": 512,
        "LIGHTRAG_TIMEOUT_SECONDS": 8,
        "RETRIEVAL_GRAPH_TIMEOUT_SECONDS": 1.8,
        "RETRIEVAL_RAG_TIMEOUT_SECONDS": 1.8,
        "RETRIEVAL_CACHE_TTL_SECONDS": 5,
        "RETRIEVAL_CACHE_MAX_ENTRIES": 512,
        "CONTEXT_COMPRESSION_MIN_CHARS": 1800,
        "CONTEXT_COMPRESSION_MAX_CHARS": 1000,
        "CONTEXT_COMPRESSION_LLM_TIMEOUT_SECONDS": 1.8,
        "CONTEXT_COMPRESSION_RERANKER_TOP_K": 12,
        "CONTEXT_COMPRESSION_RERANKER_MIN_SCORE": 0.5,
        "CONTEXT_COMPRESSION_RERANKER_BATCH_SIZE": 8,
        "GRAPH_SYNC_MAX_RETRIES": 2,
        "GRAPH_SYNC_RETRY_DELAY_SECONDS": 2,
        "INDEX_LIFECYCLE_MAX_RETRIES": 2,
        "INDEX_LIFECYCLE_RETRY_DELAY_SECONDS": 2,
        "ENTITY_MERGE_SCAN_NODE_LIMIT": 300,
        "ENTITY_MERGE_SCAN_MAX_PROPOSALS": 2,
        "CONSISTENCY_AUDIT_MAX_CHAPTERS": 2,
        "CONSISTENCY_AUDIT_MAX_ITEMS": 8,
        "CONSISTENCY_AUDIT_LLM_TIMEOUT_SECONDS": 2.2,
        "JOB_QUEUE_POLL_INTERVAL_SECONDS": 0.3,
        "JOB_PROCESSING_TIMEOUT_SECONDS": 90,
    },
    "quality-first": {
        "LLM_TIMEOUT_SECONDS": 90,
        "LLM_MAX_OUTPUT_TOKENS": 2400,
        "CONTEXT_CACHE_TTL_SECONDS": 5400,
        "CONTEXT_CACHE_MAX_ENTRIES": 2048,
        "LIGHTRAG_TIMEOUT_SECONDS": 12,
        "RETRIEVAL_GRAPH_TIMEOUT_SECONDS": 2.8,
        "RETRIEVAL_RAG_TIMEOUT_SECONDS": 2.8,
        "RETRIEVAL_CACHE_TTL_SECONDS": 10,
        "RETRIEVAL_CACHE_MAX_ENTRIES": 2048,
        "CONTEXT_COMPRESSION_MIN_CHARS": 2800,
        "CONTEXT_COMPRESSION_MAX_CHARS": 1500,
        "CONTEXT_COMPRESSION_LLM_TIMEOUT_SECONDS": 3.2,
        "CONTEXT_COMPRESSION_RERANKER_TOP_K": 24,
        "CONTEXT_COMPRESSION_RERANKER_MIN_SCORE": 0.58,
        "CONTEXT_COMPRESSION_RERANKER_BATCH_SIZE": 24,
        "GRAPH_SYNC_MAX_RETRIES": 4,
        "GRAPH_SYNC_RETRY_DELAY_SECONDS": 3,
        "INDEX_LIFECYCLE_MAX_RETRIES": 4,
        "INDEX_LIFECYCLE_RETRY_DELAY_SECONDS": 3,
        "ENTITY_MERGE_SCAN_NODE_LIMIT": 600,
        "ENTITY_MERGE_SCAN_MAX_PROPOSALS": 5,
        "CONSISTENCY_AUDIT_MAX_CHAPTERS": 5,
        "CONSISTENCY_AUDIT_MAX_ITEMS": 20,
        "CONSISTENCY_AUDIT_LLM_TIMEOUT_SECONDS": 4,
        "JOB_QUEUE_POLL_INTERVAL_SECONDS": 0.2,
        "JOB_PROCESSING_TIMEOUT_SECONDS": 180,
    },
}


def _resolve_profile(raw_profile: str) -> str:
    candidate = PROFILE_ALIASES.get(raw_profile, raw_profile)
    if candidate in STRATEGY_DEFAULTS:
        return candidate
    return "local-dev"


def _apply_profile_defaults() -> str:
    profile = os.getenv("CONFIG_PROFILE", "local-dev").strip().lower()
    resolved_profile = _resolve_profile(profile)
    os.environ.setdefault("CONFIG_PROFILE", resolved_profile)

    merged_defaults = {
        **STRATEGY_DEFAULTS[resolved_profile],
        **IMPLEMENTATION_DEFAULTS[resolved_profile],
    }
    for key, value in merged_defaults.items():
        os.environ.setdefault(key, _stringify_env_default(value))

    return resolved_profile


_ACTIVE_CONFIG_PROFILE = _apply_profile_defaults()


class Settings:
    def __init__(self) -> None:
        self.config_profile = _ACTIVE_CONFIG_PROFILE
        self.core = CoreSettings(self)
        self.policy = PolicySettings(self)
        self.runtime = RuntimeSettings(self)

    api_prefix = "/api"
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/novel_platform",
    )
    auth_enabled = _parse_bool(os.getenv("AUTH_ENABLED"), True)
    auth_tokens = os.getenv("AUTH_TOKENS", "")
    auth_token = os.getenv("AUTH_TOKEN", "")
    auth_user = os.getenv("AUTH_USER", "local-user")
    auth_project_owners = os.getenv("AUTH_PROJECT_OWNERS", "")
    auth_disabled_user = os.getenv("AUTH_DISABLED_USER", "local-user")
    llm_provider = os.getenv("LLM_PROVIDER", "stub")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_api_key = os.getenv("LLM_API_KEY", "")
    llm_timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    llm_max_output_tokens = max(int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1800")), 128)
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    llm_temperature_chat = _parse_float(os.getenv("LLM_TEMPERATURE_CHAT"), llm_temperature)
    llm_temperature_action = _parse_float(os.getenv("LLM_TEMPERATURE_ACTION"), 0.0)
    llm_temperature_ghost = _parse_float(os.getenv("LLM_TEMPERATURE_GHOST"), 0.7)
    llm_temperature_brainstorm = _parse_float(os.getenv("LLM_TEMPERATURE_BRAINSTORM"), 0.95)
    context_cache_enabled = _parse_bool(os.getenv("CONTEXT_CACHE_ENABLED"), True)
    context_cache_ttl_seconds = max(int(os.getenv("CONTEXT_CACHE_TTL_SECONDS", "3600")), 60)
    context_cache_max_entries = max(int(os.getenv("CONTEXT_CACHE_MAX_ENTRIES", "1024")), 32)
    openai_prompt_cache_key_enabled = _parse_bool(os.getenv("OPENAI_PROMPT_CACHE_KEY_ENABLED"), True)
    openai_prompt_cache_key_salt = os.getenv("OPENAI_PROMPT_CACHE_KEY_SALT", "")
    anthropic_enabled = _parse_bool(os.getenv("ANTHROPIC_ENABLED"), False)
    anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-4-sonnet")
    anthropic_version = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
    anthropic_prompt_caching_beta = os.getenv("ANTHROPIC_PROMPT_CACHING_BETA", "prompt-caching-2024-07-31")
    gemini_enabled = _parse_bool(os.getenv("GEMINI_ENABLED"), False)
    gemini_base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    gemini_cache_enabled = _parse_bool(os.getenv("GEMINI_CACHE_ENABLED"), True)
    gemini_cache_ttl_seconds = max(int(os.getenv("GEMINI_CACHE_TTL_SECONDS", "3600")), 60)
    deepseek_enabled = _parse_bool(os.getenv("DEEPSEEK_ENABLED"), False)
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    langfuse_enabled = _parse_bool(os.getenv("LANGFUSE_ENABLED"), False)
    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")

    lightrag_enabled = _parse_bool(os.getenv("LIGHTRAG_ENABLED"), False)
    lightrag_base_url = os.getenv("LIGHTRAG_BASE_URL", "")
    lightrag_llm_model = os.getenv("LIGHTRAG_LLM_MODEL", "")
    lightrag_llm_base_url = os.getenv("LIGHTRAG_LLM_BASE_URL", "")
    lightrag_llm_api_key = os.getenv("LIGHTRAG_LLM_API_KEY", "")
    lightrag_query_path = os.getenv("LIGHTRAG_QUERY_PATH", "/query/data")
    lightrag_query_mode = os.getenv("LIGHTRAG_QUERY_MODE", "mix")
    lightrag_extract_path = os.getenv("LIGHTRAG_EXTRACT_PATH", "/extract_graph")
    lightrag_graph_query_mode = os.getenv("LIGHTRAG_GRAPH_QUERY_MODE", "global")
    lightrag_graph_from_query_enabled = _parse_bool(
        os.getenv("LIGHTRAG_GRAPH_FROM_QUERY_ENABLED"),
        True,
    )
    lightrag_rerank_enabled = _parse_bool(os.getenv("LIGHTRAG_RERANK_ENABLED"), False)
    lightrag_rerank_model = os.getenv("LIGHTRAG_RERANK_MODEL", "")
    lightrag_api_key = os.getenv("LIGHTRAG_API_KEY", "")
    lightrag_timeout_seconds = int(os.getenv("LIGHTRAG_TIMEOUT_SECONDS", "8"))
    lightrag_documents_text_path = os.getenv("LIGHTRAG_DOCUMENTS_TEXT_PATH", "/documents/text")
    lightrag_documents_texts_path = os.getenv("LIGHTRAG_DOCUMENTS_TEXTS_PATH", "/documents/texts")
    lightrag_documents_paginated_path = os.getenv(
        "LIGHTRAG_DOCUMENTS_PAGINATED_PATH",
        "/documents/paginated",
    )
    lightrag_documents_delete_path = os.getenv(
        "LIGHTRAG_DOCUMENTS_DELETE_PATH",
        "/documents/delete_document",
    )
    lightrag_documents_pipeline_status_path = os.getenv(
        "LIGHTRAG_DOCUMENTS_PIPELINE_STATUS_PATH",
        "/documents/pipeline_status",
    )
    rag_route_policy = os.getenv("RAG_ROUTE_POLICY", "mix")
    citation_policy = os.getenv("CITATION_POLICY", "off")
    citation_min_count = int(os.getenv("CITATION_MIN_COUNT", "1"))
    citation_block_actions = _parse_bool(os.getenv("CITATION_BLOCK_ACTIONS"), False)
    quality_gate_enforce = _parse_bool(os.getenv("QUALITY_GATE_ENFORCE"), False)
    reranker_required = _parse_bool(os.getenv("RERANKER_REQUIRED"), False)
    context_pack_enabled = _parse_bool(os.getenv("CONTEXT_PACK_ENABLED"), True)
    context_pack_ttl_seconds = int(os.getenv("CONTEXT_PACK_TTL_SECONDS", "45"))
    context_pack_max_settings = int(os.getenv("CONTEXT_PACK_MAX_SETTINGS", "256"))
    context_pack_max_cards = int(os.getenv("CONTEXT_PACK_MAX_CARDS", "256"))
    context_window_profile = os.getenv("CONTEXT_WINDOW_PROFILE", "balanced")
    retrieval_parallel_enabled = _parse_bool(os.getenv("RETRIEVAL_PARALLEL_ENABLED"), True)
    retrieval_graph_timeout_seconds = _parse_float(os.getenv("RETRIEVAL_GRAPH_TIMEOUT_SECONDS"), 2.0)
    retrieval_rag_timeout_seconds = _parse_float(os.getenv("RETRIEVAL_RAG_TIMEOUT_SECONDS"), 2.0)
    retrieval_cache_ttl_seconds = _parse_float(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS"), 6.0)
    retrieval_cache_max_entries = int(os.getenv("RETRIEVAL_CACHE_MAX_ENTRIES", "1024"))
    deterministic_short_circuit_enabled = _parse_bool(
        os.getenv("DETERMINISTIC_SHORT_CIRCUIT_ENABLED"),
        True,
    )
    deterministic_min_dsl_hits = int(os.getenv("DETERMINISTIC_MIN_DSL_HITS", "1"))
    deterministic_min_graph_hits = int(os.getenv("DETERMINISTIC_MIN_GRAPH_HITS", "1"))
    semantic_router_enabled = _parse_bool(os.getenv("SEMANTIC_ROUTER_ENABLED"), True)
    semantic_router_mode = os.getenv("SEMANTIC_ROUTER_MODE", "auto")
    semantic_router_llm_timeout_seconds = _parse_float(os.getenv("SEMANTIC_ROUTER_LLM_TIMEOUT_SECONDS"), 1.2)
    semantic_router_low_confidence_threshold = _parse_float(
        os.getenv("SEMANTIC_ROUTER_LOW_CONFIDENCE_THRESHOLD"),
        0.55,
    )
    context_compression_enabled = _parse_bool(os.getenv("CONTEXT_COMPRESSION_ENABLED"), True)
    context_compression_mode = os.getenv("CONTEXT_COMPRESSION_MODE", "rerank")
    context_compression_min_chars = max(int(os.getenv("CONTEXT_COMPRESSION_MIN_CHARS", "2200")), 200)
    context_compression_max_chars = max(int(os.getenv("CONTEXT_COMPRESSION_MAX_CHARS", "1200")), 200)
    context_compression_reranker_enabled = _parse_bool(
        os.getenv("CONTEXT_COMPRESSION_RERANKER_ENABLED"),
        True,
    )
    context_compression_reranker_model = os.getenv(
        "CONTEXT_COMPRESSION_RERANKER_MODEL",
        "BAAI/bge-reranker-v2-minicpm-layerwise",
    )
    context_compression_reranker_runtime = os.getenv(
        "CONTEXT_COMPRESSION_RERANKER_RUNTIME",
        "onnx",
    )
    context_compression_reranker_onnx_path = os.getenv(
        "CONTEXT_COMPRESSION_RERANKER_ONNX_PATH",
        "./models/bge-reranker-v2-minicpm-layerwise-int8-onnx",
    )
    context_compression_reranker_onnx_provider = os.getenv(
        "CONTEXT_COMPRESSION_RERANKER_ONNX_PROVIDER",
        "CPUExecutionProvider",
    )
    context_compression_reranker_device = os.getenv(
        "CONTEXT_COMPRESSION_RERANKER_DEVICE",
        "cpu",
    )
    context_compression_reranker_top_k = max(
        int(os.getenv("CONTEXT_COMPRESSION_RERANKER_TOP_K", "18")),
        1,
    )
    context_compression_reranker_min_score = max(
        min(_parse_float(os.getenv("CONTEXT_COMPRESSION_RERANKER_MIN_SCORE"), 0.52), 1.0),
        0.0,
    )
    context_compression_reranker_batch_size = max(
        int(os.getenv("CONTEXT_COMPRESSION_RERANKER_BATCH_SIZE", "16")),
        1,
    )
    context_compression_reranker_max_length = max(
        int(os.getenv("CONTEXT_COMPRESSION_RERANKER_MAX_LENGTH", "384")),
        64,
    )
    context_compression_reranker_min_dsl_keep = max(
        int(os.getenv("CONTEXT_COMPRESSION_RERANKER_MIN_DSL_KEEP", "2")),
        0,
    )
    context_compression_reranker_dsl_bias = _parse_float(
        os.getenv("CONTEXT_COMPRESSION_RERANKER_DSL_BIAS"),
        0.12,
    )
    context_compression_reranker_graph_bias = _parse_float(
        os.getenv("CONTEXT_COMPRESSION_RERANKER_GRAPH_BIAS"),
        0.06,
    )
    context_compression_reranker_rag_bias = _parse_float(
        os.getenv("CONTEXT_COMPRESSION_RERANKER_RAG_BIAS"),
        0.03,
    )
    context_compression_llm_timeout_seconds = _parse_float(
        os.getenv("CONTEXT_COMPRESSION_LLM_TIMEOUT_SECONDS"),
        2.2,
    )
    self_reflective_enabled = _parse_bool(os.getenv("SELF_REFLECTIVE_ENABLED"), True)
    self_reflective_mode = os.getenv("SELF_REFLECTIVE_MODE", "auto")
    self_reflective_brainstorm_trigger_enabled = _parse_bool(
        os.getenv("SELF_REFLECTIVE_BRAINSTORM_TRIGGER_ENABLED"),
        True,
    )
    self_reflective_low_confidence_trigger_enabled = _parse_bool(
        os.getenv("SELF_REFLECTIVE_LOW_CONFIDENCE_TRIGGER_ENABLED"),
        False,
    )
    self_reflective_low_confidence_threshold = _parse_float(
        os.getenv("SELF_REFLECTIVE_LOW_CONFIDENCE_THRESHOLD"),
        0.5,
    )
    self_reflective_max_rounds = max(int(os.getenv("SELF_REFLECTIVE_MAX_ROUNDS", "1")), 1)
    self_reflective_max_followup_queries = max(int(os.getenv("SELF_REFLECTIVE_MAX_FOLLOWUP_QUERIES", "1")), 1)
    self_reflective_llm_timeout_seconds = _parse_float(
        os.getenv("SELF_REFLECTIVE_LLM_TIMEOUT_SECONDS"),
        1.8,
    )
    consistency_audit_enabled = _parse_bool(os.getenv("CONSISTENCY_AUDIT_ENABLED"), True)
    consistency_audit_auto_enqueue = _parse_bool(os.getenv("CONSISTENCY_AUDIT_AUTO_ENQUEUE"), True)
    consistency_audit_queue_name = os.getenv("CONSISTENCY_AUDIT_QUEUE_NAME", "consistency_audit_jobs")
    consistency_audit_max_retries = int(os.getenv("CONSISTENCY_AUDIT_MAX_RETRIES", "2"))
    consistency_audit_retry_delay_seconds = int(os.getenv("CONSISTENCY_AUDIT_RETRY_DELAY_SECONDS", "15"))
    consistency_audit_worker_block_seconds = int(os.getenv("CONSISTENCY_AUDIT_WORKER_BLOCK_SECONDS", "2"))
    consistency_audit_scheduler_interval_seconds = _parse_float(
        os.getenv("CONSISTENCY_AUDIT_SCHEDULER_INTERVAL_SECONDS"),
        120.0,
    )
    consistency_audit_scheduler_project_scan_limit = max(
        int(os.getenv("CONSISTENCY_AUDIT_SCHEDULER_PROJECT_SCAN_LIMIT", "200")),
        1,
    )
    consistency_audit_idle_minutes = max(int(os.getenv("CONSISTENCY_AUDIT_IDLE_MINUTES", "30")), 1)
    consistency_audit_daily_hour_utc = min(max(int(os.getenv("CONSISTENCY_AUDIT_DAILY_HOUR_UTC", "2")), 0), 23)
    consistency_audit_max_chapters = max(int(os.getenv("CONSISTENCY_AUDIT_MAX_CHAPTERS", "3")), 1)
    consistency_audit_max_items = max(int(os.getenv("CONSISTENCY_AUDIT_MAX_ITEMS", "12")), 1)
    consistency_audit_foreshadow_gap = max(int(os.getenv("CONSISTENCY_AUDIT_FORESHADOW_GAP", "8")), 1)
    consistency_audit_chapter_preview_chars = max(
        int(os.getenv("CONSISTENCY_AUDIT_CHAPTER_PREVIEW_CHARS", "1400")),
        300,
    )
    consistency_audit_graph_facts_limit = max(int(os.getenv("CONSISTENCY_AUDIT_GRAPH_FACTS_LIMIT", "8")), 1)
    consistency_audit_llm_enabled = _parse_bool(os.getenv("CONSISTENCY_AUDIT_LLM_ENABLED"), True)
    consistency_audit_llm_timeout_seconds = _parse_float(
        os.getenv("CONSISTENCY_AUDIT_LLM_TIMEOUT_SECONDS"),
        2.8,
    )
    budget_router_enabled = _parse_bool(os.getenv("BUDGET_ROUTER_ENABLED"), True)
    budget_router_mode = os.getenv("BUDGET_ROUTER_MODE", "auto")
    budget_router_llm_timeout_seconds = _parse_float(os.getenv("BUDGET_ROUTER_LLM_TIMEOUT_SECONDS"), 1.2)
    budget_router_low_confidence_threshold = _parse_float(os.getenv("BUDGET_ROUTER_LOW_CONFIDENCE_THRESHOLD"), 0.55)
    graph_temporal_enabled = _parse_bool(os.getenv("GRAPH_TEMPORAL_ENABLED"), True)
    graph_temporal_filter_without_chapter = _parse_bool(os.getenv("GRAPH_TEMPORAL_FILTER_WITHOUT_CHAPTER"), True)
    memory_decay_half_life_days = max(int(os.getenv("MEMORY_DECAY_HALF_LIFE_DAYS", "45")), 1)
    memory_decay_floor = _parse_float(os.getenv("MEMORY_DECAY_FLOOR"), 0.25)
    memory_consolidation_enabled = _parse_bool(os.getenv("MEMORY_CONSOLIDATION_ENABLED"), True)
    memory_consolidation_max_facts = max(int(os.getenv("MEMORY_CONSOLIDATION_MAX_FACTS", "8")), 3)
    memory_consolidation_preview_chars = max(int(os.getenv("MEMORY_CONSOLIDATION_PREVIEW_CHARS", "1200")), 200)
    memory_consolidation_llm_timeout_seconds = _parse_float(
        os.getenv("MEMORY_CONSOLIDATION_LLM_TIMEOUT_SECONDS"),
        8.0,
    )
    memory_semantic_key_prefix = os.getenv("MEMORY_SEMANTIC_KEY_PREFIX", "memory.semantic.volume.")
    tot_enabled = _parse_bool(os.getenv("TOT_ENABLED"), True)
    tot_max_branches = max(int(os.getenv("TOT_MAX_BRANCHES", "3")), 1)
    tot_max_depth = max(int(os.getenv("TOT_MAX_DEPTH", "3")), 1)
    tot_timeout_seconds = _parse_float(os.getenv("TOT_TIMEOUT_SECONDS"), 6.0)
    tot_trigger_keywords = _parse_csv(
        os.getenv(
            "TOT_TRIGGER_KEYWORDS",
            "推演,下一步剧情,分支,如果,可能性,权谋,悬疑,伏笔",
        )
    )
    spatial_enabled = _parse_bool(os.getenv("SPATIAL_ENABLED"), True)
    spatial_max_hops = max(int(os.getenv("SPATIAL_MAX_HOPS", "6")), 1)
    spatial_penalty_lambda = _parse_float(os.getenv("SPATIAL_PENALTY_LAMBDA"), 0.5)
    spatial_relation_keys = _parse_csv(
        os.getenv(
            "SPATIAL_RELATION_KEYS",
            "belongs_to,parent,region,zone,location,located_in,隶属,所属,地点,区域",
        )
    )
    prompt_template_guard_enabled = _parse_bool(os.getenv("PROMPT_TEMPLATE_GUARD_ENABLED"), True)
    prompt_template_guard_mode = os.getenv("PROMPT_TEMPLATE_GUARD_MODE", "warn")
    prompt_template_guard_warn_score = _parse_float(os.getenv("PROMPT_TEMPLATE_GUARD_WARN_SCORE"), 0.45)
    prompt_template_guard_block_score = _parse_float(os.getenv("PROMPT_TEMPLATE_GUARD_BLOCK_SCORE"), 0.75)
    prompt_template_guard_max_risk_terms = max(int(os.getenv("PROMPT_TEMPLATE_GUARD_MAX_RISK_TERMS", "2")), 1)
    prompt_template_guard_terms = _parse_csv(
        os.getenv(
            "PROMPT_TEMPLATE_GUARD_TERMS",
            "ignore previous,忽略以上,覆盖系统,system prompt,越权,泄露,reveal policy,开发者消息",
        )
    )

    neo4j_enabled = _parse_bool(os.getenv("NEO4J_ENABLED"), False)
    neo4j_uri = os.getenv("NEO4J_URI", "")
    neo4j_username = os.getenv("NEO4J_USERNAME", "")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    graph_sync_async_enabled = _parse_bool(os.getenv("GRAPH_SYNC_ASYNC_ENABLED"), True)
    graph_sync_queue_name = os.getenv("GRAPH_SYNC_QUEUE_NAME", "graph_sync_jobs")
    graph_sync_max_retries = int(os.getenv("GRAPH_SYNC_MAX_RETRIES", "3"))
    graph_sync_retry_delay_seconds = int(os.getenv("GRAPH_SYNC_RETRY_DELAY_SECONDS", "2"))
    graph_sync_worker_block_seconds = int(os.getenv("GRAPH_SYNC_WORKER_BLOCK_SECONDS", "5"))
    entity_merge_scan_enabled = _parse_bool(os.getenv("ENTITY_MERGE_SCAN_ENABLED"), True)
    entity_merge_scan_auto_enqueue = _parse_bool(os.getenv("ENTITY_MERGE_SCAN_AUTO_ENQUEUE"), True)
    entity_merge_scan_queue_name = os.getenv("ENTITY_MERGE_SCAN_QUEUE_NAME", "entity_merge_scan_jobs")
    entity_merge_scan_max_retries = int(os.getenv("ENTITY_MERGE_SCAN_MAX_RETRIES", "2"))
    entity_merge_scan_retry_delay_seconds = int(os.getenv("ENTITY_MERGE_SCAN_RETRY_DELAY_SECONDS", "4"))
    entity_merge_scan_worker_block_seconds = int(os.getenv("ENTITY_MERGE_SCAN_WORKER_BLOCK_SECONDS", "2"))
    entity_merge_scan_enqueue_interval_seconds = int(os.getenv("ENTITY_MERGE_SCAN_ENQUEUE_INTERVAL_SECONDS", "180"))
    entity_merge_scan_node_limit = int(os.getenv("ENTITY_MERGE_SCAN_NODE_LIMIT", "400"))
    entity_merge_scan_max_proposals = int(os.getenv("ENTITY_MERGE_SCAN_MAX_PROPOSALS", "3"))
    entity_merge_scan_min_degree = int(os.getenv("ENTITY_MERGE_SCAN_MIN_DEGREE", "2"))
    entity_merge_scan_min_shared_neighbors = int(os.getenv("ENTITY_MERGE_SCAN_MIN_SHARED_NEIGHBORS", "3"))
    entity_merge_scan_min_jaccard = _parse_float(os.getenv("ENTITY_MERGE_SCAN_MIN_JACCARD"), 0.74)
    entity_merge_scan_min_relation_overlap = int(os.getenv("ENTITY_MERGE_SCAN_MIN_RELATION_OVERLAP", "1"))
    entity_merge_scan_min_name_similarity = _parse_float(os.getenv("ENTITY_MERGE_SCAN_MIN_NAME_SIMILARITY"), 0.28)
    graph_coref_preprocess_enabled = _parse_bool(os.getenv("GRAPH_COREF_PREPROCESS_ENABLED"), False)
    graph_coref_max_replacements = int(os.getenv("GRAPH_COREF_MAX_REPLACEMENTS", "12"))
    graph_coref_overlap_enabled = _parse_bool(os.getenv("GRAPH_COREF_OVERLAP_ENABLED"), False)
    graph_coref_chunk_size = int(os.getenv("GRAPH_COREF_CHUNK_SIZE", "420"))
    graph_coref_chunk_overlap = int(os.getenv("GRAPH_COREF_CHUNK_OVERLAP", "120"))
    graph_coref_max_chunks = int(os.getenv("GRAPH_COREF_MAX_CHUNKS", "6"))
    graph_coref_llm_enabled = _parse_bool(os.getenv("GRAPH_COREF_LLM_ENABLED"), False)
    graph_coref_llm_model = os.getenv("GRAPH_COREF_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
    graph_coref_llm_timeout_seconds = int(os.getenv("GRAPH_COREF_LLM_TIMEOUT_SECONDS", "12"))
    job_queue_poll_interval_seconds = _parse_float(os.getenv("JOB_QUEUE_POLL_INTERVAL_SECONDS"), 0.25)
    job_processing_timeout_seconds = int(os.getenv("JOB_PROCESSING_TIMEOUT_SECONDS", "120"))
    job_cleanup_enabled = _parse_bool(os.getenv("JOB_CLEANUP_ENABLED"), True)
    job_cleanup_retention_seconds = int(os.getenv("JOB_CLEANUP_RETENTION_SECONDS", str(7 * 86400)))
    job_cleanup_batch_size = int(os.getenv("JOB_CLEANUP_BATCH_SIZE", "200"))
    job_cleanup_interval_seconds = _parse_float(os.getenv("JOB_CLEANUP_INTERVAL_SECONDS"), 60.0)
    project_advisory_lock_enabled = _parse_bool(os.getenv("PROJECT_ADVISORY_LOCK_ENABLED"), True)
    index_lifecycle_enabled = _parse_bool(os.getenv("INDEX_LIFECYCLE_ENABLED"), True)
    index_lifecycle_queue_name = os.getenv("INDEX_LIFECYCLE_QUEUE_NAME", "index_lifecycle_jobs")
    index_lifecycle_dead_letter_queue_name = os.getenv(
        "INDEX_LIFECYCLE_DEAD_LETTER_QUEUE_NAME",
        "index_lifecycle_dead_letters",
    )
    index_lifecycle_max_retries = int(os.getenv("INDEX_LIFECYCLE_MAX_RETRIES", "3"))
    index_lifecycle_retry_delay_seconds = int(os.getenv("INDEX_LIFECYCLE_RETRY_DELAY_SECONDS", "2"))
    lightrag_rebuild_enabled = _parse_bool(os.getenv("LIGHTRAG_REBUILD_ENABLED"), False)
    lightrag_rebuild_path = os.getenv("LIGHTRAG_REBUILD_PATH", "/documents/rebuild")


settings = Settings()
