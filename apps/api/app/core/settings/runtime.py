from __future__ import annotations

from typing import Any


class RuntimeSettings:
    """Proxy view for runtime/infrastructure and queue settings on root Settings."""

    FIELD_NAMES: tuple[str, ...] = (
        "lightrag_enabled",
        "lightrag_base_url",
        "lightrag_llm_model",
        "lightrag_llm_base_url",
        "lightrag_llm_api_key",
        "lightrag_query_path",
        "lightrag_query_mode",
        "lightrag_extract_path",
        "lightrag_graph_query_mode",
        "lightrag_graph_from_query_enabled",
        "lightrag_rerank_enabled",
        "lightrag_rerank_model",
        "lightrag_api_key",
        "lightrag_timeout_seconds",
        "lightrag_documents_text_path",
        "lightrag_documents_texts_path",
        "lightrag_documents_paginated_path",
        "lightrag_documents_delete_path",
        "lightrag_documents_pipeline_status_path",
        "neo4j_enabled",
        "neo4j_uri",
        "neo4j_username",
        "neo4j_password",
        "neo4j_database",
        "graph_sync_async_enabled",
        "graph_sync_queue_name",
        "graph_sync_max_retries",
        "graph_sync_retry_delay_seconds",
        "graph_sync_worker_block_seconds",
        "entity_merge_scan_enabled",
        "entity_merge_scan_auto_enqueue",
        "entity_merge_scan_queue_name",
        "entity_merge_scan_max_retries",
        "entity_merge_scan_retry_delay_seconds",
        "entity_merge_scan_worker_block_seconds",
        "entity_merge_scan_enqueue_interval_seconds",
        "entity_merge_scan_node_limit",
        "entity_merge_scan_max_proposals",
        "entity_merge_scan_min_degree",
        "entity_merge_scan_min_shared_neighbors",
        "entity_merge_scan_min_jaccard",
        "entity_merge_scan_min_relation_overlap",
        "entity_merge_scan_min_name_similarity",
        "graph_coref_preprocess_enabled",
        "graph_coref_max_replacements",
        "graph_coref_overlap_enabled",
        "graph_coref_chunk_size",
        "graph_coref_chunk_overlap",
        "graph_coref_max_chunks",
        "graph_coref_llm_enabled",
        "graph_coref_llm_model",
        "graph_coref_llm_timeout_seconds",
        "job_queue_poll_interval_seconds",
        "job_processing_timeout_seconds",
        "job_cleanup_enabled",
        "job_cleanup_retention_seconds",
        "job_cleanup_batch_size",
        "job_cleanup_interval_seconds",
        "project_advisory_lock_enabled",
        "index_lifecycle_enabled",
        "index_lifecycle_queue_name",
        "index_lifecycle_dead_letter_queue_name",
        "index_lifecycle_max_retries",
        "index_lifecycle_retry_delay_seconds",
        "lightrag_rebuild_enabled",
        "lightrag_rebuild_path",
        "consistency_audit_enabled",
        "consistency_audit_auto_enqueue",
        "consistency_audit_queue_name",
        "consistency_audit_max_retries",
        "consistency_audit_retry_delay_seconds",
        "consistency_audit_worker_block_seconds",
        "consistency_audit_scheduler_interval_seconds",
        "consistency_audit_scheduler_project_scan_limit",
        "consistency_audit_idle_minutes",
        "consistency_audit_daily_hour_utc",
        "consistency_audit_max_chapters",
        "consistency_audit_max_items",
        "consistency_audit_foreshadow_gap",
        "consistency_audit_chapter_preview_chars",
        "consistency_audit_graph_facts_limit",
        "consistency_audit_llm_enabled",
        "consistency_audit_llm_timeout_seconds",
    )

    def __init__(self, root: Any) -> None:
        object.__setattr__(self, "_root", root)

    def __getattr__(self, name: str) -> Any:
        if name in self.FIELD_NAMES:
            return getattr(self._root, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.FIELD_NAMES:
            setattr(self._root, name, value)
            return
        object.__setattr__(self, name, value)
