from __future__ import annotations

from typing import Any


class PolicySettings:
    """Proxy view for retrieval/policy behavior settings on root Settings."""

    FIELD_NAMES: tuple[str, ...] = (
        "rag_route_policy",
        "citation_policy",
        "citation_min_count",
        "citation_block_actions",
        "quality_gate_enforce",
        "reranker_required",
        "context_pack_enabled",
        "context_pack_ttl_seconds",
        "context_pack_max_settings",
        "context_pack_max_cards",
        "context_window_profile",
        "retrieval_parallel_enabled",
        "retrieval_graph_timeout_seconds",
        "retrieval_rag_timeout_seconds",
        "retrieval_cache_ttl_seconds",
        "retrieval_cache_max_entries",
        "deterministic_short_circuit_enabled",
        "deterministic_min_dsl_hits",
        "deterministic_min_graph_hits",
        "semantic_router_enabled",
        "semantic_router_mode",
        "semantic_router_llm_timeout_seconds",
        "semantic_router_low_confidence_threshold",
        "context_compression_enabled",
        "context_compression_mode",
        "context_compression_min_chars",
        "context_compression_max_chars",
        "context_compression_reranker_enabled",
        "context_compression_reranker_model",
        "context_compression_reranker_runtime",
        "context_compression_reranker_onnx_path",
        "context_compression_reranker_onnx_provider",
        "context_compression_reranker_device",
        "context_compression_reranker_top_k",
        "context_compression_reranker_min_score",
        "context_compression_reranker_batch_size",
        "context_compression_reranker_max_length",
        "context_compression_reranker_min_dsl_keep",
        "context_compression_reranker_dsl_bias",
        "context_compression_reranker_graph_bias",
        "context_compression_reranker_rag_bias",
        "context_compression_llm_timeout_seconds",
        "self_reflective_enabled",
        "self_reflective_mode",
        "self_reflective_brainstorm_trigger_enabled",
        "self_reflective_low_confidence_trigger_enabled",
        "self_reflective_low_confidence_threshold",
        "self_reflective_max_rounds",
        "self_reflective_max_followup_queries",
        "self_reflective_llm_timeout_seconds",
        "budget_router_enabled",
        "budget_router_mode",
        "budget_router_llm_timeout_seconds",
        "budget_router_low_confidence_threshold",
        "graph_temporal_enabled",
        "graph_temporal_filter_without_chapter",
        "memory_decay_half_life_days",
        "memory_decay_floor",
        "memory_consolidation_enabled",
        "memory_consolidation_max_facts",
        "memory_consolidation_preview_chars",
        "memory_consolidation_llm_timeout_seconds",
        "memory_semantic_key_prefix",
        "tot_enabled",
        "tot_max_branches",
        "tot_max_depth",
        "tot_timeout_seconds",
        "tot_trigger_keywords",
        "spatial_enabled",
        "spatial_max_hops",
        "spatial_penalty_lambda",
        "spatial_relation_keys",
        "prompt_template_guard_enabled",
        "prompt_template_guard_mode",
        "prompt_template_guard_warn_score",
        "prompt_template_guard_block_score",
        "prompt_template_guard_max_risk_terms",
        "prompt_template_guard_terms",
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
