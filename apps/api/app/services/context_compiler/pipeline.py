import time
from concurrent.futures import Future
from typing import Any

from sqlmodel import Session

from app.core.config import settings
from app.services.chat_service import (
    get_project_chapter,
    get_project_volume,
    get_prompt_template,
    list_cards,
    list_scene_beats,
    list_messages,
    list_settings,
)
from app.services.context_compiler._types import CompiledContextBundle
from app.services.context_compiler._state import (
    _GRAPH_HITS_CACHE,
    _RAG_HITS_CACHE,
    _WEB_SEARCH_CACHE,
    _LOGGER,
)
from app.services.context_compiler._utils import (
    _truncate_text,
    _extract_query_terms,
    _safe_iso,
    _setting_value_text,
    _card_content_text,
    _setting_source_text,
    _card_source_text,
    _freshness_days,
)
from app.services.context_compiler.normalization import (
    _normalize_pov,
    _normalize_timeout,
    _normalize_self_reflective_mode,
    _normalize_temperature_profile,
    _normalize_followup_queries,
)
from app.services.context_compiler.routing import (
    _resolve_semantic_route,
    _resolve_dynamic_budget_plan,
)
from app.services.context_compiler.normalization import _resolve_context_window_policy
from app.services.context_compiler.caching import (
    _cache_ttl_seconds,
    _cache_get,
    _cache_set,
    _submit_graph_future,
    _submit_rag_future,
    _await_hits_future,
    _strict_graph_mode_enabled,
    _graph_cache_key,
    _rag_cache_key,
    _web_search_cache_key,
    _submit_web_search_future,
)
from app.services.context_compiler.circuit_breaker import (
    _circuit_breaker_should_short_circuit,
    _circuit_breaker_record_failure,
    _circuit_breaker_record_success,
    _circuit_breaker_snapshot,
)
from app.services.context_compiler.context_pack import (
    _load_context_pack,
    _normalize_reference_project_ids,
    _load_reference_context,
)
from app.services.context_compiler.quality_gate import (
    _build_quality_gate,
    _resolve_rag_short_circuit,
)
from app.services.context_compiler.retrieval import (
    _window_retrieval_rows,
    _apply_pov_filter,
    _build_dsl_hits,
    _build_graph_facts,
    _build_semantic_hits,
)
from app.services.context_compiler.compression import (
    _build_negative_constraints,
    _build_context_compression,
    _hit_preview_text,
)
from app.services.context_compiler.self_review import (
    _heuristic_self_reflective_review,
    _call_self_reflective_judge_llm,
    _merge_unique_hits,
    _run_reflective_followup_retrieval,
    _build_context_cache_layers,
)
from app.services.context_compiler.memory import (
    _apply_memory_decay,
    _apply_spatial_penalty,
    _split_memory_layers,
)
from app.services.context_compiler.normalization import _resolve_rag_route


def compile_context_bundle(
    db: Session,
    *,
    session_id: int | None,
    project_id: int,
    chapter_id: int | None,
    scene_beat_id: int | None,
    prompt_template_id: int | None,
    user_input: str,
    pov_mode: str | None,
    pov_anchor: str | None,
    rag_mode_override: str | None = None,
    deterministic_first: bool = False,
    thinking_enabled: bool = False,
    reference_project_ids: list[int] | None = None,
    context_window_profile: str | None = None,
    budget_mode: str | None = None,
    current_location: str | None = None,
    temperature_profile: str | None = None,
    web_search_enabled: bool = False,
) -> CompiledContextBundle:
    compile_started_at = time.perf_counter()
    mode, anchor, notes = _normalize_pov(pov_mode, pov_anchor)
    terms = _extract_query_terms(user_input)
    context_window, context_window_source = _resolve_context_window_policy(context_window_profile)

    chapter_preview_for_router = ""
    scene_beat_text_for_router = ""
    preloaded_chapter = get_project_chapter(db, project_id, int(chapter_id)) if chapter_id is not None else None
    if preloaded_chapter is not None:
        chapter_preview_for_router = str(getattr(preloaded_chapter, "content", "") or "")[:900]
        try:
            beats_for_router = list(
                list_scene_beats(
                    db,
                    project_id=project_id,
                    chapter_id=int(getattr(preloaded_chapter, "id", 0) or 0),
                )
            )
        except ValueError:
            beats_for_router = []
        if beats_for_router:
            selected = None
            if scene_beat_id is not None:
                selected = next(
                    (item for item in beats_for_router if int(getattr(item, "id", 0) or 0) == int(scene_beat_id)),
                    None,
                )
            if selected is None:
                selected = next((item for item in beats_for_router if str(getattr(item, "status", "")) == "pending"), None)
            if selected is None:
                selected = beats_for_router[0]
            scene_beat_text_for_router = str(getattr(selected, "content", "") or "")

    semantic_route = _resolve_semantic_route(
        user_input=user_input,
        chapter_preview=chapter_preview_for_router,
        scene_beat_text=scene_beat_text_for_router,
    )

    effective_rag_mode_override = rag_mode_override
    rag_override_source = "request_override" if rag_mode_override else ""
    if effective_rag_mode_override is None and semantic_route.rag_mode:
        effective_rag_mode_override = semantic_route.rag_mode
        rag_override_source = "semantic_router"

    rag_mode, rag_route_reason, rag_route_source = _resolve_rag_route(user_input, terms, effective_rag_mode_override)
    if rag_override_source == "semantic_router":
        rag_route_reason = f"semantic_router_{semantic_route.intent}"
        rag_route_source = "semantic_router"

    effective_budget_mode = budget_mode
    if effective_budget_mode is None and semantic_route.budget_mode:
        effective_budget_mode = semantic_route.budget_mode
        notes.append(
            f"semantic_router 命中 {semantic_route.intent}，budget_mode 自动路由为 {semantic_route.budget_mode}。"
        )

    dynamic_budget_plan = _resolve_dynamic_budget_plan(
        base_policy=context_window,
        user_input=user_input,
        request_budget_mode=effective_budget_mode,
        chapter_preview=chapter_preview_for_router,
        scene_beat_text=scene_beat_text_for_router,
    )

    recent_messages = (
        list_messages(db, session_id, limit=dynamic_budget_plan.recent_messages_limit) if session_id is not None else []
    )
    all_settings, all_cards, context_pack_meta = _load_context_pack(db, project_id)
    scoped_settings, scoped_cards = _apply_pov_filter(all_settings, all_cards, mode, anchor)
    normalized_reference_project_ids = _normalize_reference_project_ids(
        reference_project_ids,
        current_project_id=project_id,
    )
    referenced_settings, referenced_cards, reference_project_meta = _load_reference_context(
        db,
        normalized_reference_project_ids,
    )
    scoped_reference_settings, scoped_reference_cards = _apply_pov_filter(
        referenced_settings,
        referenced_cards,
        mode,
        anchor,
    )
    project_working_settings, project_semantic_settings = _split_memory_layers(scoped_settings)
    ref_working_settings, ref_semantic_settings = _split_memory_layers(scoped_reference_settings)
    retrieval_settings = [*project_working_settings, *ref_working_settings, *project_semantic_settings, *ref_semantic_settings]
    retrieval_cards = [*scoped_cards, *scoped_reference_cards]
    windowed_retrieval_settings = _window_retrieval_rows(
        retrieval_settings,
        limit=dynamic_budget_plan.retrieval_settings_limit,
        terms=terms,
        source_text_getter=_setting_source_text,
    )
    windowed_retrieval_cards = _window_retrieval_rows(
        retrieval_cards,
        limit=dynamic_budget_plan.retrieval_cards_limit,
        terms=terms,
        source_text_getter=_card_source_text,
    )
    context_rows_meta = {
        "recent_messages": len(recent_messages),
        "project_settings": len(all_settings),
        "project_cards": len(all_cards),
        "project_settings_scoped": len(scoped_settings),
        "project_cards_scoped": len(scoped_cards),
        "reference_settings_scoped": len(scoped_reference_settings),
        "reference_cards_scoped": len(scoped_reference_cards),
        "working_settings": len(project_working_settings) + len(ref_working_settings),
        "semantic_settings": len(project_semantic_settings) + len(ref_semantic_settings),
        "retrieval_settings": len(retrieval_settings),
        "retrieval_cards": len(retrieval_cards),
        "windowed_retrieval_settings": len(windowed_retrieval_settings),
        "windowed_retrieval_cards": len(windowed_retrieval_cards),
    }

    prompt_workshop_template: dict[str, Any] | None = None
    prompt_workshop_knowledge_settings: list[dict[str, Any]] = []
    prompt_workshop_knowledge_cards: list[dict[str, Any]] = []
    prompt_workshop_reason = "not_selected"
    if prompt_template_id is not None:
        prompt_template = get_prompt_template(db, project_id, int(prompt_template_id))
        if prompt_template is None:
            prompt_workshop_reason = "template_not_found"
            notes.append(f"prompt_template_id={prompt_template_id} 未命中，已忽略模板注入。")
        else:
            knowledge_setting_keys = {
                str(item).strip()
                for item in (getattr(prompt_template, "knowledge_setting_keys", []) or [])
                if str(item).strip()
            }
            knowledge_card_ids = {
                int(item)
                for item in (getattr(prompt_template, "knowledge_card_ids", []) or [])
                if isinstance(item, int) and int(item) > 0
            }
            if knowledge_setting_keys:
                for row in scoped_settings:
                    if row.key not in knowledge_setting_keys:
                        continue
                    prompt_workshop_knowledge_settings.append(
                        {
                            "id": row.id,
                            "key": row.key,
                            "value_preview": _truncate_text(_setting_value_text(row), 500),
                            "freshness_days": _freshness_days(row.updated_at),
                        }
                    )
            if knowledge_card_ids:
                for row in scoped_cards:
                    if int(row.id) not in knowledge_card_ids:
                        continue
                    prompt_workshop_knowledge_cards.append(
                        {
                            "id": row.id,
                            "title": row.title,
                            "content_preview": _truncate_text(_card_content_text(row), 500),
                            "freshness_days": _freshness_days(row.updated_at),
                        }
                    )
            prompt_workshop_template = {
                "id": int(getattr(prompt_template, "id")),
                "name": str(getattr(prompt_template, "name", "") or ""),
                "system_prompt": str(getattr(prompt_template, "system_prompt", "") or ""),
                "user_prompt_prefix": str(getattr(prompt_template, "user_prompt_prefix", "") or ""),
            }
            prompt_workshop_reason = "ok"

    prompt_workshop_meta: dict[str, Any] = {
        "enabled": prompt_workshop_template is not None,
        "reason": prompt_workshop_reason,
        "requested_template_id": prompt_template_id,
        "injected_settings": len(prompt_workshop_knowledge_settings),
        "injected_cards": len(prompt_workshop_knowledge_cards),
    }
    if prompt_workshop_template is not None:
        prompt_workshop_meta.update(
            {
                "template_id": prompt_workshop_template.get("id"),
                "template_name": prompt_workshop_template.get("name"),
            }
        )

    chapter_context_reason = "not_requested"
    current_chapter: dict[str, Any] | None = None
    current_chapter_row = None
    current_volume: dict[str, Any] | None = None
    scene_beat_context: dict[str, Any] | None = None
    outline_context_reason = "chapter_not_requested"
    if chapter_id is not None:
        chapter = preloaded_chapter if preloaded_chapter is not None else get_project_chapter(db, project_id, int(chapter_id))
        if chapter is None:
            chapter_context_reason = "chapter_not_found"
            notes.append(f"chapter_id={chapter_id} 未命中，已忽略章节上下文。")
        else:
            current_chapter_row = chapter
            chapter_content = str(getattr(chapter, "content", "") or "")
            current_chapter = {
                "id": int(getattr(chapter, "id")),
                "volume_id": int(getattr(chapter, "volume_id", 0) or 0) or None,
                "chapter_index": int(getattr(chapter, "chapter_index")),
                "title": str(getattr(chapter, "title", "") or ""),
                "version": int(getattr(chapter, "version", 0) or 0),
                "updated_at": _safe_iso(getattr(chapter, "updated_at", None)),
                "content": _truncate_text(chapter_content, context_window.chapter_content_chars),
                "content_preview": _truncate_text(chapter_content, context_window.chapter_preview_chars),
                "total_chars": len(chapter_content),
            }
            chapter_context_reason = "ok"

    chapter_context_meta: dict[str, Any] = {
        "enabled": current_chapter is not None,
        "reason": chapter_context_reason,
        "requested_chapter_id": chapter_id,
    }
    if current_chapter is not None:
        chapter_context_meta.update(
            {
                "chapter_id": current_chapter.get("id"),
                "chapter_index": current_chapter.get("chapter_index"),
                "chapter_version": current_chapter.get("version"),
                "updated_at": current_chapter.get("updated_at"),
                "total_chars": current_chapter.get("total_chars"),
            }
        )

    if current_chapter_row is not None:
        resolved_volume_id = int(getattr(current_chapter_row, "volume_id", 0) or 0)
        if resolved_volume_id > 0:
            volume_row = get_project_volume(db, project_id, resolved_volume_id)
            if volume_row is not None:
                current_volume = {
                    "id": int(getattr(volume_row, "id", 0) or 0),
                    "volume_index": int(getattr(volume_row, "volume_index", 0) or 0),
                    "title": str(getattr(volume_row, "title", "") or ""),
                    "outline": _truncate_text(str(getattr(volume_row, "outline", "") or ""), 1800),
                    "outline_preview": _truncate_text(str(getattr(volume_row, "outline", "") or ""), 420),
                    "updated_at": _safe_iso(getattr(volume_row, "updated_at", None)),
                }
                outline_context_reason = "ok"
            else:
                outline_context_reason = "volume_not_found"
                notes.append(f"volume_id={resolved_volume_id} 未命中，已忽略卷纲注入。")
        else:
            outline_context_reason = "volume_not_bound"

        try:
            beats = list(list_scene_beats(db, project_id=project_id, chapter_id=int(getattr(current_chapter_row, "id"))))
        except ValueError:
            beats = []

        active_beat_row = None
        if scene_beat_id is not None and beats:
            active_beat_row = next((item for item in beats if int(getattr(item, "id", 0) or 0) == int(scene_beat_id)), None)
            if active_beat_row is None:
                notes.append(f"scene_beat_id={scene_beat_id} 未命中，已退回章节默认 Beat。")
        if active_beat_row is None and beats:
            active_beat_row = next((item for item in beats if str(getattr(item, "status", "")) == "pending"), None)
        if active_beat_row is None and beats:
            active_beat_row = beats[0]

        if active_beat_row is not None:
            active_index = next(
                (idx for idx, item in enumerate(beats) if int(getattr(item, "id", 0) or 0) == int(getattr(active_beat_row, "id", 0) or 0)),
                0,
            )
            prev_row = beats[active_index - 1] if active_index > 0 else None
            next_row = beats[active_index + 1] if active_index + 1 < len(beats) else None
            scene_beat_context = {
                "active": {
                    "id": int(getattr(active_beat_row, "id", 0) or 0),
                    "beat_index": int(getattr(active_beat_row, "beat_index", 0) or 0),
                    "content": _truncate_text(str(getattr(active_beat_row, "content", "") or ""), 600),
                    "status": str(getattr(active_beat_row, "status", "") or "pending"),
                },
                "previous": (
                    {
                        "id": int(getattr(prev_row, "id", 0) or 0),
                        "beat_index": int(getattr(prev_row, "beat_index", 0) or 0),
                        "content": _truncate_text(str(getattr(prev_row, "content", "") or ""), 260),
                    }
                    if prev_row is not None
                    else None
                ),
                "next": (
                    {
                        "id": int(getattr(next_row, "id", 0) or 0),
                        "beat_index": int(getattr(next_row, "beat_index", 0) or 0),
                        "content": _truncate_text(str(getattr(next_row, "content", "") or ""), 260),
                    }
                    if next_row is not None
                    else None
                ),
                "total": len(beats),
            }
        elif beats:
            scene_beat_context = None
        else:
            outline_context_reason = "ok_no_beats" if outline_context_reason.startswith("ok") else outline_context_reason

    outline_context_meta: dict[str, Any] = {
        "enabled": bool(current_volume),
        "reason": outline_context_reason,
        "requested_scene_beat_id": scene_beat_id,
        "selected_scene_beat_id": (
            int(scene_beat_context["active"]["id"]) if scene_beat_context and isinstance(scene_beat_context.get("active"), dict) else None
        ),
    }

    dsl_hits = _build_dsl_hits(
        terms,
        windowed_retrieval_settings,
        windowed_retrieval_cards,
        limit=dynamic_budget_plan.dsl_limit,
    )
    dsl_hits = _apply_memory_decay(dsl_hits)[: max(dynamic_budget_plan.dsl_limit, 1)]
    graph_anchor = anchor if mode == "character" else None
    rag_anchor = anchor if mode == "character" else None
    current_chapter_index = int(current_chapter.get("chapter_index", 0) or 0) if isinstance(current_chapter, dict) else None
    graph_limit = max(dynamic_budget_plan.graph_limit, 1)
    rag_limit = max(dynamic_budget_plan.rag_limit, 1)
    parallel_enabled = bool(settings.retrieval_parallel_enabled)
    graph_timeout_seconds = _normalize_timeout(settings.retrieval_graph_timeout_seconds, 2.0)
    rag_timeout_seconds = _normalize_timeout(settings.retrieval_rag_timeout_seconds, 2.0)
    retrieval_cache_ttl = _cache_ttl_seconds()

    graph_cache_key = _graph_cache_key(
        project_id,
        terms,
        graph_anchor,
        graph_limit,
        current_chapter=current_chapter_index,
    )
    graph_hits_remote, graph_cache_status = _cache_get(_GRAPH_HITS_CACHE, graph_cache_key)
    graph_future: Future[list[dict[str, Any]]] | None = None
    graph_timed_out = False
    graph_failed = False
    graph_circuit_open = False
    graph_circuit_open_remaining = 0.0
    if graph_hits_remote is None:
        graph_circuit_open, graph_circuit_open_remaining = _circuit_breaker_should_short_circuit("graph")
        if graph_circuit_open:
            graph_cache_status = "circuit_open"
            if _strict_graph_mode_enabled():
                notes.append(
                    f"Neo4j circuit breaker 已开启（剩余约 {max(int(round(graph_circuit_open_remaining)), 1)}s），strict graph 模式下不再降级到本地图谱。"
                )
            else:
                notes.append(
                    f"Neo4j circuit breaker 已开启（剩余约 {max(int(round(graph_circuit_open_remaining)), 1)}s），已切换本地图谱回退。"
                )
        else:
            graph_future = _submit_graph_future(
                project_id,
                terms,
                graph_anchor,
                graph_limit,
                current_chapter=current_chapter_index,
            )

    rag_cache_key = _rag_cache_key(user_input, rag_anchor, rag_mode, rag_limit)
    semantic_hits_remote, rag_cache_status = _cache_get(_RAG_HITS_CACHE, rag_cache_key)
    rag_future: Future[list[dict[str, Any]]] | None = None
    rag_future_started_at: float | None = None
    rag_timed_out = False
    rag_failed = False
    rag_circuit_open = False
    rag_circuit_open_remaining = 0.0
    rag_circuit_note_added = False

    if semantic_hits_remote is None and parallel_enabled and not deterministic_first:
        rag_circuit_open, rag_circuit_open_remaining = _circuit_breaker_should_short_circuit("rag")
        if rag_circuit_open:
            rag_cache_status = "circuit_open"
            notes.append(
                f"LightRAG circuit breaker 已开启（剩余约 {max(int(round(rag_circuit_open_remaining)), 1)}s），已切换本地语义回退。"
            )
            rag_circuit_note_added = True
        else:
            rag_future_started_at = time.monotonic()
            rag_future = _submit_rag_future(user_input, rag_anchor, rag_limit, rag_mode)

    # --- Web Search Layer ---
    web_search_hits: list[dict[str, Any]] = []
    web_search_provider = "disabled"
    web_search_cache_status = "disabled"
    web_limit = max(int(settings.exa_default_num_results), 1)
    web_future: Future[list[dict[str, Any]]] | None = None

    if web_search_enabled and settings.exa_enabled and settings.exa_api_key:
        web_cache_key = _web_search_cache_key(user_input, web_limit)
        web_hits_cached, web_search_cache_status = _cache_get(_WEB_SEARCH_CACHE, web_cache_key)
        if web_hits_cached is not None:
            web_search_hits = web_hits_cached
            web_search_provider = "exa_cache"
        else:
            web_circuit_open, _ = _circuit_breaker_should_short_circuit("web_search")
            if not web_circuit_open:
                web_future = _submit_web_search_future(user_input, web_limit)
            else:
                web_search_cache_status = "circuit_open"
                web_search_provider = "exa_circuit_open"

    if graph_hits_remote is None and graph_future is not None:
        graph_hits_remote, graph_timed_out, graph_failed = _await_hits_future(
            graph_future,
            graph_timeout_seconds,
        )
        if graph_timed_out or graph_failed:
            _circuit_breaker_record_failure("graph")
        else:
            _circuit_breaker_record_success("graph")
        if graph_hits_remote:
            _cache_set(_GRAPH_HITS_CACHE, graph_cache_key, graph_hits_remote)
            graph_cache_status = "set"
        elif graph_timed_out:
            graph_cache_status = "timeout"
        elif graph_failed:
            graph_cache_status = "error"
        else:
            graph_cache_status = "empty"

    if graph_hits_remote:
        graph_facts = _apply_memory_decay(graph_hits_remote)[:graph_limit]
        graph_provider = "neo4j_cache" if graph_cache_status == "hit" else "neo4j"
    else:
        if _strict_graph_mode_enabled():
            graph_facts = []
            if graph_circuit_open:
                graph_provider = "neo4j_circuit_open_strict_no_fallback"
            elif graph_timed_out:
                graph_provider = "neo4j_timeout_strict_no_fallback"
            elif graph_failed:
                graph_provider = "neo4j_error_strict_no_fallback"
            else:
                graph_provider = "neo4j_strict_no_fallback"
        else:
            graph_facts = _apply_memory_decay(
                _build_graph_facts(
                    windowed_retrieval_cards,
                    windowed_retrieval_settings,
                    graph_anchor,
                    limit=graph_limit,
                )
            )[:graph_limit]
            if graph_circuit_open:
                graph_provider = "neo4j_circuit_open_local_graph_fallback"
            elif graph_timed_out:
                graph_provider = "neo4j_timeout_local_graph_fallback"
            elif graph_failed:
                graph_provider = "neo4j_error_local_graph_fallback"
            else:
                graph_provider = "local_graph_fallback"

    rag_short_circuit_enabled, rag_short_circuit_reason = _resolve_rag_short_circuit(
        deterministic_first=deterministic_first,
        dsl_hits=dsl_hits,
        graph_facts=graph_facts,
    )
    if rag_short_circuit_enabled:
        semantic_hits = []
        rag_provider = "skipped_by_deterministic_short_circuit"
        if rag_future is not None:
            rag_future.cancel()
    else:
        if semantic_hits_remote is None:
            if rag_future is None:
                rag_circuit_open, rag_circuit_open_remaining = _circuit_breaker_should_short_circuit("rag")
                if rag_circuit_open:
                    rag_cache_status = "circuit_open"
                    if not rag_circuit_note_added:
                        notes.append(
                            f"LightRAG circuit breaker 已开启（剩余约 {max(int(round(rag_circuit_open_remaining)), 1)}s），已切换本地语义回退。"
                        )
                        rag_circuit_note_added = True
                else:
                    rag_future_started_at = time.monotonic()
                    rag_future = _submit_rag_future(user_input, rag_anchor, rag_limit, rag_mode)
            if rag_future is not None:
                elapsed = (time.monotonic() - rag_future_started_at) if rag_future_started_at is not None else 0.0
                wait_timeout = max(rag_timeout_seconds - elapsed, 0.05)
                semantic_hits_remote, rag_timed_out, rag_failed = _await_hits_future(rag_future, wait_timeout)
                if rag_timed_out or rag_failed:
                    _circuit_breaker_record_failure("rag")
                else:
                    _circuit_breaker_record_success("rag")
                if semantic_hits_remote:
                    _cache_set(_RAG_HITS_CACHE, rag_cache_key, semantic_hits_remote)
                    rag_cache_status = "set"
                elif rag_timed_out:
                    rag_cache_status = "timeout"
                elif rag_failed:
                    rag_cache_status = "error"
                else:
                    rag_cache_status = "empty"
        if semantic_hits_remote:
            semantic_hits = _apply_memory_decay(semantic_hits_remote)[:rag_limit]
            rag_provider = "lightrag_cache" if rag_cache_status == "hit" else "lightrag"
        else:
            semantic_hits = _apply_memory_decay(
                _build_semantic_hits(
                    user_input,
                    windowed_retrieval_settings,
                    windowed_retrieval_cards,
                    rag_anchor,
                    limit=rag_limit,
                )
            )[:rag_limit]
            if rag_circuit_open:
                rag_provider = "lightrag_circuit_open_local_semantic_fallback"
            elif rag_timed_out:
                rag_provider = "lightrag_timeout_local_semantic_fallback"
            elif rag_failed:
                rag_provider = "lightrag_error_local_semantic_fallback"
            else:
                rag_provider = "local_semantic_fallback"

    # --- Await Web Search ---
    if web_future is not None:
        web_hits_remote, web_timed_out, web_failed = _await_hits_future(
            web_future, _normalize_timeout(settings.exa_timeout_seconds, 3.0)
        )
        if web_timed_out or web_failed:
            _circuit_breaker_record_failure("web_search")
            web_search_provider = "exa_timeout" if web_timed_out else "exa_error"
        else:
            _circuit_breaker_record_success("web_search")
            if web_hits_remote:
                _cache_set(_WEB_SEARCH_CACHE, web_cache_key, web_hits_remote)
                web_search_hits = web_hits_remote[:web_limit]
                web_search_provider = "exa"
            else:
                web_search_provider = "exa_empty"

    # 反思护栏在 followup 前先基于当前证据抽取一次禁忌约束。
    reflective_negative_constraints, _ = _build_negative_constraints(
        user_input=user_input,
        dsl_hits=dsl_hits,
        graph_facts=graph_facts,
        semantic_hits=semantic_hits,
        limit=8,
    )
    reflective_mode = _normalize_self_reflective_mode(settings.self_reflective_mode)
    normalized_temperature_profile = _normalize_temperature_profile(temperature_profile)
    reflective_triggered = False
    reflective_trigger_reason = "not_enabled"
    if settings.self_reflective_enabled and reflective_mode != "off":
        reflective_trigger_reason = "condition_not_met"
        if (
            bool(settings.self_reflective_brainstorm_trigger_enabled)
            and normalized_temperature_profile == "brainstorm"
        ):
            reflective_triggered = True
            reflective_trigger_reason = "brainstorm_temperature_profile"
        elif (
            bool(settings.self_reflective_low_confidence_trigger_enabled)
            and float(semantic_route.confidence) < float(settings.self_reflective_low_confidence_threshold)
        ):
            reflective_triggered = True
            reflective_trigger_reason = "semantic_router_low_confidence"
    elif reflective_mode == "off":
        reflective_trigger_reason = "mode_off"

    reflective_meta: dict[str, Any] = {
        "enabled": bool(settings.self_reflective_enabled),
        "mode": reflective_mode,
        "max_rounds": max(int(settings.self_reflective_max_rounds), 1),
        "triggered": reflective_triggered,
        "trigger_reason": reflective_trigger_reason,
        "source": "none",
        "needs_refine": False,
        "confidence": 0.0,
        "issues": [],
        "followup_queries": [],
        "query_count": 0,
        "applied": False,
        "added": {"dsl": 0, "graph": 0, "rag": 0},
        "negative_constraint_count": len(reflective_negative_constraints),
        "negative_conflicts": [],
        "elapsed_ms": 0,
        "followup_runtime": {
            "elapsed_ms": 0,
            "query_count": 0,
            "graph_remote_hits": 0,
            "rag_remote_hits": 0,
            "graph_timeouts": 0,
            "rag_timeouts": 0,
        },
    }
    if reflective_triggered:
        reflective_started_at = time.perf_counter()
        max_queries = max(int(settings.self_reflective_max_followup_queries), 1)
        judge_result: dict[str, Any] | None = None
        if reflective_mode in {"llm", "auto"}:
            judge_result = _call_self_reflective_judge_llm(
                user_input=user_input,
                intent=semantic_route.intent,
                temperature_profile=normalized_temperature_profile,
                chapter_preview=chapter_preview_for_router,
                scene_beat_text=scene_beat_text_for_router,
                dsl_hits=dsl_hits,
                graph_facts=graph_facts,
                semantic_hits=semantic_hits,
                negative_constraints=reflective_negative_constraints,
                max_queries=max_queries,
            )
            if judge_result:
                reflective_meta["source"] = "llm"
        if judge_result is None:
            judge_result = _heuristic_self_reflective_review(
                user_input=user_input,
                intent=semantic_route.intent,
                dsl_hits=dsl_hits,
                graph_facts=graph_facts,
                semantic_hits=semantic_hits,
                negative_constraints=reflective_negative_constraints,
                max_queries=max_queries,
            )
            reflective_meta["source"] = str(judge_result.get("source") or "heuristic")

        followup_queries = _normalize_followup_queries(
            judge_result.get("followup_queries"),
            limit=max_queries,
        )
        issues = (
            [str(item) for item in judge_result.get("issues", []) if str(item).strip()]
            if isinstance(judge_result.get("issues"), list)
            else []
        )
        has_negative_conflict = "negative_constraint_conflict" in issues
        needs_refine = bool(judge_result.get("needs_refine")) and (
            bool(followup_queries) or has_negative_conflict
        )
        confidence = 0.0
        try:
            confidence = max(0.0, min(float(judge_result.get("confidence", 0.0)), 1.0))
        except Exception:
            confidence = 0.0
        negative_conflicts = (
            [item for item in judge_result.get("negative_conflicts", []) if isinstance(item, dict)]
            if isinstance(judge_result.get("negative_conflicts"), list)
            else []
        )

        reflective_meta.update(
            {
                "needs_refine": needs_refine,
                "confidence": round(confidence, 4),
                "issues": issues[:6],
                "followup_queries": followup_queries,
                "query_count": len(followup_queries),
                "negative_conflicts": negative_conflicts[:4],
            }
        )

        if needs_refine and followup_queries:
            dsl_extra, graph_extra, rag_extra, followup_runtime = _run_reflective_followup_retrieval(
                project_id=project_id,
                followup_queries=followup_queries,
                graph_anchor=graph_anchor,
                rag_anchor=rag_anchor,
                rag_mode=rag_mode,
                current_chapter_index=current_chapter_index,
                rag_short_circuit_enabled=rag_short_circuit_enabled,
                windowed_retrieval_settings=windowed_retrieval_settings,
                windowed_retrieval_cards=windowed_retrieval_cards,
                dsl_limit=max(dynamic_budget_plan.dsl_limit + max_queries * 2, dynamic_budget_plan.dsl_limit),
                graph_limit=max(dynamic_budget_plan.graph_limit + max_queries * 2, dynamic_budget_plan.graph_limit),
                rag_limit=max(dynamic_budget_plan.rag_limit + max_queries * 2, dynamic_budget_plan.rag_limit),
            )
            before_counts = (len(dsl_hits), len(graph_facts), len(semantic_hits))
            dsl_hits = _merge_unique_hits(
                dsl_hits,
                dsl_extra,
                kind="dsl",
                limit=max(dynamic_budget_plan.dsl_limit + max_queries * 2, dynamic_budget_plan.dsl_limit),
            )
            graph_facts = _merge_unique_hits(
                graph_facts,
                graph_extra,
                kind="graph",
                limit=max(dynamic_budget_plan.graph_limit + max_queries * 2, dynamic_budget_plan.graph_limit),
            )
            semantic_hits = _merge_unique_hits(
                semantic_hits,
                rag_extra,
                kind="rag",
                limit=max(dynamic_budget_plan.rag_limit + max_queries * 2, dynamic_budget_plan.rag_limit),
            )
            reflective_meta["applied"] = True
            reflective_meta["added"] = {
                "dsl": max(len(dsl_hits) - before_counts[0], 0),
                "graph": max(len(graph_facts) - before_counts[1], 0),
                "rag": max(len(semantic_hits) - before_counts[2], 0),
            }
            reflective_meta["followup_runtime"] = followup_runtime

        reflective_meta["elapsed_ms"] = int((time.perf_counter() - reflective_started_at) * 1000)

    dsl_hits, spatial_dsl_meta = _apply_spatial_penalty(
        dsl_hits,
        current_location=current_location,
        settings_rows=windowed_retrieval_settings,
        cards_rows=windowed_retrieval_cards,
    )
    graph_facts, spatial_graph_meta = _apply_spatial_penalty(
        graph_facts,
        current_location=current_location,
        settings_rows=windowed_retrieval_settings,
        cards_rows=windowed_retrieval_cards,
    )
    semantic_hits, spatial_rag_meta = _apply_spatial_penalty(
        semantic_hits,
        current_location=current_location,
        settings_rows=windowed_retrieval_settings,
        cards_rows=windowed_retrieval_cards,
    )
    negative_constraints, negative_constraints_meta = _build_negative_constraints(
        user_input=user_input,
        dsl_hits=dsl_hits,
        graph_facts=graph_facts,
        semantic_hits=semantic_hits,
    )

    compressed_context, context_compression_meta = _build_context_compression(
        user_input=user_input,
        intent=semantic_route.intent,
        dsl_hits=dsl_hits,
        graph_facts=graph_facts,
        semantic_hits=semantic_hits,
        budget_mode=semantic_route.budget_mode,
    )
    context_cache_layers, context_cache_meta = _build_context_cache_layers(
        mode=mode,
        anchor=anchor,
        prompt_workshop_template=prompt_workshop_template,
        working_settings=[*project_working_settings, *ref_working_settings],
        working_cards=[*scoped_cards, *scoped_reference_cards],
        semantic_settings=[*project_semantic_settings, *ref_semantic_settings],
        current_chapter=current_chapter,
        current_volume=current_volume,
        scene_beat_context=scene_beat_context,
        latest_messages=recent_messages,
        dsl_hits=dsl_hits,
        graph_facts=graph_facts,
        semantic_hits=semantic_hits,
        negative_constraints=negative_constraints,
    )

    quality_gate = _build_quality_gate(user_input, rag_provider, semantic_hits)
    retrieval_runtime_meta = {
        "parallel_enabled": parallel_enabled,
        "graph_timeout_seconds": graph_timeout_seconds,
        "rag_timeout_seconds": rag_timeout_seconds,
        "circuit_breaker": {
            "enabled": bool(_circuit_breaker_snapshot("graph").get("enabled", True)),
            "graph": _circuit_breaker_snapshot("graph"),
            "rag": _circuit_breaker_snapshot("rag"),
        },
        "cache_ttl_seconds": retrieval_cache_ttl,
        "graph_cache": graph_cache_status,
        "rag_cache": rag_cache_status,
        "web_search_cache": web_search_cache_status,
        "graph_timeout": graph_timed_out,
        "rag_timeout": rag_timed_out,
        "context_window": {
            "profile": context_window.profile,
            "source": context_window_source,
            "recent_messages_limit": dynamic_budget_plan.recent_messages_limit,
            "retrieval_settings_limit": dynamic_budget_plan.retrieval_settings_limit,
            "retrieval_cards_limit": dynamic_budget_plan.retrieval_cards_limit,
            "model_settings_limit": dynamic_budget_plan.model_settings_limit,
            "model_cards_limit": dynamic_budget_plan.model_cards_limit,
            "chapter_content_chars": context_window.chapter_content_chars,
            "chapter_preview_chars": context_window.chapter_preview_chars,
        },
        "dynamic_budget": {
            "mode": dynamic_budget_plan.mode,
            "source": dynamic_budget_plan.source,
            "confidence": dynamic_budget_plan.confidence,
            "weights": {
                "dsl": round(dynamic_budget_plan.weights.dsl, 4),
                "graph": round(dynamic_budget_plan.weights.graph, 4),
                "rag": round(dynamic_budget_plan.weights.rag, 4),
                "history": round(dynamic_budget_plan.weights.history, 4),
            },
            "dsl_limit": dynamic_budget_plan.dsl_limit,
            "graph_limit": dynamic_budget_plan.graph_limit,
            "rag_limit": dynamic_budget_plan.rag_limit,
        },
        "intent_router": {
            "intent": semantic_route.intent,
            "confidence": semantic_route.confidence,
            "source": semantic_route.source,
            "budget_mode": semantic_route.budget_mode,
            "rag_mode": semantic_route.rag_mode,
            "signals": semantic_route.signals[:8],
        },
        "context_compression": context_compression_meta,
        "negative_constraints": negative_constraints_meta,
        "self_reflective": reflective_meta,
        "context_cache": context_cache_meta,
        "spatial": {
            "dsl": spatial_dsl_meta,
            "graph": spatial_graph_meta,
            "rag": spatial_rag_meta,
        },
        "memory_layers": {
            "working_settings": len(project_working_settings) + len(ref_working_settings),
            "semantic_settings": len(project_semantic_settings) + len(ref_semantic_settings),
            "decay_half_life_days": max(int(settings.memory_decay_half_life_days), 1),
        },
        "context_rows": context_rows_meta,
        "compile_elapsed_ms": 0,
    }
    reference_projects_meta = {
        "requested": normalized_reference_project_ids,
        "resolved": reference_project_meta,
        "settings_count": len(scoped_reference_settings),
        "cards_count": len(scoped_reference_cards),
    }
    runtime_options = {
        "thinking_enabled": bool(thinking_enabled),
        "context_window_profile": context_window.profile,
        "scene_beat_id": scene_beat_id,
        "current_chapter_index": current_chapter_index,
        "budget_mode": dynamic_budget_plan.mode,
        "intent": semantic_route.intent,
        "rag_mode": rag_mode,
        "temperature_profile": normalized_temperature_profile or None,
        "current_location": str(current_location or "").strip() or None,
    }

    model_context = {
        "pov": {
            "mode": mode,
            "anchor": anchor,
            "notes": notes,
        },
        "latest_messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": _truncate_text(msg.content, 500),
            }
            for msg in recent_messages
            if msg.content
        ],
        "settings": [
            {
                "id": row.id,
                "project_id": row.project_id,
                "key": row.key,
                "value": row.value,
                "aliases": row.aliases,
            }
            for row in windowed_retrieval_settings[: dynamic_budget_plan.model_settings_limit]
        ],
        "cards": [
            {
                "id": row.id,
                "project_id": row.project_id,
                "title": row.title,
                "content": row.content if isinstance(row.content, dict) else {},
                "aliases": row.aliases,
                "content_preview": _truncate_text(_card_content_text(row), 500),
            }
            for row in windowed_retrieval_cards[: dynamic_budget_plan.model_cards_limit]
        ],
        "memory_layers": {
            "l1_working_memory": {
                "settings_count": len(project_working_settings) + len(ref_working_settings),
                "cards_count": len(scoped_cards) + len(scoped_reference_cards),
            },
            "l2_episodic_memory": {
                "semantic_hits_count": len(semantic_hits),
                "provider": rag_provider,
            },
            "l3_semantic_memory": {
                "settings_count": len(project_semantic_settings) + len(ref_semantic_settings),
                "setting_prefix": str(settings.memory_semantic_key_prefix or "memory.semantic.volume."),
            },
        },
        "current_chapter": current_chapter,
        "story_outline": {
            "volume": current_volume,
            "scene_beat": scene_beat_context,
            "meta": outline_context_meta,
        },
        "runtime_options": runtime_options,
        "reference_projects": reference_projects_meta,
        "context_cache": context_cache_layers,
        "compressed_context": compressed_context,
        "negative_constraints": {
            "items": negative_constraints,
            "meta": negative_constraints_meta,
        },
        "prompt_workshop": {
            "template": prompt_workshop_template,
            "knowledge_injection": {
                "settings": prompt_workshop_knowledge_settings[:24],
                "cards": prompt_workshop_knowledge_cards[:20],
            },
            "meta": prompt_workshop_meta,
        },
        "evidence": {
            "resolver_order": ["DSL", "GRAPH", "RAG", "WEB_SEARCH"],
            "ranking_dimensions": ["freshness", "confidence", "relevance"],
            "providers": {
                "dsl": "local_dsl",
                "graph": graph_provider,
                "rag": rag_provider,
                "web_search": web_search_provider,
            },
            "rag_route": {
                "mode": rag_mode,
                "reason": rag_route_reason,
                "source": rag_route_source,
            },
            "rag_short_circuit": {
                "enabled": rag_short_circuit_enabled,
                "reason": rag_short_circuit_reason,
            },
            "retrieval_runtime": retrieval_runtime_meta,
            "context_pack": context_pack_meta,
            "reference_projects": reference_projects_meta,
            "runtime_options": runtime_options,
            "prompt_workshop": prompt_workshop_meta,
            "chapter_context": chapter_context_meta,
            "outline_context": outline_context_meta,
            "quality_gate": quality_gate,
            "dsl_hits": dsl_hits,
            "graph_facts": graph_facts,
            "semantic_hits": semantic_hits,
            "web_search_hits": web_search_hits,
            "negative_constraints": {
                "items": negative_constraints,
                "meta": negative_constraints_meta,
            },
        },
    }

    evidence_event = {
        "type": "evidence",
        "policy": {
            "mode": mode,
            "anchor": anchor,
            "notes": notes,
            "resolver_order": "DSL > GRAPH > RAG > WEB_SEARCH",
            "ranking_dimensions": "freshness + confidence + relevance",
            "providers": {
                "dsl": "local_dsl",
                "graph": graph_provider,
                "rag": rag_provider,
                "web_search": web_search_provider,
            },
            "rag_route": {
                "mode": rag_mode,
                "reason": rag_route_reason,
                "source": rag_route_source,
            },
            "rag_short_circuit": {
                "enabled": rag_short_circuit_enabled,
                "reason": rag_short_circuit_reason,
            },
            "retrieval_runtime": retrieval_runtime_meta,
            "context_pack": context_pack_meta,
            "reference_projects": reference_projects_meta,
            "runtime_options": runtime_options,
            "prompt_workshop": prompt_workshop_meta,
            "chapter_context": chapter_context_meta,
            "outline_context": outline_context_meta,
            "quality_gate": quality_gate,
            "negative_constraints": negative_constraints_meta,
        },
        "summary": {
            "dsl": len(dsl_hits),
            "graph": len(graph_facts),
            "rag": len(semantic_hits),
            "web_search": len(web_search_hits),
            "negative_constraints": len(negative_constraints),
        },
        "sources": {
            "dsl": dsl_hits,
            "graph": graph_facts,
            "rag": semantic_hits,
            "web_search": web_search_hits,
            "negative_constraints": negative_constraints,
        },
    }

    compile_elapsed_ms = int((time.perf_counter() - compile_started_at) * 1000)
    retrieval_runtime_meta["compile_elapsed_ms"] = compile_elapsed_ms
    _LOGGER.info(
        "context_compiled project_id=%s session_id=%s elapsed_ms=%s profile=%s recent_messages=%s retrieval_settings=%s retrieval_cards=%s",
        project_id,
        session_id,
        compile_elapsed_ms,
        context_window.profile,
        context_rows_meta["recent_messages"],
        context_rows_meta["windowed_retrieval_settings"],
        context_rows_meta["windowed_retrieval_cards"],
    )

    return CompiledContextBundle(model_context=model_context, evidence_event=evidence_event)


