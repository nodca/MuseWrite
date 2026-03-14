import hashlib
import json
import re
import time
from typing import Any

from pydantic import BaseModel, Field
from sqlmodel import Session

from app.core.config import settings
from app.services.context_compiler._utils import _truncate_text, _extract_query_terms, _setting_value_text, _card_content_text
from app.services.context_compiler.normalization import (
    _normalize_followup_queries,
    _normalize_self_reflective_mode,
    _normalize_temperature_profile,
    _normalize_timeout,
)
from app.services.context_compiler.caching import (
    _submit_graph_future,
    _submit_rag_future,
    _await_hits_future,
    _strict_graph_mode_enabled,
)
from app.services.context_compiler.circuit_breaker import (
    _circuit_breaker_should_short_circuit,
    _circuit_breaker_record_failure,
    _circuit_breaker_record_success,
)
from app.services.context_compiler.retrieval import (
    _build_dsl_hits,
    _build_graph_facts,
    _build_semantic_hits,
)
from app.services.context_compiler.memory import _apply_memory_decay
from app.services.context_compiler.compression import _hit_preview_text
from app.services.llm_provider import generate_structured_sync


class _SelfReflectiveJudgeOutput(BaseModel):
    needs_refine: bool = False
    issues: list[str] = Field(default_factory=list)
    followup_queries: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""


def _detect_negative_constraint_conflicts(
    *,
    user_input: str,
    negative_constraints: list[dict[str, Any]],
    max_items: int = 4,
) -> list[dict[str, Any]]:
    text = str(user_input or "").strip()
    if not text:
        return []
    normalized_text = re.sub(r"\s+", "", text).lower()
    if not normalized_text:
        return []

    # 用户显式“不要写/避免”时，视为主动遵守禁忌，不判定为冲突。
    if any(
        token in normalized_text
        for token in ("不要", "别写", "避免", "禁止", "不可", "不能", "不得", "不写", "别提")
    ):
        return []

    input_terms = {token.lower() for token in _extract_query_terms(text)}
    conflicts: list[dict[str, Any]] = []
    for item in negative_constraints:
        if not isinstance(item, dict):
            continue
        constraint_text = str(item.get("text", "") or "").strip()
        if not constraint_text:
            continue
        constraint_terms = {
            token.lower()
            for token in _extract_query_terms(constraint_text)
            if len(str(token or "").strip()) >= 2
        }
        if not constraint_terms:
            continue
        matched = sorted(term for term in constraint_terms if term in input_terms)
        if not matched:
            continue
        conflicts.append(
            {
                "text": _truncate_text(constraint_text, 160),
                "source": str(item.get("source") or "").strip(),
                "matched_terms": matched[:4],
            }
        )
        if len(conflicts) >= max(max_items, 1):
            break
    return conflicts


def _heuristic_self_reflective_review(
    *,
    user_input: str,
    intent: str,
    dsl_hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
    negative_constraints: list[dict[str, Any]],
    max_queries: int,
) -> dict[str, Any]:
    issues: list[str] = []
    followup_queries: list[str] = []
    text = str(user_input or "").strip()
    if not text:
        return {
            "needs_refine": False,
            "issues": issues,
            "followup_queries": followup_queries,
            "confidence": 0.2,
            "source": "heuristic_empty_input",
        }

    if intent in {"brainstorm", "writing_help"} and len(semantic_hits) <= 1:
        issues.append("missing_semantic_evidence")
    if len(graph_facts) <= 1 and any(token in text for token in ("线索", "真相", "伏笔", "冲突", "推演", "智斗", "权谋")):
        issues.append("missing_graph_links")
    if len(dsl_hits) <= 1 and any(token in text for token in ("设定", "世界观", "规则", "身份", "地点", "物品")):
        issues.append("missing_dsl_constraints")

    conflicts = _detect_negative_constraint_conflicts(
        user_input=text,
        negative_constraints=negative_constraints,
        max_items=3,
    )
    if conflicts:
        issues.append("negative_constraint_conflict")

    if issues:
        base_query = _truncate_text(text, 96)
        if "missing_graph_links" in issues:
            followup_queries.append(base_query + " 关键关系 伏笔")
        if "missing_semantic_evidence" in issues:
            followup_queries.append(base_query + " 相关前情 章节证据")
        if "missing_dsl_constraints" in issues:
            followup_queries.append(base_query + " 世界观设定 规则")
        if "negative_constraint_conflict" in issues:
            followup_queries.append(base_query + " 禁忌约束 校验 重写")

    followup_queries = _normalize_followup_queries(followup_queries, limit=max_queries)
    return {
        "needs_refine": bool(followup_queries) or bool(conflicts),
        "issues": issues[:6],
        "followup_queries": followup_queries,
        "confidence": 0.62 if conflicts else (0.46 if followup_queries else 0.35),
        "negative_conflicts": conflicts[:3],
        "source": "heuristic",
    }


def _call_self_reflective_judge_llm(
    *,
    user_input: str,
    intent: str,
    temperature_profile: str,
    chapter_preview: str,
    scene_beat_text: str,
    dsl_hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
    negative_constraints: list[dict[str, Any]],
    max_queries: int,
) -> dict[str, Any] | None:
    model = str(settings.lightrag_llm_model or "").strip()
    base_url = str(settings.lightrag_llm_base_url or "").strip()
    api_key = str(settings.lightrag_llm_api_key or "").strip()
    if not (model and base_url and api_key):
        return None

    dsl_preview = [
        {
            "title": str(item.get("title") or item.get("kind") or ""),
            "snippet": _truncate_text(str(item.get("snippet") or ""), 120),
        }
        for item in dsl_hits[:4]
        if isinstance(item, dict)
    ]
    graph_preview = [
        {
            "fact": _truncate_text(str(item.get("fact") or ""), 120),
            "confidence": item.get("confidence"),
        }
        for item in graph_facts[:4]
        if isinstance(item, dict)
    ]
    rag_preview = [
        {
            "title": str(item.get("title") or ""),
            "snippet": _truncate_text(str(item.get("snippet") or ""), 120),
            "citation": (
                item.get("citation")
                if isinstance(item.get("citation"), dict)
                else None
            ),
        }
        for item in semantic_hits[:4]
        if isinstance(item, dict)
    ]
    negative_preview = [
        {
            "text": _truncate_text(str(item.get("text") or ""), 140),
            "source": str(item.get("source") or ""),
            "title": str(item.get("title") or ""),
        }
        for item in negative_constraints[:6]
        if isinstance(item, dict)
    ]

    system_prompt = (
        "你是小说上下文审视器（Judge）。"
        "任务：检查当前检索是否遗漏关键事实、是否存在时序风险、是否有明显噪声，"
        "并判断用户请求是否可能违反 negative_constraints 禁忌约束。"
        "若命中 negative_constraint_conflict，needs_refine 必须为 true。"
        "followup_queries 最多给 2 条，每条短且可直接检索。"
    )
    payload_text = json.dumps(
        {
            "intent": intent,
            "temperature_profile": temperature_profile,
            "user_input": _truncate_text(user_input, 820),
            "chapter_preview": _truncate_text(chapter_preview, 540),
            "scene_beat": _truncate_text(scene_beat_text, 360),
            "retrieved": {
                "dsl": dsl_preview,
                "graph": graph_preview,
                "rag": rag_preview,
                "negative_constraints": negative_preview,
            },
            "max_queries": max(max_queries, 1),
        },
        ensure_ascii=False,
    )
    try:
        structured = generate_structured_sync(
            payload_text,
            output_model=_SelfReflectiveJudgeOutput,
            schema_name="self_reflective_judge_output",
            context={"system": system_prompt, "raw_prompt": True},
            runtime_config={
                "provider": "openai_compatible",
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
            },
            temperature_override=0.0,
        )
    except Exception:
        return None

    parsed = structured.parsed
    if not parsed:
        return None

    needs_refine = bool(getattr(parsed, "needs_refine", False))
    try:
        confidence = max(0.0, min(float(getattr(parsed, "confidence", 0.0) or 0.0), 1.0))
    except Exception:
        confidence = 0.0
    issues: list[str] = []
    for item in list(getattr(parsed, "issues", []) or [])[:16]:
        token = str(item or "").strip().lower()
        if not token or token in issues:
            continue
        issues.append(token[:40])
        if len(issues) >= 6:
            break

    followup_queries = _normalize_followup_queries(getattr(parsed, "followup_queries", []), limit=max_queries)
    if "negative_constraint_conflict" in issues and not followup_queries:
        followup_queries = _normalize_followup_queries(
            [_truncate_text(str(user_input or ""), 96) + " 禁忌约束 校验 重写"],
            limit=max_queries,
        )
    if not followup_queries and "negative_constraint_conflict" not in issues:
        needs_refine = False
    return {
        "needs_refine": needs_refine,
        "issues": issues,
        "followup_queries": followup_queries,
        "confidence": confidence,
        "source": "llm",
    }

def _hit_identity(item: dict[str, Any], *, kind: str) -> str:
    if kind == "dsl":
        if item.get("id") is not None:
            return f"id:{item.get('project_id')}:{item.get('id')}:{item.get('kind')}"
        return f"title:{item.get('project_id')}:{item.get('title')}:{item.get('kind')}"
    if kind == "graph":
        if item.get("id") is not None:
            return f"id:{item.get('project_id')}:{item.get('id')}"
        fact_key = str(item.get("fact_key") or "").strip()
        if fact_key:
            return f"fact_key:{fact_key}"
        return f"fact:{item.get('project_id')}:{item.get('fact')}"
    if item.get("id") is not None:
        return f"id:{item.get('project_id')}:{item.get('id')}"
    citation = item.get("citation") if isinstance(item.get("citation"), dict) else {}
    citation_key = f"{citation.get('source')}|{citation.get('chunk')}"
    if citation_key != "None|None":
        return f"citation:{citation_key}"
    return f"title:{item.get('project_id')}:{item.get('title')}:{item.get('snippet')}"


def _merge_unique_hits(
    primary: list[dict[str, Any]],
    extra: list[dict[str, Any]],
    *,
    kind: str,
    limit: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for collection in (primary, extra):
        for item in collection:
            if not isinstance(item, dict):
                continue
            key = _hit_identity(item, kind=kind)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= max(limit, 1):
                return merged
    return merged


def _run_reflective_followup_retrieval(
    *,
    project_id: int,
    followup_queries: list[str],
    graph_anchor: str | None,
    rag_anchor: str | None,
    rag_mode: str,
    current_chapter_index: int | None,
    rag_short_circuit_enabled: bool,
    windowed_retrieval_settings: list[Any],
    windowed_retrieval_cards: list[Any],
    dsl_limit: int,
    graph_limit: int,
    rag_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    start = time.perf_counter()
    dsl_extra: list[dict[str, Any]] = []
    graph_extra: list[dict[str, Any]] = []
    rag_extra: list[dict[str, Any]] = []
    graph_timeout_seconds = _normalize_timeout(settings.retrieval_graph_timeout_seconds, 2.0)
    rag_timeout_seconds = _normalize_timeout(settings.retrieval_rag_timeout_seconds, 2.0)
    graph_remote_hits = 0
    rag_remote_hits = 0
    graph_timeouts = 0
    rag_timeouts = 0
    graph_circuit_open_count = 0
    rag_circuit_open_count = 0

    dsl_step_limit = max(min(dsl_limit, 4), 1)
    graph_step_limit = max(min(graph_limit, 4), 1)
    rag_step_limit = max(min(rag_limit, 4), 1)

    for query in followup_queries:
        terms = _extract_query_terms(query)
        dsl_extra.extend(
            _build_dsl_hits(
                terms,
                windowed_retrieval_settings,
                windowed_retrieval_cards,
                limit=dsl_step_limit,
            )
        )

        graph_circuit_open, _ = _circuit_breaker_should_short_circuit("graph")
        if graph_circuit_open:
            graph_remote, graph_timed_out, graph_failed = [], False, False
            graph_circuit_open_count += 1
        else:
            graph_future = _submit_graph_future(
                project_id,
                terms,
                graph_anchor,
                graph_step_limit,
                current_chapter=current_chapter_index,
            )
            graph_remote, graph_timed_out, graph_failed = _await_hits_future(graph_future, graph_timeout_seconds)
            if graph_timed_out or graph_failed:
                _circuit_breaker_record_failure("graph")
            else:
                _circuit_breaker_record_success("graph")
        if graph_remote:
            graph_extra.extend(graph_remote[:graph_step_limit])
            graph_remote_hits += len(graph_remote[:graph_step_limit])
        elif not _strict_graph_mode_enabled():
            graph_extra.extend(
                _build_graph_facts(
                    windowed_retrieval_cards,
                    windowed_retrieval_settings,
                    graph_anchor,
                    limit=graph_step_limit,
                )
            )
        if graph_timed_out:
            graph_timeouts += 1
        if graph_failed:
            continue

        if rag_short_circuit_enabled:
            continue
        rag_circuit_open, _ = _circuit_breaker_should_short_circuit("rag")
        if rag_circuit_open:
            rag_remote, rag_timed_out, rag_failed = [], False, False
            rag_circuit_open_count += 1
        else:
            rag_future = _submit_rag_future(query, rag_anchor, rag_step_limit, rag_mode)
            rag_remote, rag_timed_out, rag_failed = _await_hits_future(rag_future, rag_timeout_seconds)
            if rag_timed_out or rag_failed:
                _circuit_breaker_record_failure("rag")
            else:
                _circuit_breaker_record_success("rag")
        if rag_remote:
            rag_extra.extend(rag_remote[:rag_step_limit])
            rag_remote_hits += len(rag_remote[:rag_step_limit])
        else:
            rag_extra.extend(
                _build_semantic_hits(
                    query,
                    windowed_retrieval_settings,
                    windowed_retrieval_cards,
                    rag_anchor,
                    limit=rag_step_limit,
                )
            )
        if rag_timed_out:
            rag_timeouts += 1
        if rag_failed:
            continue

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return (
        _apply_memory_decay(dsl_extra),
        _apply_memory_decay(graph_extra),
        _apply_memory_decay(rag_extra),
        {
            "elapsed_ms": elapsed_ms,
            "query_count": len(followup_queries),
            "graph_remote_hits": graph_remote_hits,
            "rag_remote_hits": rag_remote_hits,
            "graph_timeouts": graph_timeouts,
            "rag_timeouts": rag_timeouts,
            "graph_circuit_open_count": graph_circuit_open_count,
            "rag_circuit_open_count": rag_circuit_open_count,
        },
    )


def _build_context_cache_layers(
    *,
    mode: str,
    anchor: str | None,
    prompt_workshop_template: dict[str, Any] | None,
    working_settings: list[Any],
    working_cards: list[Any],
    semantic_settings: list[Any],
    current_chapter: dict[str, Any] | None,
    current_volume: dict[str, Any] | None,
    scene_beat_context: dict[str, Any] | None,
    latest_messages: list[Any],
    dsl_hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
    negative_constraints: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    static_lines: list[str] = [
        f"pov_mode={mode}",
        f"pov_anchor={anchor or ''}",
        "resolver_order=DSL>GRAPH>RAG",
    ]
    if isinstance(prompt_workshop_template, dict):
        static_lines.append(f"template_name={str(prompt_workshop_template.get('name', '') or '')}")
        static_lines.append(
            "template_system_prompt="
            + _truncate_text(str(prompt_workshop_template.get("system_prompt", "") or ""), 1400)
        )

    ordered_working_settings = sorted(
        [row for row in working_settings if getattr(row, "key", None)],
        key=lambda row: str(getattr(row, "key", "") or ""),
    )
    for row in ordered_working_settings[:90]:
        static_lines.append(
            f"[setting] {str(getattr(row, 'key', '') or '')}: "
            + _truncate_text(_setting_value_text(row), 220)
        )
    ordered_working_cards = sorted(
        [row for row in working_cards if getattr(row, "title", None)],
        key=lambda row: str(getattr(row, "title", "") or ""),
    )
    for row in ordered_working_cards[:72]:
        static_lines.append(
            f"[card] {str(getattr(row, 'title', '') or '')}: "
            + _truncate_text(_card_content_text(row), 200)
        )
    # semantic settings 在同一项目内很少变化，放入 static 层提高 cache hit rate
    ordered_semantic_settings = sorted(
        [row for row in semantic_settings if getattr(row, "key", None)],
        key=lambda row: str(getattr(row, "key", "") or ""),
    )
    for row in ordered_semantic_settings[:40]:
        static_lines.append(
            f"[semantic] {str(getattr(row, 'key', '') or '')}: "
            + _truncate_text(_setting_value_text(row), 260)
        )
    # volume outline 在整卷写作期间几乎不变，放入 static 层提高 cache hit rate
    if isinstance(current_volume, dict):
        static_lines.append(
            "volume_outline="
            + _truncate_text(str(current_volume.get("outline", "") or ""), 2000)
        )
    static_prefix = _truncate_text("\n".join(static_lines), 36000)

    persistent_lines: list[str] = []
    if isinstance(current_chapter, dict):
        persistent_lines.append(
            "chapter_preview="
            + _truncate_text(str(current_chapter.get("content_preview", "") or ""), 2200)
        )
    if isinstance(scene_beat_context, dict):
        active = scene_beat_context.get("active")
        if isinstance(active, dict):
            persistent_lines.append(
                "scene_beat_active="
                + _truncate_text(str(active.get("content", "") or ""), 600)
            )
    persistent_prefix = _truncate_text("\n".join(persistent_lines), 20000)

    session_lines: list[str] = []
    for msg in latest_messages[-12:]:
        role = str(getattr(msg, "role", "") or "")
        content = _truncate_text(str(getattr(msg, "content", "") or ""), 320)
        session_lines.append(f"[{role}] {content}")
    for source, rows in (("DSL", dsl_hits), ("GRAPH", graph_facts), ("RAG", semantic_hits)):
        for row in rows[:8]:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or row.get("kind") or source)
            snippet = _truncate_text(_hit_preview_text(row), 180)
            session_lines.append(f"[{source}] {title}: {snippet}")
    for item in negative_constraints[:8]:
        if not isinstance(item, dict):
            continue
        text = _truncate_text(str(item.get("text", "") or ""), 180)
        if not text:
            continue
        source_name = str(item.get("source") or "NEG").strip().upper()
        session_lines.append(f"[NEGATIVE/{source_name}] {text}")
    session_prefix = _truncate_text("\n".join(session_lines), 12000)

    stable_prefix_hash = hashlib.sha1(
        (static_prefix + "\n\n" + persistent_prefix).encode("utf-8")
    ).hexdigest()
    layers = {
        "static_prefix": static_prefix,
        "persistent_prefix": persistent_prefix,
        "session_prefix": session_prefix,
        "stable_prefix_hash": stable_prefix_hash,
    }
    meta = {
        "enabled": bool(settings.context_cache_enabled),
        "static_chars": len(static_prefix),
        "persistent_chars": len(persistent_prefix),
        "session_chars": len(session_prefix),
        "stable_prefix_hash": stable_prefix_hash,
    }
    return layers, meta
