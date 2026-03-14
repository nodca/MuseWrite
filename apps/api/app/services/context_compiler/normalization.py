import re
from typing import Any

from app.core.config import settings
from app.services.context_compiler._types import ContextWindowPolicy, BudgetWeights
from app.services.context_compiler._constants import (
    _RAG_MODES,
    _CONTEXT_WINDOW_PROFILES,
    _BUDGET_MODE_PRESETS,
    _SEMANTIC_ROUTE_INTENTS,
)


def _normalize_pov(pov_mode: str | None, pov_anchor: str | None) -> tuple[str, str | None, list[str]]:
    mode = (pov_mode or "global").strip().lower()
    if mode not in {"global", "character"}:
        mode = "global"
    anchor = (pov_anchor or "").strip() or None
    notes: list[str] = []
    if mode == "character" and not anchor:
        notes.append("pov_mode=character 但未提供 pov_anchor，已退化为全局上下文。")
        mode = "global"
    return mode, anchor, notes


def _normalize_rag_mode(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    mode = value.strip().lower()
    if mode == "auto":
        return "auto"
    if mode in _RAG_MODES:
        return mode
    return None


def _resolve_rag_route(user_input: str, terms: list[str], request_override: str | None) -> tuple[str, str, str]:
    _ = user_input, terms
    request_mode = _normalize_rag_mode(request_override)
    if request_mode and request_mode in _RAG_MODES:
        return request_mode, "forced_by_request", "request_override"

    configured_mode = _normalize_rag_mode(settings.rag_route_policy)
    if configured_mode and configured_mode in _RAG_MODES:
        return configured_mode, "forced_by_config", "config_policy"

    return "mix", "default_mix_policy", "static_default"


def _normalize_context_window_profile(value: str | None) -> str | None:
    profile = str(value or "").strip().lower()
    if profile in _CONTEXT_WINDOW_PROFILES:
        return profile
    return None


def _resolve_context_window_policy(request_profile: str | None) -> tuple[ContextWindowPolicy, str]:
    if request_profile:
        requested = _normalize_context_window_profile(request_profile)
        if requested is not None:
            return _CONTEXT_WINDOW_PROFILES[requested], "request_override"
        return _CONTEXT_WINDOW_PROFILES["balanced"], "request_invalid_fallback"

    configured = _normalize_context_window_profile(getattr(settings, "context_window_profile", "balanced"))
    if configured is not None:
        return _CONTEXT_WINDOW_PROFILES[configured], "config_default"

    return _CONTEXT_WINDOW_PROFILES["balanced"], "fallback_balanced"


def _normalize_budget_mode(value: str | None) -> str | None:
    mode = str(value or "").strip().lower()
    return mode if mode in _BUDGET_MODE_PRESETS else None


def _normalize_semantic_intent(value: str | None) -> str | None:
    intent = str(value or "").strip().lower()
    return intent if intent in _SEMANTIC_ROUTE_INTENTS else None


def _normalize_semantic_router_mode(value: str | None) -> str:
    mode = str(value or "auto").strip().lower()
    if mode in {"llm", "heuristic", "auto"}:
        return mode
    return "auto"


def _normalize_weight_dict(raw: Any) -> BudgetWeights | None:
    if not isinstance(raw, dict):
        return None
    keys = ("dsl", "graph", "rag", "history")
    values: list[float] = []
    for key in keys:
        try:
            parsed = float(raw.get(key))
        except Exception:
            return None
        values.append(max(parsed, 0.0))
    total = sum(values)
    if total <= 0.0:
        return None
    normalized = [item / total for item in values]
    return BudgetWeights(
        dsl=normalized[0],
        graph=normalized[1],
        rag=normalized[2],
        history=normalized[3],
    )


def _normalize_timeout(seconds: float | int | None, fallback: float) -> float:
    try:
        value = float(seconds)
    except Exception:
        value = fallback
    return max(value, 0.1)


def _normalize_context_compression_mode(value: str | None) -> str:
    mode = str(value or "rerank").strip().lower()
    if mode in {"off", "rerank"}:
        return mode
    return "rerank"


def _normalize_self_reflective_mode(value: str | None) -> str:
    mode = str(value or "auto").strip().lower()
    if mode in {"off", "heuristic", "llm", "auto"}:
        return mode
    return "auto"


def _normalize_temperature_profile(value: str | None) -> str:
    profile = str(value or "").strip().lower()
    if profile in {"action", "chat", "suggestion", "brainstorm"}:
        return profile
    return ""


def _normalize_followup_queries(raw: Any, *, limit: int) -> list[str]:
    if limit <= 0:
        return []
    values: list[str] = []
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = [str(item or "") for item in raw]
    normalized: list[str] = []
    for item in values:
        query = re.sub(r"\s+", " ", str(item or "").strip())
        if not query:
            continue
        if len(query) > 120:
            query = query[:120].rstrip()
        if len(query) < 2 or query in normalized:
            continue
        normalized.append(query)
        if len(normalized) >= limit:
            break
    return normalized
