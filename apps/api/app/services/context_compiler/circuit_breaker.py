import time
from collections import deque
from typing import Any

from app.services.context_compiler._state import (
    _RETRIEVAL_CIRCUIT_BREAKER_LOCK,
    _RETRIEVAL_CIRCUIT_BREAKERS,
)

# ---------------------------------------------------------------------------
# Module-internal defaults (previously exposed as env vars).
# These values are only consumed by this module and do not need
# user-facing configuration knobs.
# ---------------------------------------------------------------------------
_CB_ENABLED = True
_GRAPH_CB_FAILURE_THRESHOLD = 3
_GRAPH_CB_WINDOW_SECONDS = 30.0
_GRAPH_CB_OPEN_SECONDS = 15.0
_RAG_CB_FAILURE_THRESHOLD = 3
_RAG_CB_WINDOW_SECONDS = 30.0
_RAG_CB_OPEN_SECONDS = 15.0
_WEB_SEARCH_CB_FAILURE_THRESHOLD = 3
_WEB_SEARCH_CB_WINDOW_SECONDS = 30.0
_WEB_SEARCH_CB_OPEN_SECONDS = 15.0

_VALID_CB_KINDS = {"graph", "rag", "web_search"}


def _normalize_cb_kind(kind: str) -> str:
    lowered = str(kind).strip().lower()
    if lowered in _VALID_CB_KINDS:
        return lowered
    return "rag"


def _circuit_breaker_settings(kind: str) -> tuple[int, float, float]:
    normalized = _normalize_cb_kind(kind)
    if normalized == "graph":
        return (
            max(int(_GRAPH_CB_FAILURE_THRESHOLD), 1),
            max(float(_GRAPH_CB_WINDOW_SECONDS), 1.0),
            max(float(_GRAPH_CB_OPEN_SECONDS), 1.0),
        )
    if normalized == "web_search":
        return (
            max(int(_WEB_SEARCH_CB_FAILURE_THRESHOLD), 1),
            max(float(_WEB_SEARCH_CB_WINDOW_SECONDS), 1.0),
            max(float(_WEB_SEARCH_CB_OPEN_SECONDS), 1.0),
        )
    return (
        max(int(_RAG_CB_FAILURE_THRESHOLD), 1),
        max(float(_RAG_CB_WINDOW_SECONDS), 1.0),
        max(float(_RAG_CB_OPEN_SECONDS), 1.0),
    )


def _circuit_breaker_prune_failures(failures: deque[float], *, now: float, window_seconds: float) -> None:
    while failures and (now - failures[0]) > window_seconds:
        failures.popleft()


def _circuit_breaker_should_short_circuit(kind: str) -> tuple[bool, float]:
    if not _CB_ENABLED:
        return False, 0.0
    normalized = _normalize_cb_kind(kind)
    _, window_seconds, _ = _circuit_breaker_settings(normalized)
    now = time.monotonic()
    with _RETRIEVAL_CIRCUIT_BREAKER_LOCK:
        state = _RETRIEVAL_CIRCUIT_BREAKERS[normalized]
        failures: deque[float] = state["failures"]
        _circuit_breaker_prune_failures(failures, now=now, window_seconds=window_seconds)
        open_until = float(state.get("open_until") or 0.0)
        if open_until > now:
            return True, max(open_until - now, 0.0)
        return False, 0.0


def _circuit_breaker_record_failure(kind: str) -> None:
    if not _CB_ENABLED:
        return
    normalized = _normalize_cb_kind(kind)
    failure_threshold, window_seconds, open_seconds = _circuit_breaker_settings(normalized)
    now = time.monotonic()
    with _RETRIEVAL_CIRCUIT_BREAKER_LOCK:
        state = _RETRIEVAL_CIRCUIT_BREAKERS[normalized]
        failures: deque[float] = state["failures"]
        _circuit_breaker_prune_failures(failures, now=now, window_seconds=window_seconds)
        failures.append(now)
        if len(failures) >= failure_threshold:
            state["open_until"] = now + open_seconds
            state["opened_count"] = int(state.get("opened_count") or 0) + 1
            failures.clear()


def _circuit_breaker_record_success(kind: str) -> None:
    if not _CB_ENABLED:
        return
    normalized = _normalize_cb_kind(kind)
    _, window_seconds, _ = _circuit_breaker_settings(normalized)
    now = time.monotonic()
    with _RETRIEVAL_CIRCUIT_BREAKER_LOCK:
        state = _RETRIEVAL_CIRCUIT_BREAKERS[normalized]
        failures: deque[float] = state["failures"]
        _circuit_breaker_prune_failures(failures, now=now, window_seconds=window_seconds)
        failures.clear()
        if float(state.get("open_until") or 0.0) <= now:
            state["open_until"] = 0.0


def _circuit_breaker_snapshot(kind: str) -> dict[str, Any]:
    normalized = _normalize_cb_kind(kind)
    failure_threshold, window_seconds, open_seconds = _circuit_breaker_settings(normalized)
    now = time.monotonic()
    with _RETRIEVAL_CIRCUIT_BREAKER_LOCK:
        state = _RETRIEVAL_CIRCUIT_BREAKERS[normalized]
        failures: deque[float] = state["failures"]
        _circuit_breaker_prune_failures(failures, now=now, window_seconds=window_seconds)
        open_until = float(state.get("open_until") or 0.0)
        open_remaining = max(open_until - now, 0.0)
        return {
            "kind": normalized,
            "enabled": bool(_CB_ENABLED),
            "failure_threshold": failure_threshold,
            "window_seconds": window_seconds,
            "open_seconds": open_seconds,
            "recent_failures": len(failures),
            "open": bool(open_remaining > 0),
            "open_remaining_seconds": round(open_remaining, 3),
            "opened_count": int(state.get("opened_count") or 0),
        }

