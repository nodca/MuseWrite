import hashlib
import time
from concurrent.futures import Future
from typing import Any

from app.core.config import settings
from app.services.context_compiler._state import (
    _GRAPH_HITS_CACHE,
    _RAG_HITS_CACHE,
    _WEB_SEARCH_CACHE,
    _RETRIEVAL_CACHE_LOCK,
    _RETRIEVAL_EXECUTOR,
)
from app.services.context_compiler._utils import _safe_int
from app.services.context_compiler.normalization import _normalize_timeout
from app.services.retrieval_adapters import (
    fetch_neo4j_graph_facts,
    fetch_lightrag_semantic_hits,
)


def _cache_ttl_seconds() -> float:
    try:
        ttl = float(settings.retrieval_cache_ttl_seconds)
    except Exception:
        ttl = 0.0
    return max(ttl, 0.0)


def _cache_enabled() -> bool:
    return _cache_ttl_seconds() > 0.0


def _clone_hits(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in items if isinstance(item, dict)]


def _cache_get(
    cache: dict[str, tuple[float, list[dict[str, Any]]]],
    key: str,
) -> tuple[list[dict[str, Any]] | None, str]:
    if not _cache_enabled():
        return None, "disabled"
    now = time.time()
    with _RETRIEVAL_CACHE_LOCK:
        cached = cache.get(key)
        if cached is None:
            return None, "miss"
        expires_at, rows = cached
        if expires_at <= now:
            cache.pop(key, None)
            return None, "expired"
        return _clone_hits(rows), "hit"


def _cache_set(
    cache: dict[str, tuple[float, list[dict[str, Any]]]],
    key: str,
    rows: list[dict[str, Any]],
) -> None:
    ttl = _cache_ttl_seconds()
    if ttl <= 0:
        return
    normalized = _clone_hits(rows)
    if not normalized:
        return
    max_entries = max(int(settings.retrieval_cache_max_entries), 16)
    expires_at = time.time() + ttl
    with _RETRIEVAL_CACHE_LOCK:
        cache[key] = (expires_at, normalized)
        if len(cache) > max_entries:
            oldest_key = min(cache.items(), key=lambda item: item[1][0])[0]
            cache.pop(oldest_key, None)


def invalidate_graph_retrieval_cache(project_id: int) -> int:
    prefix = f"p:{int(project_id)}|"
    removed = 0
    with _RETRIEVAL_CACHE_LOCK:
        keys = [key for key in _GRAPH_HITS_CACHE.keys() if key.startswith(prefix)]
        for key in keys:
            _GRAPH_HITS_CACHE.pop(key, None)
            removed += 1
    return removed


def _graph_cache_key(
    project_id: int,
    terms: list[str],
    anchor: str | None,
    limit: int,
    *,
    current_chapter: int | None = None,
) -> str:
    normalized_terms = sorted(
        {
            str(item).strip().lower()
            for item in terms
            if isinstance(item, str) and str(item).strip()
        }
    )
    anchor_norm = str(anchor or "").strip().lower()
    chapter_norm = _safe_int(current_chapter, 0)
    return f"p:{project_id}|a:{anchor_norm}|l:{int(limit)}|c:{chapter_norm}|terms:{','.join(normalized_terms)}"


def _rag_cache_key(
    user_input: str,
    anchor: str | None,
    mode: str | None,
    limit: int,
) -> str:
    query_hash = hashlib.sha1((user_input or "").strip().encode("utf-8")).hexdigest()[:20]
    anchor_norm = str(anchor or "").strip().lower()
    mode_norm = str(mode or "").strip().lower()
    return f"q:{query_hash}|a:{anchor_norm}|m:{mode_norm}|l:{int(limit)}"


def _submit_graph_future(
    project_id: int,
    terms: list[str],
    anchor: str | None,
    limit: int,
    *,
    current_chapter: int | None = None,
) -> Future[list[dict[str, Any]]]:
    return _RETRIEVAL_EXECUTOR.submit(
        fetch_neo4j_graph_facts,
        project_id,
        terms,
        anchor=anchor,
        limit=limit,
        current_chapter=current_chapter,
        raise_on_error=True,
    )


def _strict_graph_mode_enabled() -> bool:
    return bool(settings.neo4j_enabled and getattr(settings, "neo4j_gds_required", False))


def _submit_rag_future(
    user_input: str,
    anchor: str | None,
    limit: int,
    rag_mode: str,
) -> Future[list[dict[str, Any]]]:
    return _RETRIEVAL_EXECUTOR.submit(
        fetch_lightrag_semantic_hits,
        user_input,
        anchor=anchor,
        limit=limit,
        mode_override=rag_mode,
        raise_on_error=True,
    )


def _await_hits_future(
    future: Future[list[dict[str, Any]]],
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], bool, bool]:
    try:
        rows = future.result(timeout=_normalize_timeout(timeout_seconds, 1.0))
    except TimeoutError:
        future.cancel()
        return [], True, False
    except Exception:
        return [], False, True
    if not isinstance(rows, list):
        return [], False, True
    return [item for item in rows if isinstance(item, dict)], False, False


def _web_search_cache_key(query: str, limit: int) -> str:
    query_hash = hashlib.sha1((query or "").strip().encode("utf-8")).hexdigest()[:20]
    return f"ws:{query_hash}|l:{int(limit)}"


def _submit_web_search_future(user_input: str, limit: int) -> Future[list[dict[str, Any]]]:
    from app.services.exa_search import search as _exa_search

    return _RETRIEVAL_EXECUTOR.submit(_exa_search, user_input, num_results=limit)

