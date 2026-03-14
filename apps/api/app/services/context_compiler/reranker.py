"""Remote reranker scoring via API (e.g. Qwen3-Reranker-8B).

All local model loading code has been removed — reranking is now
done exclusively via a remote API endpoint configured by:
  CONTEXT_COMPRESSION_RERANKER_REMOTE_URL
  CONTEXT_COMPRESSION_RERANKER_REMOTE_API_KEY
  CONTEXT_COMPRESSION_RERANKER_REMOTE_MODEL
"""
from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings

_LOGGER = logging.getLogger("context_compiler.reranker")


def _score_lines_with_remote_reranker(
    *,
    query: str,
    lines: list[str],
) -> list[float] | None:
    """Call remote reranker API (OpenAI-compatible /v1/rerank). Returns None if unavailable."""
    url = str(getattr(settings, "context_compression_reranker_remote_url", "") or "").strip()
    if not url:
        return None
    api_key = str(getattr(settings, "context_compression_reranker_remote_api_key", "") or "").strip()
    model = str(getattr(settings, "context_compression_reranker_remote_model", "") or "").strip()
    try:
        import httpx

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload: dict[str, object] = {"query": query, "documents": lines}
        if model:
            payload["model"] = model
        resp = httpx.post(
            url.rstrip("/") + "/rerank",
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(8.0),
        )
        if resp.status_code != 200:
            _LOGGER.warning("remote reranker returned %s", resp.status_code)
            return None
        data = resp.json()
        # Format A: {"results": [{"index": 0, "relevance_score": 0.99}, ...]}
        results = data.get("results")
        if isinstance(results, list) and len(results) == len(lines):
            scores = [0.0] * len(lines)
            for item in results:
                idx = int(item.get("index", -1))
                if 0 <= idx < len(lines):
                    scores[idx] = float(item.get("relevance_score", 0.0))
            return scores
        # Format B (legacy microservice): {"scores": [0.99, ...]}
        flat_scores = data.get("scores")
        if isinstance(flat_scores, list) and len(flat_scores) == len(lines):
            return [float(s) for s in flat_scores]
        return None
    except Exception as exc:
        _LOGGER.warning("remote reranker failed: %s", exc)
        return None


def _score_lines_with_reranker(
    *,
    query: str,
    lines: list[str],
) -> list[float] | None:
    """Score lines using remote reranker API."""
    if not lines:
        return None
    return _score_lines_with_remote_reranker(query=query, lines=lines)
