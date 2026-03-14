"""Exa Web Search integration.

Provides a thin wrapper around the Exa search API for retrieving
real-world information (e.g. fan-fiction references, historical facts).

Pattern follows ``lightrag_documents.py`` — httpx + timeout + error handling.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import settings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_exa_enabled() -> None:
    """Raise if Exa integration is not configured."""
    if not settings.exa_enabled:
        raise RuntimeError("EXA_ENABLED=false")
    if not settings.exa_api_key:
        raise RuntimeError("EXA_API_KEY is empty")


def _exa_headers() -> dict[str, str]:
    return {
        "x-api-key": settings.exa_api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _exa_request(method: str, path: str, json_body: dict[str, Any]) -> dict[str, Any]:
    """Send a request to the Exa API and return the parsed JSON response."""
    _ensure_exa_enabled()
    base = settings.exa_base_url.rstrip("/")
    url = f"{base}{path}"
    timeout = httpx.Timeout(float(settings.exa_timeout_seconds))

    with httpx.Client(timeout=timeout) as client:
        resp = client.request(
            method=method,
            url=url,
            json=json_body,
            headers=_exa_headers(),
        )

    if int(resp.status_code) >= 400:
        detail = resp.text.strip()[:300]
        raise RuntimeError(f"Exa request failed: {resp.status_code} {detail}")

    try:
        payload = resp.json()
    except Exception:
        payload = {"status": "ok", "raw_text": resp.text}
    return payload if isinstance(payload, dict) else {"status": "ok", "data": payload}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(query: str, *, num_results: int = 5) -> list[dict[str, Any]]:
    """Search Exa and return normalized hits.

    Returns a list of dicts, each containing:
        kind, title, snippet, confidence, source_url
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    if not settings.exa_enabled:
        return []

    if not settings.exa_api_key:
        return []

    limit = max(int(num_results), 1)
    body: dict[str, Any] = {
        "query": query.strip(),
        "numResults": limit,
        "type": "auto",
        "contents": {
            "text": True,
            "highlights": {"numSentences": 3},
        },
    }

    payload = _exa_request("POST", "/search", body)

    results_raw = payload.get("results", [])
    if not isinstance(results_raw, list):
        results_raw = []

    normalized: list[dict[str, Any]] = []
    for item in results_raw:
        if not isinstance(item, dict):
            continue
        highlights = item.get("highlights", [])
        snippet = ""
        if isinstance(highlights, list) and highlights:
            snippet = str(highlights[0]) if highlights[0] else ""
        if not snippet:
            text = str(item.get("text", ""))
            snippet = text[:300] if text else ""

        normalized.append({
            "kind": "web_search",
            "title": str(item.get("title", "")),
            "snippet": snippet,
            "confidence": 0.75,
            "source_url": str(item.get("url", "")),
        })

    return normalized[:limit]
