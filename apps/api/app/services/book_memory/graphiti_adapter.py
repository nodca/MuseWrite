"""Graphiti Temporal Graph Adapter for Book Memory OS.

Wraps ``graphiti-core`` to provide:

1. **Sync API** for the codebase's synchronous call paths
   (``asyncio.run`` under the hood — safe because callers are
   sync worker threads or sync FastAPI path operations).
2. **Project-scoped graph partitioning** via Graphiti ``group_id``.
3. **Fiction-domain entity types** for guided extraction.
4. **Chapter-indexed temporal ordering** via ``reference_time``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Module-level singleton ──────────────────────────────────────────

_graphiti_instance = None
_graphiti_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None

# Epoch used as base for chapter→datetime mapping.
# Chapter 1 → epoch + 1 day, chapter N → epoch + N days.
_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


# ── Async bridge ────────────────────────────────────────────────────

def _get_loop() -> asyncio.AbstractEventLoop:
    """Return a dedicated event loop for running Graphiti coroutines."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop


def _run_async(coro: Any) -> Any:
    """Execute *coro* on the module event loop (blocking)."""
    return _get_loop().run_until_complete(coro)


# ── Factory ─────────────────────────────────────────────────────────

def _is_available() -> bool:
    """Check whether Graphiti can be initialised with current config."""
    if not settings.graphiti_enabled:
        return False
    if not (settings.neo4j_uri and settings.neo4j_username):
        return False
    if not (settings.lightrag_llm_api_key and settings.lightrag_llm_base_url):
        return False
    return True


def get_graphiti():
    """Return the shared Graphiti instance, creating it on first call.

    Returns ``None`` if Graphiti is disabled or misconfigured.
    """
    global _graphiti_instance
    if _graphiti_instance is not None:
        return _graphiti_instance

    with _graphiti_lock:
        # Double-check under lock.
        if _graphiti_instance is not None:
            return _graphiti_instance

        if not _is_available():
            logger.info("graphiti: disabled or missing config — skipping init")
            return None

        try:
            _graphiti_instance = _build_graphiti()
            logger.info("graphiti: initialised successfully")
        except Exception:
            logger.exception("graphiti: failed to initialise")
            _graphiti_instance = None

    return _graphiti_instance


def _build_graphiti():
    """Construct and initialise a Graphiti instance."""
    # Late imports so the module loads even when graphiti-core is absent.
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_client import OpenAIClient

    llm_api_key = settings.lightrag_llm_api_key or settings.llm_api_key
    llm_base_url = settings.lightrag_llm_base_url or settings.llm_base_url
    llm_model = settings.lightrag_llm_model or settings.llm_model

    llm_config = LLMConfig(
        api_key=llm_api_key,
        base_url=llm_base_url,
        model=llm_model,
        small_model=llm_model,
        temperature=0.0,
    )
    llm_client = OpenAIClient(config=llm_config, reasoning="medium")

    embedder_config = OpenAIEmbedderConfig(
        api_key=settings.graphiti_embedding_api_key,
        base_url=settings.graphiti_embedding_base_url,
        embedding_model=settings.graphiti_embedding_model,
        embedding_dim=settings.graphiti_embedding_dim,
    )
    embedder = OpenAIEmbedder(config=embedder_config)

    # Cross-encoder: reuse LLM config (Graphiti uses it for reranking).
    cross_encoder = OpenAIRerankerClient(config=llm_config)

    driver = Neo4jDriver(
        uri=settings.neo4j_uri,
        user=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )

    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
        store_raw_episode_content=True,
    )

    # Create Neo4j indexes / constraints that Graphiti needs.
    _run_async(graphiti.build_indices_and_constraints())

    return graphiti


# ── Helpers ─────────────────────────────────────────────────────────

def _chapter_to_datetime(chapter_index: int) -> datetime:
    """Map a chapter index to a synthetic datetime for temporal ordering."""
    return _EPOCH + timedelta(days=max(chapter_index, 0))


def _group_id(project_id: int) -> str:
    return f"project-{project_id}"


def _format_episode_body(ep: dict[str, Any]) -> str:
    """Build a plain-text body from an episode payload dict."""
    parts: list[str] = []
    title = str(ep.get("title") or "").strip()
    if title:
        parts.append(f"## {title}")
    summary = str(ep.get("summary") or "").strip()
    if summary:
        parts.append(summary)
    participants = ep.get("participants") or []
    if participants:
        parts.append(f"参与角色: {', '.join(str(p) for p in participants)}")
    location = str(ep.get("location") or "").strip()
    if location:
        parts.append(f"地点: {location}")
    event_type = str(ep.get("event_type") or "").strip()
    if event_type:
        parts.append(f"事件类型: {event_type}")
    return "\n".join(parts)


# ── Public API ──────────────────────────────────────────────────────

def ingest_chapter_episodes(
    *,
    project_id: int,
    chapter_id: int,
    chapter_index: int,
    episodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Feed extracted story episodes into the Graphiti temporal graph.

    Each episode is ingested as a separate Graphiti episode with
    ``group_id`` scoped to the project and ``reference_time`` derived
    from ``chapter_index``.

    Returns a list of per-episode status dicts.
    """
    from graphiti_core.nodes import EpisodeType

    from app.services.book_memory.entity_types import (
        FICTION_ENTITY_TYPES,
        FICTION_EXTRACTION_INSTRUCTIONS,
    )

    graphiti = get_graphiti()
    if graphiti is None:
        return [{"status": "skipped", "reason": "graphiti_disabled"}]

    gid = _group_id(project_id)
    ref_time = _chapter_to_datetime(chapter_index)
    results: list[dict[str, Any]] = []

    for ep in episodes:
        ep_title = str(ep.get("title") or "未命名事件").strip()
        ep_index = int(ep.get("episode_index") or 0)
        body = _format_episode_body(ep)
        if not body.strip():
            results.append({"status": "skipped", "episode": ep_title, "reason": "empty_body"})
            continue

        try:
            _run_async(
                graphiti.add_episode(
                    name=ep_title,
                    episode_body=body,
                    source_description=f"chapter:{chapter_id}:episode:{ep_index}",
                    reference_time=ref_time,
                    source=EpisodeType.text,
                    group_id=gid,
                    entity_types=FICTION_ENTITY_TYPES,
                    custom_extraction_instructions=FICTION_EXTRACTION_INSTRUCTIONS,
                )
            )
            results.append({"status": "ok", "episode": ep_title})
        except Exception as exc:
            logger.warning("graphiti: ingest failed for '%s': %s", ep_title, exc)
            results.append({"status": "error", "episode": ep_title, "error": str(exc)})

    return results


def search_temporal_facts(
    *,
    project_id: int,
    query: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search the temporal knowledge graph for facts relevant to *query*.

    Returns a list of dicts with keys: ``name``, ``fact``, ``valid_at``,
    ``invalid_at``, ``source_description``, ``score``.
    """
    graphiti = get_graphiti()
    if graphiti is None:
        return []

    gid = _group_id(project_id)

    try:
        results = _run_async(
            graphiti.search(
                query=query,
                group_ids=[gid],
                num_results=limit,
            )
        )
    except Exception as exc:
        logger.warning("graphiti: search failed: %s", exc)
        return []

    return _normalize_search_results(results)


def search_character_knowledge(
    *,
    project_id: int,
    character_name: str,
    at_chapter: int | None = None,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Query what *character_name* knows, optionally bounded to *at_chapter*.

    Combines a Graphiti semantic search with post-hoc chapter filtering.
    """
    query = f"角色 {character_name} 知道的事实和关系"
    facts = search_temporal_facts(
        project_id=project_id,
        query=query,
        limit=limit * 2,  # over-fetch, then filter
    )

    if at_chapter is not None:
        chapter_dt = _chapter_to_datetime(at_chapter)
        filtered: list[dict[str, Any]] = []
        for f in facts:
            valid_at = f.get("valid_at")
            invalid_at = f.get("invalid_at")
            # Keep facts that are valid at the requested chapter.
            if valid_at and valid_at > chapter_dt:
                continue
            if invalid_at and invalid_at <= chapter_dt:
                continue
            filtered.append(f)
            if len(filtered) >= limit:
                break
        return filtered

    return facts[:limit]


# ── Result normalisation ────────────────────────────────────────────

def _normalize_search_results(results: Any) -> list[dict[str, Any]]:
    """Convert Graphiti SearchResults into plain dicts."""
    out: list[dict[str, Any]] = []

    # SearchResults has .edges and .nodes attributes.
    edges = getattr(results, "edges", None) or []
    for edge in edges:
        out.append({
            "type": "edge",
            "name": getattr(edge, "name", ""),
            "fact": getattr(edge, "fact", getattr(edge, "name", "")),
            "valid_at": getattr(edge, "valid_at", None),
            "invalid_at": getattr(edge, "invalid_at", None),
            "source_description": getattr(edge, "source_description", ""),
            "uuid": getattr(edge, "uuid", ""),
        })

    nodes = getattr(results, "nodes", None) or []
    for node in nodes:
        out.append({
            "type": "node",
            "name": getattr(node, "name", ""),
            "fact": getattr(node, "summary", getattr(node, "name", "")),
            "valid_at": None,
            "invalid_at": None,
            "source_description": "",
            "uuid": getattr(node, "uuid", ""),
        })

    return out


# ── Teardown ────────────────────────────────────────────────────────

def close_graphiti() -> None:
    """Gracefully close the Graphiti driver (call on process shutdown)."""
    global _graphiti_instance
    if _graphiti_instance is not None:
        try:
            _run_async(_graphiti_instance.close())
        except Exception:
            logger.exception("graphiti: error during close")
        finally:
            _graphiti_instance = None
