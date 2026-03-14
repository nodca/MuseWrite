from __future__ import annotations

import logging
import threading
from typing import Any

_logger = logging.getLogger(__name__)


def _lightrag_enabled() -> bool:
    try:
        from app.core.config import settings
        return bool(settings.lightrag_enabled and settings.lightrag_base_url)
    except Exception:
        return False


def _ingest_chapter_background(
    *,
    project_id: int,
    chapter_id: int,
    title: str,
    content: str,
) -> None:
    try:
        from app.services.lightrag_documents import insert_text_document
        text = f"{title}\n\n{content}".strip()
        if not text:
            return
        file_source = f"np://project/{project_id}/chapter-{chapter_id}.txt"
        insert_text_document(
            project_id=project_id,
            text=text,
            file_source=file_source,
            enable_scene_chunking=True,
        )
        _logger.debug("lightrag_ingestion: chapter %s/%s ok", project_id, chapter_id)
    except Exception as exc:
        _logger.warning("lightrag_ingestion: chapter %s/%s failed: %s", project_id, chapter_id, exc)


def _ingest_storycard_background(
    *,
    project_id: int,
    card_id: int,
    title: str,
    content: dict[str, Any],
    aliases: list[str] | None = None,
) -> None:
    try:
        from app.services.lightrag_documents import insert_storycard_document
        insert_storycard_document(
            project_id=project_id,
            card_id=card_id,
            title=title,
            content=content,
            aliases=aliases,
        )
        _logger.debug("lightrag_ingestion: card %s/%s ok", project_id, card_id)
    except Exception as exc:
        _logger.warning("lightrag_ingestion: card %s/%s failed: %s", project_id, card_id, exc)


def enqueue_chapter_ingestion(
    *,
    project_id: int,
    chapter_id: int,
    title: str,
    content: str,
) -> None:
    """Async fire-and-forget ingestion for a chapter. No-op if LightRAG disabled."""
    if not _lightrag_enabled():
        return
    if not (content or "").strip():
        return
    threading.Thread(
        target=_ingest_chapter_background,
        kwargs={
            "project_id": project_id,
            "chapter_id": chapter_id,
            "title": title,
            "content": content,
        },
        daemon=True,
        name=f"lightrag-chapter-{project_id}-{chapter_id}",
    ).start()


def enqueue_storycard_ingestion(
    *,
    project_id: int,
    card_id: int,
    title: str,
    content: dict[str, Any],
    aliases: list[str] | None = None,
) -> None:
    """Async fire-and-forget ingestion for a StoryCard. No-op if LightRAG disabled."""
    if not _lightrag_enabled():
        return
    threading.Thread(
        target=_ingest_storycard_background,
        kwargs={
            "project_id": project_id,
            "card_id": card_id,
            "title": title,
            "content": content,
            "aliases": aliases,
        },
        daemon=True,
        name=f"lightrag-card-{project_id}-{card_id}",
    ).start()
