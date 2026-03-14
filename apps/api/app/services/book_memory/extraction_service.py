"""Structured memory extraction from chapter content.

Uses **Instructor** (when available) for type-safe structured LLM output
with automatic retry/validation.  Falls back to the legacy
``generate_structured_sync`` path if Instructor cannot be loaded.
"""

from __future__ import annotations

import json
import logging
from typing import Sequence

from pydantic import BaseModel, Field

from app.core.config import settings
from app.models.book_memory import CharacterProfile, StoryStateSnapshot
from app.models.content import ChapterSceneBeat, ProjectChapter

logger = logging.getLogger(__name__)

# ── Output schemas ──────────────────────────────────────────────────


class BookMemoryEpisodeExtraction(BaseModel):
    beat_index: int | None = Field(default=None, ge=1)
    scene_beat_id: int | None = Field(default=None, ge=1)
    title: str = Field(default="", max_length=255)
    summary: str = Field(default="", max_length=500)
    event_type: str = Field(default="scene", max_length=64)
    participants: list[str] = Field(default_factory=list, max_length=12)
    location: str = Field(default="", max_length=191)
    visibility: str = Field(default="public", max_length=32)
    importance: int = Field(default=50, ge=0, le=100)
    source_excerpt: str = Field(default="", max_length=220)


class BookMemoryKnowledgeClaimExtraction(BaseModel):
    character_name: str = Field(default="", max_length=191)
    fact: str = Field(default="", max_length=240)
    known: bool = True
    source_episode_index: int | None = Field(default=None, ge=1)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class BookMemoryStructuredExtraction(BaseModel):
    chapter_goal: str = Field(default="", max_length=191)
    active_characters: list[str] = Field(default_factory=list, max_length=16)
    current_location: str = Field(default="", max_length=191)
    active_conflicts: list[str] = Field(default_factory=list, max_length=8)
    open_questions: list[str] = Field(default_factory=list, max_length=8)
    episodes: list[BookMemoryEpisodeExtraction] = Field(default_factory=list, max_length=16)
    knowledge_claims: list[BookMemoryKnowledgeClaimExtraction] = Field(
        default_factory=list, max_length=24
    )


# ── System prompt ───────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "你是小说 Book Memory 抽取器。"
    "请从章节与 scene beats 中提取结构化记忆，严格输出符合 schema 的 JSON。"
    "目标不是摘要，而是为长期 memory OS 生成可验证的事件、剧情状态、以及「角色知道什么」的记忆。"
    "要求："
    "1. 只抽取文本明确表达或强烈暗示的内容，禁止编造。"
    "2. active_characters 使用提供的 canonical_name。"
    "3. knowledge_claims 只记录角色已经知道/发现/确认的事实，不要写旁白全知信息。"
    "4. source_episode_index 指向 episodes 的 1-based 顺序。"
    "5. 如果不确定，返回空字段或空数组。"
)


# ── Config helpers ──────────────────────────────────────────────────


def _book_memory_runtime_config() -> dict[str, str] | None:
    model = str(settings.lightrag_llm_model or "").strip()
    base_url = str(settings.lightrag_llm_base_url or "").strip()
    api_key = str(settings.lightrag_llm_api_key or "").strip()
    if not (model and base_url and api_key):
        return None
    return {
        "provider": "openai_compatible",
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }


def _build_payload_text(
    *,
    project_id: int,
    chapter: ProjectChapter,
    scene_beats: Sequence[ChapterSceneBeat] | None,
    character_profiles: Sequence[CharacterProfile] | None,
    previous_snapshot: StoryStateSnapshot | None,
) -> str:
    beats_payload = [
        {
            "scene_beat_id": int(item.id or 0) or None,
            "beat_index": int(item.beat_index),
            "content": str(item.content or "").strip()[:1200],
        }
        for item in sorted(
            list(scene_beats or []),
            key=lambda row: (int(row.beat_index), int(row.id or 0)),
        )
        if str(item.content or "").strip()
    ]
    profile_payload = [
        {
            "canonical_name": str(item.canonical_name or "").strip(),
            "aliases": [
                str(a or "").strip()
                for a in list(item.aliases or [])
                if str(a or "").strip()
            ],
            "core_goals": [
                str(g or "").strip()
                for g in list(item.core_goals or [])
                if str(g or "").strip()
            ][:4],
            "public_traits": [
                str(t or "").strip()
                for t in list(item.public_traits or [])
                if str(t or "").strip()
            ][:4],
        }
        for item in list(character_profiles or [])
        if str(item.canonical_name or "").strip()
    ]
    previous_payload = (
        {
            "chapter_goal": str(previous_snapshot.chapter_goal or "").strip(),
            "active_characters": list(previous_snapshot.active_characters or []),
            "current_location": str(previous_snapshot.current_location or "").strip(),
            "active_conflicts": list(previous_snapshot.active_conflicts or []),
            "open_questions": list(previous_snapshot.open_questions or []),
        }
        if previous_snapshot is not None
        else None
    )
    return json.dumps(
        {
            "project_id": int(project_id),
            "chapter": {
                "chapter_id": int(chapter.id or 0) or None,
                "volume_id": int(chapter.volume_id or 0) or None,
                "chapter_index": int(chapter.chapter_index),
                "title": str(chapter.title or "").strip(),
                "content": str(chapter.content or "").strip()[:4000],
            },
            "scene_beats": beats_payload,
            "character_profiles": profile_payload,
            "previous_story_state": previous_payload,
        },
        ensure_ascii=False,
    )


# ── Public API ──────────────────────────────────────────────────────


def extract_book_memory_structured(
    *,
    project_id: int,
    chapter: ProjectChapter,
    scene_beats: Sequence[ChapterSceneBeat] | None = None,
    character_profiles: Sequence[CharacterProfile] | None = None,
    previous_snapshot: StoryStateSnapshot | None = None,
) -> BookMemoryStructuredExtraction | None:
    """Extract structured memory from a chapter using Instructor.

    Returns ``None`` if LLM config is missing or extraction fails.
    """
    runtime_cfg = _book_memory_runtime_config()
    if runtime_cfg is None:
        return None

    payload_text = _build_payload_text(
        project_id=project_id,
        chapter=chapter,
        scene_beats=scene_beats,
        character_profiles=character_profiles,
        previous_snapshot=previous_snapshot,
    )

    import instructor
    from openai import OpenAI

    client = instructor.from_openai(
        OpenAI(
            api_key=runtime_cfg["api_key"],
            base_url=runtime_cfg["base_url"],
            timeout=float(settings.memory_consolidation_llm_timeout_seconds),
        ),
        mode=instructor.Mode.JSON,
    )

    try:
        return client.chat.completions.create(
            model=runtime_cfg["model"],
            response_model=BookMemoryStructuredExtraction,
            max_retries=2,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": payload_text},
            ],
        )
    except Exception:
        logger.exception("structured extraction failed")
        return None
