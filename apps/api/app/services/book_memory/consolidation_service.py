from __future__ import annotations

import logging
from typing import Any

from sqlmodel import Session, select

from app.models.book_memory import (
    CharacterKnowledgeState,
    MemoryMaterialization,
    StoryEpisode,
    StoryStateSnapshot,
)
from app.models.content import ChapterSceneBeat, ProjectChapter
from app.services.book_memory.character_service import list_character_profiles
from app.services.book_memory.episode_extractor import extract_story_episode_candidates
from app.services.book_memory.extraction_service import extract_book_memory_structured
from app.services.book_memory.story_state_compiler import compile_story_state_payload
from app.services.chat_service.chapters import get_project_chapter

logger = logging.getLogger(__name__)


def _list_chapter_scene_beats(db: Session, *, project_id: int, chapter_id: int) -> list[ChapterSceneBeat]:
    stmt = (
        select(ChapterSceneBeat)
        .where(
            ChapterSceneBeat.project_id == project_id,
            ChapterSceneBeat.chapter_id == chapter_id,
        )
        .order_by(ChapterSceneBeat.beat_index.asc(), ChapterSceneBeat.id.asc())
    )
    return db.exec(stmt).all()


def _get_previous_chapter_snapshot(
    db: Session,
    *,
    project_id: int,
    chapter: ProjectChapter,
) -> StoryStateSnapshot | None:
    stmt = (
        select(ProjectChapter)
        .where(
            ProjectChapter.project_id == project_id,
            ProjectChapter.chapter_index < int(chapter.chapter_index),
        )
        .order_by(ProjectChapter.chapter_index.desc(), ProjectChapter.id.desc())
        .limit(1)
    )
    previous_chapter = db.exec(stmt).first()
    if previous_chapter is None:
        return None

    snapshot_stmt = (
        select(StoryStateSnapshot)
        .where(
            StoryStateSnapshot.project_id == project_id,
            StoryStateSnapshot.chapter_id == int(previous_chapter.id or 0),
            StoryStateSnapshot.scene_beat_id.is_(None),
        )
        .order_by(StoryStateSnapshot.updated_at.desc(), StoryStateSnapshot.id.desc())
        .limit(1)
    )
    return db.exec(snapshot_stmt).first()


def _replace_story_episodes(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    episodes: list[dict[str, Any]],
) -> list[StoryEpisode]:
    existing_stmt = select(StoryEpisode).where(
        StoryEpisode.project_id == project_id,
        StoryEpisode.chapter_id == chapter_id,
    )
    for row in db.exec(existing_stmt).all():
        db.delete(row)
    db.flush()

    persisted: list[StoryEpisode] = []
    for item in episodes:
        row = StoryEpisode(
            project_id=project_id,
            chapter_id=chapter_id,
            scene_beat_id=item.get("scene_beat_id"),
            episode_index=int(item.get("episode_index") or (len(persisted) + 1)),
            title=str(item.get("title") or "").strip() or "未命名事件",
            summary=str(item.get("summary") or "").strip(),
            event_type=str(item.get("event_type") or "scene").strip() or "scene",
            participants=list(item.get("participants") or []),
            location=str(item.get("location") or "").strip(),
            visibility=str(item.get("visibility") or "public").strip() or "public",
            importance=int(item.get("importance") or 50),
            source_text_ref=dict(item.get("source_text_ref") or {}),
        )
        db.add(row)
        persisted.append(row)
    db.flush()
    for row in persisted:
        db.refresh(row)
    return persisted


def _upsert_story_state_snapshot(
    db: Session,
    *,
    project_id: int,
    payload: dict[str, Any],
) -> StoryStateSnapshot:
    stmt = (
        select(StoryStateSnapshot)
        .where(
            StoryStateSnapshot.project_id == project_id,
            StoryStateSnapshot.chapter_id == payload.get("chapter_id"),
            StoryStateSnapshot.scene_beat_id.is_(None),
        )
        .order_by(StoryStateSnapshot.updated_at.desc(), StoryStateSnapshot.id.desc())
        .limit(1)
    )
    snapshot = db.exec(stmt).first()
    if snapshot is None:
        snapshot = StoryStateSnapshot(
            project_id=project_id,
            volume_id=payload.get("volume_id"),
            chapter_id=payload.get("chapter_id"),
            scene_beat_id=None,
        )
    snapshot.volume_id = payload.get("volume_id")
    snapshot.chapter_id = payload.get("chapter_id")
    snapshot.scene_beat_id = None
    snapshot.chapter_goal = str(payload.get("chapter_goal") or "").strip()
    snapshot.active_characters = list(payload.get("active_characters") or [])
    snapshot.current_location = str(payload.get("current_location") or "").strip()
    snapshot.active_conflicts = list(payload.get("active_conflicts") or [])
    snapshot.open_questions = list(payload.get("open_questions") or [])
    snapshot.source_refs = list(payload.get("source_refs") or [])
    db.add(snapshot)
    db.flush()
    db.refresh(snapshot)
    return snapshot


def _replace_chapter_knowledge_states(
    db: Session,
    *,
    project_id: int,
    chapter_index: int,
    knowledge_updates: list[dict[str, Any]],
    character_id_by_name: dict[str, int],
    episode_id_by_index: dict[int, int],
) -> list[CharacterKnowledgeState]:
    existing_stmt = select(CharacterKnowledgeState).where(
        CharacterKnowledgeState.project_id == project_id,
        CharacterKnowledgeState.gained_at_chapter == chapter_index,
    )
    for row in db.exec(existing_stmt).all():
        db.delete(row)
    db.flush()

    persisted: list[CharacterKnowledgeState] = []
    for item in knowledge_updates:
        character_name = str(item.get("character_name") or "").strip()
        character_profile_id = character_id_by_name.get(character_name)
        if not character_profile_id:
            continue
        episode_index = int(item.get("source_episode_index") or 0) or None
        row = CharacterKnowledgeState(
            project_id=project_id,
            character_profile_id=character_profile_id,
            knowledge_key=str(item.get("knowledge_key") or "").strip(),
            knowledge_value=dict(item.get("knowledge_value") or {}),
            gained_at_chapter=chapter_index,
            lost_at_chapter=item.get("lost_at_chapter"),
            source_episode_id=episode_id_by_index.get(episode_index or 0),
            confidence=max(0.0, min(float(item.get("confidence") or 0.8), 1.0)),
        )
        if not row.knowledge_key:
            continue
        db.add(row)
        persisted.append(row)
    db.flush()
    for row in persisted:
        db.refresh(row)
    return persisted


def _upsert_materialization(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    payload: dict[str, Any],
    episodes: list[StoryEpisode],
    knowledge_states: list[CharacterKnowledgeState],
) -> MemoryMaterialization:
    scope_key = f"chapter:{chapter_id}"
    stmt = (
        select(MemoryMaterialization)
        .where(
            MemoryMaterialization.project_id == project_id,
            MemoryMaterialization.materialization_type == "book_memory_chapter_pack",
            MemoryMaterialization.scope_key == scope_key,
        )
        .order_by(MemoryMaterialization.updated_at.desc(), MemoryMaterialization.id.desc())
        .limit(1)
    )
    materialization = db.exec(stmt).first()
    if materialization is None:
        materialization = MemoryMaterialization(
            project_id=project_id,
            materialization_type="book_memory_chapter_pack",
            scope_key=scope_key,
        )
    materialization.payload = {
        "chapter_goal": payload.get("chapter_goal"),
        "active_characters": list(payload.get("active_characters") or []),
        "current_location": payload.get("current_location"),
        "active_conflicts": list(payload.get("active_conflicts") or []),
        "open_questions": list(payload.get("open_questions") or []),
        "episode_ids": [int(item.id or 0) for item in episodes if int(item.id or 0) > 0],
        "knowledge_state_ids": [int(item.id or 0) for item in knowledge_states if int(item.id or 0) > 0],
    }
    materialization.source_versions = {
        "chapter_version": int(payload.get("chapter_version") or 0),
        "episode_count": len(episodes),
        "knowledge_count": len(knowledge_states),
    }
    db.add(materialization)
    db.flush()
    db.refresh(materialization)
    return materialization


def run_book_memory_consolidation(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    operator_id: str = "system",
    reason: str = "book_memory_consolidation",
    scene_beat_id: int | None = None,
) -> dict[str, Any]:
    chapter = get_project_chapter(db, project_id, chapter_id)
    if chapter is None or chapter.id is None:
        raise ValueError("chapter not found")

    scene_beats = _list_chapter_scene_beats(db, project_id=project_id, chapter_id=chapter_id)
    if scene_beat_id is not None and not any(int(item.id or 0) == int(scene_beat_id) for item in scene_beats):
        raise ValueError("scene beat not found")

    character_profiles = list_character_profiles(db, project_id)
    previous_snapshot = _get_previous_chapter_snapshot(db, project_id=project_id, chapter=chapter)
    structured = extract_book_memory_structured(
        project_id=project_id,
        chapter=chapter,
        scene_beats=scene_beats,
        character_profiles=character_profiles,
        previous_snapshot=previous_snapshot,
    )
    episode_payloads = extract_story_episode_candidates(
        project_id=project_id,
        chapter=chapter,
        scene_beats=scene_beats,
        character_profiles=character_profiles,
        previous_snapshot=previous_snapshot,
        extraction_result=structured,
    )
    story_state_payload = compile_story_state_payload(
        project_id=project_id,
        chapter=chapter,
        scene_beats=scene_beats,
        character_profiles=character_profiles,
        previous_snapshot=previous_snapshot,
        episodes=episode_payloads,
        extraction_result=structured,
    )
    story_state_payload["chapter_version"] = int(chapter.version)
    story_state_payload["operator_id"] = str(operator_id or "system")
    story_state_payload["reason"] = str(reason or "book_memory_consolidation")
    story_state_payload["scene_beat_id"] = int(scene_beat_id) if scene_beat_id else None

    episode_rows = _replace_story_episodes(
        db,
        project_id=project_id,
        chapter_id=int(chapter.id),
        episodes=episode_payloads,
    )
    snapshot = _upsert_story_state_snapshot(
        db,
        project_id=project_id,
        payload=story_state_payload,
    )
    character_id_by_name = {
        str(item.canonical_name or "").strip(): int(item.id or 0)
        for item in character_profiles
        if int(item.id or 0) > 0 and str(item.canonical_name or "").strip()
    }
    episode_id_by_index = {
        int(item.episode_index): int(item.id or 0)
        for item in episode_rows
        if int(item.id or 0) > 0
    }
    knowledge_rows = _replace_chapter_knowledge_states(
        db,
        project_id=project_id,
        chapter_index=int(chapter.chapter_index),
        knowledge_updates=list(story_state_payload.get("knowledge_updates") or []),
        character_id_by_name=character_id_by_name,
        episode_id_by_index=episode_id_by_index,
    )
    materialization = _upsert_materialization(
        db,
        project_id=project_id,
        chapter_id=int(chapter.id),
        payload=story_state_payload,
        episodes=episode_rows,
        knowledge_states=knowledge_rows,
    )
    db.commit()

    # ── Phase A: Inject episodes into Graphiti temporal graph ────────
    graphiti_results: list[dict[str, Any]] = []
    try:
        from app.services.book_memory.graphiti_adapter import ingest_chapter_episodes

        graphiti_results = ingest_chapter_episodes(
            project_id=project_id,
            chapter_id=int(chapter.id),
            chapter_index=int(chapter.chapter_index),
            episodes=episode_payloads,
        )
    except Exception:
        logger.warning(
            "graphiti ingest skipped for chapter %s (project %s)",
            chapter_id,
            project_id,
            exc_info=True,
        )

    return {
        "status": "ok",
        "project_id": project_id,
        "chapter_id": int(chapter.id),
        "scene_beat_id": int(scene_beat_id) if scene_beat_id else None,
        "snapshot_id": int(snapshot.id or 0),
        "episode_count": len(episode_rows),
        "knowledge_count": len(knowledge_rows),
        "materialization_id": int(materialization.id or 0),
        "graphiti_ingest": graphiti_results,
    }
