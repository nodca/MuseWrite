"""Unified temporal query layer for Book Memory OS.

Combines **Postgres** structured memory (L1-L3) with **Graphiti** temporal
graph facts (L4) into a single query surface.  Every query function accepts
an optional ``at_chapter`` parameter to scope truth to a narrative time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import or_
from sqlmodel import Session, select

from app.models.book_memory import (
    CharacterKnowledgeState,
    CharacterProfile,
    StoryEpisode,
    StoryStateSnapshot,
    WorldRule,
)

logger = logging.getLogger(__name__)


# ── Result types ────────────────────────────────────────────────────


@dataclass
class TemporalFact:
    """A single fact with an optional validity window."""

    subject: str = ""
    predicate: str = ""
    object: str = ""
    valid_from_chapter: int | None = None
    valid_to_chapter: int | None = None
    confidence: float = 1.0
    source: str = ""  # "postgres" | "graphiti"
    evidence: str = ""


@dataclass
class CharacterKnowledgeBundle:
    """What a character knows (and does not know) at a given chapter."""

    character_name: str = ""
    aliases: list[str] = field(default_factory=list)
    known_facts: list[TemporalFact] = field(default_factory=list)
    withheld_facts: list[TemporalFact] = field(default_factory=list)
    core_goals: list[str] = field(default_factory=list)
    public_traits: list[str] = field(default_factory=list)


# ── L1: Story State ────────────────────────────────────────────────


def query_story_state(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    scene_beat_id: int | None = None,
) -> dict[str, Any]:
    """Return the latest L1 story state snapshot for a chapter."""
    stmt = (
        select(StoryStateSnapshot)
        .where(
            StoryStateSnapshot.project_id == project_id,
            StoryStateSnapshot.chapter_id == chapter_id,
        )
    )
    if scene_beat_id is not None:
        stmt = stmt.where(StoryStateSnapshot.scene_beat_id == scene_beat_id)
    else:
        stmt = stmt.where(StoryStateSnapshot.scene_beat_id.is_(None))

    stmt = stmt.order_by(
        StoryStateSnapshot.updated_at.desc(),
        StoryStateSnapshot.id.desc(),
    ).limit(1)

    snapshot = db.exec(stmt).first()
    if snapshot is None:
        return {}

    return {
        "chapter_goal": str(snapshot.chapter_goal or ""),
        "active_characters": list(snapshot.active_characters or []),
        "current_location": str(snapshot.current_location or ""),
        "active_conflicts": list(snapshot.active_conflicts or []),
        "open_questions": list(snapshot.open_questions or []),
    }


# ── L2: Episodic Memory ────────────────────────────────────────────


def query_recent_episodes(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Return recent L2 story episodes for a chapter, newest first."""
    stmt = (
        select(StoryEpisode)
        .where(
            StoryEpisode.project_id == project_id,
            StoryEpisode.chapter_id == chapter_id,
        )
        .order_by(StoryEpisode.episode_index.asc(), StoryEpisode.id.asc())
        .limit(max(limit, 1))
    )
    rows = db.exec(stmt).all()

    return [
        {
            "title": str(getattr(r, "title", "") or ""),
            "summary": str(getattr(r, "summary", "") or "")[:280],
            "event_type": str(getattr(r, "event_type", "") or ""),
            "participants": list(getattr(r, "participants", []) or []),
            "location": str(getattr(r, "location", "") or ""),
            "importance": int(getattr(r, "importance", 0) or 0),
            "episode_index": int(getattr(r, "episode_index", 0) or 0),
        }
        for r in rows
    ]


# ── L3: Canon Memory ───────────────────────────────────────────────


def query_active_world_rules(
    db: Session,
    *,
    project_id: int,
    status: str = "active",
) -> list[dict[str, Any]]:
    """Return active L3 world rules."""
    stmt = (
        select(WorldRule)
        .where(WorldRule.project_id == project_id, WorldRule.status == status)
        .order_by(WorldRule.priority.asc(), WorldRule.id.asc())
    )
    rows = db.exec(stmt).all()

    return [
        {
            "title": str(r.title or ""),
            "statement": str(r.statement or ""),
            "scope": str(r.scope or ""),
            "priority": int(r.priority or 100),
        }
        for r in rows
    ]


def query_character_profiles(
    db: Session,
    *,
    project_id: int,
    names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return L3 character profiles, optionally filtered by name."""
    stmt = select(CharacterProfile).where(CharacterProfile.project_id == project_id)
    stmt = stmt.order_by(CharacterProfile.id.asc())
    rows = db.exec(stmt).all()

    if names:
        name_set = {n.lower() for n in names}
        rows = [
            r
            for r in rows
            if str(r.canonical_name or "").lower() in name_set
            or any(str(a or "").lower() in name_set for a in (r.aliases or []))
        ]

    return [
        {
            "canonical_name": str(r.canonical_name or ""),
            "aliases": list(r.aliases or []),
            "core_goals": list(r.core_goals or []),
            "public_traits": list(r.public_traits or []),
            "private_traits": list(r.private_traits or []),
            "fears": list(r.fears or []),
            "taboos": list(r.taboos or []),
            "voice_notes": str(r.default_voice_notes or ""),
        }
        for r in rows
    ]


# ── L3+L4: Character Knowledge ─────────────────────────────────────


def query_character_knowledge_at_chapter(
    db: Session,
    *,
    project_id: int,
    character_name: str,
    at_chapter: int,
    facts_limit: int = 12,
) -> CharacterKnowledgeBundle:
    """What does *character_name* know at *at_chapter*?

    Merges Postgres ``CharacterKnowledgeState`` (L3) with Graphiti
    temporal graph facts (L4) for a unified view.
    """
    # ── Resolve character profile ──
    profile_stmt = select(CharacterProfile).where(
        CharacterProfile.project_id == project_id,
    )
    profiles = db.exec(profile_stmt).all()

    target_profile: CharacterProfile | None = None
    for p in profiles:
        if str(p.canonical_name or "").lower() == character_name.lower():
            target_profile = p
            break
        if any(str(a or "").lower() == character_name.lower() for a in (p.aliases or [])):
            target_profile = p
            break

    if target_profile is None or target_profile.id is None:
        return CharacterKnowledgeBundle(character_name=character_name)

    # ── L3: Postgres knowledge states ──
    pg_stmt = select(CharacterKnowledgeState).where(
        CharacterKnowledgeState.project_id == project_id,
        CharacterKnowledgeState.character_profile_id == int(target_profile.id),
        CharacterKnowledgeState.gained_at_chapter.is_not(None),
        CharacterKnowledgeState.gained_at_chapter <= at_chapter,
        or_(
            CharacterKnowledgeState.lost_at_chapter.is_(None),
            CharacterKnowledgeState.lost_at_chapter > at_chapter,
        ),
    ).order_by(
        CharacterKnowledgeState.gained_at_chapter.desc(),
        CharacterKnowledgeState.id.desc(),
    ).limit(facts_limit)

    pg_rows = db.exec(pg_stmt).all()

    known_facts: list[TemporalFact] = []
    for row in pg_rows:
        value = row.knowledge_value if isinstance(row.knowledge_value, dict) else {}
        fact_text = str(value.get("fact") or row.knowledge_key or "").strip()
        if not fact_text:
            continue
        known_facts.append(
            TemporalFact(
                subject=character_name,
                predicate="knows",
                object=fact_text[:200],
                valid_from_chapter=int(row.gained_at_chapter) if row.gained_at_chapter else None,
                valid_to_chapter=int(row.lost_at_chapter) if row.lost_at_chapter else None,
                confidence=float(row.confidence or 0.8),
                source="postgres",
                evidence=str(value.get("evidence") or ""),
            )
        )

    # ── L4: Graphiti temporal facts ──
    graphiti_facts: list[TemporalFact] = []
    try:
        from app.services.book_memory.graphiti_adapter import search_character_knowledge

        raw = search_character_knowledge(
            project_id=project_id,
            character_name=character_name,
            at_chapter=at_chapter,
            limit=facts_limit,
        )
        for item in raw:
            graphiti_facts.append(
                TemporalFact(
                    subject=character_name,
                    predicate=str(item.get("name") or "related_to"),
                    object=str(item.get("fact") or ""),
                    confidence=0.7,
                    source="graphiti",
                )
            )
    except Exception:
        logger.debug("graphiti knowledge query skipped", exc_info=True)

    # ── Merge: Postgres facts win on duplicates ──
    seen_objects = {f.object.lower() for f in known_facts}
    for gf in graphiti_facts:
        if gf.object.lower() not in seen_objects:
            known_facts.append(gf)
            seen_objects.add(gf.object.lower())
        if len(known_facts) >= facts_limit:
            break

    # ── Withheld: facts known by OTHER characters but not this one ──
    withheld_facts = _compute_withheld_facts(
        db,
        project_id=project_id,
        exclude_profile_id=int(target_profile.id),
        at_chapter=at_chapter,
        known_objects=seen_objects,
        limit=6,
    )

    return CharacterKnowledgeBundle(
        character_name=str(target_profile.canonical_name or character_name),
        aliases=list(target_profile.aliases or []),
        known_facts=known_facts[:facts_limit],
        withheld_facts=withheld_facts,
        core_goals=list(target_profile.core_goals or []),
        public_traits=list(target_profile.public_traits or []),
    )


# ── L4: Graphiti temporal search ────────────────────────────────────


def query_temporal_facts(
    *,
    project_id: int,
    query: str,
    limit: int = 10,
) -> list[TemporalFact]:
    """Search the L4 Graphiti temporal graph for relevant facts."""
    try:
        from app.services.book_memory.graphiti_adapter import search_temporal_facts

        raw = search_temporal_facts(project_id=project_id, query=query, limit=limit)
    except Exception:
        logger.debug("graphiti temporal query failed", exc_info=True)
        return []

    return [
        TemporalFact(
            subject="",
            predicate=str(item.get("name") or ""),
            object=str(item.get("fact") or ""),
            confidence=0.7,
            source="graphiti",
        )
        for item in raw
        if str(item.get("fact") or "").strip()
    ]


# ── Helpers ─────────────────────────────────────────────────────────


def _compute_withheld_facts(
    db: Session,
    *,
    project_id: int,
    exclude_profile_id: int,
    at_chapter: int,
    known_objects: set[str],
    limit: int = 6,
) -> list[TemporalFact]:
    """Find facts known by other characters but NOT by the target."""
    stmt = select(CharacterKnowledgeState).where(
        CharacterKnowledgeState.project_id == project_id,
        CharacterKnowledgeState.character_profile_id != exclude_profile_id,
        CharacterKnowledgeState.gained_at_chapter.is_not(None),
        CharacterKnowledgeState.gained_at_chapter <= at_chapter,
        or_(
            CharacterKnowledgeState.lost_at_chapter.is_(None),
            CharacterKnowledgeState.lost_at_chapter > at_chapter,
        ),
    ).order_by(
        CharacterKnowledgeState.gained_at_chapter.desc(),
    ).limit(limit * 3)

    rows = db.exec(stmt).all()
    withheld: list[TemporalFact] = []
    for row in rows:
        value = row.knowledge_value if isinstance(row.knowledge_value, dict) else {}
        fact_text = str(value.get("fact") or row.knowledge_key or "").strip()
        if not fact_text or fact_text.lower() in known_objects:
            continue
        withheld.append(
            TemporalFact(
                subject="other_character",
                predicate="knows_but_target_does_not",
                object=fact_text[:200],
                valid_from_chapter=int(row.gained_at_chapter) if row.gained_at_chapter else None,
                confidence=float(row.confidence or 0.8),
                source="postgres",
            )
        )
        if len(withheld) >= limit:
            break

    return withheld
