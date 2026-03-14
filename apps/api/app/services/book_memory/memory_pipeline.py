"""Memory-first context assembly pipeline for Book Memory OS.

Replaces the retrieval-centric ``context_compiler/pipeline.py`` with a
**memory-first** architecture where structured memory layers (L1-L4) are
the primary truth source and LightRAG (L5) is a cold fallback.

Authority order::

    L1 Active Writing State  >  L4 Temporal Graph  >  L3 Canon Memory
    >  L2 Episodic Memory  >  L5 Cold Retrieval

Switching is controlled by the ``USE_MEMORY_PIPELINE`` feature flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlmodel import Session

from app.core.config import settings
from app.services.book_memory.temporal_query import (
    CharacterKnowledgeBundle,
    TemporalFact,
    query_active_world_rules,
    query_character_knowledge_at_chapter,
    query_character_profiles,
    query_recent_episodes,
    query_story_state,
    query_temporal_facts,
)

logger = logging.getLogger(__name__)


# ── Result types ────────────────────────────────────────────────────


@dataclass
class MemoryLayer:
    """Tracks which layer contributed what, for Context X-Ray."""

    layer: str  # "L1", "L2", "L3", "L4", "L5"
    label: str  # human-readable label
    items: int = 0
    tokens_estimate: int = 0


@dataclass
class MemoryContext:
    """Full memory context assembled by the pipeline."""

    # L1 — Active Writing State
    story_state: dict[str, Any] = field(default_factory=dict)

    # L2 — Episodic Memory
    recent_episodes: list[dict[str, Any]] = field(default_factory=list)

    # L3 — Canon Memory
    world_rules: list[dict[str, Any]] = field(default_factory=list)
    character_profiles: list[dict[str, Any]] = field(default_factory=list)

    # L3+L4 — Character Knowledge (merged Postgres + Graphiti)
    character_knowledge: list[CharacterKnowledgeBundle] = field(default_factory=list)

    # L4 — Temporal Graph (Graphiti)
    temporal_facts: list[TemporalFact] = field(default_factory=list)

    # L5 — Cold Retrieval (LightRAG, only on fallback)
    cold_retrieval: list[dict[str, Any]] = field(default_factory=list)

    # Provenance
    layers_used: list[MemoryLayer] = field(default_factory=list)
    l5_triggered: bool = False

    def to_prompt_sections(self) -> list[dict[str, str]]:
        """Serialise memory context into prompt-ready sections."""
        sections: list[dict[str, str]] = []

        # L1 — Story State
        if self.story_state:
            lines = []
            goal = self.story_state.get("chapter_goal", "")
            if goal:
                lines.append(f"当前章节目标：{goal}")
            loc = self.story_state.get("current_location", "")
            if loc:
                lines.append(f"当前地点：{loc}")
            chars = self.story_state.get("active_characters", [])
            if chars:
                lines.append(f"活跃角色：{'、'.join(str(c) for c in chars)}")
            conflicts = self.story_state.get("active_conflicts", [])
            if conflicts:
                lines.append(f"活跃冲突：{'；'.join(str(c) for c in conflicts[:4])}")
            questions = self.story_state.get("open_questions", [])
            if questions:
                lines.append(f"未解问题：{'；'.join(str(q) for q in questions[:3])}")
            if lines:
                sections.append({"label": "当前写作状态", "content": "\n".join(lines)})

        # L3 — World Rules
        if self.world_rules:
            rule_lines = [
                f"- [{r.get('scope', '')}] {r.get('title', '')}: {r.get('statement', '')}"
                for r in self.world_rules[:8]
            ]
            sections.append({"label": "世界规则", "content": "\n".join(rule_lines)})

        # L3+L4 — Character Knowledge
        for ck in self.character_knowledge[:4]:
            lines = [f"## {ck.character_name}"]
            if ck.core_goals:
                lines.append(f"目标：{'、'.join(ck.core_goals[:3])}")
            for f in ck.known_facts[:8]:
                src_tag = f" [ch{f.valid_from_chapter}]" if f.valid_from_chapter else ""
                lines.append(f"  ✓ {f.object}{src_tag}")
            for f in ck.withheld_facts[:4]:
                lines.append(f"  ✗ 不知道：{f.object}")
            sections.append({"label": f"角色记忆：{ck.character_name}", "content": "\n".join(lines)})

        # L4 — Temporal Facts
        if self.temporal_facts:
            fact_lines = [
                f"- {f.predicate}: {f.object}" for f in self.temporal_facts[:6]
            ]
            sections.append({"label": "时间图事实", "content": "\n".join(fact_lines)})

        # L2 — Recent Episodes
        if self.recent_episodes:
            ep_lines = [
                f"- [{e.get('event_type', '')}] {e.get('title', '')}: {e.get('summary', '')[:120]}"
                for e in self.recent_episodes[:6]
            ]
            sections.append({"label": "近期事件", "content": "\n".join(ep_lines)})

        # L5 — Cold Retrieval
        if self.cold_retrieval:
            cr_lines = [str(item.get("text", ""))[:200] for item in self.cold_retrieval[:4]]
            sections.append({"label": "补充检索", "content": "\n---\n".join(cr_lines)})

        return sections

    def to_prompt_text(self) -> str:
        """Flatten all sections into a single prompt-insertable string."""
        sections = self.to_prompt_sections()
        if not sections:
            return ""
        parts = [f"【{s['label']}】\n{s['content']}" for s in sections]
        return "\n\n".join(parts)


# ── Pipeline entry points ───────────────────────────────────────────


def build_planning_memory_context(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    scene_beat_id: int | None = None,
    chapter_index: int | None = None,
) -> MemoryContext:
    """Assemble memory for the **planning assistant**.

    Read order: L1 → L3 → L4 → L2 → L5 (fallback).
    """
    ctx = MemoryContext()

    # ── L1: Story State ──
    state = query_story_state(
        db, project_id=project_id, chapter_id=chapter_id, scene_beat_id=scene_beat_id,
    )
    ctx.story_state = state
    ctx.layers_used.append(MemoryLayer("L1", "story_state", items=1 if state else 0))

    # ── L3: World Rules ──
    rules = query_active_world_rules(db, project_id=project_id)
    ctx.world_rules = rules
    ctx.layers_used.append(MemoryLayer("L3", "world_rules", items=len(rules)))

    # ── L3: Character Profiles ──
    active_names = list(state.get("active_characters") or [])
    profiles = query_character_profiles(db, project_id=project_id, names=active_names or None)
    ctx.character_profiles = profiles
    ctx.layers_used.append(MemoryLayer("L3", "character_profiles", items=len(profiles)))

    # ── L3+L4: Character Knowledge ──
    effective_chapter = chapter_index or 999
    for name in active_names[:6]:
        bundle = query_character_knowledge_at_chapter(
            db,
            project_id=project_id,
            character_name=name,
            at_chapter=effective_chapter,
            facts_limit=8,
        )
        if bundle.known_facts or bundle.withheld_facts:
            ctx.character_knowledge.append(bundle)
    ctx.layers_used.append(
        MemoryLayer("L3+L4", "character_knowledge", items=len(ctx.character_knowledge))
    )

    # ── L4: Temporal Graph Facts (planning-specific) ──
    chapter_goal = str(state.get("chapter_goal") or "").strip()
    if chapter_goal:
        facts = query_temporal_facts(
            project_id=project_id,
            query=chapter_goal,
            limit=8,
        )
        ctx.temporal_facts = facts
    ctx.layers_used.append(MemoryLayer("L4", "temporal_graph", items=len(ctx.temporal_facts)))

    # ── L2: Recent Episodes ──
    episodes = query_recent_episodes(db, project_id=project_id, chapter_id=chapter_id, limit=6)
    ctx.recent_episodes = episodes
    ctx.layers_used.append(MemoryLayer("L2", "episodes", items=len(episodes)))

    # ── L5: Cold Retrieval (only if memory is sparse) ──
    total_items = (
        len(ctx.world_rules)
        + len(ctx.character_knowledge)
        + len(ctx.temporal_facts)
        + len(ctx.recent_episodes)
    )
    if total_items < 3 and chapter_goal:
        ctx.cold_retrieval = _l5_cold_retrieval(project_id=project_id, query=chapter_goal)
        ctx.l5_triggered = True
    ctx.layers_used.append(
        MemoryLayer("L5", "cold_retrieval", items=len(ctx.cold_retrieval))
    )

    return ctx


def build_rewrite_memory_context(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    selection_text: str = "",
    mode: str = "polish",
    chapter_index: int | None = None,
) -> MemoryContext:
    """Assemble memory for **rewrite** operations.

    - **polish**: L1 + L3 hard constraints only.  No L5.
    - **expand**: all polish + L2 + L4.  Targeted L5 only.
    """
    ctx = MemoryContext()

    # ── L1: Story State (both modes) ──
    state = query_story_state(db, project_id=project_id, chapter_id=chapter_id)
    ctx.story_state = state
    ctx.layers_used.append(MemoryLayer("L1", "story_state", items=1 if state else 0))

    # ── L3: World Rules (both modes, hard constraints) ──
    rules = query_active_world_rules(db, project_id=project_id)
    ctx.world_rules = rules
    ctx.layers_used.append(MemoryLayer("L3", "world_rules", items=len(rules)))

    if mode == "expand":
        # ── L3: Character Profiles ──
        active_names = list(state.get("active_characters") or [])
        profiles = query_character_profiles(db, project_id=project_id, names=active_names or None)
        ctx.character_profiles = profiles

        # ── L3+L4: Character Knowledge ──
        effective_chapter = chapter_index or 999
        for name in active_names[:4]:
            bundle = query_character_knowledge_at_chapter(
                db,
                project_id=project_id,
                character_name=name,
                at_chapter=effective_chapter,
                facts_limit=6,
            )
            if bundle.known_facts or bundle.withheld_facts:
                ctx.character_knowledge.append(bundle)

        # ── L4: Temporal Facts ──
        if selection_text:
            facts = query_temporal_facts(
                project_id=project_id,
                query=selection_text[:500],
                limit=6,
            )
            ctx.temporal_facts = facts

        # ── L2: Recent Episodes ──
        episodes = query_recent_episodes(
            db, project_id=project_id, chapter_id=chapter_id, limit=4,
        )
        ctx.recent_episodes = episodes

    # Polish: no L5.  Expand: targeted L5 only if memory is sparse.
    if mode == "expand":
        total_items = len(ctx.character_knowledge) + len(ctx.temporal_facts)
        if total_items < 2 and selection_text:
            ctx.cold_retrieval = _l5_cold_retrieval(
                project_id=project_id, query=selection_text[:300],
            )
            ctx.l5_triggered = True

    return ctx


def build_consistency_memory_context(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    draft_text: str = "",
    chapter_index: int | None = None,
) -> MemoryContext:
    """Assemble memory for **consistency checking**.

    Focuses on world rules, temporal facts, and character knowledge
    boundaries to detect contradictions.
    """
    ctx = MemoryContext()

    # L3: World Rules
    ctx.world_rules = query_active_world_rules(db, project_id=project_id)

    # L1: Story State
    ctx.story_state = query_story_state(db, project_id=project_id, chapter_id=chapter_id)

    # L3+L4: Character Knowledge for all active characters
    active_names = list(ctx.story_state.get("active_characters") or [])
    effective_chapter = chapter_index or 999
    for name in active_names[:6]:
        bundle = query_character_knowledge_at_chapter(
            db,
            project_id=project_id,
            character_name=name,
            at_chapter=effective_chapter,
            facts_limit=10,
        )
        ctx.character_knowledge.append(bundle)

    # L4: Temporal Facts (draft-specific)
    if draft_text:
        ctx.temporal_facts = query_temporal_facts(
            project_id=project_id, query=draft_text[:500], limit=10,
        )

    # L2: Recent Episodes
    ctx.recent_episodes = query_recent_episodes(
        db, project_id=project_id, chapter_id=chapter_id, limit=8,
    )

    return ctx


# ── L5 Cold Retrieval (LightRAG fallback) ──────────────────────────


def _l5_cold_retrieval(
    *,
    project_id: int,
    query: str,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Call LightRAG for cold semantic retrieval.  Returns empty on failure."""
    if not settings.lightrag_enabled:
        return []

    try:
        from app.services.retrieval_adapters import lightrag_query

        raw = lightrag_query(
            project_id=project_id,
            query=query,
            mode=str(settings.lightrag_query_mode or "mix"),
        )
        if not raw:
            return []

        results = []
        for item in raw[:limit]:
            text = ""
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "")
            elif isinstance(item, str):
                text = item
            if text.strip():
                results.append({"text": text.strip()[:400], "source": "lightrag"})
        return results
    except Exception:
        logger.debug("L5 cold retrieval failed", exc_info=True)
        return []
