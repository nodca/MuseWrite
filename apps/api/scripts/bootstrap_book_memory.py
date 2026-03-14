#!/usr/bin/env python3
"""Bootstrap Book Memory from novel chapters.

Reads chapters from a JSONL file and runs the full Book Memory OS
pipeline with REAL LLM calls:

1. Auto-extract WorldRules + CharacterProfiles from chapter 1
2. Run consolidation for each chapter (extraction → episodes → knowledge)
3. Query character knowledge at different chapters
4. Run consistency checker
5. Print report

Usage::

    cd apps/api
    python scripts/bootstrap_book_memory.py \
        --data /path/to/chapters.jsonl \
        --book "诡秘之主" \
        --chapters 5 \
        --llm-base-url http://127.0.0.1:8081/v1 \
        --llm-api-key proxy \
        --llm-model gpt-5.2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure app is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os


def _setup_env(args: argparse.Namespace) -> None:
    """Set environment variables for the app config."""
    os.environ.setdefault("DATABASE_URL", "sqlite:///book_memory_bootstrap.db")
    os.environ["LLM_PROVIDER"] = "openai_compatible"
    os.environ["LLM_BASE_URL"] = args.llm_base_url
    os.environ["LLM_API_KEY"] = args.llm_api_key
    os.environ["LLM_MODEL"] = args.llm_model
    # LightRAG LLM inherits from LLM_* via config fallback.
    os.environ["NEO4J_ENABLED"] = "false"
    os.environ["GRAPHITI_ENABLED"] = "false"
    os.environ["LIGHTRAG_ENABLED"] = "false"
    os.environ["AUTH_ENABLED"] = "false"
    os.environ["MEMORY_CONSOLIDATION_LLM_TIMEOUT_SECONDS"] = "60"
    os.environ["LLM_TIMEOUT_SECONDS"] = "60"


def _load_chapters(data_path: str, book: str, max_chapters: int) -> list[dict[str, Any]]:
    chapters = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["book"] == book and d["chapter_index"] <= max_chapters:
                chapters.append(d)
    return sorted(chapters, key=lambda x: x["chapter_index"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Book Memory from novel chapters")
    parser.add_argument("--data", required=True, help="Path to chapters JSONL")
    parser.add_argument("--book", required=True, help="Book name to filter")
    parser.add_argument("--chapters", type=int, default=5, help="Max chapters to process")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--llm-api-key", default="proxy")
    parser.add_argument("--llm-model", default="gpt-5.2")
    parser.add_argument("--db-url", default=None, help="Database URL (default: sqlite in-memory)")
    args = parser.parse_args()

    _setup_env(args)

    if args.db_url:
        os.environ["DATABASE_URL"] = args.db_url

    # ── Load chapters ──
    print(f"Loading chapters from: {args.data}")
    chapters = _load_chapters(args.data, args.book, args.chapters)
    if not chapters:
        print(f"No chapters found for book '{args.book}'")
        sys.exit(1)
    print(f"Loaded {len(chapters)} chapters of '{args.book}'")
    for ch in chapters:
        print(f"  ch{ch['chapter_index']:3d}: {ch['chapter_title']} ({ch['char_count']}字)")

    # ── Initialize DB ──
    from sqlalchemy.pool import StaticPool
    from sqlmodel import Session, SQLModel, create_engine, select

    import app.models  # noqa: F401 — register tables

    db_url = os.environ.get("DATABASE_URL", "sqlite:///book_memory_bootstrap.db")
    connect_args = {"check_same_thread": False} if "sqlite" in db_url else {}
    engine = create_engine(db_url, connect_args=connect_args, poolclass=StaticPool if "sqlite" in db_url else None)
    SQLModel.metadata.create_all(engine)

    from app.models.book_memory import (
        CharacterKnowledgeState,
        CharacterProfile,
        StoryEpisode,
        StoryStateSnapshot,
        WorldRule,
    )
    from app.models.content import ProjectChapter, ProjectVolume

    # ── Seed project structure ──
    print("\n── Seeding project structure ──")
    with Session(engine) as db:
        volume = ProjectVolume(project_id=1, volume_index=1, title="第一卷")
        db.add(volume)
        db.flush()

        for ch_data in chapters:
            chapter = ProjectChapter(
                project_id=1,
                volume_id=int(volume.id),
                chapter_index=ch_data["chapter_index"],
                title=ch_data["chapter_title"],
                content=ch_data["text"],
                version=1,
            )
            db.add(chapter)
        db.commit()
        print(f"  Created volume + {len(chapters)} chapters")

    # ── Step 1: Extract canon from chapter 1 using LLM ──
    print("\n── Step 1: Auto-extract canon (WorldRules + Characters) via LLM ──")
    t0 = time.time()

    import instructor
    from openai import OpenAI
    from pydantic import BaseModel, Field

    class ExtractedWorldRule(BaseModel):
        title: str
        statement: str
        scope: str = "universal"

    class ExtractedCharacter(BaseModel):
        name: str
        aliases: list[str] = Field(default_factory=list)
        role: str = ""
        traits: list[str] = Field(default_factory=list)
        goals: list[str] = Field(default_factory=list)

    class CanonExtraction(BaseModel):
        world_rules: list[ExtractedWorldRule] = Field(default_factory=list)
        characters: list[ExtractedCharacter] = Field(default_factory=list)

    client = instructor.from_openai(
        OpenAI(api_key=args.llm_api_key, base_url=args.llm_base_url, timeout=60),
        mode=instructor.Mode.JSON,
    )

    # Use first 2 chapters for richer extraction.
    canon_text = "\n\n".join(ch["text"][:3000] for ch in chapters[:2])
    try:
        canon = client.chat.completions.create(
            model=args.llm_model,
            response_model=CanonExtraction,
            max_retries=2,
            temperature=0.0,
            messages=[
                {"role": "system", "content": (
                    "你是小说世界设定分析器。从给定的章节文本中提取：\n"
                    "1. world_rules: 世界规则/法则/限制（如修炼体系、魔法规则、社会制度等）\n"
                    "2. characters: 出场角色（名字、别名、角色定位、性格特点、目标）\n"
                    "只提取文本明确提到或强烈暗示的内容，不要编造。"
                )},
                {"role": "user", "content": canon_text},
            ],
        )
        print(f"  LLM extraction done in {time.time() - t0:.1f}s")
        print(f"  World rules: {len(canon.world_rules)}")
        for r in canon.world_rules:
            print(f"    - [{r.scope}] {r.title}: {r.statement[:60]}")
        print(f"  Characters: {len(canon.characters)}")
        for c in canon.characters:
            print(f"    - {c.name} ({', '.join(c.aliases[:3])}): {c.role}")
    except Exception as exc:
        print(f"  ERROR: Canon extraction failed: {exc}")
        canon = CanonExtraction()

    # Persist canon to DB.
    with Session(engine) as db:
        for r in canon.world_rules:
            db.add(WorldRule(
                project_id=1, title=r.title, statement=r.statement,
                scope=r.scope, priority=10, status="active",
            ))
        for c in canon.characters:
            db.add(CharacterProfile(
                project_id=1,
                canonical_name=c.name,
                aliases=c.aliases,
                core_goals=c.goals,
                public_traits=c.traits,
            ))
        db.commit()
        rules_count = db.exec(select(WorldRule).where(WorldRule.project_id == 1)).all()
        profiles_count = db.exec(select(CharacterProfile).where(CharacterProfile.project_id == 1)).all()
        print(f"  Persisted: {len(rules_count)} rules, {len(profiles_count)} profiles")

    # ── Step 2: Run consolidation per chapter ──
    print("\n── Step 2: Run consolidation per chapter (real LLM extraction) ──")
    from app.services.book_memory.consolidation_service import run_book_memory_consolidation

    with Session(engine) as db:
        db_chapters = db.exec(
            select(ProjectChapter)
            .where(ProjectChapter.project_id == 1)
            .order_by(ProjectChapter.chapter_index)
        ).all()

        for ch in db_chapters:
            t1 = time.time()
            try:
                result = run_book_memory_consolidation(db, project_id=1, chapter_id=int(ch.id))
                elapsed = time.time() - t1
                print(f"  ch{ch.chapter_index}: {result['status']} "
                      f"({result['episode_count']} episodes, "
                      f"{result['knowledge_count']} knowledge, "
                      f"{elapsed:.1f}s)")
            except Exception as exc:
                print(f"  ch{ch.chapter_index}: ERROR — {exc}")

    # ── Step 3: Query character knowledge ──
    print("\n── Step 3: Query character knowledge at different chapters ──")
    from app.services.book_memory.temporal_query import query_character_knowledge_at_chapter

    with Session(engine) as db:
        profiles = db.exec(select(CharacterProfile).where(CharacterProfile.project_id == 1)).all()
        char_names = [p.canonical_name for p in profiles[:3]]

        for name in char_names:
            for at_ch in [1, 3, 5]:
                bundle = query_character_knowledge_at_chapter(
                    db, project_id=1, character_name=name, at_chapter=at_ch,
                )
                facts = [f.object[:40] for f in bundle.known_facts[:3]]
                withheld = len(bundle.withheld_facts)
                print(f"  {name} @ch{at_ch}: knows={len(bundle.known_facts)}, "
                      f"withheld={withheld} | {'; '.join(facts)}")

    # ── Step 4: Build planning memory context ──
    print("\n── Step 4: Build planning memory context for last chapter ──")
    from app.services.book_memory.memory_pipeline import build_planning_memory_context

    with Session(engine) as db:
        last_ch = db.exec(
            select(ProjectChapter)
            .where(ProjectChapter.project_id == 1)
            .order_by(ProjectChapter.chapter_index.desc())
        ).first()

        ctx = build_planning_memory_context(
            db, project_id=1, chapter_id=int(last_ch.id),
            chapter_index=int(last_ch.chapter_index),
        )
        prompt = ctx.to_prompt_text()
        print(f"  Story state: {ctx.story_state.get('chapter_goal', 'N/A')}")
        print(f"  World rules: {len(ctx.world_rules)}")
        print(f"  Character knowledge bundles: {len(ctx.character_knowledge)}")
        print(f"  Recent episodes: {len(ctx.recent_episodes)}")
        print(f"  L5 triggered: {ctx.l5_triggered}")
        print(f"  Prompt length: {len(prompt)} chars")

    # ── Step 5: Consistency check ──
    print("\n── Step 5: Consistency check with deliberate error ──")
    from app.services.book_memory.consistency_checker import check_draft_consistency

    bad_draft = f"{'、'.join(char_names[:2])}心中暗想，这个世界的一切规则他都已了然于胸，没有什么能限制他了。"
    with Session(engine) as db:
        report = check_draft_consistency(
            db, project_id=1, chapter_id=int(last_ch.id),
            draft_text=bad_draft, chapter_index=int(last_ch.chapter_index),
        )
        print(f"  Issues: {len(report.issues)} (critical={report.critical_count}, warnings={report.warning_count})")
        for issue in report.issues[:5]:
            print(f"    [{issue.severity}] {issue.category}: {issue.title}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Bootstrap complete!")
    with Session(engine) as db:
        total_episodes = len(db.exec(select(StoryEpisode).where(StoryEpisode.project_id == 1)).all())
        total_knowledge = len(db.exec(select(CharacterKnowledgeState).where(CharacterKnowledgeState.project_id == 1)).all())
        total_snapshots = len(db.exec(select(StoryStateSnapshot).where(StoryStateSnapshot.project_id == 1)).all())
        print(f"  Chapters:   {len(chapters)}")
        print(f"  Rules:      {len(rules_count)}")
        print(f"  Profiles:   {len(profiles_count)}")
        print(f"  Episodes:   {total_episodes}")
        print(f"  Knowledge:  {total_knowledge}")
        print(f"  Snapshots:  {total_snapshots}")
    print("=" * 60)


if __name__ == "__main__":
    main()
