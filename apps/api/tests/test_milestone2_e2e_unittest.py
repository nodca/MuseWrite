"""Milestone 2: End-to-end integration test with real novel data.

Uses the first 5 chapters of 《诡秘之主》 (Lord of Mysteries) to verify
the full Book Memory OS pipeline:

1. Import chapters → Postgres
2. Auto-extract WorldRules + CharacterProfiles (mocked LLM)
3. Run consolidation per chapter → Episodes + KnowledgeState
4. Query character knowledge at different chapters
5. Run consistency checker against draft with deliberate errors

LLM calls are mocked with curated fiction-accurate responses.
"""

import json
import os
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import app.models  # noqa: F401
from app.models.book_memory import (
    CharacterKnowledgeState,
    CharacterProfile,
    StoryEpisode,
    StoryStateSnapshot,
    WorldRule,
)
from app.models.content import ProjectChapter, ProjectVolume
from sqlmodel import select as sq_select

NOVEL_DATA_PATH = Path("/home/wcn/Desktop/sft-scripts-backup/cpt_weighted/all_books_by_chapter.jsonl")
BOOK_NAME = "诡秘之主"
MAX_CHAPTERS = 5


def _load_chapters() -> list[dict[str, Any]]:
    """Load first N chapters of the target book."""
    if not NOVEL_DATA_PATH.exists():
        return []
    chapters = []
    with open(NOVEL_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["book"] == BOOK_NAME and d["chapter_index"] <= MAX_CHAPTERS:
                chapters.append(d)
    return sorted(chapters, key=lambda x: x["chapter_index"])


# ── Curated mock LLM responses (fiction-accurate for 诡秘之主) ──

_MOCK_EXTRACTION_CH1 = {
    "chapter_goal": "主角穿越醒来，发现自己身处异世界",
    "active_characters": ["周明瑞"],
    "current_location": "克莱恩的房间",
    "active_conflicts": ["头痛异常", "身份危机"],
    "open_questions": ["为什么会穿越？", "克莱恩是谁？"],
    "episodes": [
        {
            "title": "穿越醒来",
            "summary": "周明瑞从梦中醒来，发现自己穿越到了异世界，头痛剧烈",
            "event_type": "revelation",
            "participants": ["周明瑞"],
            "location": "克莱恩的房间",
            "visibility": "private",
            "importance": 90,
        }
    ],
    "knowledge_claims": [
        {
            "character_name": "周明瑞",
            "fact": "自己穿越到了另一个人的身体里",
            "known": True,
            "confidence": 0.95,
        }
    ],
}

_MOCK_EXTRACTION_CH2 = {
    "chapter_goal": "周明瑞检查伤口，了解克莱恩的身份和处境",
    "active_characters": ["周明瑞"],
    "current_location": "克莱恩的房间",
    "active_conflicts": ["贯穿伤口", "身份伪装"],
    "open_questions": ["伤口为何还活着？", "克莱恩为何自杀？"],
    "episodes": [
        {
            "title": "发现贯穿伤",
            "summary": "周明瑞在镜中发现太阳穴有贯穿伤口，震惊不已",
            "event_type": "revelation",
            "participants": ["周明瑞"],
            "location": "克莱恩的房间",
            "visibility": "private",
            "importance": 85,
        }
    ],
    "knowledge_claims": [
        {
            "character_name": "周明瑞",
            "fact": "克莱恩的太阳穴有贯穿伤口，疑似自杀",
            "known": True,
            "confidence": 0.9,
        },
        {
            "character_name": "周明瑞",
            "fact": "克莱恩是廷根市一名大学毕业生",
            "known": True,
            "confidence": 0.85,
        },
    ],
}

_MOCK_EXTRACTION_CH3 = {
    "chapter_goal": "梅丽莎登场，周明瑞决定伪装成克莱恩",
    "active_characters": ["周明瑞", "梅丽莎"],
    "current_location": "克莱恩的住所",
    "active_conflicts": ["身份伪装", "梅丽莎的信任"],
    "open_questions": ["能否瞒过梅丽莎？"],
    "episodes": [
        {
            "title": "梅丽莎来访",
            "summary": "克莱恩的妹妹梅丽莎来看望他，周明瑞决定冒充克莱恩",
            "event_type": "dialogue",
            "participants": ["周明瑞", "梅丽莎"],
            "location": "克莱恩的住所",
            "visibility": "public",
            "importance": 80,
        }
    ],
    "knowledge_claims": [
        {
            "character_name": "周明瑞",
            "fact": "梅丽莎是克莱恩的妹妹",
            "known": True,
            "confidence": 0.95,
        },
        {
            "character_name": "梅丽莎",
            "fact": "哥哥克莱恩身体不适",
            "known": True,
            "confidence": 0.8,
        },
    ],
}

_MOCK_EXTRACTION_CH4 = {
    "chapter_goal": "周明瑞准备去应聘，探索世界",
    "active_characters": ["周明瑞"],
    "current_location": "廷根市街道",
    "active_conflicts": ["经济困难", "身份维持"],
    "open_questions": ["能否找到工作？"],
    "episodes": [
        {
            "title": "出门探索",
            "summary": "周明瑞穿戴整齐出门，观察这个维多利亚风格的世界",
            "event_type": "movement",
            "participants": ["周明瑞"],
            "location": "廷根市",
            "visibility": "public",
            "importance": 60,
        }
    ],
    "knowledge_claims": [
        {
            "character_name": "周明瑞",
            "fact": "这个世界类似维多利亚时代的蒸汽朋克",
            "known": True,
            "confidence": 0.9,
        },
    ],
}

_MOCK_EXTRACTION_CH5 = {
    "chapter_goal": "周明瑞遭遇占卜仪式，接触超凡力量",
    "active_characters": ["周明瑞", "占卜女"],
    "current_location": "占卜帐篷",
    "active_conflicts": ["超凡力量的诱惑与危险"],
    "open_questions": ["占卜是否真实？", "超凡力量的代价是什么？"],
    "episodes": [
        {
            "title": "占卜仪式",
            "summary": "周明瑞在集市进入占卜帐篷，经历了一次神秘的占卜",
            "event_type": "revelation",
            "participants": ["周明瑞", "占卜女"],
            "location": "占卜帐篷",
            "visibility": "public",
            "importance": 85,
        }
    ],
    "knowledge_claims": [
        {
            "character_name": "周明瑞",
            "fact": "这个世界存在超凡力量和占卜术",
            "known": True,
            "confidence": 0.85,
        },
    ],
}

_MOCK_EXTRACTIONS = {
    1: _MOCK_EXTRACTION_CH1,
    2: _MOCK_EXTRACTION_CH2,
    3: _MOCK_EXTRACTION_CH3,
    4: _MOCK_EXTRACTION_CH4,
    5: _MOCK_EXTRACTION_CH5,
}


def _mock_extract_structured(**kwargs):
    """Return curated extraction for the given chapter."""
    chapter = kwargs.get("chapter")
    if chapter is None:
        return None
    ch_idx = int(getattr(chapter, "chapter_index", 0))
    data = _MOCK_EXTRACTIONS.get(ch_idx)
    if data is None:
        return None

    from app.services.book_memory.extraction_service import BookMemoryStructuredExtraction

    return BookMemoryStructuredExtraction.model_validate(data)


@unittest.skipUnless(NOVEL_DATA_PATH.exists(), "Novel data not available")
class TestMilestone2E2E(unittest.TestCase):
    """End-to-end test: real novel data → full pipeline → knowledge queries."""

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)
        self.chapters_data = _load_chapters()
        self._seed_project()

    def tearDown(self) -> None:
        SQLModel.metadata.drop_all(self.engine)
        self.engine.dispose()

    def _seed_project(self) -> None:
        """Import novel chapters into the test DB."""
        with Session(self.engine) as db:
            volume = ProjectVolume(
                project_id=1,
                volume_index=1,
                title="第一卷",
                outline="克莱恩穿越，发现异世界的超凡力量",
            )
            db.add(volume)
            db.flush()

            for ch_data in self.chapters_data:
                chapter = ProjectChapter(
                    project_id=1,
                    volume_id=int(volume.id),
                    chapter_index=ch_data["chapter_index"],
                    title=ch_data["chapter_title"],
                    content=ch_data["text"],
                    version=1,
                )
                db.add(chapter)

            # Seed canon: world rules (hand-crafted for 诡秘之主)
            rules = [
                ("超凡序列", "超凡者通过服用魔药获得能力，序列越低越强大", "universal"),
                ("魔药消化", "服用魔药后必须扮演对应职业来消化，否则失控", "universal"),
                ("非凡力量代价", "超凡力量使用过度会导致失控，变成怪物", "universal"),
            ]
            for title, statement, scope in rules:
                db.add(WorldRule(
                    project_id=1, title=title, statement=statement,
                    scope=scope, priority=10, status="active",
                ))

            # Seed canon: character profiles
            profiles = [
                CharacterProfile(
                    project_id=1,
                    canonical_name="周明瑞",
                    aliases=["克莱恩", "克莱恩·莫雷蒂"],
                    core_goals=["在异世界生存", "弄清穿越真相"],
                    public_traits=["谨慎", "善于观察"],
                    private_traits=["穿越者身份"],
                    fears=["暴露穿越者身份"],
                    taboos=["不暴露现代知识"],
                    default_voice_notes="内心活泼但表面沉稳",
                ),
                CharacterProfile(
                    project_id=1,
                    canonical_name="梅丽莎",
                    aliases=["梅丽莎·莫雷蒂"],
                    core_goals=["关心哥哥", "好好读书"],
                    public_traits=["善良", "单纯"],
                    fears=["失去亲人"],
                ),
            ]
            for p in profiles:
                db.add(p)

            db.commit()

    @patch("app.services.book_memory.consolidation_service.extract_book_memory_structured")
    @patch("app.services.book_memory.graphiti_adapter.get_graphiti")
    def test_full_consolidation_pipeline(
        self, mock_graphiti: MagicMock, mock_extract: MagicMock
    ) -> None:
        """Run consolidation for all 5 chapters and verify state."""
        mock_graphiti.return_value = None  # Disable Graphiti for this test
        mock_extract.side_effect = _mock_extract_structured

        from app.services.book_memory.consolidation_service import run_book_memory_consolidation

        with Session(self.engine) as db:
            chapters = db.exec(
                sq_select(ProjectChapter)
                .where(ProjectChapter.project_id == 1)
                .order_by(ProjectChapter.chapter_index)
            ).all()

            results = []
            for ch in chapters:
                result = run_book_memory_consolidation(
                    db, project_id=1, chapter_id=int(ch.id),
                )
                results.append(result)

        # All 5 chapters consolidated
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r["status"] == "ok" for r in results))

        # Verify episodes created
        with Session(self.engine) as db:
            all_episodes = db.exec(
                sq_select(StoryEpisode).where(StoryEpisode.project_id == 1)
            ).all()
            self.assertGreaterEqual(len(all_episodes), 5)

            # Verify story state snapshots
            snapshots = db.exec(
                sq_select(StoryStateSnapshot).where(StoryStateSnapshot.project_id == 1)
            ).all()
            self.assertEqual(len(snapshots), 5)

            # Verify knowledge states
            knowledge = db.exec(
                sq_select(CharacterKnowledgeState).where(CharacterKnowledgeState.project_id == 1)
            ).all()
            self.assertGreaterEqual(len(knowledge), 5)

    @patch("app.services.book_memory.consolidation_service.extract_book_memory_structured")
    @patch("app.services.book_memory.graphiti_adapter.get_graphiti")
    def test_character_knowledge_boundary(
        self, mock_graphiti: MagicMock, mock_extract: MagicMock
    ) -> None:
        """Verify knowledge boundaries: what 周明瑞 knows at different chapters."""
        mock_graphiti.return_value = None
        mock_extract.side_effect = _mock_extract_structured

        from app.services.book_memory.consolidation_service import run_book_memory_consolidation
        from app.services.book_memory.temporal_query import query_character_knowledge_at_chapter

        with Session(self.engine) as db:
            chapters = db.exec(
                sq_select(ProjectChapter)
                .where(ProjectChapter.project_id == 1)
                .order_by(ProjectChapter.chapter_index)
            ).all()
            for ch in chapters:
                run_book_memory_consolidation(db, project_id=1, chapter_id=int(ch.id))

        # Query: what does 周明瑞 know at chapter 2?
        with Session(self.engine) as db:
            bundle_ch2 = query_character_knowledge_at_chapter(
                db, project_id=1, character_name="周明瑞", at_chapter=2,
            )
            known_ch2 = [f.object for f in bundle_ch2.known_facts]
            # Should know about: 穿越, 贯穿伤口, 大学毕业生
            self.assertTrue(
                any("穿越" in f for f in known_ch2),
                f"周明瑞 at ch2 should know about 穿越. Got: {known_ch2}",
            )

        # Query: what does 周明瑞 know at chapter 5?
        with Session(self.engine) as db:
            bundle_ch5 = query_character_knowledge_at_chapter(
                db, project_id=1, character_name="周明瑞", at_chapter=5,
            )
            known_ch5 = [f.object for f in bundle_ch5.known_facts]
            # Should know MORE at ch5 than ch2
            self.assertGreater(len(known_ch5), len(known_ch2))
            # Should know about 超凡力量 by ch5
            self.assertTrue(
                any("超凡" in f or "占卜" in f for f in known_ch5),
                f"周明瑞 at ch5 should know about 超凡/占卜. Got: {known_ch5}",
            )

        # Query: what does 梅丽莎 know at chapter 3?
        with Session(self.engine) as db:
            bundle_mel = query_character_knowledge_at_chapter(
                db, project_id=1, character_name="梅丽莎", at_chapter=3,
            )
            known_mel = [f.object for f in bundle_mel.known_facts]
            # 梅丽莎 should NOT know about 穿越
            self.assertFalse(
                any("穿越" in f for f in known_mel),
                f"梅丽莎 should NOT know about 穿越. Got: {known_mel}",
            )

    @patch("app.services.book_memory.consolidation_service.extract_book_memory_structured")
    @patch("app.services.book_memory.graphiti_adapter.get_graphiti")
    def test_planning_memory_context_output(
        self, mock_graphiti: MagicMock, mock_extract: MagicMock
    ) -> None:
        """Verify the planning memory pipeline produces usable output."""
        mock_graphiti.return_value = None
        mock_extract.side_effect = _mock_extract_structured

        from app.services.book_memory.consolidation_service import run_book_memory_consolidation
        from app.services.book_memory.memory_pipeline import build_planning_memory_context

        with Session(self.engine) as db:
            chapters = db.exec(
                sq_select(ProjectChapter)
                .where(ProjectChapter.project_id == 1)
                .order_by(ProjectChapter.chapter_index)
            ).all()
            for ch in chapters:
                run_book_memory_consolidation(db, project_id=1, chapter_id=int(ch.id))

        # Build planning context for chapter 5
        with Session(self.engine) as db:
            ch5 = db.exec(
                sq_select(ProjectChapter)
                .where(ProjectChapter.project_id == 1, ProjectChapter.chapter_index == 5)
            ).first()
            ctx = build_planning_memory_context(
                db, project_id=1, chapter_id=int(ch5.id), chapter_index=5,
            )

        # Verify context is non-empty and useful
        self.assertTrue(ctx.story_state)
        self.assertIn("占卜", ctx.story_state.get("chapter_goal", ""))
        self.assertTrue(len(ctx.world_rules) >= 3)
        self.assertTrue(len(ctx.recent_episodes) >= 1)

        # Verify prompt text is generated
        prompt = ctx.to_prompt_text()
        self.assertIn("超凡序列", prompt)  # World rule
        self.assertIn("周明瑞", prompt)  # Character
        self.assertFalse(ctx.l5_triggered)  # Should NOT fallback to LightRAG

    @patch("app.services.book_memory.consolidation_service.extract_book_memory_structured")
    @patch("app.services.book_memory.graphiti_adapter.get_graphiti")
    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_consistency_checker_with_real_data(
        self,
        mock_temporal: MagicMock,
        mock_knowledge: MagicMock,
        mock_graphiti: MagicMock,
        mock_extract: MagicMock,
    ) -> None:
        """Verify consistency checker catches errors in draft text."""
        mock_graphiti.return_value = None
        mock_extract.side_effect = _mock_extract_structured
        mock_temporal.return_value = []
        mock_knowledge.return_value = []

        from app.services.book_memory.consolidation_service import run_book_memory_consolidation
        from app.services.book_memory.consistency_checker import check_draft_consistency

        with Session(self.engine) as db:
            chapters = db.exec(
                sq_select(ProjectChapter)
                .where(ProjectChapter.project_id == 1)
                .order_by(ProjectChapter.chapter_index)
            ).all()
            for ch in chapters:
                run_book_memory_consolidation(db, project_id=1, chapter_id=int(ch.id))

        # Draft with knowledge leak: 梅丽莎 mentions 穿越 (she shouldn't know)
        bad_draft = "梅丽莎看着哥哥，心想他一定是穿越来的，这种异世界的人怎么可能是原来的克莱恩呢。"

        with Session(self.engine) as db:
            ch3 = db.exec(
                sq_select(ProjectChapter)
                .where(ProjectChapter.project_id == 1, ProjectChapter.chapter_index == 3)
            ).first()
            report = check_draft_consistency(
                db, project_id=1, chapter_id=int(ch3.id),
                draft_text=bad_draft, chapter_index=3,
            )

        # Should detect knowledge leak
        leak_issues = [i for i in report.issues if i.category == "knowledge_leak"]
        self.assertTrue(
            len(leak_issues) > 0,
            f"Should detect 梅丽莎 knowing about 穿越. Issues: {[i.title for i in report.issues]}",
        )


if __name__ == "__main__":
    unittest.main()
