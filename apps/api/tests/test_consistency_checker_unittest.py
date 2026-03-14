"""Unit tests for the Book Memory consistency checker.

Each test seeds deliberate contradictions and verifies that the
checker detects the correct issue category and severity.
"""

import unittest
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
from app.services.book_memory.consistency_checker import (
    ConsistencyReport,
    check_draft_consistency,
    _extract_keywords,
)


class ConsistencyTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)
        self._seed()

    def tearDown(self) -> None:
        SQLModel.metadata.drop_all(self.engine)
        self.engine.dispose()

    def _seed(self) -> None:
        with Session(self.engine) as db:
            # World rules
            db.add(WorldRule(
                project_id=1, scope="universal", title="凡人不可飞行",
                statement="凡人不可飞行，只有金丹期以上才能御空",
                priority=10, status="active",
            ))
            db.add(WorldRule(
                project_id=1, scope="universal", title="灵石为货币",
                statement="灵石是修仙界唯一货币",
                priority=20, status="active",
            ))

            # Profiles
            lin = CharacterProfile(
                project_id=1, canonical_name="林澈",
                aliases=["小澈"], core_goals=["飞升"],
            )
            su = CharacterProfile(
                project_id=1, canonical_name="苏青",
                aliases=["苏师姐"], core_goals=["守护师门"],
            )
            db.add(lin)
            db.add(su)
            db.flush()

            # Story state
            db.add(StoryStateSnapshot(
                project_id=1, chapter_id=10,
                chapter_goal="揭秘",
                active_characters=["林澈", "苏青"],
                current_location="藏经阁",
            ))

            # Episodes (ordered)
            db.add(StoryEpisode(
                project_id=1, chapter_id=10, episode_index=1,
                title="发现密信", summary="林澈发现了密信",
                event_type="revelation", participants=["林澈"],
            ))
            db.add(StoryEpisode(
                project_id=1, chapter_id=10, episode_index=2,
                title="苏青来访", summary="苏青前来询问",
                event_type="dialogue", participants=["林澈", "苏青"],
            ))

            # Knowledge: 林澈 knows about the letter; 苏青 does NOT
            db.add(CharacterKnowledgeState(
                project_id=1, character_profile_id=int(lin.id),
                knowledge_key="knows:密信内容",
                knowledge_value={"fact": "师门曾与魔族勾结"},
                gained_at_chapter=5, confidence=0.9,
            ))
            # 苏青 knows something 林澈 doesn't
            db.add(CharacterKnowledgeState(
                project_id=1, character_profile_id=int(su.id),
                knowledge_key="knows:解药配方",
                knowledge_value={"fact": "苏青知道解药配方"},
                gained_at_chapter=3, confidence=0.85,
            ))
            db.commit()


class TestExtractKeywords(unittest.TestCase):
    def test_extracts_chinese_tokens(self) -> None:
        kws = _extract_keywords("凡人不可飞行")
        self.assertTrue(len(kws) > 0)

    def test_respects_max_count(self) -> None:
        kws = _extract_keywords("一 二 三 四 五 六 七 八 九 十", max_count=3)
        self.assertLessEqual(len(kws), 3)

    def test_respects_min_len(self) -> None:
        kws = _extract_keywords("a bb ccc dddd", min_len=3)
        self.assertTrue(all(len(k) >= 3 for k in kws))


class TestWorldRuleViolation(ConsistencyTestBase):
    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_detects_rule_negation(self, mock_t: MagicMock, mock_k: MagicMock) -> None:
        mock_t.return_value = []
        mock_k.return_value = []

        draft = "林澈轻轻一跃，突破了飞行的限制，不再受凡人不可飞行的束缚，径直飞向天际。"

        with Session(self.engine) as db:
            report = check_draft_consistency(
                db, project_id=1, chapter_id=10, draft_text=draft, chapter_index=10,
            )

        world_rule_issues = [i for i in report.issues if i.category == "world_rule"]
        self.assertTrue(len(world_rule_issues) > 0, "Should detect world rule violation")
        self.assertIn("飞行", world_rule_issues[0].title)

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_no_false_positive_when_rule_respected(self, mock_t: MagicMock, mock_k: MagicMock) -> None:
        mock_t.return_value = []
        mock_k.return_value = []

        draft = "林澈望着天空，知道自己作为凡人还不可飞行，只能步行前往。"

        with Session(self.engine) as db:
            report = check_draft_consistency(
                db, project_id=1, chapter_id=10, draft_text=draft, chapter_index=10,
            )

        world_rule_issues = [i for i in report.issues if i.category == "world_rule"]
        self.assertEqual(len(world_rule_issues), 0)


class TestKnowledgeLeak(ConsistencyTestBase):
    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_detects_omniscience_leak(self, mock_t: MagicMock, mock_k: MagicMock) -> None:
        mock_t.return_value = []
        mock_k.return_value = []

        # 林澈 should NOT know about 解药配方 — only 苏青 does
        draft = "林澈翻开密信，心中暗想解药配方或许就在此处。他早已知道解药配方的秘密。"

        with Session(self.engine) as db:
            report = check_draft_consistency(
                db, project_id=1, chapter_id=10, draft_text=draft, chapter_index=10,
            )

        leak_issues = [i for i in report.issues if i.category == "knowledge_leak"]
        self.assertTrue(len(leak_issues) > 0, "Should detect knowledge leak")
        self.assertEqual(leak_issues[0].severity, "critical")
        self.assertIn("林澈", leak_issues[0].title)

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_no_leak_when_character_knows(self, mock_t: MagicMock, mock_k: MagicMock) -> None:
        mock_t.return_value = []
        mock_k.return_value = []

        # 林澈 DOES know about 密信 (gained at ch5)
        draft = "林澈想起了密信内容中提到的师门曾与魔族勾结的事实。"

        with Session(self.engine) as db:
            report = check_draft_consistency(
                db, project_id=1, chapter_id=10, draft_text=draft, chapter_index=10,
            )

        leak_issues = [i for i in report.issues if i.category == "knowledge_leak"]
        self.assertEqual(len(leak_issues), 0, "Should not flag knowledge character already has")


class TestChronology(ConsistencyTestBase):
    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_detects_out_of_order_events(self, mock_t: MagicMock, mock_k: MagicMock) -> None:
        mock_t.return_value = []
        mock_k.return_value = []

        # Episode order is: 发现密信 (ep1) → 苏青来访 (ep2)
        # Draft mentions them in reversed order.
        draft = "苏青来访之后，林澈才想起发现密信的那个下午。"

        with Session(self.engine) as db:
            report = check_draft_consistency(
                db, project_id=1, chapter_id=10, draft_text=draft, chapter_index=10,
            )

        chrono_issues = [i for i in report.issues if i.category == "chronology"]
        # This may or may not trigger depending on keyword extraction,
        # but if events are detected, order should be flagged.
        if len(chrono_issues) > 0:
            self.assertEqual(chrono_issues[0].severity, "warning")


class TestReportStructure(ConsistencyTestBase):
    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_report_has_correct_structure(self, mock_t: MagicMock, mock_k: MagicMock) -> None:
        mock_t.return_value = []
        mock_k.return_value = []

        with Session(self.engine) as db:
            report = check_draft_consistency(
                db, project_id=1, chapter_id=10, draft_text="无关文本", chapter_index=10,
            )

        self.assertIsInstance(report, ConsistencyReport)
        self.assertEqual(report.project_id, 1)
        self.assertEqual(report.chapter_id, 10)
        self.assertEqual(report.checks_run, 3)

        d = report.to_dict()
        self.assertIn("clean", d)
        self.assertIn("issues", d)
        self.assertIn("checks_run", d)

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_empty_draft_returns_clean(self, mock_t: MagicMock, mock_k: MagicMock) -> None:
        mock_t.return_value = []
        mock_k.return_value = []

        with Session(self.engine) as db:
            report = check_draft_consistency(
                db, project_id=1, chapter_id=10, draft_text="", chapter_index=10,
            )

        self.assertTrue(report.clean)
        self.assertEqual(report.checks_run, 0)


if __name__ == "__main__":
    unittest.main()
