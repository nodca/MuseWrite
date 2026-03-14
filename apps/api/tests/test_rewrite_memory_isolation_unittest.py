"""Unit tests for rewrite memory pipeline layer isolation.

Verifies that:
- **polish** mode only uses L1 + L3 (no episodes, no temporal, no L5)
- **expand** mode uses L1 + L2 + L3 + L4 (with targeted L5 fallback)
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
from app.services.book_memory.memory_pipeline import (
    build_rewrite_memory_context,
)


class RewriteMemoryTestBase(unittest.TestCase):
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
            db.add(WorldRule(
                project_id=1, scope="universal", title="修炼需灵石",
                statement="修炼必须消耗灵石", priority=10, status="active",
            ))
            profile = CharacterProfile(
                project_id=1, canonical_name="林澈",
                aliases=["小澈"], core_goals=["飞升"],
            )
            db.add(profile)
            db.flush()
            db.add(StoryStateSnapshot(
                project_id=1, chapter_id=10,
                chapter_goal="突破瓶颈",
                active_characters=["林澈"],
                current_location="丹房",
                active_conflicts=["灵石不足"],
                open_questions=["如何获取灵石？"],
            ))
            db.add(StoryEpisode(
                project_id=1, chapter_id=10, episode_index=1,
                title="购买灵石", summary="林澈在集市购买灵石",
                event_type="action", participants=["林澈"],
                location="集市", importance=50,
            ))
            db.add(CharacterKnowledgeState(
                project_id=1, character_profile_id=int(profile.id),
                knowledge_key="knows:灵石价格",
                knowledge_value={"fact": "上品灵石价值千金"},
                gained_at_chapter=5, confidence=0.85,
            ))
            db.commit()


class TestPolishIsolation(RewriteMemoryTestBase):
    """Polish mode should only access L1 + L3."""

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_polish_has_story_state(self, mock_t: MagicMock, mock_k: MagicMock) -> None:
        mock_t.return_value = []
        mock_k.return_value = []
        with Session(self.engine) as db:
            ctx = build_rewrite_memory_context(
                db, project_id=1, chapter_id=10, mode="polish",
            )
        self.assertTrue(ctx.story_state)
        self.assertEqual(ctx.story_state.get("chapter_goal"), "突破瓶颈")

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_polish_has_world_rules(self, mock_t: MagicMock, mock_k: MagicMock) -> None:
        mock_t.return_value = []
        mock_k.return_value = []
        with Session(self.engine) as db:
            ctx = build_rewrite_memory_context(
                db, project_id=1, chapter_id=10, mode="polish",
            )
        self.assertEqual(len(ctx.world_rules), 1)
        self.assertEqual(ctx.world_rules[0]["title"], "修炼需灵石")

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_polish_no_episodes(self, mock_t: MagicMock, mock_k: MagicMock) -> None:
        mock_t.return_value = []
        mock_k.return_value = []
        with Session(self.engine) as db:
            ctx = build_rewrite_memory_context(
                db, project_id=1, chapter_id=10, mode="polish",
            )
        self.assertEqual(ctx.recent_episodes, [])

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_polish_no_temporal_facts(self, mock_t: MagicMock, mock_k: MagicMock) -> None:
        mock_t.return_value = []
        mock_k.return_value = []
        with Session(self.engine) as db:
            ctx = build_rewrite_memory_context(
                db, project_id=1, chapter_id=10, mode="polish",
            )
        self.assertEqual(ctx.temporal_facts, [])

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_polish_no_character_knowledge(self, mock_t: MagicMock, mock_k: MagicMock) -> None:
        mock_t.return_value = []
        mock_k.return_value = []
        with Session(self.engine) as db:
            ctx = build_rewrite_memory_context(
                db, project_id=1, chapter_id=10, mode="polish",
            )
        self.assertEqual(ctx.character_knowledge, [])

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_polish_never_triggers_l5(self, mock_t: MagicMock, mock_k: MagicMock) -> None:
        mock_t.return_value = []
        mock_k.return_value = []
        with Session(self.engine) as db:
            ctx = build_rewrite_memory_context(
                db, project_id=1, chapter_id=10, mode="polish",
            )
        self.assertFalse(ctx.l5_triggered)


class TestExpandInclusion(RewriteMemoryTestBase):
    """Expand mode should include L1+L2+L3+L4."""

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_expand_has_episodes(self, mock_t: MagicMock, mock_k: MagicMock) -> None:
        mock_t.return_value = []
        mock_k.return_value = []
        with Session(self.engine) as db:
            ctx = build_rewrite_memory_context(
                db, project_id=1, chapter_id=10, mode="expand",
                selection_text="林澈开始修炼", chapter_index=10,
            )
        self.assertTrue(len(ctx.recent_episodes) > 0)

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_expand_has_character_knowledge(self, mock_t: MagicMock, mock_k: MagicMock) -> None:
        mock_t.return_value = []
        mock_k.return_value = []
        with Session(self.engine) as db:
            ctx = build_rewrite_memory_context(
                db, project_id=1, chapter_id=10, mode="expand",
                selection_text="林澈开始修炼", chapter_index=10,
            )
        self.assertTrue(len(ctx.character_knowledge) > 0)
        names = [ck.character_name for ck in ctx.character_knowledge]
        self.assertIn("林澈", names)

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_expand_has_temporal_facts(self, mock_t: MagicMock, mock_k: MagicMock) -> None:
        from app.services.book_memory.temporal_query import TemporalFact

        mock_t.return_value = [
            TemporalFact(predicate="at", object="林澈在丹房", source="graphiti"),
        ]
        mock_k.return_value = []
        with Session(self.engine) as db:
            ctx = build_rewrite_memory_context(
                db, project_id=1, chapter_id=10, mode="expand",
                selection_text="林澈开始修炼", chapter_index=10,
            )
        self.assertTrue(len(ctx.temporal_facts) > 0)


if __name__ == "__main__":
    unittest.main()
