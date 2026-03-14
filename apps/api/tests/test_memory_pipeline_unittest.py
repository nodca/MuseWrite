"""Unit tests for temporal_query.py and memory_pipeline.py.

Uses in-memory SQLite to verify the Postgres query paths.
Graphiti calls are mocked.
"""

import unittest
from unittest.mock import MagicMock, patch

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import app.models  # noqa: F401  — register all SQLModel tables
from app.models.book_memory import (
    CharacterKnowledgeState,
    CharacterProfile,
    StoryEpisode,
    StoryStateSnapshot,
    WorldRule,
)
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


class TemporalQueryTestBase(unittest.TestCase):
    """Shared setup for temporal query tests."""

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)
        self._seed_data()

    def tearDown(self) -> None:
        SQLModel.metadata.drop_all(self.engine)
        self.engine.dispose()

    def _seed_data(self) -> None:
        with Session(self.engine) as db:
            # World rules
            db.add(WorldRule(
                project_id=1, scope="universal", title="禁止飞行",
                statement="凡人不可飞行", priority=10, status="active",
            ))
            db.add(WorldRule(
                project_id=1, scope="regional", title="灵气稀薄",
                statement="此地灵气稀薄", priority=20, status="active",
            ))
            db.add(WorldRule(
                project_id=1, scope="obsolete", title="旧规则",
                statement="已废弃", priority=99, status="deprecated",
            ))

            # Character profiles
            profile_a = CharacterProfile(
                project_id=1, canonical_name="林澈",
                aliases=["小澈", "林师弟"],
                core_goals=["复仇", "飞升"],
                public_traits=["冷静", "坚毅"],
                private_traits=["内心恐惧"],
                fears=["失去同伴"],
                taboos=["不伤无辜"],
                default_voice_notes="语气冷淡",
            )
            profile_b = CharacterProfile(
                project_id=1, canonical_name="苏青",
                aliases=["苏师姐"],
                core_goals=["守护师门"],
                public_traits=["温柔"],
            )
            db.add(profile_a)
            db.add(profile_b)
            db.flush()

            # Story state snapshot
            db.add(StoryStateSnapshot(
                project_id=1, chapter_id=10, scene_beat_id=None,
                chapter_goal="揭示师门秘密",
                active_characters=["林澈", "苏青"],
                current_location="藏经阁",
                active_conflicts=["师门秘密"],
                open_questions=["谁是叛徒？"],
            ))

            # Episodes
            db.add(StoryEpisode(
                project_id=1, chapter_id=10, episode_index=1,
                title="发现密信", summary="林澈在藏经阁发现了一封密信",
                event_type="revelation", participants=["林澈"],
                location="藏经阁", importance=80,
            ))
            db.add(StoryEpisode(
                project_id=1, chapter_id=10, episode_index=2,
                title="苏青来访", summary="苏青前来询问情况",
                event_type="dialogue", participants=["林澈", "苏青"],
                location="藏经阁", importance=60,
            ))

            # Character knowledge
            db.add(CharacterKnowledgeState(
                project_id=1, character_profile_id=int(profile_a.id),
                knowledge_key="knows:密信内容",
                knowledge_value={"fact": "师门曾与魔族勾结", "evidence": "密信原文"},
                gained_at_chapter=5, lost_at_chapter=None, confidence=0.9,
            ))
            db.add(CharacterKnowledgeState(
                project_id=1, character_profile_id=int(profile_a.id),
                knowledge_key="knows:后期秘密",
                knowledge_value={"fact": "掌门已死"},
                gained_at_chapter=15, lost_at_chapter=None, confidence=0.8,
            ))
            db.add(CharacterKnowledgeState(
                project_id=1, character_profile_id=int(profile_b.id),
                knowledge_key="knows:苏青独知",
                knowledge_value={"fact": "苏青知道解药配方"},
                gained_at_chapter=3, lost_at_chapter=None, confidence=0.85,
            ))

            db.commit()


class TestQueryStoryState(TemporalQueryTestBase):
    def test_returns_state_for_chapter(self) -> None:
        with Session(self.engine) as db:
            state = query_story_state(db, project_id=1, chapter_id=10)
        self.assertEqual(state["chapter_goal"], "揭示师门秘密")
        self.assertIn("林澈", state["active_characters"])
        self.assertEqual(state["current_location"], "藏经阁")

    def test_returns_empty_for_missing_chapter(self) -> None:
        with Session(self.engine) as db:
            state = query_story_state(db, project_id=1, chapter_id=999)
        self.assertEqual(state, {})


class TestQueryRecentEpisodes(TemporalQueryTestBase):
    def test_returns_episodes_ordered(self) -> None:
        with Session(self.engine) as db:
            eps = query_recent_episodes(db, project_id=1, chapter_id=10)
        self.assertEqual(len(eps), 2)
        self.assertEqual(eps[0]["title"], "发现密信")
        self.assertEqual(eps[1]["title"], "苏青来访")

    def test_returns_empty_for_missing_chapter(self) -> None:
        with Session(self.engine) as db:
            eps = query_recent_episodes(db, project_id=1, chapter_id=999)
        self.assertEqual(eps, [])

    def test_respects_limit(self) -> None:
        with Session(self.engine) as db:
            eps = query_recent_episodes(db, project_id=1, chapter_id=10, limit=1)
        self.assertEqual(len(eps), 1)


class TestQueryWorldRules(TemporalQueryTestBase):
    def test_returns_active_rules_only(self) -> None:
        with Session(self.engine) as db:
            rules = query_active_world_rules(db, project_id=1)
        self.assertEqual(len(rules), 2)
        titles = [r["title"] for r in rules]
        self.assertIn("禁止飞行", titles)
        self.assertNotIn("旧规则", titles)

    def test_ordered_by_priority(self) -> None:
        with Session(self.engine) as db:
            rules = query_active_world_rules(db, project_id=1)
        self.assertEqual(rules[0]["title"], "禁止飞行")  # priority 10


class TestQueryCharacterProfiles(TemporalQueryTestBase):
    def test_returns_all_profiles(self) -> None:
        with Session(self.engine) as db:
            profiles = query_character_profiles(db, project_id=1)
        self.assertEqual(len(profiles), 2)

    def test_filter_by_name(self) -> None:
        with Session(self.engine) as db:
            profiles = query_character_profiles(db, project_id=1, names=["林澈"])
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0]["canonical_name"], "林澈")

    def test_filter_by_alias(self) -> None:
        with Session(self.engine) as db:
            profiles = query_character_profiles(db, project_id=1, names=["苏师姐"])
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0]["canonical_name"], "苏青")


class TestQueryCharacterKnowledgeAtChapter(TemporalQueryTestBase):
    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    def test_knows_fact_at_chapter_10(self, mock_graphiti: MagicMock) -> None:
        mock_graphiti.return_value = []
        with Session(self.engine) as db:
            bundle = query_character_knowledge_at_chapter(
                db, project_id=1, character_name="林澈", at_chapter=10,
            )
        self.assertIsInstance(bundle, CharacterKnowledgeBundle)
        self.assertEqual(bundle.character_name, "林澈")
        known_objects = [f.object for f in bundle.known_facts]
        self.assertIn("师门曾与魔族勾结", known_objects)
        # "掌门已死" gained at ch15 should NOT be known at ch10
        self.assertNotIn("掌门已死", known_objects)

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    def test_knows_all_at_chapter_20(self, mock_graphiti: MagicMock) -> None:
        mock_graphiti.return_value = []
        with Session(self.engine) as db:
            bundle = query_character_knowledge_at_chapter(
                db, project_id=1, character_name="林澈", at_chapter=20,
            )
        known_objects = [f.object for f in bundle.known_facts]
        self.assertIn("师门曾与魔族勾结", known_objects)
        self.assertIn("掌门已死", known_objects)

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    def test_withheld_facts(self, mock_graphiti: MagicMock) -> None:
        mock_graphiti.return_value = []
        with Session(self.engine) as db:
            bundle = query_character_knowledge_at_chapter(
                db, project_id=1, character_name="林澈", at_chapter=10,
            )
        withheld_objects = [f.object for f in bundle.withheld_facts]
        # 苏青 knows "解药配方" but 林澈 does not
        self.assertIn("苏青知道解药配方", withheld_objects)

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    def test_graphiti_facts_merged(self, mock_graphiti: MagicMock) -> None:
        mock_graphiti.return_value = [
            {"name": "trusts", "fact": "林澈信任苏青"},
        ]
        with Session(self.engine) as db:
            bundle = query_character_knowledge_at_chapter(
                db, project_id=1, character_name="林澈", at_chapter=10,
            )
        known_objects = [f.object for f in bundle.known_facts]
        self.assertIn("林澈信任苏青", known_objects)
        # Check source attribution
        graphiti_fact = [f for f in bundle.known_facts if f.source == "graphiti"]
        self.assertTrue(len(graphiti_fact) > 0)

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    def test_unknown_character_returns_empty(self, mock_graphiti: MagicMock) -> None:
        mock_graphiti.return_value = []
        with Session(self.engine) as db:
            bundle = query_character_knowledge_at_chapter(
                db, project_id=1, character_name="不存在", at_chapter=10,
            )
        self.assertEqual(bundle.known_facts, [])


class TestQueryTemporalFacts(unittest.TestCase):
    @patch("app.services.book_memory.graphiti_adapter.search_temporal_facts")
    def test_returns_empty_on_graphiti_failure(self, mock_search: MagicMock) -> None:
        mock_search.side_effect = RuntimeError("offline")
        facts = query_temporal_facts(project_id=1, query="test")
        self.assertEqual(facts, [])


class TestMemoryPipeline(TemporalQueryTestBase):
    """Integration tests for the memory pipeline assemblers."""

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_planning_context_assembles_all_layers(
        self, mock_temporal: MagicMock, mock_knowledge: MagicMock
    ) -> None:
        mock_knowledge.return_value = []
        mock_temporal.return_value = [
            TemporalFact(predicate="conflict", object="师门内斗", source="graphiti"),
        ]

        from app.services.book_memory.memory_pipeline import build_planning_memory_context

        with Session(self.engine) as db:
            ctx = build_planning_memory_context(
                db, project_id=1, chapter_id=10, chapter_index=10,
            )

        # L1
        self.assertEqual(ctx.story_state.get("chapter_goal"), "揭示师门秘密")
        # L3
        self.assertEqual(len(ctx.world_rules), 2)
        # L4
        self.assertEqual(len(ctx.temporal_facts), 1)
        # L2
        self.assertEqual(len(ctx.recent_episodes), 2)
        # L5 should NOT trigger (we have enough memory)
        self.assertFalse(ctx.l5_triggered)
        # Provenance layers
        layer_labels = [l.label for l in ctx.layers_used]
        self.assertIn("story_state", layer_labels)
        self.assertIn("world_rules", layer_labels)
        self.assertIn("temporal_graph", layer_labels)
        self.assertIn("episodes", layer_labels)

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_planning_context_to_prompt_text(
        self, mock_temporal: MagicMock, mock_knowledge: MagicMock
    ) -> None:
        mock_knowledge.return_value = []
        mock_temporal.return_value = []

        from app.services.book_memory.memory_pipeline import build_planning_memory_context

        with Session(self.engine) as db:
            ctx = build_planning_memory_context(
                db, project_id=1, chapter_id=10, chapter_index=10,
            )

        text = ctx.to_prompt_text()
        self.assertIn("揭示师门秘密", text)
        self.assertIn("藏经阁", text)
        self.assertIn("禁止飞行", text)

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_rewrite_polish_skips_l2_l4_l5(
        self, mock_temporal: MagicMock, mock_knowledge: MagicMock
    ) -> None:
        mock_knowledge.return_value = []
        mock_temporal.return_value = []

        from app.services.book_memory.memory_pipeline import build_rewrite_memory_context

        with Session(self.engine) as db:
            ctx = build_rewrite_memory_context(
                db, project_id=1, chapter_id=10, mode="polish",
            )

        # Polish should only have L1 + L3
        self.assertTrue(ctx.story_state)
        self.assertTrue(ctx.world_rules)
        self.assertEqual(ctx.recent_episodes, [])
        self.assertEqual(ctx.temporal_facts, [])
        self.assertFalse(ctx.l5_triggered)

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_rewrite_expand_includes_l2_l4(
        self, mock_temporal: MagicMock, mock_knowledge: MagicMock
    ) -> None:
        mock_knowledge.return_value = []
        mock_temporal.return_value = [
            TemporalFact(predicate="at", object="林澈在藏经阁", source="graphiti"),
        ]

        from app.services.book_memory.memory_pipeline import build_rewrite_memory_context

        with Session(self.engine) as db:
            ctx = build_rewrite_memory_context(
                db,
                project_id=1,
                chapter_id=10,
                mode="expand",
                selection_text="林澈翻开了密信",
                chapter_index=10,
            )

        self.assertTrue(ctx.recent_episodes)
        self.assertTrue(ctx.temporal_facts)


if __name__ == "__main__":
    unittest.main()
