"""Unit tests for the Graphiti temporal graph adapter.

These tests verify the adapter layer without requiring a live Neo4j
instance or real LLM calls.  All external dependencies are mocked.
"""

import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.book_memory.entity_types import (
    FICTION_ENTITY_TYPES,
    FICTION_EXTRACTION_INSTRUCTIONS,
    FictionCharacter,
    FictionFaction,
    FictionItem,
    FictionLocation,
    FictionWorldRule,
)
from app.services.book_memory.graphiti_adapter import (
    _chapter_to_datetime,
    _format_episode_body,
    _group_id,
    _normalize_search_results,
)


class TestEntityTypes(unittest.TestCase):
    """Verify the fiction entity type registry."""

    def test_all_entity_types_present(self) -> None:
        self.assertIn("Character", FICTION_ENTITY_TYPES)
        self.assertIn("Location", FICTION_ENTITY_TYPES)
        self.assertIn("Faction", FICTION_ENTITY_TYPES)
        self.assertIn("Item", FICTION_ENTITY_TYPES)
        self.assertIn("WorldRule", FICTION_ENTITY_TYPES)

    def test_entity_types_are_pydantic_models(self) -> None:
        from pydantic import BaseModel

        for name, model in FICTION_ENTITY_TYPES.items():
            self.assertTrue(
                issubclass(model, BaseModel),
                f"{name} should be a Pydantic BaseModel subclass",
            )

    def test_fiction_character_fields(self) -> None:
        char = FictionCharacter(role="protagonist", status="alive", affiliation="天剑宗")
        self.assertEqual(char.role, "protagonist")
        self.assertEqual(char.status, "alive")
        self.assertEqual(char.affiliation, "天剑宗")

    def test_fiction_location_fields(self) -> None:
        loc = FictionLocation(location_type="mountain", controlled_by="玄天门")
        self.assertEqual(loc.location_type, "mountain")
        self.assertEqual(loc.controlled_by, "玄天门")

    def test_fiction_world_rule_fields(self) -> None:
        rule = FictionWorldRule(scope="universal", enforceable=True)
        self.assertTrue(rule.enforceable)

    def test_extraction_instructions_non_empty(self) -> None:
        self.assertTrue(len(FICTION_EXTRACTION_INSTRUCTIONS) > 50)
        self.assertIn("角色", FICTION_EXTRACTION_INSTRUCTIONS)


class TestHelpers(unittest.TestCase):
    """Test adapter helper functions."""

    def test_chapter_to_datetime_ordering(self) -> None:
        dt1 = _chapter_to_datetime(1)
        dt5 = _chapter_to_datetime(5)
        dt10 = _chapter_to_datetime(10)
        self.assertLess(dt1, dt5)
        self.assertLess(dt5, dt10)

    def test_chapter_to_datetime_zero(self) -> None:
        dt0 = _chapter_to_datetime(0)
        self.assertEqual(dt0, datetime(2000, 1, 1, tzinfo=timezone.utc))

    def test_chapter_to_datetime_negative_clamped(self) -> None:
        dt_neg = _chapter_to_datetime(-5)
        dt0 = _chapter_to_datetime(0)
        self.assertEqual(dt_neg, dt0)

    def test_group_id_format(self) -> None:
        self.assertEqual(_group_id(42), "project-42")
        self.assertEqual(_group_id(1), "project-1")

    def test_format_episode_body_full(self) -> None:
        ep = {
            "title": "秘密揭露",
            "summary": "林澈发现了师门的秘密",
            "participants": ["林澈", "苏青"],
            "location": "藏经阁",
            "event_type": "revelation",
        }
        body = _format_episode_body(ep)
        self.assertIn("秘密揭露", body)
        self.assertIn("林澈发现了师门的秘密", body)
        self.assertIn("林澈", body)
        self.assertIn("藏经阁", body)
        self.assertIn("revelation", body)

    def test_format_episode_body_empty(self) -> None:
        body = _format_episode_body({})
        self.assertEqual(body, "")

    def test_format_episode_body_partial(self) -> None:
        body = _format_episode_body({"title": "决战", "summary": "最终对决"})
        self.assertIn("决战", body)
        self.assertIn("最终对决", body)
        self.assertNotIn("参与角色", body)


class TestNormalizeSearchResults(unittest.TestCase):
    """Test search result normalisation."""

    def test_empty_results(self) -> None:
        mock_results = MagicMock()
        mock_results.edges = []
        mock_results.nodes = []
        out = _normalize_search_results(mock_results)
        self.assertEqual(out, [])

    def test_edge_normalisation(self) -> None:
        edge = MagicMock()
        edge.name = "trusts"
        edge.fact = "A trusts B"
        edge.valid_at = datetime(2000, 1, 5, tzinfo=timezone.utc)
        edge.invalid_at = None
        edge.source_description = "chapter:1:episode:1"
        edge.uuid = "edge-uuid-1"

        mock_results = MagicMock()
        mock_results.edges = [edge]
        mock_results.nodes = []

        out = _normalize_search_results(mock_results)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["type"], "edge")
        self.assertEqual(out[0]["name"], "trusts")
        self.assertEqual(out[0]["fact"], "A trusts B")
        self.assertIsNotNone(out[0]["valid_at"])

    def test_node_normalisation(self) -> None:
        node = MagicMock()
        node.name = "林澈"
        node.summary = "天剑宗弟子，主角"
        node.uuid = "node-uuid-1"

        mock_results = MagicMock()
        mock_results.edges = []
        mock_results.nodes = [node]

        out = _normalize_search_results(mock_results)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["type"], "node")
        self.assertEqual(out[0]["name"], "林澈")
        self.assertIn("天剑宗", out[0]["fact"])


class TestIngestChapterEpisodes(unittest.TestCase):
    """Test the ingest_chapter_episodes function with mocked Graphiti."""

    @patch("app.services.book_memory.graphiti_adapter.get_graphiti")
    def test_skipped_when_disabled(self, mock_get: MagicMock) -> None:
        mock_get.return_value = None
        from app.services.book_memory.graphiti_adapter import ingest_chapter_episodes

        results = ingest_chapter_episodes(
            project_id=1,
            chapter_id=10,
            chapter_index=3,
            episodes=[{"title": "test", "summary": "test summary"}],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "skipped")
        self.assertEqual(results[0]["reason"], "graphiti_disabled")

    @patch("app.services.book_memory.graphiti_adapter.get_graphiti")
    @patch("app.services.book_memory.graphiti_adapter._run_async")
    def test_successful_ingest(self, mock_run: MagicMock, mock_get: MagicMock) -> None:
        mock_graphiti = MagicMock()
        mock_get.return_value = mock_graphiti
        mock_run.return_value = MagicMock()  # AddEpisodeResults

        from app.services.book_memory.graphiti_adapter import ingest_chapter_episodes

        episodes = [
            {"title": "事件一", "summary": "第一个事件", "episode_index": 1},
            {"title": "事件二", "summary": "第二个事件", "episode_index": 2},
        ]
        results = ingest_chapter_episodes(
            project_id=1,
            chapter_id=10,
            chapter_index=3,
            episodes=episodes,
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["status"] == "ok" for r in results))

    @patch("app.services.book_memory.graphiti_adapter.get_graphiti")
    @patch("app.services.book_memory.graphiti_adapter._run_async")
    def test_partial_failure(self, mock_run: MagicMock, mock_get: MagicMock) -> None:
        mock_graphiti = MagicMock()
        mock_get.return_value = mock_graphiti
        mock_run.side_effect = [MagicMock(), RuntimeError("LLM timeout")]

        from app.services.book_memory.graphiti_adapter import ingest_chapter_episodes

        episodes = [
            {"title": "成功", "summary": "ok", "episode_index": 1},
            {"title": "失败", "summary": "fail", "episode_index": 2},
        ]
        results = ingest_chapter_episodes(
            project_id=1,
            chapter_id=10,
            chapter_index=3,
            episodes=episodes,
        )
        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(results[1]["status"], "error")

    @patch("app.services.book_memory.graphiti_adapter.get_graphiti")
    def test_empty_body_skipped(self, mock_get: MagicMock) -> None:
        mock_get.return_value = MagicMock()

        from app.services.book_memory.graphiti_adapter import ingest_chapter_episodes

        results = ingest_chapter_episodes(
            project_id=1,
            chapter_id=10,
            chapter_index=3,
            episodes=[{"title": "", "summary": "", "episode_index": 1}],
        )
        self.assertEqual(results[0]["status"], "skipped")
        self.assertEqual(results[0]["reason"], "empty_body")


class TestSearchTemporalFacts(unittest.TestCase):
    """Test temporal fact search with mocked Graphiti."""

    @patch("app.services.book_memory.graphiti_adapter.get_graphiti")
    def test_returns_empty_when_disabled(self, mock_get: MagicMock) -> None:
        mock_get.return_value = None
        from app.services.book_memory.graphiti_adapter import search_temporal_facts

        result = search_temporal_facts(project_id=1, query="test")
        self.assertEqual(result, [])


class TestSearchCharacterKnowledge(unittest.TestCase):
    """Test character knowledge queries with chapter filtering."""

    @patch("app.services.book_memory.graphiti_adapter.search_temporal_facts")
    def test_chapter_filtering(self, mock_search: MagicMock) -> None:
        # Fact valid from chapter 3, no invalid_at.
        ch3_dt = datetime(2000, 1, 4, tzinfo=timezone.utc)  # chapter 3 = epoch + 3 days
        mock_search.return_value = [
            {
                "type": "edge",
                "name": "knows_secret",
                "fact": "林澈知道秘密",
                "valid_at": ch3_dt,
                "invalid_at": None,
            },
            {
                "type": "edge",
                "name": "knows_late_secret",
                "fact": "林澈知道后期秘密",
                "valid_at": datetime(2000, 1, 11, tzinfo=timezone.utc),  # chapter 10
                "invalid_at": None,
            },
        ]

        from app.services.book_memory.graphiti_adapter import search_character_knowledge

        # Query at chapter 5 — only the first fact should pass.
        results = search_character_knowledge(
            project_id=1,
            character_name="林澈",
            at_chapter=5,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "knows_secret")

    @patch("app.services.book_memory.graphiti_adapter.search_temporal_facts")
    def test_no_chapter_filter(self, mock_search: MagicMock) -> None:
        mock_search.return_value = [
            {"type": "edge", "name": "f1", "fact": "fact 1", "valid_at": None, "invalid_at": None},
            {"type": "edge", "name": "f2", "fact": "fact 2", "valid_at": None, "invalid_at": None},
        ]

        from app.services.book_memory.graphiti_adapter import search_character_knowledge

        results = search_character_knowledge(
            project_id=1,
            character_name="林澈",
            at_chapter=None,
        )
        self.assertEqual(len(results), 2)


class TestExtractionServiceInstructor(unittest.TestCase):
    """Verify the Instructor integration in extraction_service."""

    def test_schemas_importable(self) -> None:
        from app.services.book_memory.extraction_service import (
            BookMemoryEpisodeExtraction,
            BookMemoryKnowledgeClaimExtraction,
            BookMemoryStructuredExtraction,
        )

        self.assertTrue(hasattr(BookMemoryStructuredExtraction, "model_fields"))
        self.assertIn("episodes", BookMemoryStructuredExtraction.model_fields)
        self.assertIn("knowledge_claims", BookMemoryStructuredExtraction.model_fields)

    def test_build_payload_text(self) -> None:
        from app.services.book_memory.extraction_service import _build_payload_text

        # Create minimal mock objects.
        chapter = MagicMock()
        chapter.id = 1
        chapter.volume_id = 1
        chapter.chapter_index = 3
        chapter.title = "第三章"
        chapter.content = "林澈走进了山门。"

        payload = _build_payload_text(
            project_id=1,
            chapter=chapter,
            scene_beats=None,
            character_profiles=None,
            previous_snapshot=None,
        )
        self.assertIn("第三章", payload)
        self.assertIn("林澈", payload)
        self.assertIn('"chapter_index": 3', payload)


if __name__ == "__main__":
    unittest.main()
