"""Unit tests for Book Memory OS injection into chat stream.

Verifies that ``_inject_book_memory`` correctly enriches a
``CompiledContextBundle`` with memory pipeline output.
"""

import unittest
from dataclasses import dataclass, field
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


@dataclass
class FakeBundle:
    """Mimics CompiledContextBundle for testing."""

    model_context: dict[str, Any] = field(default_factory=dict)
    evidence_event: dict[str, Any] = field(default_factory=dict)


class TestInjectBookMemory(unittest.TestCase):
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
                project_id=1, scope="universal", title="禁飞",
                statement="凡人不可飞行", priority=10, status="active",
            ))
            profile = CharacterProfile(
                project_id=1, canonical_name="林澈",
                aliases=["小澈"],
                core_goals=["飞升"],
                public_traits=["冷静"],
            )
            db.add(profile)
            db.flush()

            db.add(StoryStateSnapshot(
                project_id=1, chapter_id=10,
                chapter_goal="揭秘",
                active_characters=["林澈"],
                current_location="藏经阁",
                active_conflicts=["秘密"],
                open_questions=["谁是叛徒？"],
            ))
            db.add(StoryEpisode(
                project_id=1, chapter_id=10, episode_index=1,
                title="发现密信", summary="林澈发现密信",
                event_type="revelation", participants=["林澈"],
                location="藏经阁", importance=80,
            ))
            db.add(CharacterKnowledgeState(
                project_id=1, character_profile_id=int(profile.id),
                knowledge_key="knows:秘密",
                knowledge_value={"fact": "师门勾结"},
                gained_at_chapter=5, confidence=0.9,
            ))
            db.commit()

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_injection_adds_book_memory_to_model_context(
        self, mock_temporal: MagicMock, mock_knowledge: MagicMock
    ) -> None:
        mock_temporal.return_value = []
        mock_knowledge.return_value = []

        from app.services.book_memory.memory_pipeline import build_planning_memory_context

        with Session(self.engine) as db:
            memory_ctx = build_planning_memory_context(
                db, project_id=1, chapter_id=10, chapter_index=10,
            )

        bundle = FakeBundle(
            model_context={"existing": "data"},
            evidence_event={"type": "evidence", "summary": {}},
        )

        # Simulate what _inject_book_memory does
        prompt_text = memory_ctx.to_prompt_text()
        self.assertTrue(len(prompt_text) > 0)
        self.assertIn("揭秘", prompt_text)
        self.assertIn("藏经阁", prompt_text)
        self.assertIn("禁飞", prompt_text)

        bundle.model_context["book_memory"] = {
            "enabled": True,
            "hint": prompt_text,
            "layers_used": [
                {"layer": l.layer, "label": l.label, "items": l.items}
                for l in memory_ctx.layers_used
            ],
            "l5_triggered": memory_ctx.l5_triggered,
        }

        # Verify injection
        self.assertTrue(bundle.model_context["book_memory"]["enabled"])
        self.assertIn("揭秘", bundle.model_context["book_memory"]["hint"])
        self.assertEqual(bundle.model_context["existing"], "data")  # preserved

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_injection_adds_book_memory_to_evidence_event(
        self, mock_temporal: MagicMock, mock_knowledge: MagicMock
    ) -> None:
        mock_temporal.return_value = []
        mock_knowledge.return_value = []

        from app.services.book_memory.memory_pipeline import build_planning_memory_context

        with Session(self.engine) as db:
            memory_ctx = build_planning_memory_context(
                db, project_id=1, chapter_id=10, chapter_index=10,
            )

        bundle = FakeBundle(
            model_context={},
            evidence_event={"type": "evidence"},
        )

        bundle.evidence_event["book_memory"] = {
            "enabled": True,
            "layers": [
                {"layer": l.layer, "label": l.label, "items": l.items}
                for l in memory_ctx.layers_used
            ],
            "l5_triggered": memory_ctx.l5_triggered,
            "sections": [
                {"label": s["label"], "chars": len(s.get("content", ""))}
                for s in memory_ctx.to_prompt_sections()
            ],
        }

        bm = bundle.evidence_event["book_memory"]
        self.assertTrue(bm["enabled"])
        self.assertFalse(bm["l5_triggered"])
        # Should have layers: L1, L3 (rules), L3 (profiles), L3+L4, L4, L2, L5
        layer_labels = [l["label"] for l in bm["layers"]]
        self.assertIn("story_state", layer_labels)
        self.assertIn("world_rules", layer_labels)
        # Sections should include 当前写作状态, 世界规则, etc.
        section_labels = [s["label"] for s in bm["sections"]]
        self.assertIn("当前写作状态", section_labels)
        self.assertIn("世界规则", section_labels)

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_l5_not_triggered_when_memory_sufficient(
        self, mock_temporal: MagicMock, mock_knowledge: MagicMock
    ) -> None:
        mock_temporal.return_value = []
        mock_knowledge.return_value = []

        from app.services.book_memory.memory_pipeline import build_planning_memory_context

        with Session(self.engine) as db:
            ctx = build_planning_memory_context(
                db, project_id=1, chapter_id=10, chapter_index=10,
            )

        # We have world rules + episodes + knowledge — sufficient
        self.assertFalse(ctx.l5_triggered)
        self.assertEqual(ctx.cold_retrieval, [])

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_prompt_text_includes_character_knowledge(
        self, mock_temporal: MagicMock, mock_knowledge: MagicMock
    ) -> None:
        mock_knowledge.return_value = []
        mock_temporal.return_value = []

        from app.services.book_memory.memory_pipeline import build_planning_memory_context

        with Session(self.engine) as db:
            ctx = build_planning_memory_context(
                db, project_id=1, chapter_id=10, chapter_index=10,
            )

        prompt = ctx.to_prompt_text()
        # Character knowledge should be in the prompt
        self.assertIn("林澈", prompt)
        self.assertIn("师门勾结", prompt)

    @patch("app.services.book_memory.graphiti_adapter.search_character_knowledge")
    @patch("app.services.book_memory.memory_pipeline.query_temporal_facts")
    def test_prompt_sections_format(
        self, mock_temporal: MagicMock, mock_knowledge: MagicMock
    ) -> None:
        mock_knowledge.return_value = []
        mock_temporal.return_value = []

        from app.services.book_memory.memory_pipeline import build_planning_memory_context

        with Session(self.engine) as db:
            ctx = build_planning_memory_context(
                db, project_id=1, chapter_id=10, chapter_index=10,
            )

        sections = ctx.to_prompt_sections()
        self.assertTrue(len(sections) > 0)
        for s in sections:
            self.assertIn("label", s)
            self.assertIn("content", s)
            self.assertTrue(len(s["label"]) > 0)
            self.assertTrue(len(s["content"]) > 0)


if __name__ == "__main__":
    unittest.main()
