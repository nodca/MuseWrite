import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.config import settings
from app.models.content import ProjectChapter
import app.services.chat_service.entity_graph as eg_module


class StructuredSyncMigrationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "lightrag_llm_model": settings.lightrag_llm_model,
            "lightrag_llm_base_url": settings.lightrag_llm_base_url,
            "lightrag_llm_api_key": settings.lightrag_llm_api_key,
            "graph_coref_llm_enabled": eg_module._COREF_LLM_ENABLED,
            "memory_consolidation_enabled": settings.memory_consolidation_enabled,
            "consistency_audit_llm_enabled": settings.consistency_audit_llm_enabled,
        }
        settings.lightrag_llm_model = "mock-model"
        settings.lightrag_llm_base_url = "https://mock-llm.local/v1"
        settings.lightrag_llm_api_key = "sk-test"
        eg_module._COREF_LLM_ENABLED = True
        settings.memory_consolidation_enabled = True
        settings.consistency_audit_llm_enabled = True

    def tearDown(self) -> None:
        settings.lightrag_llm_model = self._snapshot["lightrag_llm_model"]
        settings.lightrag_llm_base_url = self._snapshot["lightrag_llm_base_url"]
        settings.lightrag_llm_api_key = self._snapshot["lightrag_llm_api_key"]
        eg_module._COREF_LLM_ENABLED = self._snapshot["graph_coref_llm_enabled"]
        settings.memory_consolidation_enabled = self._snapshot["memory_consolidation_enabled"]
        settings.consistency_audit_llm_enabled = self._snapshot["consistency_audit_llm_enabled"]

    @patch("app.services.context_compiler.self_review.generate_structured_sync")
    def test_context_compiler_self_reflective_judge_uses_generate_structured_sync(
        self,
        mock_generate_structured_sync,
    ) -> None:
        from app.services.context_compiler import _call_self_reflective_judge_llm

        mock_generate_structured_sync.return_value = SimpleNamespace(
            parsed=SimpleNamespace(
                needs_refine=True,
                confidence=0.91,
                issues=["negative_constraint_conflict"],
                followup_queries=["主角 禁忌约束 重写"],
            ),
            raw_text="{}",
            usage={"provider": "openai_compatible"},
        )

        result = _call_self_reflective_judge_llm(
            user_input="让主角直接杀掉无辜者。",
            intent="brainstorm",
            temperature_profile="brainstorm",
            chapter_preview="主角正在犹豫。",
            scene_beat_text="关键转折",
            dsl_hits=[],
            graph_facts=[],
            semantic_hits=[],
            negative_constraints=[{"text": "禁止主角滥杀无辜", "source": "dsl", "title": "角色底线"}],
            max_queries=2,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result["needs_refine"])
        self.assertIn("negative_constraint_conflict", result["issues"])
        self.assertEqual(result["followup_queries"], ["主角 禁忌约束 重写"])
        mock_generate_structured_sync.assert_called_once()

    @patch("app.services.chat_service.entity_graph.generate_structured_sync")
    def test_chat_service_coref_rewrite_uses_generate_structured_sync(self, mock_generate_structured_sync) -> None:
        from app.services.chat_service import _rewrite_chunk_with_llm_coref

        mock_generate_structured_sync.return_value = SimpleNamespace(
            parsed=SimpleNamespace(rewritten_text="林默握紧长剑。", applied=True, confidence=0.93),
            raw_text="{}",
            usage={"provider": "openai_compatible"},
        )

        rewritten, meta = _rewrite_chunk_with_llm_coref(
            "他握紧长剑。",
            context_summary="林默正在对峙。",
            anchor_canonical="林默",
        )

        self.assertEqual(rewritten, "林默握紧长剑。")
        self.assertTrue(meta["applied"])
        self.assertEqual(meta["reason"], "llm_applied")
        mock_generate_structured_sync.assert_called_once()

    @patch("app.services.chat_service.volumes.generate_structured_sync")
    def test_chat_service_memory_consolidation_uses_generate_structured_sync(self, mock_generate_structured_sync) -> None:
        from app.services.chat_service import _call_volume_memory_consolidation_llm

        mock_generate_structured_sync.return_value = SimpleNamespace(
            parsed=SimpleNamespace(
                facts=["林默在第一章获得戒指。", "第二章他开始怀疑戒指来历。"],
            ),
            raw_text="{}",
            usage={"provider": "openai_compatible"},
        )
        chapters = [
            ProjectChapter(project_id=1, chapter_index=1, title="第一章", content="林默捡到一枚戒指。"),
            ProjectChapter(project_id=1, chapter_index=2, title="第二章", content="他开始怀疑戒指的来历。"),
        ]

        facts, source = _call_volume_memory_consolidation_llm(
            volume_title="卷一",
            volume_outline="戒指引出更深的谜团。",
            chapters=chapters,
            max_facts=4,
        )

        self.assertEqual(source, "llm")
        self.assertEqual(len(facts), 2)
        self.assertIn("林默在第一章获得戒指。", facts)
        mock_generate_structured_sync.assert_called_once()

    @patch("app.services.consistency_audit_service.generate_structured_sync")
    def test_consistency_audit_judge_uses_generate_structured_sync(self, mock_generate_structured_sync) -> None:
        from app.services.consistency_audit_service import _call_consistency_judge_llm

        mock_generate_structured_sync.return_value = SimpleNamespace(
            parsed=SimpleNamespace(
                issues=[
                    {
                        "type": "continuity_risk",
                        "severity": "high",
                        "title": "戒指去向不一致",
                        "detail": "图谱显示戒指仍在主角手中，但本章写成已经丢失。",
                        "evidence": {"fact": "第三章仍持有戒指"},
                        "suggestion": "补充戒指遗失过程或修正文稿。",
                    }
                ]
            ),
            raw_text="{}",
            usage={"provider": "openai_compatible"},
        )

        issues = _call_consistency_judge_llm(
            chapter_id=7,
            chapter_index=4,
            chapter_title="第四章",
            chapter_preview="主角说戒指昨晚不见了。",
            graph_facts=[{"fact": "第三章结尾戒指仍在主角手中", "confidence": 0.91}],
            max_items=3,
        )

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["type"], "continuity_risk")
        self.assertEqual(issues[0]["severity"], "high")
        mock_generate_structured_sync.assert_called_once()


if __name__ == "__main__":
    unittest.main()
