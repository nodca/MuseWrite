import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

import app.services.llm_provider as llm_provider_module
from app.core.config import settings
from app.models.content import StoryCard
from app.models.simulation import SimulationSession


class StructuredOutputsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._llm_provider = settings.llm_provider
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        SQLModel.metadata.drop_all(self.engine)
        self.engine.dispose()
        settings.llm_provider = self._llm_provider

    def test_generate_structured_routes_openai_like_provider(self) -> None:
        from app.services.llm_provider import StructuredGenerationResult, generate_structured
        from app.services.simulation_service import TurnAction

        original_openai = getattr(llm_provider_module, "_generate_openai_compatible_structured", None)
        calls: list[str] = []

        async def fake_openai(*args, **kwargs):
            calls.append(str(kwargs.get("provider_name") or ""))
            return StructuredGenerationResult(
                parsed=TurnAction(action_type="say", content="继续前进"),
                raw_text='{"action_type":"say","content":"继续前进"}',
                usage={"provider": kwargs.get("provider_name", "openai_compatible")},
            )

        try:
            llm_provider_module._generate_openai_compatible_structured = fake_openai
            settings.llm_provider = "openai_compatible"
            result = asyncio.run(generate_structured("测试", output_model=TurnAction, context={}))
            self.assertIsNotNone(result.parsed)
            self.assertEqual(result.parsed.action_type, "say")
            self.assertEqual(calls, ["openai_compatible"])
        finally:
            if original_openai is not None:
                llm_provider_module._generate_openai_compatible_structured = original_openai

    @patch("app.services.simulation_service.generate_structured", new_callable=AsyncMock)
    def test_run_one_turn_uses_structured_output(
        self,
        mock_generate_structured: AsyncMock,
    ) -> None:
        from app.services.llm_provider import StructuredGenerationResult
        from app.services.simulation_service import TurnAction, run_one_turn

        mock_generate_structured.return_value = StructuredGenerationResult(
            parsed=TurnAction(action_type="react", content="她没有退后。", emotion="冷静", target_card_id=None),
            raw_text='{"action_type":"react","content":"她没有退后。","emotion":"冷静","target_card_id":null}',
            usage={"provider": "openai_compatible"},
        )

        with Session(self.engine) as db:
            card = StoryCard(project_id=1, title="林默", content={"type": "角色"})
            db.add(card)
            db.commit()
            db.refresh(card)

            sim = SimulationSession(
                project_id=1,
                title="测试模拟",
                scenario="狭巷对峙",
                character_card_ids=[int(card.id or 0)],
                setting_keys=[],
                max_turns=5,
                status="idle",
                pending_events=[],
            )
            db.add(sim)
            db.commit()
            db.refresh(sim)

            turn = asyncio.run(run_one_turn(db, sim))

        self.assertEqual(turn.action_type, "react")
        self.assertEqual(turn.content, "她没有退后。")
        self.assertEqual(mock_generate_structured.await_count, 1)


if __name__ == "__main__":
    unittest.main()
