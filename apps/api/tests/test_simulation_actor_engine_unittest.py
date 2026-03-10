import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.models.content import StoryCard
from app.models.simulation import SimulationSession
from app.services.llm_provider import ChatGenerationResult
from app.services.simulation.actor_engine import ActorGenerationError, generate_turn
from app.services.simulation.types import SessionStateRecord, WorldState


class SimulationActorEngineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = SimulationSession(
            id=1,
            project_id=1,
            title="测试",
            scenario="密室里两人对峙。",
            character_card_ids=[11],
            setting_keys=[],
            max_turns=10,
            status="idle",
        )
        self.card = StoryCard(id=11, project_id=1, title="甲", content={})

    @patch("app.services.simulation.actor_engine.generate_chat", new_callable=AsyncMock)
    def test_generate_turn_returns_simulation_turn(
        self,
        mock_generate_chat: AsyncMock,
    ) -> None:
        mock_generate_chat.return_value = ChatGenerationResult(
            assistant_text="你确定要继续追问吗？",
            proposed_actions=[],
            usage={"provider": "stub"},
        )

        turn = asyncio.run(
            generate_turn(
                None,
                sim=self.sim,
                actor_card=self.card,
                turn_index=1,
                world_state=WorldState(session_id=1),
                session_state=SessionStateRecord(session_id=1),
            )
        )

        self.assertEqual(turn.actor_card_id, 11)
        self.assertEqual(turn.actor_name, "甲")
        self.assertEqual(turn.action_type, "say")
        self.assertEqual(turn.content, "你确定要继续追问吗？")

    @patch("app.services.simulation.actor_engine.generate_chat", new_callable=AsyncMock)
    def test_generate_turn_rejects_proposed_actions(
        self,
        mock_generate_chat: AsyncMock,
    ) -> None:
        mock_generate_chat.return_value = ChatGenerationResult(
            assistant_text="我不该说。",
            proposed_actions=[{"action_type": "setting.upsert", "payload": {"key": "x", "value": {}}}],
            usage={"provider": "stub"},
        )

        with self.assertRaises(ActorGenerationError) as ctx:
            asyncio.run(
                generate_turn(
                    None,
                    sim=self.sim,
                    actor_card=self.card,
                    turn_index=1,
                    world_state=WorldState(session_id=1),
                    session_state=SessionStateRecord(session_id=1),
                )
            )

        self.assertEqual(ctx.exception.code, "ACTOR_ACTIONS_NOT_ALLOWED")

    @patch("app.services.simulation.actor_engine.generate_chat", new_callable=AsyncMock)
    def test_generate_turn_rejects_non_json_provider_fallback(
        self,
        mock_generate_chat: AsyncMock,
    ) -> None:
        mock_generate_chat.return_value = ChatGenerationResult(
            assistant_text="（非 JSON 输出）",
            proposed_actions=[],
            usage={"provider": "openai_compatible", "raw_response_format": "non_json"},
        )

        with self.assertRaises(ActorGenerationError) as ctx:
            asyncio.run(
                generate_turn(
                    None,
                    sim=self.sim,
                    actor_card=self.card,
                    turn_index=1,
                    world_state=WorldState(session_id=1),
                    session_state=SessionStateRecord(session_id=1),
                )
            )

        self.assertEqual(ctx.exception.code, "ACTOR_NON_JSON")


if __name__ == "__main__":
    unittest.main()
