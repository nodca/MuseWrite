import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.services.llm_provider import ChatGenerationResult
from app.services.simulation import (
    SessionStateRecord,
    WorldState,
    update_second_order_beliefs_from_turns,
)


class SimulationToMUpdaterTestCase(unittest.TestCase):
    @patch("app.services.simulation.tom_updater.generate_chat", new_callable=AsyncMock)
    def test_update_second_order_beliefs_from_turns_skips_when_no_content(
        self,
        mock_generate_chat: AsyncMock,
    ) -> None:
        world_state = WorldState(
            session_id=1,
            recent_turns=[{"turn_index": 1, "actor_id": 12, "content": ""}],
        )
        state = SessionStateRecord(session_id=1, mind_states={"11": {}, "12": {}})

        updated = asyncio.run(
            update_second_order_beliefs_from_turns(world_state, state, character_ids=[11, 12])
        )

        self.assertEqual(updated.model_dump(), state.model_dump())
        mock_generate_chat.assert_not_awaited()

    @patch("app.services.simulation.tom_updater.generate_chat", new_callable=AsyncMock)
    def test_update_second_order_beliefs_from_turns_upserts_llm_deltas(
        self,
        mock_generate_chat: AsyncMock,
    ) -> None:
        world_state = WorldState(
            session_id=1,
            recent_turns=[
                {
                    "turn_index": 1,
                    "actor_id": 12,
                    "actor_name": "乙",
                    "content": "我不知道那封信。",
                    "target_card_id": None,
                }
            ],
        )
        state = SessionStateRecord(
            session_id=1,
            mind_states={
                "11": {
                    "beliefs_about_others": [
                        {
                            "holder_id": 11,
                            "subject_id": 12,
                            "fact_key": "secret_letter",
                            "believed_knowledge_state": "unknown",
                            "confidence": 0.4,
                            "source_turn_id": None,
                        }
                    ]
                },
                "12": {},
            },
        )

        async def _fake_generate_chat(user_input: str, *args, **kwargs) -> ChatGenerationResult:
            _ = args
            _ = kwargs
            if "holder_id=11" in user_input:
                return ChatGenerationResult(
                    assistant_text="",
                    proposed_actions=[
                        {
                            "action_type": "setting.upsert",
                            "payload": {
                                "key": "__internal.simulation.tom_delta__",
                                "value": {
                                    "beliefs_about_others": [
                                        {
                                            "holder_id": 11,
                                            "subject_id": 12,
                                            "fact_key": "secret_letter",
                                            "believed_knowledge_state": "denies",
                                            "confidence": 0.7,
                                            "source_turn_id": 1,
                                        }
                                    ]
                                },
                            },
                        }
                    ],
                    usage={"provider": "stub"},
                )
            if "holder_id=12" in user_input:
                return ChatGenerationResult(
                    assistant_text="",
                    proposed_actions=[
                        {
                            "action_type": "setting.upsert",
                            "payload": {
                                "key": "__internal.simulation.tom_delta__",
                                "value": {"beliefs_about_others": []},
                            },
                        }
                    ],
                    usage={"provider": "stub"},
                )
            raise AssertionError("unexpected user_input")

        mock_generate_chat.side_effect = _fake_generate_chat

        updated = asyncio.run(
            update_second_order_beliefs_from_turns(world_state, state, character_ids=[11, 12])
        )

        beliefs = updated.mind_states["11"]["beliefs_about_others"]
        self.assertEqual(len(beliefs), 1)
        self.assertEqual(beliefs[0]["subject_id"], 12)
        self.assertEqual(beliefs[0]["fact_key"], "secret_letter")
        self.assertEqual(beliefs[0]["believed_knowledge_state"], "denies")
        self.assertEqual(beliefs[0]["confidence"], 0.7)
        self.assertEqual(beliefs[0]["source_turn_id"], 1)


if __name__ == "__main__":
    unittest.main()

