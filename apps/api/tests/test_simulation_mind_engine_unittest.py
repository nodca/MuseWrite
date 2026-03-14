import unittest
from datetime import datetime, timedelta, timezone

from app.services.simulation import SessionStateRecord, WorldState, advance_minds


class SimulationMindEngineTestCase(unittest.TestCase):
    def test_advance_minds_adds_belief_from_targeted_event(self) -> None:
        world_state = WorldState(
            session_id=1,
            pending_events=[
                {
                    "payload": {
                        "target_id": 11,
                        "fact_key": "secret_letter",
                        "stance": "known",
                        "confidence": 0.9,
                    }
                }
            ],
        )
        state = SessionStateRecord(session_id=1)

        advanced = advance_minds(world_state, state, character_ids=[11, 12])

        mind_11 = advanced.mind_states["11"]
        self.assertEqual(mind_11["beliefs"][0]["fact_key"], "secret_letter")
        self.assertEqual(mind_11["beliefs"][0]["stance"], "known")
        self.assertEqual(advanced.mind_states["12"]["beliefs"], [])

    def test_advance_minds_builds_second_order_belief(self) -> None:
        world_state = WorldState(
            session_id=1,
            pending_events=[
                {
                    "payload": {
                        "target_id": 11,
                        "subject_id": 12,
                        "fact_key": "murder_weapon",
                        "knowledge_state": "suspects",
                    }
                }
            ],
        )

        advanced = advance_minds(
            world_state,
            SessionStateRecord(session_id=1),
            character_ids=[11],
        )

        second_order = advanced.mind_states["11"]["beliefs_about_others"][0]
        self.assertEqual(second_order["subject_id"], 12)
        self.assertEqual(second_order["fact_key"], "murder_weapon")
        self.assertEqual(second_order["believed_knowledge_state"], "suspects")

    def test_advance_minds_prunes_expired_beliefs_and_decays_confidence(self) -> None:
        expired = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        fresh = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        state = SessionStateRecord(
            session_id=1,
            mind_states={
                "11": {
                    "beliefs": [
                        {
                            "fact_key": "old_secret",
                            "confidence": 0.9,
                            "expires_at": expired,
                        },
                        {
                            "fact_key": "current_secret",
                            "confidence": 0.7,
                            "expires_at": fresh,
                        },
                    ]
                }
            },
        )

        advanced = advance_minds(
            WorldState(session_id=1),
            state,
            character_ids=[11],
        )

        beliefs = advanced.mind_states["11"]["beliefs"]
        self.assertEqual(len(beliefs), 1)
        self.assertEqual(beliefs[0]["fact_key"], "current_secret")
        self.assertLess(beliefs[0]["confidence"], 0.7)

    def test_advance_minds_updates_cooldown_and_agitation_from_turn_pressure(self) -> None:
        state = SessionStateRecord(
            session_id=1,
            mind_states={
                "11": {
                    "agitation": 0.5,
                    "cooldown": 2,
                }
            },
        )
        world_state = WorldState(
            session_id=1,
            last_speaker_id=12,
            recent_turns=[
                {
                    "actor_id": 12,
                    "target_card_id": 11,
                }
            ],
            pending_events=[
                {
                    "payload": {
                        "target_id": 11,
                        "intensity": 0.4,
                    }
                }
            ],
        )

        advanced = advance_minds(world_state, state, character_ids=[11])

        mind_11 = advanced.mind_states["11"]
        self.assertEqual(mind_11["cooldown"], 1)
        self.assertGreater(mind_11["agitation"], 0.5)
        self.assertEqual(mind_11["focus_target"], 12)
        self.assertEqual(advanced.last_speaker_id, 12)


if __name__ == "__main__":
    unittest.main()
