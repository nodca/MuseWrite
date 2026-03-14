import unittest

from app.services.simulation import (
    SessionStateRecord,
    WorldState,
    produce_candidates,
)


class SimulationBiddingTestCase(unittest.TestCase):
    def test_provocation_raises_targeted_actor_score(self) -> None:
        world_state = WorldState(
            session_id=1,
            pending_events=[
                {"payload": {"target_id": 11, "motive": "秘密被触发"}}
            ],
        )
        session_state = SessionStateRecord(
            session_id=1,
            mind_states={
                "11": {"agitation": 0.2},
                "12": {"agitation": 0.2},
            },
        )

        candidates = produce_candidates(
            world_state,
            session_state,
            character_ids=[11, 12],
        )

        self.assertEqual(candidates[0].actor_id, 11)
        self.assertGreater(
            candidates[0].agitation_score,
            candidates[1].agitation_score,
        )

    def test_cooldown_and_last_speaker_penalty_suppress_repeat_speaker(self) -> None:
        session_state = SessionStateRecord(
            session_id=1,
            last_speaker_id=11,
            mind_states={
                "11": {"agitation": 1.0, "cooldown": 1},
                "12": {"agitation": 0.8},
            },
        )

        candidates = produce_candidates(
            WorldState(session_id=1),
            session_state,
            character_ids=[11, 12],
        )

        self.assertEqual(candidates[0].actor_id, 12)

    def test_irrelevant_actor_scores_lower_than_focused_actor(self) -> None:
        world_state = WorldState(
            session_id=1,
            recent_turns=[{"actor_id": 21, "target_card_id": 11}],
        )
        session_state = SessionStateRecord(
            session_id=1,
            mind_states={
                "11": {"agitation": 0.3},
                "12": {"agitation": 0.3},
            },
        )

        candidates = produce_candidates(
            world_state,
            session_state,
            character_ids=[11, 12],
        )

        self.assertEqual(candidates[0].actor_id, 11)
        self.assertGreater(candidates[0].agitation_score, candidates[1].agitation_score)


if __name__ == "__main__":
    unittest.main()
