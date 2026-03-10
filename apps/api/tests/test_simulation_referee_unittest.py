import unittest

from app.services.simulation import RefereeDecision, TurnCandidate, pick_speaker


class SimulationRefereeTestCase(unittest.TestCase):
    def test_pick_speaker_returns_existing_candidate(self) -> None:
        candidates = [
            TurnCandidate(actor_id=11, agitation_score=0.9, motive="揭穿", confidence=0.8),
            TurnCandidate(actor_id=12, agitation_score=0.4, motive="观望", confidence=0.6),
        ]

        decision = pick_speaker(candidates)

        self.assertEqual(decision.winner_id, 11)
        self.assertIn(decision.winner_id, [item.actor_id for item in candidates])

    def test_pick_speaker_pauses_on_low_signal(self) -> None:
        decision = pick_speaker(
            [TurnCandidate(actor_id=11, agitation_score=0.1, motive="观望", confidence=0.2)]
        )

        self.assertIsNone(decision.winner_id)
        self.assertTrue(decision.pause_suggested)
        self.assertEqual(decision.reason, "signal_too_low")

    def test_pick_speaker_allows_cold_open_on_first_turn(self) -> None:
        decision = pick_speaker(
            [TurnCandidate(actor_id=11, agitation_score=0.0, motive="观望", confidence=0.1)],
            turn_index=1,
        )

        self.assertEqual(decision.winner_id, 11)
        self.assertFalse(decision.pause_suggested)

    def test_pick_speaker_is_not_round_robin(self) -> None:
        first = TurnCandidate(actor_id=11, agitation_score=0.5, motive="低压", confidence=0.6)
        second = TurnCandidate(actor_id=12, agitation_score=0.85, motive="抢话", confidence=0.7)

        decision = pick_speaker([first, second])

        self.assertEqual(decision, RefereeDecision(
            winner_id=12,
            reason="抢话",
            applied_rules=["max_agitation_score"],
            pause_suggested=False,
        ))


if __name__ == "__main__":
    unittest.main()
