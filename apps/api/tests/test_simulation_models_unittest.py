import unittest

import app.models as simulation_models
from app.models import (
    SimulationDecisionTrace,
    SimulationEvent,
    SimulationSession,
    SimulationSessionState,
    SimulationTurn,
)


class SimulationModelContractTestCase(unittest.TestCase):
    def test_session_drops_pending_events_from_runtime_contract(self) -> None:
        self.assertIn("status", SimulationSession.model_fields)
        self.assertNotIn("pending_events", SimulationSession.model_fields)

        session = SimulationSession(project_id=7)

        self.assertEqual(session.status, "idle")
        self.assertNotIn("pending_events", session.model_dump())
        self.assertFalse(
            SimulationSession.__table__.c.character_card_ids.nullable
        )
        self.assertFalse(SimulationSession.__table__.c.setting_keys.nullable)

    def test_runtime_models_are_exported_with_serializable_defaults(self) -> None:
        self.assertIn("SimulationTurn", simulation_models.__all__)
        self.assertIn("SimulationSessionState", simulation_models.__all__)
        self.assertIn("SimulationEvent", simulation_models.__all__)
        self.assertIn("SimulationDecisionTrace", simulation_models.__all__)
        self.assertIs(simulation_models.SimulationTurn, SimulationTurn)

        state = SimulationSessionState(session_id=11)
        other_state = SimulationSessionState(session_id=12)
        self.assertEqual(state.revision, 0)
        self.assertEqual(state.runtime_status, "idle")
        self.assertIsNone(state.last_speaker_id)
        self.assertEqual(state.active_pressures, [])
        self.assertEqual(state.mind_states, {})
        state.active_pressures.append({"kind": "deadline"})
        state.mind_states["lead"] = {"focus_target": 99}
        self.assertEqual(other_state.active_pressures, [])
        self.assertEqual(other_state.mind_states, {})
        self.assertEqual(
            state.model_dump(
                include={
                    "revision",
                    "runtime_status",
                    "active_pressures",
                    "mind_states",
                }
            ),
            {
                "revision": 0,
                "runtime_status": "idle",
                "active_pressures": [{"kind": "deadline"}],
                "mind_states": {"lead": {"focus_target": 99}},
            },
        )
        self.assertFalse(
            SimulationSessionState.__table__.c.active_pressures.nullable
        )
        self.assertFalse(SimulationSessionState.__table__.c.mind_states.nullable)

        event = SimulationEvent(session_id=11, event_type="director_note")
        other_event = SimulationEvent(session_id=12, event_type="system_tick")
        self.assertEqual(event.payload, {})
        self.assertEqual(event.priority, 0)
        self.assertIsNone(event.consumed_at)
        event.payload["source_turn"] = 3
        self.assertEqual(other_event.payload, {})
        self.assertEqual(
            event.model_dump(
                include={"event_type", "payload", "priority", "consumed_at"}
            ),
            {
                "event_type": "director_note",
                "payload": {"source_turn": 3},
                "priority": 0,
                "consumed_at": None,
            },
        )
        self.assertFalse(SimulationEvent.__table__.c.payload.nullable)

        trace = SimulationDecisionTrace(session_id=11, turn_index=3)
        other_trace = SimulationDecisionTrace(session_id=12, turn_index=4)
        self.assertEqual(trace.candidate_snapshot, [])
        self.assertEqual(trace.decision_payload, {})
        self.assertEqual(trace.failure_code, "")
        trace.candidate_snapshot.append({"actor_id": 9})
        trace.decision_payload["winner_id"] = 9
        self.assertEqual(other_trace.candidate_snapshot, [])
        self.assertEqual(other_trace.decision_payload, {})
        self.assertEqual(
            trace.model_dump(
                include={
                    "turn_index",
                    "candidate_snapshot",
                    "decision_payload",
                    "failure_code",
                }
            ),
            {
                "turn_index": 3,
                "candidate_snapshot": [{"actor_id": 9}],
                "decision_payload": {"winner_id": 9},
                "failure_code": "",
            },
        )
        self.assertFalse(
            SimulationDecisionTrace.__table__.c.candidate_snapshot.nullable
        )
        self.assertFalse(
            SimulationDecisionTrace.__table__.c.decision_payload.nullable
        )


if __name__ == "__main__":
    unittest.main()
