import unittest

import app.models  # noqa: F401
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models.simulation import (
    SimulationDecisionTrace,
    SimulationEvent,
    SimulationSession,
    SimulationSessionState,
    SimulationTurn,
)
from app.services.simulation import (
    RefereeDecision,
    RevisionConflictError,
    SessionStateRecord,
    TurnCandidate,
    TurnOutcome,
    append_event,
    consume_pending_events,
    load_or_create_session_state,
    record_turn_outcome,
    save_session_state,
)


class SimulationRepositoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)
        with Session(self.engine) as db:
            session = SimulationSession(
                project_id=1,
                title="审讯室",
                scenario="两人对峙",
                character_card_ids=[11, 12],
                setting_keys=["room"],
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            self.session_id = session.id

    def tearDown(self) -> None:
        SQLModel.metadata.drop_all(self.engine)
        self.engine.dispose()

    def test_load_or_create_session_state_reuses_row(self) -> None:
        with Session(self.engine) as db:
            state = load_or_create_session_state(db, self.session_id)
            again = load_or_create_session_state(db, self.session_id)
            db.commit()

            self.assertEqual(state.session_id, self.session_id)
            self.assertEqual(state.revision, 0)
            self.assertEqual(state.runtime_status, "idle")
            self.assertEqual(again.model_dump(), state.model_dump())
            count = db.exec(
                select(SimulationSessionState).where(
                    SimulationSessionState.session_id == self.session_id
                )
            ).all()
            self.assertEqual(len(count), 1)

    def test_append_and_consume_events_marks_consumed(self) -> None:
        with Session(self.engine) as db:
            low = append_event(
                db,
                self.session_id,
                event_type="ambient",
                payload={"text": "风声"},
                priority=1,
            )
            high = append_event(
                db,
                self.session_id,
                event_type="intrusion",
                payload={"text": "门被撞开"},
                priority=9,
            )

            consumed = consume_pending_events(db, self.session_id)
            db.commit()
            self.assertEqual([event.id for event in consumed], [high.id, low.id])
            self.assertTrue(all(event.consumed_at is not None for event in consumed))
            self.assertEqual(consume_pending_events(db, self.session_id), [])
            stored = db.exec(
                select(SimulationEvent).where(
                    SimulationEvent.session_id == self.session_id
                )
            ).all()
            self.assertTrue(all(event.consumed_at is not None for event in stored))

    def test_record_turn_outcome_persists_turn_and_trace(self) -> None:
        with Session(self.engine) as db:
            turn = SimulationTurn(
                session_id=self.session_id,
                turn_index=1,
                actor_card_id=11,
                actor_name="甲",
                action_type="say",
                content="你已经知道了，对吗？",
            )
            outcome = TurnOutcome(status="spoken")
            decision = RefereeDecision(
                winner_id=11,
                reason="秘密暴露压力最高",
                applied_rules=["secret_pressure"],
            )
            candidates = [
                TurnCandidate(actor_id=11, agitation_score=0.91, motive="试探"),
                TurnCandidate(actor_id=12, agitation_score=0.42, motive="克制"),
            ]

            persisted = record_turn_outcome(
                db,
                session_id=self.session_id,
                turn_index=1,
                outcome=outcome,
                decision=decision,
                candidates=candidates,
                turn=turn,
                consumed_event_ids=[7, 8],
            )
            db.commit()

            self.assertEqual(persisted.outcome.status, "spoken")
            self.assertIsNotNone(persisted.turn_id)
            self.assertEqual(persisted.consumed_event_ids, [7, 8])
            stored_turn = db.exec(
                select(SimulationTurn).where(SimulationTurn.id == persisted.turn_id)
            ).one()
            stored_trace = db.exec(
                select(SimulationDecisionTrace).where(
                    SimulationDecisionTrace.id == persisted.trace_id
                )
            ).one()
            self.assertEqual(stored_turn.content, "你已经知道了，对吗？")
            self.assertEqual(stored_trace.decision_payload["winner_id"], 11)
            self.assertEqual(len(stored_trace.candidate_snapshot), 2)

    def test_save_session_state_enforces_revision(self) -> None:
        with Session(self.engine) as db:
            current = load_or_create_session_state(db, self.session_id)
            updated = save_session_state(
                db,
                SessionStateRecord(
                    session_id=self.session_id,
                    revision=current.revision,
                    runtime_status="running",
                    last_speaker_id=12,
                    active_pressures=[{"kind": "deadline"}],
                    mind_states={"12": {"agitation": 0.8}},
                ),
                expected_revision=current.revision,
            )
            db.commit()

            self.assertEqual(updated.revision, 1)
            self.assertEqual(updated.runtime_status, "running")
            self.assertEqual(updated.last_speaker_id, 12)

            with self.assertRaises(RevisionConflictError):
                save_session_state(
                    db,
                    SessionStateRecord(
                        session_id=self.session_id,
                        revision=0,
                        runtime_status="paused",
                    ),
                    expected_revision=0,
                )


if __name__ == "__main__":
    unittest.main()
