import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import app.models  # noqa: F401
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models.content import StoryCard
from app.models.simulation import SimulationDecisionTrace, SimulationSession, SimulationTurn
from app.services.simulation import RefereeDecision, TurnOutcome, run_turn


class SimulationDirectorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)
        with Session(self.engine) as db:
            db.add(StoryCard(project_id=1, id=11, title="甲", content={}))
            db.add(StoryCard(project_id=1, id=12, title="乙", content={}))
            sim = SimulationSession(
                project_id=1,
                title="对峙",
                scenario="两人对峙，空气凝固。",
                character_card_ids=[11, 12],
                setting_keys=[],
                max_turns=5,
                status="idle",
            )
            db.add(sim)
            db.commit()
            db.refresh(sim)
            self.session_id = sim.id

    def tearDown(self) -> None:
        SQLModel.metadata.drop_all(self.engine)
        self.engine.dispose()

    @patch("app.services.simulation.director.pick_speaker")
    @patch("app.services.simulation.director.generate_turn", new_callable=AsyncMock)
    def test_director_spoken_persists_turn_and_trace(
        self,
        mock_generate_turn: AsyncMock,
        mock_pick_speaker,
    ) -> None:
        mock_pick_speaker.return_value = RefereeDecision(
            winner_id=11,
            reason="test",
            applied_rules=["test"],
            pause_suggested=False,
        )
        mock_generate_turn.return_value = SimulationTurn(
            session_id=self.session_id,
            turn_index=1,
            actor_card_id=11,
            actor_name="甲",
            action_type="say",
            content="你终于来了。",
        )

        with Session(self.engine) as db:
            sim = db.get(SimulationSession, self.session_id)
            outcome, turn = asyncio.run(run_turn(db, sim=sim))

            self.assertEqual(outcome, TurnOutcome(status="spoken", turn_index=1))
            self.assertIsNotNone(turn)

            stored_turn = db.exec(select(SimulationTurn)).one()
            stored_trace = db.exec(select(SimulationDecisionTrace)).one()
            self.assertEqual(stored_turn.content, "你终于来了。")
            self.assertEqual(stored_trace.turn_index, 1)

    @patch("app.services.simulation.director.pick_speaker")
    def test_director_paused_records_trace_without_turn(
        self,
        mock_pick_speaker,
    ) -> None:
        mock_pick_speaker.return_value = RefereeDecision(
            winner_id=None,
            reason="low_signal",
            applied_rules=["pause_on_low_signal"],
            pause_suggested=True,
        )

        with Session(self.engine) as db:
            sim = db.get(SimulationSession, self.session_id)
            outcome, turn = asyncio.run(run_turn(db, sim=sim))

            self.assertEqual(outcome.status, "paused")
            self.assertIsNone(turn)
            self.assertEqual(db.exec(select(SimulationTurn)).all(), [])
            self.assertEqual(len(db.exec(select(SimulationDecisionTrace)).all()), 1)


if __name__ == "__main__":
    unittest.main()
