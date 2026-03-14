import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.api.router import api_router
from app.core.config import settings
from app.core.database import get_session
from app.models.content import StoryCard
from app.models.simulation import SimulationEvent, SimulationSession, SimulationTurn
from app.services.simulation import TurnOutcome


class SimulationEndpointTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "auth_enabled": settings.auth_enabled,
            "auth_tokens": settings.auth_tokens,
            "auth_token": settings.auth_token,
            "auth_user": settings.auth_user,
            "auth_project_owners": settings.auth_project_owners,
            "auth_disabled_user": settings.auth_disabled_user,
        }
        settings.auth_enabled = True
        settings.auth_tokens = "human-user:human-token"
        settings.auth_token = ""
        settings.auth_user = ""
        settings.auth_project_owners = ""
        settings.auth_disabled_user = "local-user"

        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

        with Session(self.engine) as db:
            db.add(StoryCard(project_id=1, id=11, title="甲", content={}))
            db.add(StoryCard(project_id=1, id=12, title="乙", content={}))
            db.commit()

        app = FastAPI()
        app.include_router(api_router, prefix=settings.api_prefix)

        def _override_get_session():
            with Session(self.engine) as db:
                yield db

        app.dependency_overrides[get_session] = _override_get_session
        self.app = app
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        self.app.dependency_overrides.clear()
        SQLModel.metadata.drop_all(self.engine)
        self.engine.dispose()
        for key, value in self._snapshot.items():
            setattr(settings, key, value)

    @staticmethod
    def _auth_header() -> dict[str, str]:
        return {"Authorization": "Bearer human-token"}

    @staticmethod
    def _parse_sse_events(raw_body: str) -> list[dict]:
        events: list[dict] = []
        for block in str(raw_body or "").split("\n\n"):
            line = block.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                events.append(parsed)
        return events

    def _create_session(self, *, max_turns: int = 1) -> int:
        response = self.client.post(
            "/api/simulation/sessions",
            headers=self._auth_header(),
            json={
                "project_id": 1,
                "title": "对峙",
                "scenario": "两人对峙，空气凝固。",
                "character_card_ids": [11, 12],
                "setting_keys": [],
                "max_turns": max_turns,
            },
        )
        self.assertEqual(response.status_code, 201)
        payload = response.json()
        session_id = int(payload.get("id", 0))
        self.assertGreater(session_id, 0)
        return session_id

    def test_inject_event_persists_simulation_event(self) -> None:
        session_id = self._create_session(max_turns=1)

        response = self.client.post(
            f"/api/simulation/sessions/{session_id}/inject",
            headers=self._auth_header(),
            json={"event_text": "门外传来脚步声。"},
        )
        self.assertEqual(response.status_code, 200)

        with Session(self.engine) as db:
            events = db.exec(
                select(SimulationEvent).where(
                    SimulationEvent.session_id == session_id
                )
            ).all()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_type, "external_text")
            self.assertEqual(events[0].payload.get("text"), "门外传来脚步声。")

    @patch("app.api.endpoints.simulation.run_turn", new_callable=AsyncMock)
    def test_run_stream_emits_turn_and_done_when_max_turns_reached(
        self,
        mock_run_turn: AsyncMock,
    ) -> None:
        session_id = self._create_session(max_turns=1)
        mock_run_turn.return_value = (
            TurnOutcome(status="spoken", turn_index=1),
            SimulationTurn(
                id=101,
                session_id=session_id,
                turn_index=1,
                actor_card_id=11,
                actor_name="甲",
                action_type="say",
                content="你终于来了。",
            ),
        )

        response = self.client.post(
            f"/api/simulation/sessions/{session_id}/run",
            headers=self._auth_header(),
            json={},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))

        events = self._parse_sse_events(response.text)
        self.assertEqual(events[0].get("type"), "turn")
        self.assertEqual(events[-1].get("type"), "done")

        with Session(self.engine) as db:
            sim = db.get(SimulationSession, session_id)
            self.assertIsNotNone(sim)
            if sim is not None:
                self.assertEqual(sim.status, "idle")

    @patch("app.api.endpoints.simulation.run_turn", new_callable=AsyncMock)
    def test_run_stream_emits_paused_and_done(
        self,
        mock_run_turn: AsyncMock,
    ) -> None:
        session_id = self._create_session(max_turns=5)
        mock_run_turn.return_value = (
            TurnOutcome(status="paused", turn_index=1),
            None,
        )

        response = self.client.post(
            f"/api/simulation/sessions/{session_id}/run",
            headers=self._auth_header(),
            json={},
        )
        self.assertEqual(response.status_code, 200)
        events = self._parse_sse_events(response.text)
        self.assertEqual(events[0].get("type"), "paused")
        self.assertEqual(events[-1].get("type"), "done")


if __name__ == "__main__":
    unittest.main()

