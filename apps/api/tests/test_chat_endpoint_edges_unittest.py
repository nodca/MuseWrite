import unittest
from uuid import uuid4
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.api.router import api_router
from app.core.config import settings
from app.core.database import get_session
from app.models.chat import ChatAction, ChatSession


class ChatEndpointEdgesTestCase(unittest.TestCase):
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
        settings.auth_tokens = "human-user:human-token,system:system-token"
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
    def _auth_header(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def _create_session(self, *, user_id: str = "human-user", project_id: int = 1) -> int:
        with Session(self.engine) as db:
            session = ChatSession(project_id=project_id, user_id=user_id, title="edge-case-session")
            db.add(session)
            db.commit()
            db.refresh(session)
            return int(session.id or 0)

    def _create_action(
        self,
        *,
        user_id: str = "human-user",
        status: str = "proposed",
        action_type: str = "setting.upsert",
    ) -> int:
        session_id = self._create_session(user_id=user_id, project_id=1)
        with Session(self.engine) as db:
            action = ChatAction(
                session_id=session_id,
                action_type=action_type,
                status=status,
                payload={},
                idempotency_key=f"edge-{uuid4().hex[:10]}",
                operator_id=user_id,
            )
            db.add(action)
            db.commit()
            db.refresh(action)
            return int(action.id or 0)

    @staticmethod
    def _dead_letter_fixture_rows() -> list[object]:
        return [
            {
                "project_id": 1,
                "operator_id": "human-user",
                "reason": "r1",
                "action_id": 11,
                "mutation_id": "m1",
                "idempotency_key": "k-1",
            },
            {
                "project_id": "1",
                "operator_id": "human-user",
                "reason": "r2",
                "action_id": 12,
                "mutation_id": "m2",
                "idempotency_key": "k-2",
            },
            {"project_id": 1, "idempotency_key": "k-minimal"},
            {
                "project_id": 2,
                "operator_id": "human-user",
                "reason": "r3",
                "action_id": 13,
                "mutation_id": "m3",
                "idempotency_key": "k-3",
            },
            {
                "project_id": 0,
                "operator_id": "human-user",
                "reason": "r4",
                "action_id": 14,
                "mutation_id": "m4",
                "idempotency_key": "k-4",
            },
            "ignore-me",
        ]

    def test_apply_action_returns_409_when_not_proposed(self) -> None:
        action_id = self._create_action(status="rejected")
        response = self.client.post(
            f"/api/chat/actions/{action_id}/apply",
            headers=self._auth_header("human-token"),
            json={"event_payload": {}},
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json().get("detail"), "action is not in proposed state")

    def test_reject_action_returns_409_when_not_proposed(self) -> None:
        action_id = self._create_action(status="applied")
        response = self.client.post(
            f"/api/chat/actions/{action_id}/reject",
            headers=self._auth_header("human-token"),
            json={"event_payload": {}},
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json().get("detail"), "action is not in proposed state")

    def test_undo_action_returns_409_when_not_applied(self) -> None:
        action_id = self._create_action(status="rejected")
        response = self.client.post(
            f"/api/chat/actions/{action_id}/undo",
            headers=self._auth_header("human-token"),
            json={"event_payload": {}},
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json().get("detail"), "only applied action can be undone")

    def test_apply_entity_merge_requires_manual_confirmed(self) -> None:
        action_id = self._create_action(action_type="entity.merge.alias")
        response = self.client.post(
            f"/api/chat/actions/{action_id}/apply",
            headers=self._auth_header("human-token"),
            json={"event_payload": {}},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("detail"), "entity merge requires manual_confirmed=true")

    def test_apply_entity_merge_requires_human_operator(self) -> None:
        action_id = self._create_action(
            user_id="system",
            action_type="entity.merge.alias",
        )
        response = self.client.post(
            f"/api/chat/actions/{action_id}/apply",
            headers=self._auth_header("system-token"),
            json={"event_payload": {"manual_confirmed": True}},
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            response.json().get("detail"),
            "entity merge can only be applied by a human operator",
        )

    def test_graph_candidates_review_requires_manual_confirmed(self) -> None:
        response = self.client.post(
            "/api/chat/projects/1/graph-candidates/review",
            headers=self._auth_header("human-token"),
            json={"decision": "confirm", "fact_keys": ["fact-1"]},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json().get("detail"),
            "graph candidate review requires manual_confirmed=true",
        )

    def test_graph_candidates_review_requires_human_operator(self) -> None:
        response = self.client.post(
            "/api/chat/projects/1/graph-candidates/review",
            headers=self._auth_header("system-token"),
            json={
                "decision": "confirm",
                "fact_keys": ["fact-1"],
                "manual_confirmed": True,
            },
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            response.json().get("detail"),
            "graph candidate review can only be applied by a human operator",
        )

    def test_graph_candidates_review_rejects_empty_normalized_fact_keys(self) -> None:
        response = self.client.post(
            "/api/chat/projects/1/graph-candidates/review",
            headers=self._auth_header("human-token"),
            json={
                "decision": "confirm",
                "fact_keys": ["   ", "\n\t"],
                "manual_confirmed": True,
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("detail"), "fact_keys is required")

    @patch("app.api.endpoints.chat.promote_neo4j_candidate_facts")
    def test_graph_candidates_review_confirm_path_calls_promote(self, mock_promote) -> None:
        mock_promote.return_value = ["fact-1", "fact-2"]
        response = self.client.post(
            "/api/chat/projects/1/graph-candidates/review",
            headers=self._auth_header("human-token"),
            json={
                "decision": "confirm",
                "fact_keys": [" fact-1 ", "fact-1", "fact-2"],
                "manual_confirmed": True,
                "chapter_index": 3,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("decision"), "confirm")
        self.assertEqual(payload.get("requested_count"), 2)
        self.assertEqual(payload.get("reviewed_count"), 2)
        self.assertEqual(payload.get("fact_keys"), ["fact-1", "fact-2"])
        mock_promote.assert_called_once_with(
            1,
            fact_keys=["fact-1", "fact-2"],
            source_ref="",
            min_confidence=None,
            limit=2,
            current_chapter=3,
        )

    @patch("app.api.endpoints.chat.update_neo4j_graph_fact_state")
    def test_graph_candidates_review_reject_path_calls_update(self, mock_update) -> None:
        mock_update.return_value = 2
        response = self.client.post(
            "/api/chat/projects/1/graph-candidates/review",
            headers=self._auth_header("human-token"),
            json={
                "decision": "reject",
                "fact_keys": [" fact-a ", "fact-a", "fact-b"],
                "manual_confirmed": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("decision"), "reject")
        self.assertEqual(payload.get("requested_count"), 2)
        self.assertEqual(payload.get("reviewed_count"), 2)
        self.assertEqual(payload.get("fact_keys"), ["fact-a", "fact-b"])
        mock_update.assert_called_once_with(
            1,
            ["fact-a", "fact-b"],
            to_state="rejected",
            from_state="candidate",
            current_chapter=None,
        )

    def test_dead_letters_requires_project_id(self) -> None:
        response = self.client.get(
            "/api/chat/index-lifecycle/dead-letters",
            headers=self._auth_header("human-token"),
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("detail"), "project_id is required")

    @patch("app.api.endpoints.chat.peek_index_lifecycle_dead_letters")
    def test_dead_letters_filters_by_project(self, mock_peek) -> None:
        mock_peek.return_value = self._dead_letter_fixture_rows()
        response = self.client.get(
            "/api/chat/index-lifecycle/dead-letters?project_id=1&limit=5",
            headers=self._auth_header("human-token"),
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual([item.get("idempotency_key") for item in payload], ["k-1", "k-2", "k-minimal"])
        self.assertEqual(payload[2].get("operator_id"), "human-user")
        self.assertEqual(payload[2].get("reason"), "unspecified")
        self.assertEqual(payload[2].get("action_id"), 0)
        self.assertEqual(payload[2].get("mutation_id"), "")
        mock_peek.assert_called_once_with(limit=5)

    @patch("app.api.endpoints.chat.push_index_lifecycle_dead_letter")
    @patch("app.api.endpoints.chat.enqueue_index_lifecycle_job")
    @patch("app.api.endpoints.chat.pop_index_lifecycle_dead_letters")
    def test_replay_dead_letters_reports_replayed_requeue_failed_and_skipped_invalid(
        self,
        mock_pop,
        mock_enqueue,
        mock_push,
    ) -> None:
        first_item = {
            "project_id": 1,
            "action_id": 0,
            "operator_id": "human-user",
            "reason": "retry-first",
            "mutation_id": "m-1",
            "expected_version": 3,
            "idempotency_key": "job-1",
            "lifecycle_slot": "default",
        }
        second_item = {
            "project_id": 1,
            "action_id": 0,
            "operator_id": "human-user",
            "reason": "retry-second",
            "mutation_id": "m-2",
            "expected_version": 4,
            "idempotency_key": "job-2",
            "lifecycle_slot": "default",
        }
        mock_pop.return_value = [
            first_item,
            second_item,
            "bad-item",
            {"project_id": 0, "action_id": 99},
        ]
        mock_enqueue.side_effect = [True, False]

        response = self.client.post(
            "/api/chat/index-lifecycle/dead-letters/replay",
            headers=self._auth_header("human-token"),
            json={"project_id": 1, "limit": 4},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("requested"), 4)
        self.assertEqual(payload.get("project_id"), 1)
        self.assertEqual(payload.get("replayed"), 1)
        self.assertEqual(payload.get("requeue_failed"), 1)
        self.assertEqual(payload.get("skipped_invalid"), 2)
        self.assertTrue(str(payload.get("replay_request_id", "")).startswith("replay-"))

        mock_pop.assert_called_once_with(limit=4, project_id=1)
        self.assertEqual(mock_enqueue.call_count, 2)
        self.assertEqual(mock_enqueue.call_args_list[0].kwargs.get("idempotency_key"), "job-1")
        self.assertEqual(mock_enqueue.call_args_list[1].kwargs.get("idempotency_key"), "job-2")
        pushed_item = mock_push.call_args_list[0].args[0]
        self.assertEqual(str(pushed_item.get("idempotency_key") or ""), "job-2")
        self.assertEqual(str(pushed_item.get("reason") or ""), "retry-second")
        mock_push.assert_called_once()
        self.assertEqual(mock_push.call_args_list[0].args[1], "replay_enqueue_failed")

    @patch("app.api.endpoints.chat.enqueue_index_lifecycle_job")
    @patch("app.api.endpoints.chat.pop_index_lifecycle_dead_letters")
    def test_replay_dead_letters_normalizes_minimal_row_defaults(
        self,
        mock_pop,
        mock_enqueue,
    ) -> None:
        mock_pop.return_value = [{"project_id": 1, "idempotency_key": "job-min"}]
        mock_enqueue.return_value = True

        response = self.client.post(
            "/api/chat/index-lifecycle/dead-letters/replay",
            headers=self._auth_header("human-token"),
            json={"project_id": 1, "limit": 1},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("replayed"), 1)
        self.assertEqual(payload.get("requeue_failed"), 0)
        self.assertEqual(payload.get("skipped_invalid"), 0)
        mock_enqueue.assert_called_once()
        enqueue_kwargs = mock_enqueue.call_args_list[0].kwargs
        self.assertEqual(enqueue_kwargs.get("project_id"), 1)
        self.assertEqual(enqueue_kwargs.get("operator_id"), "human-user")
        self.assertEqual(enqueue_kwargs.get("reason"), "unspecified")
        self.assertEqual(enqueue_kwargs.get("action_id"), 0)
        self.assertEqual(enqueue_kwargs.get("mutation_id"), "")
        self.assertEqual(enqueue_kwargs.get("expected_version"), 0)
        self.assertEqual(enqueue_kwargs.get("idempotency_key"), "job-min")
        self.assertEqual(enqueue_kwargs.get("lifecycle_slot"), "default")

    def test_save_chapter_returns_409_on_expected_version_conflict(self) -> None:
        created = self.client.post(
            "/api/chat/projects/1/chapters",
            headers=self._auth_header("human-token"),
            json={},
        )
        self.assertEqual(created.status_code, 200)
        chapter_id = int(created.json().get("id", 0))
        self.assertGreater(chapter_id, 0)

        response = self.client.put(
            f"/api/chat/projects/1/chapters/{chapter_id}",
            headers=self._auth_header("human-token"),
            json={
                "title": "Chapter 1",
                "content": "content-v2",
                "expected_version": 999,
            },
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("chapter version conflict", str(response.json().get("detail", "")))


if __name__ == "__main__":
    unittest.main()
