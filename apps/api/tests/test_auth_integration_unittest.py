import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.api.router import api_router
from app.core.config import settings
from app.core.database import get_session
from app.models.chat import ChatMessage, ChatSession


class AuthIntegrationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "auth_enabled": settings.auth_enabled,
            "auth_tokens": settings.auth_tokens,
            "auth_token": settings.auth_token,
            "auth_user": settings.auth_user,
            "auth_project_owners": settings.auth_project_owners,
            "auth_disabled_user": settings.auth_disabled_user,
        }

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

    def test_missing_authorization_header_returns_401(self) -> None:
        settings.auth_enabled = True
        settings.auth_tokens = "local-user:secret-token"
        settings.auth_token = ""
        settings.auth_user = ""
        settings.auth_project_owners = ""

        resp = self.client.get("/api/chat/projects/1/settings")
        self.assertEqual(resp.status_code, 401)

    def test_invalid_token_returns_401(self) -> None:
        settings.auth_enabled = True
        settings.auth_tokens = "local-user:secret-token"
        settings.auth_token = ""
        settings.auth_user = ""
        settings.auth_project_owners = ""

        resp = self.client.get(
            "/api/chat/projects/1/settings",
            headers=self._auth_header("wrong-token"),
        )
        self.assertEqual(resp.status_code, 401)

    def test_valid_token_returns_200_in_single_user_mode(self) -> None:
        settings.auth_enabled = True
        settings.auth_tokens = "local-user:secret-token"
        settings.auth_token = ""
        settings.auth_user = ""
        settings.auth_project_owners = ""

        resp = self.client.get(
            "/api/chat/projects/1/settings",
            headers=self._auth_header("secret-token"),
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])

    def test_valid_token_can_still_be_forbidden_by_project_acl(self) -> None:
        settings.auth_enabled = True
        settings.auth_tokens = "local-user:secret-token"
        settings.auth_token = ""
        settings.auth_user = ""
        settings.auth_project_owners = "1:another-user"

        resp = self.client.get(
            "/api/chat/projects/1/settings",
            headers=self._auth_header("secret-token"),
        )
        self.assertEqual(resp.status_code, 403)

    def test_session_access_enforces_owner_identity(self) -> None:
        settings.auth_enabled = True
        settings.auth_tokens = "owner:owner-token,guest:guest-token"
        settings.auth_token = ""
        settings.auth_user = ""
        settings.auth_project_owners = ""

        with Session(self.engine) as db:
            session = ChatSession(project_id=1, user_id="owner", title="owner-session")
            db.add(session)
            db.commit()
            db.refresh(session)
            session_id = int(session.id or 0)

        denied = self.client.get(
            f"/api/chat/sessions/{session_id}/messages",
            headers=self._auth_header("guest-token"),
        )
        self.assertEqual(denied.status_code, 403)

        allowed = self.client.get(
            f"/api/chat/sessions/{session_id}/messages",
            headers=self._auth_header("owner-token"),
        )
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.json(), [])

    def test_project_sessions_lists_only_current_user(self) -> None:
        settings.auth_enabled = True
        settings.auth_tokens = "owner:owner-token,guest:guest-token"
        settings.auth_token = ""
        settings.auth_user = ""
        settings.auth_project_owners = ""

        with Session(self.engine) as db:
            owner_old = ChatSession(project_id=1, user_id="owner", title="owner-old")
            guest_session = ChatSession(project_id=1, user_id="guest", title="guest-only")
            owner_new = ChatSession(project_id=1, user_id="owner", title="owner-new")
            db.add(owner_old)
            db.add(guest_session)
            db.add(owner_new)
            db.commit()
            db.refresh(owner_old)
            db.refresh(guest_session)
            db.refresh(owner_new)

        resp = self.client.get(
            "/api/chat/projects/1/sessions",
            headers=self._auth_header("owner-token"),
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        returned_ids = [int(item.get("id")) for item in payload]
        self.assertEqual(returned_ids, [int(owner_new.id or 0), int(owner_old.id or 0)])
        self.assertTrue(all(int(item.get("project_id", 0)) == 1 for item in payload))

    def test_project_session_rename_and_delete(self) -> None:
        settings.auth_enabled = True
        settings.auth_tokens = "owner:owner-token,guest:guest-token"
        settings.auth_token = ""
        settings.auth_user = ""
        settings.auth_project_owners = ""

        with Session(self.engine) as db:
            session = ChatSession(project_id=1, user_id="owner", title="old-title")
            db.add(session)
            db.commit()
            db.refresh(session)
            session_id = int(session.id or 0)

            db.add(ChatMessage(session_id=session_id, role="user", content="hello"))
            db.commit()

        renamed = self.client.put(
            f"/api/chat/projects/1/sessions/{session_id}",
            headers=self._auth_header("owner-token"),
            json={"title": "new-title"},
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json().get("title"), "new-title")

        denied = self.client.delete(
            f"/api/chat/projects/1/sessions/{session_id}",
            headers=self._auth_header("guest-token"),
        )
        self.assertEqual(denied.status_code, 403)

        removed = self.client.delete(
            f"/api/chat/projects/1/sessions/{session_id}",
            headers=self._auth_header("owner-token"),
        )
        self.assertEqual(removed.status_code, 200)
        self.assertEqual(int(removed.json().get("deleted_session_id", 0)), session_id)

        missing = self.client.get(
            f"/api/chat/sessions/{session_id}/messages",
            headers=self._auth_header("owner-token"),
        )
        self.assertEqual(missing.status_code, 404)

    def test_consistency_audit_run_and_list_with_auth(self) -> None:
        settings.auth_enabled = True
        settings.auth_tokens = "local-user:secret-token"
        settings.auth_token = ""
        settings.auth_user = ""
        settings.auth_project_owners = ""

        run_resp = self.client.post(
            "/api/chat/projects/1/consistency-audits/run",
            headers=self._auth_header("secret-token"),
            json={
                "run_mode": "sync",
                "reason": "manual_test",
                "force": True,
                "max_chapters": 2,
            },
        )
        self.assertEqual(run_resp.status_code, 200)
        run_payload = run_resp.json()
        self.assertEqual(run_payload.get("run_mode"), "sync")
        self.assertFalse(bool(run_payload.get("queued")))
        self.assertIsNotNone(run_payload.get("report"))

        list_resp = self.client.get(
            "/api/chat/projects/1/consistency-audits?limit=5",
            headers=self._auth_header("secret-token"),
        )
        self.assertEqual(list_resp.status_code, 200)
        reports = list_resp.json()
        self.assertTrue(isinstance(reports, list))
        self.assertGreaterEqual(len(reports), 1)


if __name__ == "__main__":
    unittest.main()
