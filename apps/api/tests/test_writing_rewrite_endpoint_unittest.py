import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.api.router import api_router
from app.core.config import settings
from app.core.database import get_session
from app.services.llm_provider import ChatGenerationResult


class WritingRewriteEndpointTestCase(unittest.TestCase):
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

    @patch("app.api.endpoints.writing.resolve_model_profile_runtime")
    @patch("app.api.endpoints.writing.generate_chat", new_callable=AsyncMock)
    def test_rewrite_polish_uses_runtime_profile(
        self,
        mock_generate_chat: AsyncMock,
        mock_resolve_model_profile_runtime,
    ) -> None:
        runtime_profile = {
            "profile_id": "writer",
            "provider": "openai_compatible",
            "base_url": "http://example/v1",
            "api_key": "sk-test",
            "model": "writer-polish",
        }
        mock_resolve_model_profile_runtime.return_value = runtime_profile
        mock_generate_chat.return_value = ChatGenerationResult(
            assistant_text="润色后的正文",
            proposed_actions=[],
            usage={"provider": "stub"},
        )

        response = self.client.post(
            "/api/writing/rewrite",
            headers=self._auth_header(),
            json={
                "project_id": 1,
                "mode": "polish",
                "text": "原文段落",
                "model_profile_id": "writer",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("result"), "润色后的正文")

        prompt = str(mock_generate_chat.await_args_list[0].args[0])
        self.assertIn("### Instruction", prompt)
        self.assertIn("任务：对给定原文做润色", prompt)
        self.assertIn("原文", prompt)
        self.assertIn("原文段落", prompt)
        self.assertEqual(mock_generate_chat.await_args_list[0].kwargs.get("runtime_config"), runtime_profile)
        self.assertEqual(mock_generate_chat.await_args_list[0].kwargs.get("temperature_profile"), "chat")
        self.assertEqual(
            mock_generate_chat.await_args_list[0].kwargs.get("context", {})
            .get("runtime_options", {})
            .get("source"),
            "rewrite_shim",
        )
        self.assertEqual(
            response.headers.get("X-Deprecated-Endpoint"),
            "/api/writing/rewrite",
        )

    @patch("app.api.endpoints.writing.resolve_model_profile_runtime")
    @patch("app.api.endpoints.writing.generate_chat", new_callable=AsyncMock)
    def test_rewrite_expand_builds_expand_prompt(
        self,
        mock_generate_chat: AsyncMock,
        mock_resolve_model_profile_runtime,
    ) -> None:
        mock_resolve_model_profile_runtime.return_value = None
        mock_generate_chat.return_value = ChatGenerationResult(
            assistant_text="扩写后的正文",
            proposed_actions=[],
            usage={"provider": "stub"},
        )

        response = self.client.post(
            "/api/writing/rewrite",
            headers=self._auth_header(),
            json={
                "project_id": 1,
                "mode": "expand",
                "text": "原文",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("result"), "扩写后的正文")

        prompt = str(mock_generate_chat.await_args_list[0].args[0])
        self.assertIn("任务：对给定原文做扩写", prompt)
        self.assertIn("原文", prompt)


if __name__ == "__main__":
    unittest.main()

