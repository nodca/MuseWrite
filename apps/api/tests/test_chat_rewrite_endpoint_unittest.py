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


# Rewrite endpoint coverage.
class ChatRewriteEndpointTestCase(unittest.TestCase):
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

    @patch("app.api.endpoints.chat_writing_assist.resolve_model_profile_runtime")
    @patch("app.api.endpoints.chat_writing_assist.generate_chat", new_callable=AsyncMock)
    def test_chat_rewrite_endpoint_skips_entities_hint_even_when_enabled(
        self,
        mock_generate_chat: AsyncMock,
        mock_resolve_model_profile_runtime,
    ) -> None:
        mock_resolve_model_profile_runtime.return_value = None
        mock_generate_chat.return_value = ChatGenerationResult(
            assistant_text="他停住脚步，抬眼看向巷口的阴影。",
            proposed_actions=[],
            usage={"provider": "stub"},
        )

        endpoint_path = "/api/chat/rewrite/polish"
        response = self.client.post(
            endpoint_path,
            headers=self._auth_header(),
            json={
                "project_id": 1,
                "text": "克莱恩跟着值夜者穿过廷根街，塔罗会的低语在耳边回响。",
                "active_roles": ["克莱恩", "邓恩"],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_generate_chat.await_count, 1)
        generation_prompt = str(mock_generate_chat.await_args_list[0].args[0])
        self.assertNotIn("实体锚定抽取器", generation_prompt)
        self.assertNotIn("实体锚定（严格遵守）", generation_prompt)
        self.assertNotIn("<Entities>", generation_prompt)

    @patch("app.api.endpoints.chat_writing_assist.resolve_model_profile_runtime")
    @patch("app.api.endpoints.chat_writing_assist.generate_chat", new_callable=AsyncMock)
    def test_chat_rewrite_endpoint_skips_entities_hint_when_disabled(
        self,
        mock_generate_chat: AsyncMock,
        mock_resolve_model_profile_runtime,
    ) -> None:
        mock_resolve_model_profile_runtime.return_value = None
        mock_generate_chat.return_value = ChatGenerationResult(
            assistant_text="风从台阶间掠过，他没有回头。",
            proposed_actions=[],
            usage={"provider": "stub"},
        )

        endpoint_path = "/api/chat/rewrite/polish"
        response = self.client.post(
            endpoint_path,
            headers=self._auth_header(),
            json={
                "project_id": 1,
                "text": "克莱恩跟着值夜者穿过廷根街，塔罗会的低语在耳边回响。",
                "active_roles": ["克莱恩", "邓恩"],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_generate_chat.await_count, 1)
        prompt = str(mock_generate_chat.await_args_list[0].args[0])
        self.assertNotIn("实体锚定（严格遵守）", prompt)
        self.assertNotIn("<Entities>", prompt)
    @patch("app.api.endpoints.chat_writing_assist.resolve_model_profile_runtime")
    @patch("app.api.endpoints.chat_writing_assist.generate_chat", new_callable=AsyncMock)
    def test_chat_rewrite_endpoint_polish_mode_uses_text_payload(
        self,
        mock_generate_chat: AsyncMock,
        mock_resolve_model_profile_runtime,
    ) -> None:
        mock_resolve_model_profile_runtime.return_value = None
        mock_generate_chat.return_value = ChatGenerationResult(
            assistant_text="润色后的文本\n第二行",
            proposed_actions=[],
            usage={"provider": "stub"},
        )

        endpoint_path = "/api/chat/rewrite/polish"
        response = self.client.post(
            endpoint_path,
            headers=self._auth_header(),
            json={
                "project_id": 1,
                "text": "原文段落",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("suggestion"), "润色后的文本\n第二行")

        prompt = str(mock_generate_chat.await_args_list[0].args[0])
        self.assertIn("### Instruction", prompt)
        self.assertIn("任务：对给定原文做润色", prompt)
        self.assertIn("原文段落", prompt)

    @patch("app.api.endpoints.chat_writing_assist.resolve_model_profile_runtime")
    @patch("app.api.endpoints.chat_writing_assist.generate_chat", new_callable=AsyncMock)
    def test_chat_rewrite_endpoint_rejects_empty_text_for_expand_mode(
        self,
        mock_generate_chat: AsyncMock,
        mock_resolve_model_profile_runtime,
    ) -> None:
        mock_resolve_model_profile_runtime.return_value = None

        endpoint_path = "/api/chat/rewrite/expand"
        response = self.client.post(
            endpoint_path,
            headers=self._auth_header(),
            json={
                "project_id": 1,
                "text": "",
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("String should have at least 1 character", str(response.json().get("detail", "")))


if __name__ == "__main__":
    unittest.main()


