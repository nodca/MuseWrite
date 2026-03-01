import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.api.router import api_router
from app.core.database import get_session
from app.core.config import settings
from app.models.chat import ActionAuditLog, ChatAction, ChatMessage, ChatSession
from app.services.context_compiler import CompiledContextBundle
from app.services.llm_provider import ChatGenerationResult


class ChatStreamEndpointTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "auth_enabled": settings.auth_enabled,
            "auth_tokens": settings.auth_tokens,
            "auth_token": settings.auth_token,
            "auth_user": settings.auth_user,
            "auth_project_owners": settings.auth_project_owners,
            "auth_disabled_user": settings.auth_disabled_user,
            "tot_enabled": settings.tot_enabled,
        }
        settings.auth_enabled = True
        settings.auth_tokens = "human-user:human-token"
        settings.auth_token = ""
        settings.auth_user = ""
        settings.auth_project_owners = ""
        settings.auth_disabled_user = "local-user"
        settings.tot_enabled = False

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

    @staticmethod
    def _mock_compiled_bundle() -> CompiledContextBundle:
        return CompiledContextBundle(
            model_context={"workspace_context": {}, "runtime_options": {}},
            evidence_event={
                "type": "evidence",
                "summary": {"dsl": 0, "graph": 0, "rag": 0},
                "policy": {"resolver_order": ["DSL", "GRAPH", "RAG"]},
            },
        )

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

    def _update_message_content_local(self, message_id: int, content: str, db=None) -> None:
        _ = db
        with Session(self.engine) as local_db:
            message = local_db.get(ChatMessage, int(message_id))
            if message is None:
                return
            message.content = str(content or "")
            session = local_db.get(ChatSession, int(message.session_id))
            if session is not None:
                session.updated_at = datetime.now(timezone.utc)
                local_db.add(session)
            local_db.add(message)
            local_db.commit()

    @patch("app.api.endpoints.chat.emit_chat_trace")
    @patch("app.api.endpoints.chat.update_message_content")
    @patch("app.api.endpoints.chat.resolve_model_profile_runtime")
    @patch("app.api.endpoints.chat.generate_chat")
    @patch("app.api.endpoints.chat.compile_context_bundle")
    def test_chat_stream_emits_meta_evidence_delta_done_and_persists_assistant_message(
        self,
        mock_compile_context_bundle,
        mock_generate_chat,
        mock_resolve_model_profile_runtime,
        mock_update_message_content,
        _mock_emit_chat_trace,
    ) -> None:
        mock_resolve_model_profile_runtime.return_value = None
        mock_update_message_content.side_effect = self._update_message_content_local
        mock_compile_context_bundle.return_value = self._mock_compiled_bundle()
        mock_generate_chat.return_value = ChatGenerationResult(
            assistant_text="你好，世界。这是一段稳定输出。",
            proposed_actions=[],
            usage={"provider": "stub"},
        )

        response = self.client.post(
            "/api/chat/stream",
            headers=self._auth_header(),
            json={"project_id": 1, "content": "帮我续写一句话"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))
        events = self._parse_sse_events(response.text)
        self.assertGreaterEqual(len(events), 4)
        self.assertEqual(events[0].get("type"), "meta")
        self.assertEqual(events[1].get("type"), "evidence")
        self.assertEqual(events[-1].get("type"), "done")
        self.assertIn("delta", [str(item.get("type")) for item in events])

        meta_event = events[0]
        session_id = int(meta_event.get("session_id", 0))
        self.assertGreater(session_id, 0)

        with Session(self.engine) as db:
            session = db.get(ChatSession, session_id)
            self.assertIsNotNone(session)
            if session is not None:
                self.assertEqual(session.user_id, "human-user")

            rows = db.exec(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.id.asc())
            ).all()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0].role, "user")
            self.assertEqual(rows[0].content, "帮我续写一句话")
            self.assertEqual(rows[1].role, "assistant")
            self.assertEqual(rows[1].content, "你好，世界。这是一段稳定输出。")

    @patch("app.api.endpoints.chat.emit_chat_trace")
    @patch("app.api.endpoints.chat.update_message_content")
    @patch("app.api.endpoints.chat.resolve_model_profile_runtime")
    @patch("app.api.endpoints.chat.generate_chat")
    @patch("app.api.endpoints.chat.compile_context_bundle")
    def test_chat_stream_creates_session_when_session_id_missing(
        self,
        mock_compile_context_bundle,
        mock_generate_chat,
        mock_resolve_model_profile_runtime,
        mock_update_message_content,
        _mock_emit_chat_trace,
    ) -> None:
        mock_resolve_model_profile_runtime.return_value = None
        mock_update_message_content.side_effect = self._update_message_content_local
        mock_compile_context_bundle.return_value = self._mock_compiled_bundle()
        mock_generate_chat.return_value = ChatGenerationResult(
            assistant_text="创建成功",
            proposed_actions=[],
            usage={"provider": "stub"},
        )

        with Session(self.engine) as db:
            before_count = len(db.exec(select(ChatSession)).all())
            self.assertEqual(before_count, 0)

        response = self.client.post(
            "/api/chat/stream",
            headers=self._auth_header(),
            json={"project_id": 1, "content": "创建会话"},
        )
        self.assertEqual(response.status_code, 200)
        events = self._parse_sse_events(response.text)
        meta_event = next(item for item in events if item.get("type") == "meta")
        session_id = int(meta_event.get("session_id", 0))
        self.assertGreater(session_id, 0)

        with Session(self.engine) as db:
            sessions = db.exec(select(ChatSession)).all()
            self.assertEqual(len(sessions), 1)
            self.assertEqual(int(sessions[0].id or 0), session_id)

    @patch("app.api.endpoints.chat.emit_chat_trace")
    @patch("app.api.endpoints.chat.resolve_model_profile_runtime")
    @patch("app.api.endpoints.chat.update_message_content")
    @patch("app.api.endpoints.chat.generate_chat")
    @patch("app.api.endpoints.chat.compile_context_bundle")
    def test_chat_stream_emits_error_event_when_stream_finalize_fails(
        self,
        mock_compile_context_bundle,
        mock_generate_chat,
        mock_update_message_content,
        mock_resolve_model_profile_runtime,
        _mock_emit_chat_trace,
    ) -> None:
        mock_resolve_model_profile_runtime.return_value = None
        mock_compile_context_bundle.return_value = self._mock_compiled_bundle()
        mock_generate_chat.return_value = ChatGenerationResult(
            assistant_text="故障分支",
            proposed_actions=[],
            usage={"provider": "stub"},
        )
        mock_update_message_content.side_effect = [RuntimeError("persist failed"), None]

        response = self.client.post(
            "/api/chat/stream",
            headers=self._auth_header(),
            json={"project_id": 1, "content": "触发错误"},
        )
        self.assertEqual(response.status_code, 200)
        events = self._parse_sse_events(response.text)
        event_types = [str(item.get("type")) for item in events]
        self.assertIn("meta", event_types)
        self.assertIn("evidence", event_types)
        self.assertIn("error", event_types)
        self.assertNotIn("done", event_types)

        error_event = next(item for item in events if item.get("type") == "error")
        self.assertIn("persist failed", str(error_event.get("message", "")))
        self.assertEqual(mock_update_message_content.call_count, 2)

    @patch("app.api.endpoints.chat.resolve_model_profile_runtime")
    def test_chat_stream_returns_400_when_model_profile_invalid(self, mock_resolve_model_profile_runtime) -> None:
        mock_resolve_model_profile_runtime.side_effect = ValueError("model profile not found")
        response = self.client.post(
            "/api/chat/stream",
            headers=self._auth_header(),
            json={
                "project_id": 1,
                "content": "测试 profile 校验",
                "model_profile_id": "missing-profile",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("detail"), "model profile not found")

    @patch("app.api.endpoints.chat.resolve_model_profile_runtime")
    def test_chat_stream_rejects_cross_user_session_access(self, mock_resolve_model_profile_runtime) -> None:
        mock_resolve_model_profile_runtime.return_value = None
        settings.auth_tokens = "human-user:human-token,guest-user:guest-token"

        with Session(self.engine) as db:
            session = ChatSession(project_id=1, user_id="guest-user", title="guest-session")
            db.add(session)
            db.commit()
            db.refresh(session)
            denied_session_id = int(session.id or 0)

        response = self.client.post(
            "/api/chat/stream",
            headers=self._auth_header(),
            json={"project_id": 1, "session_id": denied_session_id, "content": "越权测试"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json().get("detail"), "session access denied")

    @patch("app.api.endpoints.chat.emit_chat_trace")
    @patch("app.api.endpoints.chat.update_message_content")
    @patch("app.api.endpoints.chat.resolve_model_profile_runtime")
    @patch("app.api.endpoints.chat.generate_chat")
    @patch("app.api.endpoints.chat.compile_context_bundle")
    def test_chat_stream_persists_proposed_actions_and_meta_ids(
        self,
        mock_compile_context_bundle,
        mock_generate_chat,
        mock_resolve_model_profile_runtime,
        mock_update_message_content,
        _mock_emit_chat_trace,
    ) -> None:
        mock_resolve_model_profile_runtime.return_value = None
        mock_update_message_content.side_effect = self._update_message_content_local
        mock_compile_context_bundle.return_value = self._mock_compiled_bundle()
        mock_generate_chat.return_value = ChatGenerationResult(
            assistant_text="我建议更新一个设定。",
            proposed_actions=[
                {"action_type": "setting.upsert", "payload": {"key": "世界观", "value": {"风格": "蒸汽朋克"}}},
                {"action_type": "setting.delete", "payload": "invalid-payload-should-be-dropped"},
            ],
            usage={"provider": "stub"},
        )

        response = self.client.post(
            "/api/chat/stream",
            headers=self._auth_header(),
            json={"project_id": 1, "content": "给我一个设定建议"},
        )
        self.assertEqual(response.status_code, 200)
        events = self._parse_sse_events(response.text)
        meta_event = next(item for item in events if item.get("type") == "meta")
        proposed_ids = meta_event.get("proposed_action_ids")
        self.assertTrue(isinstance(proposed_ids, list))
        self.assertEqual(len(proposed_ids), 1)
        created_action_id = int(proposed_ids[0])
        self.assertGreater(created_action_id, 0)

        with Session(self.engine) as db:
            action = db.get(ChatAction, created_action_id)
            self.assertIsNotNone(action)
            if action is not None:
                self.assertEqual(action.action_type, "setting.upsert")
                self.assertEqual(action.status, "proposed")
                self.assertEqual(action.payload.get("key"), "世界观")

            logs = db.exec(
                select(ActionAuditLog)
                .where(ActionAuditLog.action_id == created_action_id)
                .order_by(ActionAuditLog.id.asc())
            ).all()
            self.assertGreaterEqual(len(logs), 1)
            self.assertEqual(logs[0].event_type, "proposed")

    @patch("app.api.endpoints.chat.emit_chat_trace")
    @patch("app.api.endpoints.chat.update_message_content")
    @patch("app.api.endpoints.chat.resolve_model_profile_runtime")
    @patch("app.api.endpoints.chat.generate_chat")
    @patch("app.api.endpoints.chat.compile_context_bundle")
    def test_chat_stream_falls_back_when_generate_chat_raises(
        self,
        mock_compile_context_bundle,
        mock_generate_chat,
        mock_resolve_model_profile_runtime,
        mock_update_message_content,
        _mock_emit_chat_trace,
    ) -> None:
        mock_resolve_model_profile_runtime.return_value = None
        mock_update_message_content.side_effect = self._update_message_content_local
        mock_compile_context_bundle.return_value = self._mock_compiled_bundle()
        mock_generate_chat.side_effect = RuntimeError("model boom")

        response = self.client.post(
            "/api/chat/stream",
            headers=self._auth_header(),
            json={"project_id": 1, "content": "触发模型异常"},
        )
        self.assertEqual(response.status_code, 200)

        events = self._parse_sse_events(response.text)
        event_types = [str(item.get("type")) for item in events]
        self.assertIn("done", event_types)
        done_event = next(item for item in events if item.get("type") == "done")
        usage = done_event.get("usage") if isinstance(done_event.get("usage"), dict) else {}
        self.assertEqual(usage.get("provider"), "error")

        with Session(self.engine) as db:
            assistant_rows = db.exec(
                select(ChatMessage)
                .where(ChatMessage.role == "assistant")
                .order_by(ChatMessage.id.desc())
            ).all()
            self.assertGreaterEqual(len(assistant_rows), 1)
            self.assertTrue(str(assistant_rows[0].content).startswith("模型调用失败：model boom"))


if __name__ == "__main__":
    unittest.main()
