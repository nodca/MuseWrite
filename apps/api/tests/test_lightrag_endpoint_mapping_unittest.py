import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.api.router import api_router
from app.core.config import settings
from app.core.database import get_session


class LightRAGEndpointMappingTestCase(unittest.TestCase):
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

    @patch("app.api.endpoints.chat.insert_text_document")
    def test_documents_text_value_error_maps_to_400(self, mock_insert_text) -> None:
        mock_insert_text.side_effect = ValueError("invalid text payload")
        response = self.client.post(
            "/api/chat/projects/1/documents/text",
            headers=self._auth_header(),
            json={"text": "hello"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("detail"), "invalid text payload")

    @patch("app.api.endpoints.chat.insert_text_document")
    def test_documents_text_runtime_error_maps_to_502(self, mock_insert_text) -> None:
        mock_insert_text.side_effect = RuntimeError("upstream unavailable")
        response = self.client.post(
            "/api/chat/projects/1/documents/text",
            headers=self._auth_header(),
            json={"text": "hello"},
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json().get("detail"), "upstream unavailable")

    @patch("app.api.endpoints.chat.list_project_documents")
    def test_documents_paginated_runtime_error_maps_to_502(self, mock_list_documents) -> None:
        mock_list_documents.side_effect = RuntimeError("list failed")
        response = self.client.post(
            "/api/chat/projects/1/documents/paginated",
            headers=self._auth_header(),
            json={},
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json().get("detail"), "list failed")

    @patch("app.api.endpoints.chat.delete_documents")
    def test_documents_delete_value_error_maps_to_400(self, mock_delete_documents) -> None:
        mock_delete_documents.side_effect = ValueError("invalid doc ids")
        response = self.client.request(
            "DELETE",
            "/api/chat/projects/1/documents",
            headers=self._auth_header(),
            json={"doc_ids": ["doc-1"]},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("detail"), "invalid doc ids")

    @patch("app.api.endpoints.chat.delete_documents")
    def test_documents_delete_runtime_error_maps_to_502(self, mock_delete_documents) -> None:
        mock_delete_documents.side_effect = RuntimeError("delete failed")
        response = self.client.request(
            "DELETE",
            "/api/chat/projects/1/documents",
            headers=self._auth_header(),
            json={"doc_ids": ["doc-1"]},
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json().get("detail"), "delete failed")

    @patch("app.api.endpoints.chat.get_pipeline_status")
    def test_documents_pipeline_status_runtime_error_maps_to_502(self, mock_get_pipeline_status) -> None:
        mock_get_pipeline_status.side_effect = RuntimeError("pipeline status failed")
        response = self.client.get(
            "/api/chat/projects/1/documents/pipeline-status",
            headers=self._auth_header(),
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json().get("detail"), "pipeline status failed")


if __name__ == "__main__":
    unittest.main()
