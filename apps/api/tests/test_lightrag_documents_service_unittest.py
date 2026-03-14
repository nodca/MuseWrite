import unittest
from unittest.mock import patch

from app.core.config import settings
from app.services import lightrag_documents as docs


class LightRAGDocumentsServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "lightrag_enabled": settings.lightrag_enabled,
            "lightrag_base_url": settings.lightrag_base_url,
        }
        settings.lightrag_enabled = True
        settings.lightrag_base_url = "http://lightrag.local"

    def tearDown(self) -> None:
        for key, value in self._snapshot.items():
            setattr(settings, key, value)

    def test_insert_text_document_rejects_empty_text(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            docs.insert_text_document(project_id=1, text="   ")
        self.assertEqual(str(ctx.exception), "text is required")

    @patch("app.services.lightrag_documents._request_json")
    def test_list_project_documents_filters_by_project_scope(self, mock_request_json) -> None:
        mock_request_json.return_value = {
            "status": "ok",
            "data": {
                "documents": [
                    {
                        "doc_id": "d1",
                        "file_source": "np://project/7/plot-a.txt",
                        "metadata": {},
                        "status": "done",
                    },
                    {
                        "doc_id": "d2",
                        "file_source": "np://project/9/plot-b.txt",
                        "metadata": {"project_id": "7"},
                        "status": "done",
                    },
                    {
                        "doc_id": "d3",
                        "file_source": "np://project/9/plot-c.txt",
                        "metadata": {"project_id": "9"},
                        "status": "done",
                    },
                ],
                "pagination": {"page": 1, "page_size": 50, "total": 3},
            },
        }

        result = docs.list_project_documents(
            project_id=7,
            page=1,
            page_size=50,
            status_filter=None,
            sort_field="updated_at",
            sort_direction="desc",
        )
        self.assertEqual(result.get("project_id"), 7)
        self.assertEqual(result.get("page_scan_count"), 3)
        self.assertEqual(result.get("project_hit_count"), 2)
        doc_ids = [str(item.get("doc_id")) for item in result.get("documents", [])]
        self.assertEqual(doc_ids, ["d1", "d2"])

    @patch("app.services.lightrag_documents._request_json")
    def test_delete_documents_uses_batch_mode(self, mock_request_json) -> None:
        mock_request_json.return_value = {"status": "ok", "deleted": 2}
        result = docs.delete_documents(doc_ids=["a", "b"], delete_file=True, delete_llm_cache=False)

        self.assertEqual(result.get("provider"), "lightrag_native")
        self.assertEqual(result.get("mode"), "batch")
        self.assertEqual(int(result.get("requested", 0)), 2)
        self.assertEqual(result.get("result"), {"status": "ok", "deleted": 2})
        self.assertEqual(mock_request_json.call_count, 1)
        self.assertEqual(mock_request_json.call_args.args[0], "DELETE")
        payload = mock_request_json.call_args.kwargs.get("json_body", {})
        self.assertEqual(payload.get("doc_ids"), ["a", "b"])

    @patch("app.services.lightrag_documents._request_json")
    def test_get_pipeline_status_uses_get_only(self, mock_request_json) -> None:
        mock_request_json.return_value = {"status": "ok", "pipeline": {"queued": 0, "running": 1}}
        result = docs.get_pipeline_status()
        self.assertEqual(result.get("status"), "ok")
        pipeline = result.get("pipeline") if isinstance(result.get("pipeline"), dict) else {}
        self.assertEqual(int(pipeline.get("running", -1)), 1)
        self.assertEqual(mock_request_json.call_count, 1)
        self.assertEqual(mock_request_json.call_args_list[0].args[0], "GET")


if __name__ == "__main__":
    unittest.main()
