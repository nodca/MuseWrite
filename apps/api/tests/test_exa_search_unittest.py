"""Unit tests for exa_search service."""

import unittest
from unittest import mock

from app.core.config import settings


class TestExaSearch(unittest.TestCase):
    """Tests for app.services.exa_search."""

    def test_search_returns_normalized_hits(self) -> None:
        fake_payload = {
            "results": [
                {
                    "title": "三国演义角色解析",
                    "url": "https://example.com/sanguo",
                    "text": "刘备是蜀汉的开国皇帝...",
                    "highlights": ["刘备字玄德，涿郡人。"],
                },
                {
                    "title": "武侠小说设定参考",
                    "url": "https://example.com/wuxia",
                    "text": "江湖门派体系...",
                    "highlights": [],
                },
            ]
        }

        original_enabled = settings.exa_enabled
        original_key = settings.exa_api_key
        try:
            settings.exa_enabled = True
            settings.exa_api_key = "test-key"

            with mock.patch(
                "app.services.exa_search._exa_request",
                return_value=fake_payload,
            ):
                from app.services.exa_search import search

                hits = search("刘备", num_results=5)

            self.assertEqual(len(hits), 2)
            self.assertEqual(hits[0]["kind"], "web_search")
            self.assertEqual(hits[0]["title"], "三国演义角色解析")
            self.assertEqual(hits[0]["snippet"], "刘备字玄德，涿郡人。")
            self.assertEqual(hits[0]["source_url"], "https://example.com/sanguo")
            self.assertAlmostEqual(hits[0]["confidence"], 0.75)
            # Second hit falls back to text[:300] since highlights is empty
            self.assertIn("江湖门派体系", hits[1]["snippet"])
        finally:
            settings.exa_enabled = original_enabled
            settings.exa_api_key = original_key

    def test_search_disabled_returns_empty(self) -> None:
        original_enabled = settings.exa_enabled
        try:
            settings.exa_enabled = False
            from app.services.exa_search import search

            hits = search("test query")
            self.assertEqual(hits, [])
        finally:
            settings.exa_enabled = original_enabled

    def test_search_empty_query_raises(self) -> None:
        original_enabled = settings.exa_enabled
        original_key = settings.exa_api_key
        try:
            settings.exa_enabled = True
            settings.exa_api_key = "test-key"
            from app.services.exa_search import search

            with self.assertRaises(ValueError):
                search("")
            with self.assertRaises(ValueError):
                search("   ")
        finally:
            settings.exa_enabled = original_enabled
            settings.exa_api_key = original_key


if __name__ == "__main__":
    unittest.main()
