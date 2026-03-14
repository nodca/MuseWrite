import time
import unittest

import app.services.context_compiler as context_compiler_module
from app.services.context_compiler._state import _RETRIEVAL_CACHE_LOCK, _GRAPH_HITS_CACHE


class GraphCacheInvalidationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        with _RETRIEVAL_CACHE_LOCK:
            self._graph_cache_snapshot = dict(_GRAPH_HITS_CACHE)

    def tearDown(self) -> None:
        with _RETRIEVAL_CACHE_LOCK:
            _GRAPH_HITS_CACHE.clear()
            _GRAPH_HITS_CACHE.update(self._graph_cache_snapshot)

    def test_invalidate_graph_retrieval_cache_removes_project_entries_only(self) -> None:
        expires_at = time.time() + 60
        with _RETRIEVAL_CACHE_LOCK:
            _GRAPH_HITS_CACHE.clear()
            _GRAPH_HITS_CACHE["p:7|a:戒指|l:4|c:47|terms:废宅,戒指"] = (
                expires_at,
                [{"fact": "project7"}],
            )
            _GRAPH_HITS_CACHE["p:8|a:戒指|l:4|c:47|terms:废宅,戒指"] = (
                expires_at,
                [{"fact": "project8"}],
            )

        cleared = context_compiler_module.invalidate_graph_retrieval_cache(7)

        self.assertEqual(cleared, 1)
        with _RETRIEVAL_CACHE_LOCK:
            self.assertNotIn("p:7|a:戒指|l:4|c:47|terms:废宅,戒指", _GRAPH_HITS_CACHE)
            self.assertIn("p:8|a:戒指|l:4|c:47|terms:废宅,戒指", _GRAPH_HITS_CACHE)


if __name__ == "__main__":
    unittest.main()
