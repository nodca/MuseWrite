import unittest
from unittest import mock

import app.services.retrieval_adapters as retrieval_adapters_module
import app.services.runtime_guards as runtime_guards_module
from app.core.config import settings


class StartupGdsGateTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "neo4j_enabled": settings.neo4j_enabled,
            "neo4j_gds_required": getattr(settings, "neo4j_gds_required", None),
            "neo4j_gds_min_version": getattr(settings, "neo4j_gds_min_version", None),
        }

    def tearDown(self) -> None:
        for name, value in self._snapshot.items():
            setattr(settings, name, value)

    def test_probe_skips_when_neo4j_disabled(self) -> None:
        settings.neo4j_enabled = False
        settings.neo4j_gds_required = True

        with mock.patch.object(retrieval_adapters_module, "GraphDatabase", object()):
            result = retrieval_adapters_module.ensure_neo4j_gds_available()

        self.assertEqual(result.get("status"), "skipped")
        self.assertEqual(result.get("reason"), "neo4j_disabled")

    def test_probe_skips_when_gds_not_required(self) -> None:
        settings.neo4j_enabled = True
        settings.neo4j_gds_required = False

        with mock.patch.object(retrieval_adapters_module, "GraphDatabase", object()):
            result = retrieval_adapters_module.ensure_neo4j_gds_available()

        self.assertEqual(result.get("status"), "skipped")
        self.assertEqual(result.get("reason"), "gds_not_required")

    def test_probe_raises_when_gds_is_required_but_unavailable(self) -> None:
        settings.neo4j_enabled = True
        settings.neo4j_gds_required = True
        settings.neo4j_gds_min_version = ""

        with mock.patch.object(retrieval_adapters_module, "GraphDatabase", None):
            with self.assertRaises(RuntimeError):
                retrieval_adapters_module.ensure_neo4j_gds_available(raise_on_error=True)

    def test_probe_rejects_empty_gds_version(self) -> None:
        settings.neo4j_enabled = True
        settings.neo4j_gds_required = True
        settings.neo4j_gds_min_version = ""

        class _FakeResult:
            def single(self):
                return {"version": ""}

        class _FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def run(self, _cypher: str):
                return _FakeResult()

        class _FakeDriver:
            def session(self, database=None):
                return _FakeSession()

            def close(self):
                return None

        class _FakeGraphDatabase:
            @staticmethod
            def driver(_uri, auth=None):
                return _FakeDriver()

        with mock.patch.object(retrieval_adapters_module, "GraphDatabase", _FakeGraphDatabase):
            with self.assertRaises(RuntimeError):
                retrieval_adapters_module.ensure_neo4j_gds_available(raise_on_error=True)

    def test_runtime_guard_calls_gds_probe(self) -> None:
        settings.neo4j_enabled = True
        settings.neo4j_gds_required = True

        with mock.patch("app.services.runtime_guards.ensure_neo4j_gds_available", return_value={"status": "ok"}) as probe:
            with mock.patch("app.services.runtime_guards.ensure_lightrag_available", return_value=True):
                runtime_guards_module.assert_required_runtime_dependencies()

        probe.assert_called_once_with(raise_on_error=True)


if __name__ == "__main__":
    unittest.main()
