import unittest
from unittest import mock

import app.services.retrieval_adapters as retrieval_adapters_module
from app.core.config import settings


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return list(self._rows)

    def single(self):
        if not self._rows:
            return None
        return self._rows[0]


class _FakeSession:
    def __init__(self):
        self.queries: list[str] = []
        self.projection_version = 1
        self.scope_graph_name = ""
        self.scope_built_version = 0
        self.existing_graphs: set[str] = set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, cypher: str, **params):
        self.queries.append(cypher)
        if "ProjectionCatalog" in cypher and "ProjectionState" in cypher:
            return _FakeResult(
                [
                    {
                        "projection_version": self.projection_version,
                        "graph_name": self.scope_graph_name,
                        "built_version": self.scope_built_version,
                        "scope_key": str(params.get("scope_key") or "all"),
                        "invalidated_reason": "",
                    }
                ]
            )
        if "gds.graph.exists" in cypher:
            graph_name = str(params.get("graph_name") or "")
            return _FakeResult([{"exists": graph_name in self.existing_graphs}])
        if "CALL gds.graph.list()" in cypher:
            keep_graph_name = str(params.get("keep_graph_name") or "")
            graph_names = [name for name in self.existing_graphs if name != keep_graph_name]
            return _FakeResult([{"graph_names": graph_names}])
        if "gds.pageRank.stream" in cypher:
            return _FakeResult(
                [
                    {"name": "戒指", "name_norm": "戒指", "score": 0.91},
                    {"name": "废宅", "name_norm": "废宅", "score": 0.82},
                    {"name": "角色C", "name_norm": "角色c", "score": 0.74},
                ]
            )
        if "MATCH (seed:Entity" in cypher:
            return _FakeResult(
                [
                    {"name": "戒指", "name_norm": "戒指"},
                    {"name": "废宅", "name_norm": "废宅"},
                ]
            )
        if "gds.graph.project(" in cypher:
            graph_name = str(params.get("graph_name") or "")
            if graph_name:
                self.existing_graphs.add(graph_name)
            return _FakeResult([{"graphName": params.get("graph_name")}])
        if "SET state.graph_name = $graph_name" in cypher:
            self.scope_graph_name = str(params.get("graph_name") or "")
            self.scope_built_version = int(params.get("built_version") or 0)
            return _FakeResult(
                [
                    {
                        "graph_name": self.scope_graph_name,
                        "built_version": self.scope_built_version,
                        "scope_key": str(params.get("scope_key") or "all"),
                    }
                ]
            )
        if "MATCH (a:Entity {project_id: $project_id})-[r:FACT {project_id: $project_id}]->(b:Entity {project_id: $project_id})" in cypher:
            return _FakeResult(
                [
                    {
                        "source": "戒指",
                        "source_norm": "戒指",
                        "relation": "LOCATED_AT",
                        "target": "废宅",
                        "target_norm": "废宅",
                        "rel_props": {
                            "fact_key": "fact_ring_ruin",
                            "confidence": 0.93,
                            "updated_at": "2026-03-01T00:00:00+00:00",
                        },
                    }
                ]
            )
        if "gds.graph.drop" in cypher:
            graph_name = str(params.get("graph_name") or "")
            self.existing_graphs.discard(graph_name)
            return _FakeResult([{"graphName": params.get("graph_name")}])
        return _FakeResult([])


class _FakeDriver:
    def __init__(self):
        self.session_obj = _FakeSession()

    def session(self, database=None):
        return self.session_obj

    def close(self):
        return None


class GraphPprRetrievalTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "neo4j_enabled": settings.neo4j_enabled,
            "neo4j_uri": settings.neo4j_uri,
            "neo4j_database": settings.neo4j_database,
            "neo4j_gds_required": getattr(settings, "neo4j_gds_required", None),
            "neo4j_gds_graph_name_prefix": getattr(settings, "neo4j_gds_graph_name_prefix", None),
            "graph_temporal_enabled": settings.graph_temporal_enabled,
        }

    def tearDown(self) -> None:
        for name, value in self._snapshot.items():
            setattr(settings, name, value)

    def test_fetch_neo4j_graph_facts_uses_ppr_ranked_edges(self) -> None:
        settings.neo4j_enabled = True
        settings.neo4j_uri = "bolt://neo4j:7687"
        settings.neo4j_database = "neo4j"
        settings.neo4j_gds_required = True
        settings.neo4j_gds_graph_name_prefix = "novel_ppr"
        settings.graph_temporal_enabled = False
        fake_driver = _FakeDriver()

        class _FakeGraphDatabase:
            @staticmethod
            def driver(_uri, auth=None):
                return fake_driver

        with mock.patch.object(retrieval_adapters_module, "GraphDatabase", _FakeGraphDatabase):
            hits = retrieval_adapters_module.fetch_neo4j_graph_facts(
                7,
                ["戒指", "废宅"],
                anchor="戒指",
                limit=4,
                raise_on_error=True,
            )

        self.assertTrue(any("gds.pageRank.stream" in query for query in fake_driver.session_obj.queries))
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].get("kind"), "graph_edge")
        self.assertEqual(hits[0].get("fact_key"), "fact_ring_ruin")
        self.assertGreater(float(hits[0].get("ppr_score") or 0.0), 0.0)

    def test_fetch_neo4j_graph_facts_reuses_named_projection_when_revision_is_clean(self) -> None:
        settings.neo4j_enabled = True
        settings.neo4j_uri = "bolt://neo4j:7687"
        settings.neo4j_database = "neo4j"
        settings.neo4j_gds_required = True
        settings.neo4j_gds_graph_name_prefix = "novel_ppr"
        settings.graph_temporal_enabled = False
        fake_driver = _FakeDriver()
        fake_driver.session_obj.scope_graph_name = "novel_ppr_7_all_v1"
        fake_driver.session_obj.scope_built_version = 1
        fake_driver.session_obj.existing_graphs.add("novel_ppr_7_all_v1")

        class _FakeGraphDatabase:
            @staticmethod
            def driver(_uri, auth=None):
                return fake_driver

        with mock.patch.object(retrieval_adapters_module, "GraphDatabase", _FakeGraphDatabase):
            hits = retrieval_adapters_module.fetch_neo4j_graph_facts(
                7,
                ["戒指", "废宅"],
                anchor="戒指",
                limit=4,
                raise_on_error=True,
            )

        self.assertEqual(len(hits), 1)
        self.assertTrue(any("gds.graph.exists" in query for query in fake_driver.session_obj.queries))
        self.assertFalse(any("gds.graph.project(" in query for query in fake_driver.session_obj.queries))


if __name__ == "__main__":
    unittest.main()
