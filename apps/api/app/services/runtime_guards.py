from app.services.retrieval_adapters import ensure_neo4j_gds_available


def assert_required_runtime_dependencies() -> None:
    ensure_neo4j_gds_available(raise_on_error=True)
