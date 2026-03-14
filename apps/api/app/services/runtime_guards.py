from app.services.retrieval_adapters import ensure_neo4j_gds_available


def ensure_lightrag_available(*, raise_on_error: bool = False) -> bool:
    """Check LightRAG config is complete. Actual connectivity is verified lazily on first use."""
    from app.core.config import settings

    if not settings.lightrag_enabled:
        if raise_on_error:
            raise RuntimeError(
                "LIGHTRAG_ENABLED=false — LightRAG is a required dependency. "
                "Set LIGHTRAG_ENABLED=true and configure LIGHTRAG_BASE_URL."
            )
        return False
    if not settings.lightrag_base_url:
        if raise_on_error:
            raise RuntimeError(
                "LIGHTRAG_BASE_URL is not configured. "
                "LightRAG is a required dependency."
            )
        return False
    return True


def assert_required_runtime_dependencies() -> None:
    ensure_neo4j_gds_available(raise_on_error=True)
    ensure_lightrag_available(raise_on_error=True)
