from typing import Any

from app.core.config import settings
from app.services.context_compiler._utils import _truncate_text


def _extract_citation(hit: dict[str, Any]) -> dict[str, str] | None:
    if not isinstance(hit, dict):
        return None

    raw = hit.get("citation")
    if isinstance(raw, dict):
        source = str(raw.get("source") or "").strip()
        chunk = str(raw.get("chunk") or "").strip()
        if source or chunk:
            result: dict[str, str] = {}
            if source:
                result["source"] = source
            if chunk:
                result["chunk"] = chunk
            return result

    source = str(hit.get("file_path") or hit.get("source") or hit.get("title") or "").strip()
    chunk = str(hit.get("chunk_id") or "").strip()
    if not source and not chunk:
        return None
    result = {}
    if source:
        result["source"] = source
    if chunk:
        result["chunk"] = chunk
    return result


def _citation_required() -> bool:
    policy = str(settings.citation_policy or "off").strip().lower()
    if policy in {"off", "disabled", "none"}:
        return False
    if policy in {"always", "strict", "factual", "on", "true", "1"}:
        return True
    return False


def _build_quality_gate(user_input: str, rag_provider: str, semantic_hits: list[dict[str, Any]]) -> dict[str, Any]:
    _ = user_input
    citations = [item for hit in semantic_hits if (item := _extract_citation(hit))]
    min_required = max(int(settings.citation_min_count), 0)
    citation_required = _citation_required()
    citation_ok = (not citation_required) or len(citations) >= max(min_required, 1)

    reranker_expected = bool(settings.lightrag_rerank_enabled)
    reranker_effective = reranker_expected and rag_provider.startswith("lightrag")
    reranker_ok = (not settings.reranker_required) or reranker_effective

    reasons: list[str] = []
    if not citation_ok:
        reasons.append("missing_citation")
    if not reranker_ok:
        reasons.append("reranker_not_effective")

    unique_sources = sorted({item.get("source", "") for item in citations if item.get("source")})
    return {
        "degraded": bool(reasons),
        "degrade_reasons": reasons,
        "citation_required": citation_required,
        "citation_min_required": max(min_required, 1) if citation_required else 0,
        "citation_count": len(citations),
        "citation_sources": unique_sources[:8],
        "citation_coverage": round(len(citations) / max(len(semantic_hits), 1), 4),
        "reranker_expected": reranker_expected,
        "reranker_effective": reranker_effective,
    }


def _resolve_rag_short_circuit(
    *,
    deterministic_first: bool,
    dsl_hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
) -> tuple[bool, str]:
    if not deterministic_first:
        return False, "disabled_by_request"
    if not settings.deterministic_short_circuit_enabled:
        return False, "disabled_by_config"

    min_dsl = max(int(settings.deterministic_min_dsl_hits), 0)
    min_graph = max(int(settings.deterministic_min_graph_hits), 0)
    if min_dsl > 0 and len(dsl_hits) >= min_dsl:
        return True, "dsl_hit_threshold"
    if min_graph > 0 and len(graph_facts) >= min_graph:
        return True, "graph_hit_threshold"
    return False, "insufficient_authoritative_hits"

