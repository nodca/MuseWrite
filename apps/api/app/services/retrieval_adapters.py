import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.config import settings

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional dependency
    GraphDatabase = None


def _truncate_text(text: str, max_chars: int) -> str:
    content = (text or "").strip()
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip() + "..."


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _freshness_days(value: Any) -> int | None:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    else:
        return None
    now = datetime.now(timezone.utc)
    delta = now - dt.astimezone(timezone.utc)
    return max(int(delta.total_seconds() // 86400), 0)


def _parse_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_keywords(text: str, limit: int = 8) -> tuple[list[str], list[str]]:
    raw = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text or "")
    terms: list[str] = []
    for token in raw:
        normalized = token.strip()
        if not normalized:
            continue
        if normalized not in terms:
            terms.append(normalized)
        if len(terms) >= limit:
            break

    # LightRAG docs recommend hl_keywords + ll_keywords to bypass keyword LLM call.
    high = terms[: min(4, len(terms))]
    low = terms[: min(limit, len(terms))]
    return high, low


def _build_lightrag_query_body(
    query: str,
    *,
    mode: str,
    top_k: int,
    chunk_top_k: int,
    anchor: str | None,
) -> dict[str, Any]:
    hl_keywords, ll_keywords = _extract_keywords(query, limit=max(top_k, 8))
    if anchor:
        if anchor not in ll_keywords:
            ll_keywords = [anchor, *ll_keywords][: max(top_k, 8)]
        if anchor not in hl_keywords:
            hl_keywords = [anchor, *hl_keywords][:4]
    return {
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "chunk_top_k": chunk_top_k,
        "stream": False,
        "include_references": True,
        "include_chunk_content": True,
        "hl_keywords": hl_keywords,
        "ll_keywords": ll_keywords,
    }


_RELATION_ALIAS_MAP = {
    "relation": "RELATES_TO",
    "relationship": "RELATES_TO",
    "relationships": "RELATES_TO",
    "ally": "ALLY_OF",
    "allies": "ALLY_OF",
    "enemy": "ENEMY_OF",
    "enemies": "ENEMY_OF",
    "affiliation": "AFFILIATED_WITH",
    "faction": "AFFILIATED_WITH",
    "organization": "AFFILIATED_WITH",
    "status": "HAS_STATUS",
    "goal": "HAS_GOAL",
    "motivation": "HAS_GOAL",
    "secret": "HAS_SECRET",
    "belongs_to": "BELONGS_TO",
    "member_of": "BELONGS_TO",
}


def _normalize_entity_name(value: str) -> tuple[str, str]:
    display = str(value or "").strip()
    normalized = re.sub(r"\s+", "", display).lower()
    normalized = re.sub(r"[^\w\u4e00-\u9fff·-]", "", normalized)
    return display, normalized


def _normalize_relation_type(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "RELATES_TO"
    lowered = raw.lower()
    if lowered in _RELATION_ALIAS_MAP:
        return _RELATION_ALIAS_MAP[lowered]
    uppered = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_").upper()
    return uppered or "RELATES_TO"


def _fact_key(project_id: int, source_norm: str, relation: str, target_norm: str) -> str:
    digest = hashlib.sha1(
        f"{project_id}|{source_norm}|{relation}|{target_norm}".encode("utf-8")
    ).hexdigest()
    return f"fact_{digest[:20]}"


def _normalize_rag_item(item: dict[str, Any], index: int) -> dict[str, Any]:
    snippet = str(
        item.get("snippet")
        or item.get("content")
        or item.get("text")
        or item.get("chunk")
        or ""
    )
    title = str(item.get("title") or item.get("name") or item.get("source") or f"rag_hit_{index}")

    score = _parse_float(item.get("score", item.get("similarity")))
    file_path = str(item.get("file_path") or item.get("source") or title).strip()
    chunk_id = str(item.get("chunk_id") or item.get("id") or index).strip()
    citation = {"source": file_path, "chunk": chunk_id}

    return {
        "kind": str(item.get("kind") or item.get("type") or "rag_chunk"),
        "id": int(item.get("id", index)),
        "title": title,
        "score": round(score, 4) if isinstance(score, float) else None,
        "confidence": round(score, 4) if isinstance(score, float) else None,
        "snippet": _truncate_text(snippet, 180),
        "citation": citation,
        "file_path": file_path,
        "chunk_id": chunk_id,
    }


def make_graph_candidate(
    source: str,
    relation: str,
    target: str,
    *,
    evidence: str = "",
    origin: str = "rule",
    confidence: float | None = None,
    item_id: int | None = None,
) -> dict[str, Any] | None:
    source_display, source_norm = _normalize_entity_name(source)
    target_display, target_norm = _normalize_entity_name(target)
    if not source_norm or not target_norm:
        return None

    relation_norm = _normalize_relation_type(relation)
    confidence_norm = round(confidence, 4) if isinstance(confidence, float) else None
    return {
        "id": int(item_id or 0),
        "source_entity": source_display,
        "source_norm": source_norm,
        "relation": relation_norm,
        "target_entity": target_display,
        "target_norm": target_norm,
        "evidence": _truncate_text(
            evidence or f"{source_display} -[{relation_norm}]-> {target_display}",
            280,
        ),
        "confidence": confidence_norm,
        "origin": origin,
    }


def _normalize_graph_candidate(item: dict[str, Any], index: int, *, origin: str) -> dict[str, Any] | None:
    source_raw = item.get("source") or item.get("subject") or item.get("head") or item.get("from")
    target_raw = item.get("target") or item.get("object") or item.get("tail") or item.get("to")
    relation_raw = item.get("relation") or item.get("predicate") or item.get("rel") or "RELATES_TO"
    if not isinstance(source_raw, str) or not isinstance(target_raw, str):
        return None

    confidence = _parse_float(item.get("confidence", item.get("score")))
    return make_graph_candidate(
        source_raw,
        str(relation_raw),
        target_raw,
        evidence=str(
            item.get("evidence")
            or item.get("snippet")
            or item.get("content")
            or item.get("text")
            or ""
        ),
        origin=origin,
        confidence=confidence,
        item_id=index,
    )


def _raw_graph_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("facts", "triples", "relations", "results", "data", "items", "hits"):
        value = payload.get(key)
        if isinstance(value, list):
            normalized_list: list[dict[str, Any]] = []
            for item in value:
                if isinstance(item, dict):
                    normalized_list.append(item)
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    normalized_list.append(
                        {
                            "source": str(item[0]),
                            "relation": str(item[1]),
                            "target": str(item[2]),
                        }
                    )
            if normalized_list:
                return normalized_list
        if isinstance(value, dict):
            nested = _raw_graph_items(value)
            if nested:
                return nested
    return []


def _lightrag_relationship_items(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data_obj = payload.get("data")
    if not isinstance(data_obj, dict):
        return []
    relations = data_obj.get("relationships")
    if not isinstance(relations, list):
        return []
    return [item for item in relations if isinstance(item, dict)]


def _relation_to_candidate(item: dict[str, Any], index: int) -> dict[str, Any] | None:
    source = str(item.get("src_id") or item.get("source") or "").strip()
    target = str(item.get("tgt_id") or item.get("target") or "").strip()
    if not source or not target:
        return None

    relation_hint = str(item.get("keywords") or item.get("relation") or "RELATES_TO")
    evidence = str(
        item.get("description")
        or item.get("keywords")
        or item.get("source_id")
        or f"{source}->{target}"
    )
    return make_graph_candidate(
        source,
        relation_hint,
        target,
        evidence=evidence,
        origin="lightrag_query",
        confidence=_parse_float(item.get("weight")),
        item_id=index,
    )


def _lightrag_chunk_hits(payload: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data_obj = payload.get("data")
    if not isinstance(data_obj, dict):
        return []
    chunks = data_obj.get("chunks")
    if not isinstance(chunks, list):
        return []

    hits: list[dict[str, Any]] = []
    for idx, item in enumerate(chunks[:limit], start=1):
        if not isinstance(item, dict):
            continue
        snippet = str(item.get("content") or item.get("text") or "")
        if not snippet.strip():
            continue
        title = str(item.get("file_path") or item.get("chunk_id") or f"chunk_{idx}")
        chunk_id = str(item.get("chunk_id") or idx)
        score = _parse_float(item.get("score"))
        hits.append(
            {
                "kind": "rag_chunk",
                "id": idx,
                "title": title,
                "score": score,
                "confidence": round(score, 4) if isinstance(score, float) else None,
                "snippet": _truncate_text(snippet, 180),
                "citation": {
                    "source": title,
                    "chunk": chunk_id,
                },
                "file_path": title,
                "chunk_id": chunk_id,
            }
        )
    return hits[:limit]


def _lightrag_reference_hits(payload: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    references = payload.get("references")
    if not isinstance(references, list):
        return []

    hits: list[dict[str, Any]] = []
    for idx, item in enumerate(references[:limit], start=1):
        if not isinstance(item, dict):
            continue
        file_path = str(item.get("file_path") or item.get("title") or f"ref_{idx}")
        chunks = item.get("content")
        snippet = ""
        if isinstance(chunks, list):
            snippet = "\n".join(str(chunk) for chunk in chunks[:2] if str(chunk).strip())
        elif isinstance(chunks, str):
            snippet = chunks
        if not snippet.strip():
            continue
        hits.append(
            {
                "kind": "rag_ref",
                "id": idx,
                "title": file_path,
                "score": None,
                "confidence": None,
                "snippet": _truncate_text(snippet, 180),
                "citation": {
                    "source": file_path,
                    "chunk": str(item.get("chunk_id") or idx),
                },
                "file_path": file_path,
                "chunk_id": str(item.get("chunk_id") or idx),
            }
        )
    return hits[:limit]


def fetch_lightrag_graph_candidates(
    text: str,
    *,
    anchor: str | None = None,
    limit: int = 16,
) -> list[dict[str, Any]]:
    if not settings.lightrag_enabled:
        return []
    if not settings.lightrag_base_url:
        return []

    headers: dict[str, str] = {}
    if settings.lightrag_api_key:
        headers["Authorization"] = f"Bearer {settings.lightrag_api_key}"
    params = {"api_key_header_value": settings.lightrag_api_key} if settings.lightrag_api_key else None

    timeout = httpx.Timeout(float(settings.lightrag_timeout_seconds))
    with httpx.Client(timeout=timeout) as client:
        # First, try dedicated extract endpoint (legacy/extended deployments).
        extract_path = settings.lightrag_extract_path.strip() or "/extract_graph"
        if extract_path:
            if not extract_path.startswith("/"):
                extract_path = "/" + extract_path
            extract_endpoint = settings.lightrag_base_url.rstrip("/") + extract_path
            try:
                extract_resp = client.post(
                    extract_endpoint,
                    json={"text": text, "top_k": limit, "pov_anchor": anchor},
                    headers=headers,
                    params=params,
                )
                if extract_resp.status_code < 400:
                    extract_payload = extract_resp.json()
                    raw_items = _raw_graph_items(extract_payload)
                    if raw_items:
                        candidates: list[dict[str, Any]] = []
                        for idx, item in enumerate(raw_items[:limit], start=1):
                            normalized = _normalize_graph_candidate(item, idx, origin="lightrag_extract")
                            if normalized:
                                candidates.append(normalized)
                        if candidates:
                            return candidates
            except Exception:
                pass

        if not settings.lightrag_graph_from_query_enabled:
            return []

        # Fallback for official v1.4+ API: use /query/data relationships as graph evidence.
        query_path = settings.lightrag_query_path.strip() or "/query/data"
        if not query_path.startswith("/"):
            query_path = "/" + query_path
        query_endpoint = settings.lightrag_base_url.rstrip("/") + query_path

        body = _build_lightrag_query_body(
            text,
            mode=(settings.lightrag_graph_query_mode or "global"),
            top_k=limit,
            chunk_top_k=max(limit, 8),
            anchor=anchor,
        )
        try:
            query_resp = client.post(query_endpoint, json=body, headers=headers, params=params)
            query_resp.raise_for_status()
            payload = query_resp.json()
        except Exception:
            return []

    relation_items = _lightrag_relationship_items(payload)
    candidates: list[dict[str, Any]] = []
    for idx, item in enumerate(relation_items[:limit], start=1):
        normalized = _relation_to_candidate(item, idx)
        if normalized:
            candidates.append(normalized)
    return candidates


def fetch_lightrag_semantic_hits(
    query: str,
    *,
    anchor: str | None = None,
    limit: int = 8,
    mode_override: str | None = None,
) -> list[dict[str, Any]]:
    if not settings.lightrag_enabled:
        return []
    if not settings.lightrag_base_url:
        return []

    path = settings.lightrag_query_path.strip() or "/query"
    if not path.startswith("/"):
        path = "/" + path
    endpoint = settings.lightrag_base_url.rstrip("/") + path

    body = _build_lightrag_query_body(
        query,
        mode=(mode_override or settings.lightrag_query_mode or "mix"),
        top_k=limit,
        chunk_top_k=max(limit, 8),
        anchor=anchor,
    )

    headers: dict[str, str] = {}
    if settings.lightrag_api_key:
        headers["Authorization"] = f"Bearer {settings.lightrag_api_key}"
    params = {"api_key_header_value": settings.lightrag_api_key} if settings.lightrag_api_key else None

    try:
        timeout = httpx.Timeout(float(settings.lightrag_timeout_seconds))
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(endpoint, json=body, headers=headers, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except Exception:
        return []

    # Official LightRAG /query/data format (v1.4+).
    chunk_hits = _lightrag_chunk_hits(payload, limit)
    if chunk_hits:
        return chunk_hits

    # /query format with references+content (if include_chunk_content=true).
    reference_hits = _lightrag_reference_hits(payload, limit)
    if reference_hits:
        return reference_hits

    # Legacy/generic fallback.
    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        raw_items = (
            payload.get("results")
            or payload.get("hits")
            or payload.get("data")
            or payload.get("items")
            or []
        )
    else:
        raw_items = []

    if not isinstance(raw_items, list):
        return []

    hits: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items[:limit], start=1):
        if not isinstance(item, dict):
            continue
        normalized = _normalize_rag_item(item, idx)
        if not normalized["snippet"]:
            continue
        hits.append(normalized)
    return hits


def merge_graph_candidates(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    *,
    limit: int = 24,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for item in [*primary, *secondary]:
        if not isinstance(item, dict):
            continue
        source_norm = str(item.get("source_norm") or "").strip()
        relation = str(item.get("relation") or "").strip()
        target_norm = str(item.get("target_norm") or "").strip()
        if not source_norm or not relation or not target_norm:
            continue
        key = (source_norm, relation, target_norm)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= limit:
            break

    return merged


def upsert_neo4j_graph_facts(
    project_id: int,
    facts: list[dict[str, Any]],
    *,
    state: str = "confirmed",
    source_ref: str = "",
    current_chapter: int | None = None,
) -> list[str]:
    if not settings.neo4j_enabled:
        return []
    if not settings.neo4j_uri:
        return []
    if GraphDatabase is None:
        return []
    if not facts:
        return []

    auth: tuple[str, str] | None = None
    if settings.neo4j_username:
        auth = (settings.neo4j_username, settings.neo4j_password)

    rows: list[dict[str, Any]] = []
    chapter_value = int(current_chapter or 0) if isinstance(current_chapter, int) or str(current_chapter or "").isdigit() else 0
    use_temporal = bool(settings.graph_temporal_enabled and chapter_value > 0)
    for fact in facts:
        source_display = str(fact.get("source_entity") or "").strip()
        target_display = str(fact.get("target_entity") or "").strip()
        source_norm = str(fact.get("source_norm") or "").strip()
        target_norm = str(fact.get("target_norm") or "").strip()
        relation = str(fact.get("relation") or "RELATES_TO").strip().upper()
        if not source_display or not target_display or not source_norm or not target_norm:
            continue
        rows.append(
            {
                "source_name": source_display,
                "source_norm": source_norm,
                "target_name": target_display,
                "target_norm": target_norm,
                "relation": relation,
                "fact_key": _fact_key(project_id, source_norm, relation, target_norm),
                "interval_key": (
                    f"{_fact_key(project_id, source_norm, relation, target_norm)}:{chapter_value}"
                    if use_temporal
                    else _fact_key(project_id, source_norm, relation, target_norm)
                ),
                "evidence": _truncate_text(str(fact.get("evidence") or ""), 280),
                "confidence": _parse_float(fact.get("confidence")),
                "origin": str(fact.get("origin") or "unknown"),
                "valid_from_chapter": chapter_value if use_temporal else None,
                "valid_to_chapter": None,
            }
        )

    if not rows:
        return []

    close_temporal_cypher = """
    UNWIND $rows AS row
    MATCH (s:Entity {project_id: $project_id, name_norm: row.source_norm})-[old:FACT {project_id: $project_id}]->(x:Entity {project_id: $project_id})
    WHERE
      old.rel_type = row.relation
      AND coalesce(old.valid_to_chapter, 2147483647) >= row.valid_from_chapter
      AND x.name_norm <> row.target_norm
    SET old.valid_to_chapter = row.valid_from_chapter - 1, old.updated_at = $now
    RETURN count(old) AS closed_count
    """

    cypher = """
    UNWIND $rows AS row
    MERGE (s:Entity {project_id: $project_id, name_norm: row.source_norm})
      ON CREATE SET s.name = row.source_name, s.created_at = $now
      ON MATCH SET s.name = row.source_name, s.updated_at = $now
    MERGE (t:Entity {project_id: $project_id, name_norm: row.target_norm})
      ON CREATE SET t.name = row.target_name, t.created_at = $now
      ON MATCH SET t.name = row.target_name, t.updated_at = $now
    MERGE (s)-[r:FACT {project_id: $project_id, interval_key: row.interval_key}]->(t)
      ON CREATE SET r.created_at = $now
    SET
      r.rel_type = row.relation,
      r.source_norm = row.source_norm,
      r.target_norm = row.target_norm,
      r.state = $state,
      r.source_ref = $source_ref,
      r.origin = row.origin,
      r.evidence = row.evidence,
      r.confidence = row.confidence,
      r.fact_key = row.fact_key,
      r.valid_from_chapter = row.valid_from_chapter,
      r.valid_to_chapter = row.valid_to_chapter,
      r.updated_at = $now
    RETURN row.fact_key AS fact_key
    """

    fact_keys: list[str] = []
    driver = None
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=auth)
        with driver.session(database=settings.neo4j_database or None) as session:
            if use_temporal:
                session.run(
                    close_temporal_cypher,
                    project_id=project_id,
                    rows=rows,
                    now=_utc_iso(),
                ).consume()
            result = session.run(
                cypher,
                project_id=project_id,
                rows=rows,
                state=state,
                source_ref=source_ref,
                now=_utc_iso(),
            )
            fact_keys = [str(row["fact_key"]) for row in result]
    except Exception:
        return []
    finally:
        if driver is not None:
            driver.close()

    return fact_keys


def list_neo4j_graph_candidates(
    project_id: int,
    *,
    keyword: str = "",
    source_ref: str = "",
    min_confidence: float | None = None,
    page: int = 1,
    page_size: int = 50,
    current_chapter: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if not settings.neo4j_enabled:
        return [], 0
    if not settings.neo4j_uri:
        return [], 0
    if GraphDatabase is None:
        return [], 0

    normalized_keyword = str(keyword or "").strip().lower()
    normalized_source_ref = str(source_ref or "").strip()
    min_confidence_value = float(min_confidence) if isinstance(min_confidence, (int, float)) else -1.0
    normalized_page = max(int(page), 1)
    normalized_page_size = max(min(int(page_size), 200), 1)
    skip = (normalized_page - 1) * normalized_page_size
    chapter_value = (
        int(current_chapter or 0)
        if isinstance(current_chapter, int) or str(current_chapter or "").isdigit()
        else 0
    )
    use_chapter_filter = bool(settings.graph_temporal_enabled and chapter_value > 0)

    auth: tuple[str, str] | None = None
    if settings.neo4j_username:
        auth = (settings.neo4j_username, settings.neo4j_password)

    where_clause = """
    coalesce(r.state, 'confirmed') = 'candidate'
      AND (
        $source_ref = '' OR coalesce(r.source_ref, '') = $source_ref
      )
      AND (
        $min_confidence < 0 OR coalesce(toFloat(r.confidence), 0.0) >= $min_confidence
      )
      AND (
        $keyword = '' OR
        toLower(coalesce(r.fact_key, '')) CONTAINS $keyword OR
        toLower(coalesce(r.rel_type, '')) CONTAINS $keyword OR
        toLower(coalesce(r.source_ref, '')) CONTAINS $keyword OR
        toLower(coalesce(r.evidence, '')) CONTAINS $keyword OR
        toLower(coalesce(s.name, '')) CONTAINS $keyword OR
        toLower(coalesce(t.name, '')) CONTAINS $keyword
      )
      AND (
        $use_chapter_filter = false OR
        (
          coalesce(toInteger(r.valid_from_chapter), -2147483648) <= $current_chapter
          AND coalesce(toInteger(r.valid_to_chapter), 2147483647) >= $current_chapter
        )
      )
    """

    count_cypher = f"""
    MATCH (s:Entity)-[r:FACT {{project_id: $project_id}}]->(t:Entity)
    WHERE {where_clause}
    RETURN count(r) AS total
    """

    list_cypher = f"""
    MATCH (s:Entity)-[r:FACT {{project_id: $project_id}}]->(t:Entity)
    WHERE {where_clause}
    WITH s, t, r
    ORDER BY coalesce(toFloat(r.confidence), 0.0) DESC, coalesce(r.updated_at, '') DESC
    SKIP $skip
    LIMIT $limit
    RETURN
      coalesce(r.fact_key, '') AS fact_key,
      coalesce(s.name, '') AS source_entity,
      coalesce(r.rel_type, '') AS relation,
      coalesce(t.name, '') AS target_entity,
      toFloat(coalesce(r.confidence, 0.0)) AS confidence,
      coalesce(r.source_ref, '') AS source_ref,
      coalesce(r.origin, '') AS origin,
      coalesce(r.evidence, '') AS evidence,
      coalesce(r.state, '') AS state,
      r.valid_from_chapter AS valid_from_chapter,
      r.valid_to_chapter AS valid_to_chapter,
      coalesce(r.updated_at, '') AS updated_at
    """

    total = 0
    items: list[dict[str, Any]] = []
    driver = None
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=auth)
        with driver.session(database=settings.neo4j_database or None) as session:
            count_row = session.run(
                count_cypher,
                project_id=project_id,
                source_ref=normalized_source_ref,
                min_confidence=min_confidence_value,
                keyword=normalized_keyword,
                use_chapter_filter=use_chapter_filter,
                current_chapter=chapter_value,
            ).single()
            if count_row is not None:
                total = int(count_row.get("total", 0))

            rows = session.run(
                list_cypher,
                project_id=project_id,
                source_ref=normalized_source_ref,
                min_confidence=min_confidence_value,
                keyword=normalized_keyword,
                use_chapter_filter=use_chapter_filter,
                current_chapter=chapter_value,
                skip=skip,
                limit=normalized_page_size,
            )
            for row in rows:
                confidence = _parse_float(row.get("confidence"))
                valid_from_raw = row.get("valid_from_chapter")
                valid_to_raw = row.get("valid_to_chapter")
                valid_from: int | None = None
                valid_to: int | None = None
                if isinstance(valid_from_raw, (int, float, str)) and str(valid_from_raw).strip():
                    try:
                        valid_from = int(valid_from_raw)
                    except Exception:
                        valid_from = None
                if isinstance(valid_to_raw, (int, float, str)) and str(valid_to_raw).strip():
                    try:
                        valid_to = int(valid_to_raw)
                    except Exception:
                        valid_to = None
                updated_at = str(row.get("updated_at") or "").strip() or None
                items.append(
                    {
                        "fact_key": str(row.get("fact_key") or "").strip(),
                        "source_entity": str(row.get("source_entity") or "").strip(),
                        "relation": str(row.get("relation") or "").strip(),
                        "target_entity": str(row.get("target_entity") or "").strip(),
                        "confidence": confidence,
                        "source_ref": str(row.get("source_ref") or "").strip(),
                        "origin": str(row.get("origin") or "").strip(),
                        "evidence": str(row.get("evidence") or "").strip(),
                        "state": str(row.get("state") or "").strip() or "candidate",
                        "valid_from_chapter": valid_from,
                        "valid_to_chapter": valid_to,
                        "updated_at": updated_at,
                    }
                )
    except Exception:
        return [], 0
    finally:
        if driver is not None:
            driver.close()

    return items, max(total, 0)


def promote_neo4j_candidate_facts(
    project_id: int,
    *,
    fact_keys: list[str] | None = None,
    source_ref: str = "",
    min_confidence: float | None = None,
    limit: int = 200,
    current_chapter: int | None = None,
) -> list[str]:
    if not settings.neo4j_enabled:
        return []
    if not settings.neo4j_uri:
        return []
    if GraphDatabase is None:
        return []

    normalized_fact_keys = [str(item).strip() for item in (fact_keys or []) if str(item).strip()]
    normalized_source_ref = str(source_ref or "").strip()
    if not normalized_fact_keys and not normalized_source_ref:
        return []

    auth: tuple[str, str] | None = None
    if settings.neo4j_username:
        auth = (settings.neo4j_username, settings.neo4j_password)

    min_confidence_value = float(min_confidence) if isinstance(min_confidence, (int, float)) else -1.0
    chapter_value = int(current_chapter or 0) if isinstance(current_chapter, int) or str(current_chapter or "").isdigit() else 0
    max_limit = max(min(int(limit), 1000), 1)
    use_chapter_filter = bool(settings.graph_temporal_enabled and chapter_value > 0)

    cypher = """
    MATCH ()-[r:FACT {project_id: $project_id}]->()
    WHERE
      coalesce(r.state, 'confirmed') = 'candidate'
      AND (
        size($fact_keys) = 0 OR r.fact_key IN $fact_keys
      )
      AND (
        $source_ref = '' OR coalesce(r.source_ref, '') = $source_ref
      )
      AND (
        $min_confidence < 0 OR coalesce(toFloat(r.confidence), 0.0) >= $min_confidence
      )
      AND (
        $use_chapter_filter = false OR
        (
          coalesce(toInteger(r.valid_from_chapter), -2147483648) <= $current_chapter
          AND coalesce(toInteger(r.valid_to_chapter), 2147483647) >= $current_chapter
        )
      )
    WITH r
    ORDER BY coalesce(toFloat(r.confidence), 0.0) DESC, coalesce(r.updated_at, '') DESC
    LIMIT $limit
    SET r.state = 'confirmed', r.updated_at = $now
    RETURN coalesce(r.fact_key, '') AS fact_key
    """

    promoted_fact_keys: list[str] = []
    driver = None
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=auth)
        with driver.session(database=settings.neo4j_database or None) as session:
            rows = session.run(
                cypher,
                project_id=project_id,
                fact_keys=normalized_fact_keys,
                source_ref=normalized_source_ref,
                min_confidence=min_confidence_value,
                use_chapter_filter=use_chapter_filter,
                current_chapter=chapter_value,
                limit=max_limit,
                now=_utc_iso(),
            )
            promoted_fact_keys = [str(row.get("fact_key") or "").strip() for row in rows]
    except Exception:
        return []
    finally:
        if driver is not None:
            driver.close()

    return [item for item in promoted_fact_keys if item]


def update_neo4j_graph_fact_state(
    project_id: int,
    fact_keys: list[str],
    *,
    to_state: str,
    from_state: str | None = None,
    current_chapter: int | None = None,
) -> int:
    if not settings.neo4j_enabled:
        return 0
    if not settings.neo4j_uri:
        return 0
    if GraphDatabase is None:
        return 0
    normalized_keys = [str(item).strip() for item in fact_keys if str(item).strip()]
    if not normalized_keys:
        return 0

    target_state = str(to_state or "").strip().lower()
    if not target_state:
        return 0
    expected_state = str(from_state or "").strip().lower()

    auth: tuple[str, str] | None = None
    if settings.neo4j_username:
        auth = (settings.neo4j_username, settings.neo4j_password)

    chapter_value = int(current_chapter or 0) if isinstance(current_chapter, int) or str(current_chapter or "").isdigit() else 0
    use_chapter_filter = bool(settings.graph_temporal_enabled and chapter_value > 0)

    cypher = """
    MATCH ()-[r:FACT {project_id: $project_id}]->()
    WHERE
      r.fact_key IN $fact_keys
      AND (
        $from_state = '' OR coalesce(r.state, '') = $from_state
      )
      AND (
        $use_chapter_filter = false OR
        (
          coalesce(toInteger(r.valid_from_chapter), -2147483648) <= $current_chapter
          AND coalesce(toInteger(r.valid_to_chapter), 2147483647) >= $current_chapter
        )
      )
    SET r.state = $to_state, r.updated_at = $now
    RETURN count(r) AS updated_count
    """

    updated_count = 0
    driver = None
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=auth)
        with driver.session(database=settings.neo4j_database or None) as session:
            row = session.run(
                cypher,
                project_id=project_id,
                fact_keys=normalized_keys,
                from_state=expected_state,
                to_state=target_state,
                use_chapter_filter=use_chapter_filter,
                current_chapter=chapter_value,
                now=_utc_iso(),
            ).single()
            if row is not None:
                updated_count = int(row.get("updated_count", 0))
    except Exception:
        return 0
    finally:
        if driver is not None:
            driver.close()

    return updated_count


def delete_neo4j_graph_facts(
    project_id: int,
    fact_keys: list[str],
    *,
    current_chapter: int | None = None,
    hard_delete: bool = False,
) -> int:
    if not settings.neo4j_enabled:
        return 0
    if not settings.neo4j_uri:
        return 0
    if GraphDatabase is None:
        return 0
    normalized_keys = [str(item).strip() for item in fact_keys if str(item).strip()]
    if not normalized_keys:
        return 0

    auth: tuple[str, str] | None = None
    if settings.neo4j_username:
        auth = (settings.neo4j_username, settings.neo4j_password)

    chapter_value = int(current_chapter or 0) if isinstance(current_chapter, int) or str(current_chapter or "").isdigit() else 0
    temporal_close = bool(settings.graph_temporal_enabled and chapter_value > 0 and not hard_delete)
    if temporal_close:
        cypher = """
        MATCH ()-[r:FACT {project_id: $project_id}]->()
        WHERE
          r.fact_key IN $fact_keys
          AND coalesce(r.valid_to_chapter, 2147483647) >= $current_chapter
        SET r.valid_to_chapter = $current_chapter - 1, r.updated_at = $now
        RETURN count(r) AS deleted_count
        """
    else:
        cypher = """
        MATCH ()-[r:FACT {project_id: $project_id}]->()
        WHERE r.fact_key IN $fact_keys
        WITH collect(r) AS rels
        FOREACH (rel IN rels | DELETE rel)
        RETURN size(rels) AS deleted_count
        """

    deleted_count = 0
    driver = None
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=auth)
        with driver.session(database=settings.neo4j_database or None) as session:
            row = session.run(
                cypher,
                project_id=project_id,
                fact_keys=normalized_keys,
                current_chapter=chapter_value,
                now=_utc_iso(),
            ).single()
            if row is not None:
                deleted_count = int(row.get("deleted_count", 0))
    except Exception:
        return 0
    finally:
        if driver is not None:
            driver.close()

    return deleted_count


def delete_neo4j_graph_facts_by_sources(
    project_id: int,
    sources: list[str],
    *,
    current_chapter: int | None = None,
    hard_delete: bool = False,
) -> int:
    if not settings.neo4j_enabled:
        return 0
    if not settings.neo4j_uri:
        return 0
    if GraphDatabase is None:
        return 0

    normalized_sources: list[str] = []
    for source in sources:
        _, source_norm = _normalize_entity_name(str(source or ""))
        if source_norm and source_norm not in normalized_sources:
            normalized_sources.append(source_norm)
    if not normalized_sources:
        return 0

    auth: tuple[str, str] | None = None
    if settings.neo4j_username:
        auth = (settings.neo4j_username, settings.neo4j_password)

    chapter_value = int(current_chapter or 0) if isinstance(current_chapter, int) or str(current_chapter or "").isdigit() else 0
    temporal_close = bool(settings.graph_temporal_enabled and chapter_value > 0 and not hard_delete)
    if temporal_close:
        cypher = """
        MATCH (s:Entity {project_id: $project_id})-[r:FACT {project_id: $project_id}]->()
        WHERE
          s.name_norm IN $source_norms
          AND coalesce(r.valid_to_chapter, 2147483647) >= $current_chapter
        SET r.valid_to_chapter = $current_chapter - 1, r.updated_at = $now
        RETURN count(r) AS deleted_count
        """
    else:
        cypher = """
        MATCH (s:Entity {project_id: $project_id})-[r:FACT {project_id: $project_id}]->()
        WHERE s.name_norm IN $source_norms
        WITH collect(r) AS rels
        FOREACH (rel IN rels | DELETE rel)
        RETURN size(rels) AS deleted_count
        """

    deleted_count = 0
    driver = None
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=auth)
        with driver.session(database=settings.neo4j_database or None) as session:
            row = session.run(
                cypher,
                project_id=project_id,
                source_norms=normalized_sources,
                current_chapter=chapter_value,
                now=_utc_iso(),
            ).single()
            if row is not None:
                deleted_count = int(row.get("deleted_count", 0))
    except Exception:
        return 0
    finally:
        if driver is not None:
            driver.close()

    return deleted_count


def delete_all_neo4j_graph_facts(project_id: int) -> int:
    if not settings.neo4j_enabled:
        return 0
    if not settings.neo4j_uri:
        return 0
    if GraphDatabase is None:
        return 0

    auth: tuple[str, str] | None = None
    if settings.neo4j_username:
        auth = (settings.neo4j_username, settings.neo4j_password)

    cypher = """
    MATCH ()-[r:FACT {project_id: $project_id}]->()
    WITH collect(r) AS rels
    FOREACH (rel IN rels | DELETE rel)
    RETURN size(rels) AS deleted_count
    """

    deleted_count = 0
    driver = None
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=auth)
        with driver.session(database=settings.neo4j_database or None) as session:
            row = session.run(
                cypher,
                project_id=project_id,
            ).single()
            if row is not None:
                deleted_count = int(row.get("deleted_count", 0))
    except Exception:
        return 0
    finally:
        if driver is not None:
            driver.close()

    return deleted_count


def trigger_lightrag_rebuild(project_id: int, *, reason: str = "") -> bool:
    if not settings.lightrag_enabled:
        return False
    if not settings.lightrag_rebuild_enabled:
        return False
    if not settings.lightrag_base_url:
        return False

    path = settings.lightrag_rebuild_path.strip() or "/documents/rebuild"
    if not path.startswith("/"):
        path = "/" + path
    endpoint = settings.lightrag_base_url.rstrip("/") + path

    headers: dict[str, str] = {}
    if settings.lightrag_api_key:
        headers["Authorization"] = f"Bearer {settings.lightrag_api_key}"
    params = {"api_key_header_value": settings.lightrag_api_key} if settings.lightrag_api_key else None

    body = {
        "project_id": int(project_id),
        "reason": reason or "index_lifecycle_rebuild",
    }
    try:
        timeout = httpx.Timeout(float(settings.lightrag_timeout_seconds))
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(endpoint, json=body, headers=headers, params=params)
            return int(resp.status_code) < 400
    except Exception:
        return False


def fetch_neo4j_entity_profiles(project_id: int, *, limit: int = 400) -> list[dict[str, Any]]:
    if not settings.neo4j_enabled:
        return []
    if not settings.neo4j_uri:
        return []
    if GraphDatabase is None:
        return []
    max_rows = max(int(limit), 1)

    auth: tuple[str, str] | None = None
    if settings.neo4j_username:
        auth = (settings.neo4j_username, settings.neo4j_password)

    cypher = """
    MATCH (e:Entity {project_id: $project_id})
    OPTIONAL MATCH (e)-[out:FACT {project_id: $project_id, state: 'confirmed'}]->(out_n:Entity {project_id: $project_id})
    WITH e,
      collect(distinct coalesce(out_n.name, '')) AS out_neighbor_names,
      collect(distinct coalesce(out_n.name_norm, '')) AS out_neighbor_norms,
      collect(distinct coalesce(out.rel_type, '')) AS out_rel_types
    OPTIONAL MATCH (in_n:Entity {project_id: $project_id})-[inn:FACT {project_id: $project_id, state: 'confirmed'}]->(e)
    WITH e,
      out_neighbor_names,
      out_neighbor_norms,
      out_rel_types,
      collect(distinct coalesce(in_n.name, '')) AS in_neighbor_names,
      collect(distinct coalesce(in_n.name_norm, '')) AS in_neighbor_norms,
      collect(distinct coalesce(inn.rel_type, '')) AS in_rel_types
    RETURN
      coalesce(e.name, '') AS name,
      coalesce(e.name_norm, '') AS name_norm,
      out_neighbor_names + in_neighbor_names AS neighbor_names,
      out_neighbor_norms + in_neighbor_norms AS neighbor_norms,
      out_rel_types + in_rel_types AS relation_types
    LIMIT $limit
    """

    rows: list[dict[str, Any]] = []
    driver = None
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=auth)
        with driver.session(database=settings.neo4j_database or None) as session:
            result = session.run(
                cypher,
                project_id=project_id,
                limit=max_rows,
            )
            rows = result.data()
    except Exception:
        return []
    finally:
        if driver is not None:
            driver.close()

    profiles: list[dict[str, Any]] = []
    for row in rows[:max_rows]:
        name = str(row.get("name") or "").strip()
        name_norm = str(row.get("name_norm") or "").strip()
        if not name or not name_norm:
            continue

        neighbor_names_raw = row.get("neighbor_names")
        neighbor_norms_raw = row.get("neighbor_norms")
        relation_types_raw = row.get("relation_types")
        neighbor_names = (
            [str(item).strip() for item in neighbor_names_raw if str(item).strip()]
            if isinstance(neighbor_names_raw, list)
            else []
        )
        neighbor_norms = (
            [str(item).strip() for item in neighbor_norms_raw if str(item).strip()]
            if isinstance(neighbor_norms_raw, list)
            else []
        )
        relation_types = (
            [str(item).strip().upper() for item in relation_types_raw if str(item).strip()]
            if isinstance(relation_types_raw, list)
            else []
        )
        profiles.append(
            {
                "name": name,
                "name_norm": name_norm,
                "neighbor_names": list(dict.fromkeys(neighbor_names)),
                "neighbor_norms": list(dict.fromkeys(neighbor_norms)),
                "relation_types": list(dict.fromkeys(relation_types)),
            }
        )

    return profiles


def fetch_neo4j_graph_facts(
    project_id: int,
    terms: list[str],
    *,
    anchor: str | None = None,
    limit: int = 10,
    current_chapter: int | None = None,
) -> list[dict[str, Any]]:
    if not settings.neo4j_enabled:
        return []
    if not settings.neo4j_uri:
        return []
    if GraphDatabase is None:
        return []

    normalized_terms = [term.lower() for term in terms if term]
    normalized_anchor = (anchor or "").strip().lower()
    if not normalized_terms and not normalized_anchor:
        return []
    chapter_value = int(current_chapter or 0) if isinstance(current_chapter, int) or str(current_chapter or "").isdigit() else 0
    use_chapter_filter = bool(settings.graph_temporal_enabled)
    filter_without_chapter = bool(settings.graph_temporal_filter_without_chapter)

    auth: tuple[str, str] | None = None
    if settings.neo4j_username:
        auth = (settings.neo4j_username, settings.neo4j_password)

    cypher = """
    MATCH (a:Entity {project_id: $project_id})-[r:FACT {project_id: $project_id}]->(b:Entity {project_id: $project_id})
    WHERE
      coalesce(r.state, 'confirmed') = 'confirmed'
      AND (
        $use_chapter_filter = false OR
        (
          $current_chapter > 0
          AND coalesce(toInteger(r.valid_from_chapter), -2147483648) <= $current_chapter
          AND coalesce(toInteger(r.valid_to_chapter), 2147483647) >= $current_chapter
        )
        OR (
          $current_chapter <= 0
          AND (
            $filter_without_chapter = false
            OR r.valid_to_chapter IS NULL
          )
        )
      )
      AND (
        size($terms) = 0 OR any(term IN $terms WHERE
          toLower(coalesce(a.name, '')) CONTAINS term OR
          toLower(coalesce(b.name, '')) CONTAINS term OR
          toLower(coalesce(r.rel_type, '')) CONTAINS term
        )
      )
      AND (
        $anchor = '' OR
        toLower(coalesce(a.name, '')) CONTAINS $anchor OR
        toLower(coalesce(b.name, '')) CONTAINS $anchor
      )
    RETURN
      coalesce(a.name, toString(id(a))) AS source,
      coalesce(r.rel_type, 'RELATED_TO') AS relation,
      coalesce(b.name, toString(id(b))) AS target,
      properties(r) AS rel_props
    LIMIT $limit
    """

    records: list[dict[str, Any]] = []
    driver = None
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=auth)
        with driver.session(database=settings.neo4j_database or None) as session:
            rows = session.run(
                cypher,
                project_id=project_id,
                terms=normalized_terms,
                anchor=normalized_anchor,
                limit=limit,
                use_chapter_filter=use_chapter_filter,
                current_chapter=chapter_value,
                filter_without_chapter=filter_without_chapter,
            )
            records = rows.data()
    except Exception:
        return []
    finally:
        if driver is not None:
            driver.close()

    facts: list[dict[str, Any]] = []
    for idx, row in enumerate(records[:limit], start=1):
        source = str(row.get("source") or "unknown")
        relation = str(row.get("relation") or "RELATED_TO")
        target = str(row.get("target") or "unknown")
        rel_props = row.get("rel_props") if isinstance(row.get("rel_props"), dict) else {}
        confidence = _parse_float(rel_props.get("confidence"))
        updated_at = str(rel_props.get("updated_at") or "")
        edge = f"{source} -[{relation}]-> {target}"
        if rel_props:
            edge = f"{edge} {json.dumps(rel_props, ensure_ascii=False)}"
        facts.append(
            {
                "kind": "graph_edge",
                "id": idx,
                "title": edge[:64],
                "fact": _truncate_text(edge, 180),
                "confidence": round(confidence, 4) if isinstance(confidence, float) else None,
                "updated_at": updated_at or None,
                "freshness_days": _freshness_days(updated_at) if updated_at else None,
            }
        )
    return facts


def fetch_neo4j_graph_timeline_snapshot(
    project_id: int,
    *,
    current_chapter: int | None = None,
    limit: int = 240,
) -> dict[str, Any]:
    chapter_value = (
        int(current_chapter or 0)
        if isinstance(current_chapter, int) or str(current_chapter or "").isdigit()
        else 0
    )
    size = max(int(limit), 20)
    empty_payload = {
        "project_id": int(project_id),
        "chapter_index": chapter_value,
        "nodes": [],
        "edges": [],
        "stats": {
            "nodes": 0,
            "edges": 0,
            "limit": size,
            "truncated": False,
            "source": "neo4j",
        },
    }
    if not settings.neo4j_enabled:
        return empty_payload
    if not settings.neo4j_uri:
        return empty_payload
    if GraphDatabase is None:
        return empty_payload

    auth: tuple[str, str] | None = None
    if settings.neo4j_username:
        auth = (settings.neo4j_username, settings.neo4j_password)
    use_chapter_filter = bool(settings.graph_temporal_enabled)
    filter_without_chapter = bool(settings.graph_temporal_filter_without_chapter)

    cypher = """
    MATCH (a:Entity {project_id: $project_id})-[r:FACT {project_id: $project_id}]->(b:Entity {project_id: $project_id})
    WHERE
      coalesce(r.state, 'confirmed') = 'confirmed'
      AND (
        $use_chapter_filter = false OR
        (
          $current_chapter > 0
          AND coalesce(toInteger(r.valid_from_chapter), -2147483648) <= $current_chapter
          AND coalesce(toInteger(r.valid_to_chapter), 2147483647) >= $current_chapter
        )
        OR (
          $current_chapter <= 0
          AND (
            $filter_without_chapter = false
            OR r.valid_to_chapter IS NULL
          )
        )
      )
    RETURN
      coalesce(a.name, toString(id(a))) AS source,
      coalesce(r.rel_type, 'RELATED_TO') AS relation,
      coalesce(b.name, toString(id(b))) AS target,
      properties(r) AS rel_props
    LIMIT $limit
    """

    driver = None
    records: list[dict[str, Any]] = []
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=auth)
        with driver.session(database=settings.neo4j_database or None) as session:
            rows = session.run(
                cypher,
                project_id=project_id,
                limit=size,
                use_chapter_filter=use_chapter_filter,
                current_chapter=chapter_value,
                filter_without_chapter=filter_without_chapter,
            )
            records = rows.data()
    except Exception:
        return empty_payload
    finally:
        if driver is not None:
            driver.close()

    node_map: dict[str, dict[str, Any]] = {}
    edge_map: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(records, start=1):
        source = str(row.get("source") or "").strip()
        target = str(row.get("target") or "").strip()
        relation = str(row.get("relation") or "RELATED_TO").strip().upper()
        if not source or not target:
            continue
        rel_props = row.get("rel_props") if isinstance(row.get("rel_props"), dict) else {}
        confidence = _parse_float(rel_props.get("confidence"))
        valid_from = rel_props.get("valid_from_chapter")
        valid_to = rel_props.get("valid_to_chapter")
        key = f"{source}|{relation}|{target}"
        if key not in edge_map:
            edge_map[key] = {
                "id": f"e{len(edge_map) + 1}",
                "source": source,
                "target": target,
                "relation": relation,
                "confidence": round(confidence, 4) if isinstance(confidence, float) else None,
                "valid_from_chapter": int(valid_from) if isinstance(valid_from, int) else None,
                "valid_to_chapter": int(valid_to) if isinstance(valid_to, int) else None,
                "freshness_days": _freshness_days(rel_props.get("updated_at")),
                "_order": idx,
            }

        source_node = node_map.get(source)
        if source_node is None:
            source_node = {"id": source, "label": source, "kind": "entity", "degree": 0}
            node_map[source] = source_node
        source_node["degree"] = int(source_node.get("degree", 0)) + 1

        target_node = node_map.get(target)
        if target_node is None:
            target_node = {"id": target, "label": target, "kind": "entity", "degree": 0}
            node_map[target] = target_node
        target_node["degree"] = int(target_node.get("degree", 0)) + 1

    nodes = sorted(
        node_map.values(),
        key=lambda item: (-int(item.get("degree", 0)), str(item.get("label") or "")),
    )
    edges = sorted(
        edge_map.values(),
        key=lambda item: int(item.get("_order", 0)),
    )
    for item in edges:
        item.pop("_order", None)

    return {
        "project_id": int(project_id),
        "chapter_index": chapter_value,
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "limit": size,
            "truncated": len(records) >= size,
            "source": "neo4j",
        },
    }
