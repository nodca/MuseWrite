import json
from typing import Any

from sqlmodel import Session, select

from app.models.content import SettingEntry, StoryCard
from app.services.retrieval_adapters import (
    delete_all_neo4j_graph_facts,
    make_graph_candidate,
    trigger_lightrag_rebuild,
    upsert_neo4j_graph_facts,
)

_RELATION_FIELD_MAP = {
    "relationship": "RELATES_TO",
    "relationships": "RELATES_TO",
    "relation": "RELATES_TO",
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
}


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return ""


def _split_targets(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items = (
            value.replace("；", ",")
            .replace("、", ",")
            .replace("，", ",")
            .split(",")
        )
        return [item.strip() for item in raw_items if item and item.strip()]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            items.extend(_split_targets(item))
        return items
    if isinstance(value, dict):
        return [str(key).strip() for key in value.keys() if str(key).strip()]
    return []


def _extract_rule_candidates(source_entity: str, content_obj: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if not source_entity.strip():
        return candidates
    next_id = 1
    for raw_key, raw_value in content_obj.items():
        key = str(raw_key).strip().lower()
        if key not in _RELATION_FIELD_MAP:
            continue
        relation = _RELATION_FIELD_MAP[key]
        targets = _split_targets(raw_value)
        if not targets:
            continue
        for target in targets:
            candidate = make_graph_candidate(
                source_entity,
                relation,
                target,
                evidence=f"{source_entity} {raw_key}: {_to_text(raw_value)}",
                origin="lifecycle_rebuild",
                item_id=next_id,
            )
            if candidate:
                candidates.append(candidate)
                next_id += 1
    return candidates


def _dedupe_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
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
        deduped.append(item)
    return deduped


def process_index_lifecycle_rebuild(
    db: Session,
    *,
    project_id: int,
    reason: str,
    lifecycle_id: str,
) -> dict[str, Any]:
    # This rebuild targets the platform's deterministic fact layer (Neo4j + action state).
    # LightRAG document ingest/delete/rebuild remains handled by LightRAG native APIs/WebUI.
    settings_rows = db.exec(
        select(SettingEntry).where(SettingEntry.project_id == project_id).order_by(SettingEntry.id.asc())
    ).all()
    cards_rows = db.exec(
        select(StoryCard).where(StoryCard.project_id == project_id).order_by(StoryCard.id.asc())
    ).all()

    candidates: list[dict[str, Any]] = []
    for row in settings_rows:
        source_entity = str(row.key or "").replace("设定", "").strip() or str(row.key or "")
        value_obj = row.value if isinstance(row.value, dict) else {}
        candidates.extend(_extract_rule_candidates(source_entity, value_obj))
    for row in cards_rows:
        source_entity = str(row.title or "").strip()
        content_obj = row.content if isinstance(row.content, dict) else {}
        candidates.extend(_extract_rule_candidates(source_entity, content_obj))

    merged_candidates = _dedupe_candidates(candidates)
    deleted_before = delete_all_neo4j_graph_facts(project_id)
    written_count = 0
    if merged_candidates:
        fact_keys = upsert_neo4j_graph_facts(
            project_id,
            merged_candidates,
            state="confirmed",
            source_ref=f"lifecycle_rebuild:{lifecycle_id}",
        )
        written_count = len(fact_keys)

    rag_rebuild_ok = trigger_lightrag_rebuild(project_id, reason=reason)
    return {
        "project_id": project_id,
        "reason": reason,
        "deleted_before": deleted_before,
        "candidate_count": len(merged_candidates),
        "written_count": written_count,
        "rag_rebuild_triggered": rag_rebuild_ok,
    }
