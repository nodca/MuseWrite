from difflib import SequenceMatcher
from typing import Any

from sqlmodel import Session, select

from app.core.config import settings
from app.models.chat import ChatSession
from app.models.content import StoryCard
from app.services.retrieval_adapters import fetch_neo4j_entity_profiles
from app.services.chat_service._common import (
    _utc_now,
    _normalize_graph_entity_token,
    _normalize_aliases_payload,
    _extract_aliases_from_content,
    _extract_name_candidates_from_content,
    _GRAPH_ENTITY_ALIAS_KEYS,
    _GRAPH_ENTITY_NAME_KEYS,
)
from app.services.chat_service.project_assets import list_settings, list_cards
from app.services.chat_service.actions import create_action, create_action_audit_log


def _register_graph_entity_alias(
    alias_map: dict[str, tuple[str, int]],
    *,
    alias: str,
    canonical: str,
    priority: int,
) -> None:
    normalized_alias = _normalize_graph_entity_token(alias)
    canonical_text = str(canonical or "").strip()
    if len(normalized_alias) < 2 or not canonical_text:
        return
    existing = alias_map.get(normalized_alias)
    if existing and existing[1] >= priority:
        return
    alias_map[normalized_alias] = (canonical_text, priority)


def _build_project_entity_alias_map(db: Session, project_id: int) -> dict[str, str]:
    alias_map: dict[str, tuple[str, int]] = {}

    cards = list_cards(db, project_id)
    for card in cards:
        canonical = str(card.title or "").strip()
        if not canonical:
            continue
        _register_graph_entity_alias(alias_map, alias=canonical, canonical=canonical, priority=50)
        native_aliases = _normalize_aliases_payload(getattr(card, "aliases", []))
        for alias in native_aliases:
            _register_graph_entity_alias(alias_map, alias=alias, canonical=canonical, priority=48)
        if not native_aliases:
            content_obj = card.content if isinstance(card.content, dict) else {}
            for name in _extract_name_candidates_from_content(content_obj):
                _register_graph_entity_alias(alias_map, alias=name, canonical=canonical, priority=40)
            for alias in _extract_aliases_from_content(content_obj):
                _register_graph_entity_alias(alias_map, alias=alias, canonical=canonical, priority=30)

    settings_rows = list_settings(db, project_id)
    for item in settings_rows:
        canonical = str(item.key or "").strip()
        if not canonical:
            continue
        _register_graph_entity_alias(alias_map, alias=canonical, canonical=canonical, priority=20)
        native_aliases = _normalize_aliases_payload(getattr(item, "aliases", []))
        for alias in native_aliases:
            _register_graph_entity_alias(alias_map, alias=alias, canonical=canonical, priority=19)
        if not native_aliases:
            value_obj = item.value if isinstance(item.value, dict) else {}
            for name in _extract_name_candidates_from_content(value_obj):
                _register_graph_entity_alias(alias_map, alias=name, canonical=canonical, priority=18)
            for alias in _extract_aliases_from_content(value_obj):
                _register_graph_entity_alias(alias_map, alias=alias, canonical=canonical, priority=16)

    return {key: value for key, (value, _priority) in alias_map.items()}


def _build_project_alias_prompt_hints(
    db: Session,
    project_id: int,
    *,
    limit: int = 24,
) -> list[dict[str, str]]:
    hint_map: dict[str, tuple[str, str, int]] = {}

    def register_hint(alias: str, canonical: str, priority: int) -> None:
        alias_text = str(alias or "").strip()
        canonical_text = str(canonical or "").strip()
        normalized_alias = _normalize_graph_entity_token(alias_text)
        if len(normalized_alias) < 2 or not canonical_text:
            return
        if alias_text == canonical_text:
            return
        existing = hint_map.get(normalized_alias)
        if existing and existing[2] >= priority:
            return
        hint_map[normalized_alias] = (alias_text, canonical_text, priority)

    cards = list_cards(db, project_id)
    for card in cards:
        canonical = str(card.title or "").strip()
        if not canonical:
            continue
        native_aliases = _normalize_aliases_payload(getattr(card, "aliases", []))
        for alias in native_aliases:
            register_hint(alias, canonical, 48)
        if not native_aliases:
            content_obj = card.content if isinstance(card.content, dict) else {}
            for name in _extract_name_candidates_from_content(content_obj):
                register_hint(name, canonical, 40)
            for alias in _extract_aliases_from_content(content_obj):
                register_hint(alias, canonical, 30)

    settings_rows = list_settings(db, project_id)
    for item in settings_rows:
        canonical = str(item.key or "").strip()
        if not canonical:
            continue
        native_aliases = _normalize_aliases_payload(getattr(item, "aliases", []))
        for alias in native_aliases:
            register_hint(alias, canonical, 19)
        if not native_aliases:
            value_obj = item.value if isinstance(item.value, dict) else {}
            for name in _extract_name_candidates_from_content(value_obj):
                register_hint(name, canonical, 18)
            for alias in _extract_aliases_from_content(value_obj):
                register_hint(alias, canonical, 16)

    entries = sorted(hint_map.values(), key=lambda row: (-row[2], -len(row[0]), row[0]))
    return [{"alias": alias, "canonical": canonical} for alias, canonical, _priority in entries[:limit]]


def _resolve_entity_merge_scan_session(db: Session, project_id: int) -> ChatSession:
    stmt = (
        select(ChatSession)
        .where(
            ChatSession.project_id == project_id,
            ChatSession.user_id == "system-entity-merge",
        )
        .order_by(ChatSession.id.asc())
        .limit(1)
    )
    existing = db.exec(stmt).first()
    if existing:
        return existing
    session = ChatSession(
        project_id=project_id,
        user_id="system-entity-merge",
        title="实体合并巡检",
        updated_at=_utc_now(),
    )
    db.add(session)
    db.flush()
    return session


def _entity_name_similarity(left: str, right: str) -> float:
    a = str(left or "").strip()
    b = str(right or "").strip()
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a.lower(), b.lower()).ratio())


def _entity_neighbor_set(profile: dict[str, Any]) -> set[str]:
    values = profile.get("neighbor_norms")
    if not isinstance(values, list):
        return set()
    return {str(item).strip() for item in values if str(item).strip()}


def _entity_relation_set(profile: dict[str, Any]) -> set[str]:
    values = profile.get("relation_types")
    if not isinstance(values, list):
        return set()
    return {str(item).strip().upper() for item in values if str(item).strip()}


def run_entity_merge_scan(
    db: Session,
    *,
    project_id: int,
    operator_id: str = "system-entity-merge",
    max_proposals: int | None = None,
    source: str = "entity_merge_scan",
) -> dict[str, Any]:
    if project_id <= 0:
        return {
            "status": "invalid_project",
            "project_id": project_id,
            "proposed_count": 0,
            "proposed_action_ids": [],
        }
    if not settings.entity_merge_scan_enabled:
        return {
            "status": "disabled",
            "project_id": project_id,
            "proposed_count": 0,
            "proposed_action_ids": [],
        }

    node_limit = max(int(settings.entity_merge_scan_node_limit), 1)
    proposal_limit = max(int(max_proposals or settings.entity_merge_scan_max_proposals), 1)
    min_degree = max(int(settings.entity_merge_scan_min_degree), 1)
    min_shared = max(int(settings.entity_merge_scan_min_shared_neighbors), 1)
    min_rel_overlap = max(int(settings.entity_merge_scan_min_relation_overlap), 0)
    min_jaccard = max(float(settings.entity_merge_scan_min_jaccard), 0.0)
    min_name_similarity = max(float(settings.entity_merge_scan_min_name_similarity), 0.0)

    cards = list(list_cards(db, project_id))
    if not cards:
        return {
            "status": "no_cards",
            "project_id": project_id,
            "scanned_nodes": 0,
            "candidate_pairs": 0,
            "proposed_count": 0,
            "proposed_action_ids": [],
        }

    card_by_norm: dict[str, StoryCard] = {}
    known_alias_tokens: set[str] = set()
    for card in cards:
        card_norm = _normalize_graph_entity_token(str(card.title or ""))
        if not card_norm:
            continue
        card_by_norm[card_norm] = card
        known_alias_tokens.add(card_norm)
        for alias in _normalize_aliases_payload(getattr(card, "aliases", [])):
            alias_norm = _normalize_graph_entity_token(alias)
            if alias_norm:
                known_alias_tokens.add(alias_norm)

    if not card_by_norm:
        return {
            "status": "no_canonical_cards",
            "project_id": project_id,
            "scanned_nodes": 0,
            "candidate_pairs": 0,
            "proposed_count": 0,
            "proposed_action_ids": [],
        }

    alias_map = _build_project_entity_alias_map(db, project_id)
    profiles_raw = fetch_neo4j_entity_profiles(project_id, limit=node_limit)
    if not profiles_raw:
        return {
            "status": "no_graph_entities",
            "project_id": project_id,
            "scanned_nodes": 0,
            "candidate_pairs": 0,
            "proposed_count": 0,
            "proposed_action_ids": [],
        }

    profile_by_norm: dict[str, dict[str, Any]] = {}
    for profile in profiles_raw:
        name_norm = _normalize_graph_entity_token(str(profile.get("name_norm") or profile.get("name") or ""))
        if not name_norm:
            continue
        profile_by_norm[name_norm] = {
            **profile,
            "name_norm": name_norm,
            "name": str(profile.get("name") or "").strip(),
        }

    if not profile_by_norm:
        return {
            "status": "no_valid_profiles",
            "project_id": project_id,
            "scanned_nodes": 0,
            "candidate_pairs": 0,
            "proposed_count": 0,
            "proposed_action_ids": [],
        }

    candidates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    canonical_norms = set(card_by_norm.keys())
    for source_norm, source_profile in profile_by_norm.items():
        if source_norm in canonical_norms:
            continue
        if source_norm in known_alias_tokens:
            continue
        mapped = alias_map.get(source_norm)
        if mapped:
            mapped_norm = _normalize_graph_entity_token(mapped)
            if mapped_norm in canonical_norms:
                continue

        source_neighbors = _entity_neighbor_set(source_profile)
        source_relations = _entity_relation_set(source_profile)
        if len(source_neighbors) < min_degree:
            continue

        for target_norm in canonical_norms:
            pair_key = (source_norm, target_norm)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            target_profile = profile_by_norm.get(target_norm)
            if not target_profile:
                continue
            target_neighbors = _entity_neighbor_set(target_profile)
            target_relations = _entity_relation_set(target_profile)
            if len(target_neighbors) < min_degree:
                continue

            shared_neighbors = sorted(source_neighbors.intersection(target_neighbors))
            if len(shared_neighbors) < min_shared:
                continue

            union_neighbors = source_neighbors.union(target_neighbors)
            if not union_neighbors:
                continue
            jaccard = len(shared_neighbors) / len(union_neighbors)
            if jaccard < min_jaccard:
                continue

            relation_overlap = len(source_relations.intersection(target_relations))
            if relation_overlap < min_rel_overlap:
                continue

            source_name = str(source_profile.get("name") or "").strip()
            target_name = str(target_profile.get("name") or card_by_norm[target_norm].title or "").strip()
            name_similarity = _entity_name_similarity(source_name, target_name)
            if name_similarity < min_name_similarity:
                has_strong_graph_overlap = (
                    len(shared_neighbors) >= min_shared + 1
                    or (jaccard >= max(min_jaccard, 0.72) and relation_overlap >= max(min_rel_overlap, 2))
                )
                if not has_strong_graph_overlap:
                    continue

            confidence = min(
                0.99,
                round(jaccard * 0.78 + min(len(shared_neighbors) / 6.0, 1.0) * 0.18 + name_similarity * 0.04, 4),
            )
            candidates.append(
                {
                    "source_name": source_name,
                    "source_norm": source_norm,
                    "target_name": target_name,
                    "target_norm": target_norm,
                    "target_card_id": int(card_by_norm[target_norm].id or 0),
                    "shared_neighbors": shared_neighbors[:6],
                    "shared_neighbor_count": len(shared_neighbors),
                    "jaccard": round(jaccard, 4),
                    "relation_overlap": relation_overlap,
                    "name_similarity": round(name_similarity, 4),
                    "confidence": confidence,
                }
            )

    if not candidates:
        return {
            "status": "no_candidate",
            "project_id": project_id,
            "scanned_nodes": len(profile_by_norm),
            "candidate_pairs": 0,
            "proposed_count": 0,
            "proposed_action_ids": [],
        }

    candidates.sort(
        key=lambda item: (
            -float(item.get("confidence") or 0.0),
            -int(item.get("shared_neighbor_count") or 0),
            -float(item.get("jaccard") or 0.0),
            str(item.get("source_name") or ""),
        )
    )
    selected = candidates[:proposal_limit]
    action_session = _resolve_entity_merge_scan_session(db, project_id)

    proposed_action_ids: list[int] = []
    action_previews: list[dict[str, Any]] = []
    for item in selected:
        source_name = str(item.get("source_name") or "").strip()
        source_norm = str(item.get("source_norm") or "").strip()
        target_card_id = int(item.get("target_card_id") or 0)
        target_name = str(item.get("target_name") or "").strip()
        if not source_name or not source_norm or target_card_id <= 0:
            continue

        idempotency_key = f"entity-merge:{project_id}:{target_card_id}:{source_norm[:48]}"
        payload = {
            "target_card_id": target_card_id,
            "target_title": target_name,
            "source_entity": source_name,
            "aliases": [source_name],
            "confidence": float(item.get("confidence") or 0.0),
            "shared_neighbor_count": int(item.get("shared_neighbor_count") or 0),
            "shared_neighbors": item.get("shared_neighbors", []),
            "jaccard": float(item.get("jaccard") or 0.0),
            "relation_overlap": int(item.get("relation_overlap") or 0),
            "_provenance": {
                "source": source,
                "scan": {
                    "candidate_source_norm": source_norm,
                    "candidate_target_norm": str(item.get("target_norm") or ""),
                    "node_limit": node_limit,
                    "thresholds": {
                        "min_degree": min_degree,
                        "min_shared_neighbors": min_shared,
                        "min_jaccard": min_jaccard,
                        "min_relation_overlap": min_rel_overlap,
                        "min_name_similarity": min_name_similarity,
                    },
                },
            },
        }
        action = create_action(
            db=db,
            session_id=int(action_session.id or 0),
            action_type="entity.merge.proposal",
            payload=payload,
            operator_id=str(operator_id or "system-entity-merge"),
            idempotency_key=idempotency_key[:128],
        )
        action_id = int(action.id or 0)
        if action.status == "proposed" and action_id > 0:
            if action_id in proposed_action_ids:
                continue
            proposed_action_ids.append(action_id)
            action_previews.append(
                {
                    "action_id": action_id,
                    "source_entity": source_name,
                    "target_title": target_name,
                    "confidence": float(item.get("confidence") or 0.0),
                    "shared_neighbor_count": int(item.get("shared_neighbor_count") or 0),
                }
            )
            create_action_audit_log(
                db=db,
                action_id=action.id,
                event_type="proposed",
                operator_id=str(operator_id or "system-entity-merge"),
                event_payload={
                    "source": source,
                    "entity_merge_scan": {
                        "source_entity": source_name,
                        "target_title": target_name,
                        "confidence": float(item.get("confidence") or 0.0),
                        "shared_neighbor_count": int(item.get("shared_neighbor_count") or 0),
                    },
                },
            )

    return {
        "status": "proposed" if proposed_action_ids else "deduped_or_skipped",
        "project_id": project_id,
        "scanned_nodes": len(profile_by_norm),
        "candidate_pairs": len(candidates),
        "proposed_count": len(proposed_action_ids),
        "proposed_action_ids": proposed_action_ids,
        "proposed_actions": action_previews,
    }

