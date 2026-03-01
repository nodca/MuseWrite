from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Iterable
import copy
import json
import logging
import re
from collections import deque
import time
from uuid import uuid4

import httpx
from sqlmodel import Session, select

from app.core.config import settings
from app.core.database import engine
from app.models.chat import (
    ActionAuditLog,
    ChatAction,
    ChatMessage,
    ChatSession,
    ProjectMutationVersion,
)
from app.models.content import (
    ChapterSceneBeat,
    ForeshadowingCard,
    ProjectChapter,
    ProjectChapterRevision,
    ProjectVolume,
    PromptTemplate,
    PromptTemplateRevision,
    SettingEntry,
    StoryCard,
)
from app.services.graph_job_queue import enqueue_graph_sync_job
from app.services.index_lifecycle_queue import enqueue_index_lifecycle_job
from app.services.entity_merge_queue import enqueue_entity_merge_scan_job
from app.services.index_lifecycle_service import process_index_lifecycle_rebuild
from app.services.retrieval_adapters import (
    delete_neo4j_graph_facts,
    delete_neo4j_graph_facts_by_sources,
    fetch_neo4j_entity_profiles,
    fetch_lightrag_graph_candidates,
    make_graph_candidate,
    merge_graph_candidates,
    promote_neo4j_candidate_facts,
    update_neo4j_graph_fact_state,
    upsert_neo4j_graph_facts,
)


_LOGGER = logging.getLogger(__name__)
_INTERNAL_SETTING_PREFIXES: tuple[str, ...] = (
    "llm.profile.",
    "consistency.audit.report.",
)
_MODEL_PROFILE_PREFIX = "llm.profile."
_MODEL_PROFILE_ACTIVE_KEY = "llm.profile.active"
_MODEL_PROFILE_PROVIDER_ALIASES: dict[str, str] = {
    "openai": "openai_compatible",
    "openai_compatible": "openai_compatible",
    "gpt": "openai_compatible",
    "deepseek": "deepseek",
    "anthropic": "claude",
    "claude": "claude",
    "google": "gemini",
    "gemini": "gemini",
    "stub": "stub",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DraftVersionConflictError(ValueError):
    pass


def build_session_title(user_text: str, limit: int = 24) -> str:
    title = user_text.strip().replace("\n", " ")
    if len(title) <= limit:
        return title or "新对话"
    return title[:limit].rstrip() + "..."


def create_session(db: Session, project_id: int, user_id: str, title: str) -> ChatSession:
    session = ChatSession(project_id=project_id, user_id=user_id, title=title, updated_at=_utc_now())
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session_by_id(db: Session, session_id: int) -> ChatSession | None:
    return db.get(ChatSession, session_id)


def list_project_sessions(
    db: Session,
    *,
    project_id: int,
    user_id: str,
    limit: int = 24,
) -> Iterable[ChatSession]:
    size = max(min(int(limit), 100), 1)
    stmt = (
        select(ChatSession)
        .where(
            ChatSession.project_id == project_id,
            ChatSession.user_id == user_id,
        )
        .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        .limit(size)
    )
    return db.exec(stmt).all()


def update_session_title(
    db: Session,
    *,
    session_id: int,
    title: str,
) -> ChatSession:
    session = db.get(ChatSession, session_id)
    if not session:
        raise ValueError("chat session not found")
    normalized_title = str(title or "").strip()
    if not normalized_title:
        raise ValueError("title is required")
    session.title = normalized_title[:255]
    session.updated_at = _utc_now()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def delete_session_with_children(
    db: Session,
    *,
    session_id: int,
) -> int:
    session = db.get(ChatSession, session_id)
    if not session:
        raise ValueError("chat session not found")

    action_rows = db.exec(select(ChatAction).where(ChatAction.session_id == session_id)).all()
    for action in action_rows:
        log_rows = db.exec(select(ActionAuditLog).where(ActionAuditLog.action_id == int(action.id or 0))).all()
        for log in log_rows:
            db.delete(log)
        db.delete(action)

    message_rows = db.exec(select(ChatMessage).where(ChatMessage.session_id == session_id)).all()
    for message in message_rows:
        db.delete(message)

    deleted_session_id = int(session.id or 0)
    db.delete(session)
    db.commit()
    return deleted_session_id


def append_message(
    db: Session,
    session_id: int,
    role: str,
    content: str,
    model: str | None = None,
) -> ChatMessage:
    msg = ChatMessage(session_id=session_id, role=role, content=content, model=model)
    db.add(msg)

    session = db.get(ChatSession, session_id)
    if session:
        session.updated_at = _utc_now()
        db.add(session)

    db.commit()
    db.refresh(msg)

    return msg


def update_message_content(
    message_id: int,
    content: str,
    *,
    db: Session | None = None,
) -> None:
    def _write(target_db: Session) -> None:
        msg = target_db.get(ChatMessage, message_id)
        if not msg:
            return
        msg.content = content
        session = target_db.get(ChatSession, msg.session_id)
        if session:
            session.updated_at = _utc_now()
            target_db.add(session)
        target_db.add(msg)
        target_db.commit()

    if db is not None:
        _write(db)
        return

    with Session(engine) as managed_db:
        _write(managed_db)


def list_messages(db: Session, session_id: int, *, limit: int | None = None) -> Iterable[ChatMessage]:
    base_stmt = select(ChatMessage).where(ChatMessage.session_id == session_id)
    if limit is not None:
        size = max(int(limit), 1)
        rows = db.exec(base_stmt.order_by(ChatMessage.id.desc()).limit(size)).all()
        rows.reverse()
        return rows
    return db.exec(base_stmt.order_by(ChatMessage.id.asc())).all()


def create_action(
    db: Session,
    session_id: int,
    action_type: str,
    payload: dict,
    operator_id: str,
    idempotency_key: str,
) -> ChatAction:
    existing = get_action_by_idempotency_key(db, session_id, idempotency_key)
    if existing:
        return existing

    action = ChatAction(
        session_id=session_id,
        action_type=action_type,
        payload=payload,
        operator_id=operator_id,
        idempotency_key=idempotency_key,
        status="proposed",
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    return action


def is_entity_merge_action_type(action_type: str) -> bool:
    raw = str(action_type or "").strip().lower()
    if not raw:
        return False
    normalized = raw.replace("_", ".").replace("-", ".")
    return normalized.startswith("entity.merge") or normalized.startswith("graph.entity.merge")


def is_manual_merge_operator(operator_id: str) -> bool:
    raw = str(operator_id or "").strip().lower()
    if not raw:
        return False
    blocked_tokens = ("system", "worker", "assistant", "auto", "daemon", "scheduler", "bot")
    return not any(
        raw == token
        or raw.startswith(token)
        or raw.startswith(f"{token}_")
        or raw.startswith(f"{token}-")
        for token in blocked_tokens
    )


def get_action_by_id(db: Session, action_id: int) -> ChatAction | None:
    return db.get(ChatAction, action_id)


def get_action_by_idempotency_key(db: Session, session_id: int, idempotency_key: str) -> ChatAction | None:
    stmt = select(ChatAction).where(
        ChatAction.session_id == session_id, ChatAction.idempotency_key == idempotency_key
    )
    return db.exec(stmt).first()


def list_actions(db: Session, session_id: int) -> Iterable[ChatAction]:
    stmt = select(ChatAction).where(ChatAction.session_id == session_id).order_by(ChatAction.id.asc())
    return db.exec(stmt).all()


def list_action_logs(db: Session, action_id: int) -> Iterable[ActionAuditLog]:
    stmt = select(ActionAuditLog).where(ActionAuditLog.action_id == action_id).order_by(ActionAuditLog.id.asc())
    return db.exec(stmt).all()


def create_action_audit_log(
    db: Session,
    action_id: int,
    event_type: str,
    operator_id: str,
    event_payload: dict | None = None,
) -> ActionAuditLog:
    log = ActionAuditLog(
        action_id=action_id,
        event_type=event_type,
        operator_id=operator_id,
        event_payload=event_payload or {},
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def set_action_status(
    db: Session,
    action: ChatAction,
    status: str,
    *,
    set_applied_at: bool = False,
    set_undone_at: bool = False,
) -> ChatAction:
    action.status = status
    if set_applied_at:
        action.applied_at = _utc_now()
    if set_undone_at:
        action.undone_at = _utc_now()
    db.add(action)
    db.commit()
    db.refresh(action)
    return action


def _project_id_for_action(db: Session, action: ChatAction) -> int:
    session = db.get(ChatSession, action.session_id)
    if not session:
        raise ValueError("session not found for action")
    return session.project_id


def _current_project_mutation_version(db: Session, project_id: int) -> int:
    stmt = select(ProjectMutationVersion).where(ProjectMutationVersion.project_id == project_id)
    row = db.exec(stmt).first()
    if not row:
        return 0
    return int(row.version)


def _bump_project_mutation_version(db: Session, project_id: int) -> int:
    stmt = select(ProjectMutationVersion).where(ProjectMutationVersion.project_id == project_id)
    row = db.exec(stmt).first()
    if row:
        row.version = int(row.version) + 1
        row.updated_at = _utc_now()
        db.add(row)
        return int(row.version)

    row = ProjectMutationVersion(project_id=project_id, version=1, updated_at=_utc_now())
    db.add(row)
    db.flush()
    return int(row.version)


def _graph_sync_meta(action: ChatAction) -> dict:
    if not isinstance(action.apply_result, dict):
        return {}
    meta = action.apply_result.get("graph_sync")
    return meta if isinstance(meta, dict) else {}


def _action_graph_identifiers(action: ChatAction) -> tuple[str, int, str]:
    meta = _graph_sync_meta(action)
    mutation_id = str(meta.get("mutation_id") or "")
    expected_version_raw = meta.get("expected_version")
    expected_version = int(expected_version_raw) if isinstance(expected_version_raw, int) else 0
    idempotency_key = str(meta.get("job_idempotency_key") or "")
    return mutation_id, expected_version, idempotency_key


def _is_graph_job_stale(
    action: ChatAction,
    *,
    mutation_id: str,
    expected_version: int,
) -> tuple[bool, str]:
    meta = _graph_sync_meta(action)
    current_mutation_id = str(meta.get("mutation_id") or "")
    current_expected_raw = meta.get("expected_version")
    current_expected_version = int(current_expected_raw) if isinstance(current_expected_raw, int) else 0

    if mutation_id and current_mutation_id and current_mutation_id != mutation_id:
        return True, "mutation_id_mismatch"
    if (
        expected_version > 0
        and current_expected_version > 0
        and current_expected_version != expected_version
    ):
        return True, "expected_version_mismatch"
    if action.status != "applied":
        return True, f"action_status_{action.status}"
    return False, ""


def _index_lifecycle_key(slot: str) -> str:
    return "index_lifecycle_compensation" if slot == "compensation" else "index_lifecycle"


def _index_lifecycle_meta(action: ChatAction, *, slot: str = "default") -> dict:
    if not isinstance(action.apply_result, dict):
        return {}
    meta = action.apply_result.get(_index_lifecycle_key(slot))
    return meta if isinstance(meta, dict) else {}


def _setting_key_from_payload(payload: dict) -> str:
    key = payload.get("key") or payload.get("name")
    if not key or not isinstance(key, str):
        raise ValueError("setting key is required")
    return key


def _normalize_aliases_payload(value: Any, *, max_items: int = 64) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = re.split(r"[,\s，、;；/|]+", value)
    elif isinstance(value, list):
        raw_items = []
        for item in value:
            if isinstance(item, str):
                raw_items.extend(re.split(r"[,\s，、;；/|]+", item))
            elif isinstance(item, (int, float, bool)):
                raw_items.append(str(item))
    else:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        alias = str(raw or "").strip()
        if len(alias) < 2:
            continue
        token = _normalize_graph_entity_token(alias)
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        normalized.append(alias[:64])
        if len(normalized) >= max(max_items, 1):
            break
    return normalized


def _collect_entity_merge_aliases(payload: dict[str, Any]) -> list[str]:
    if not isinstance(payload, dict):
        return []
    raw_items: list[Any] = []
    for key in ("alias", "source_entity", "source_alias", "candidate_alias", "from_entity"):
        raw = payload.get(key)
        if raw is not None:
            raw_items.append(raw)
    aliases_raw = payload.get("aliases")
    if aliases_raw is not None:
        raw_items.append(aliases_raw)

    candidates: list[str] = []
    for item in raw_items:
        candidates.extend(_normalize_aliases_payload(item))
    return _normalize_aliases_payload(candidates)


_GRAPH_RELATION_FIELD_MAP = {
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


def _to_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return ""


def _split_targets(value: object) -> list[str]:
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
        # 关系字段常见结构：{"林澈": "师徒", "周夜": {"type":"敌对"}}
        return [str(key).strip() for key in value.keys() if str(key).strip()]
    return []


_GRAPH_ENTITY_ALIAS_KEYS = {
    "alias",
    "aliases",
    "aka",
    "别名",
    "称呼",
    "曾用名",
    "外号",
    "昵称",
    "简称",
}
_GRAPH_ENTITY_NAME_KEYS = {"name", "title", "名称", "姓名", "角色名", "实体名"}


def _normalize_graph_entity_token(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff·-]", "", text)
    return text


def _split_alias_text(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,\s，、;；/|]+", value) if item and item.strip()]


def _flatten_alias_values(value: Any, *, depth: int = 0) -> list[str]:
    if depth > 2:
        return []
    if isinstance(value, str):
        return _split_alias_text(value)
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_flatten_alias_values(item, depth=depth + 1))
        return values
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_flatten_alias_values(item, depth=depth + 1))
        return values
    return []


def _extract_aliases_from_content(content: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    for key, value in content.items():
        normalized_key = str(key).strip().lower()
        if key in _GRAPH_ENTITY_ALIAS_KEYS or normalized_key in _GRAPH_ENTITY_ALIAS_KEYS:
            aliases.extend(_flatten_alias_values(value))
    return aliases


def _extract_name_candidates_from_content(content: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key, value in content.items():
        normalized_key = str(key).strip().lower()
        if key not in _GRAPH_ENTITY_NAME_KEYS and normalized_key not in _GRAPH_ENTITY_NAME_KEYS:
            continue
        if isinstance(value, str) and value.strip():
            names.append(value.strip())
    return names


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


def _scan_alias_hints_in_text(
    text: str,
    alias_hints: list[dict[str, str]],
    *,
    limit: int = 24,
) -> list[dict[str, str]]:
    content = str(text or "")
    if not content or not alias_hints:
        return []

    candidates: list[dict[str, str]] = []
    pattern_keys: list[str] = []
    pattern_index_map: dict[str, int] = {}
    for item in alias_hints:
        alias = str(item.get("alias") or "").strip()
        canonical = str(item.get("canonical") or "").strip()
        if len(alias) < 2 or not canonical:
            continue
        key = alias.lower()
        if key in pattern_index_map:
            continue
        pattern_index_map[key] = len(pattern_keys)
        pattern_keys.append(key)
        candidates.append({"alias": alias, "canonical": canonical})
    if not candidates:
        return []

    goto: list[dict[str, int]] = [{}]
    fail: list[int] = [0]
    outputs: list[list[int]] = [[]]

    for idx, pattern in enumerate(pattern_keys):
        state = 0
        for ch in pattern:
            nxt = goto[state].get(ch)
            if nxt is None:
                nxt = len(goto)
                goto[state][ch] = nxt
                goto.append({})
                fail.append(0)
                outputs.append([])
            state = nxt
        outputs[state].append(idx)

    queue: deque[int] = deque()
    for _ch, state in goto[0].items():
        fail[state] = 0
        queue.append(state)

    while queue:
        state = queue.popleft()
        for ch, nxt in goto[state].items():
            queue.append(nxt)
            f = fail[state]
            while f and ch not in goto[f]:
                f = fail[f]
            fail_state = goto[f].get(ch, 0)
            fail[nxt] = fail_state
            if outputs[fail_state]:
                outputs[nxt].extend(outputs[fail_state])

    normalized_text = content.lower()
    matched_map: dict[str, dict[str, str]] = {}
    state = 0
    for ch in normalized_text:
        while state and ch not in goto[state]:
            state = fail[state]
        state = goto[state].get(ch, 0)
        for pattern_idx in outputs[state]:
            hit = candidates[int(pattern_idx)]
            matched_map[hit["alias"]] = hit
        if len(matched_map) >= max(limit, 1):
            break

    matched = list(matched_map.values())
    matched.sort(key=lambda row: (-len(row["alias"]), row["alias"]))
    return matched[: max(limit, 1)]


def _inject_alias_hints_into_graph_text(text: str, alias_hints: list[dict[str, str]]) -> str:
    if not text or not alias_hints:
        return text
    lines = ["[alias_hint] 请在图谱抽取时统一以下别名到标准实体名："]
    for item in alias_hints:
        alias = str(item.get("alias") or "").strip()
        canonical = str(item.get("canonical") or "").strip()
        if not alias or not canonical:
            continue
        lines.append(f"- {alias} => {canonical}")
    lines.append("[alias_hint] 仅做实体归一化，不改写原文事实。")
    return "\n".join(lines) + "\n\n" + text


def _build_overlap_chunks(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    max_chunks: int,
) -> list[str]:
    content = str(text or "")
    if not content:
        return []

    size = max(int(chunk_size), 64)
    overlap_size = max(min(int(overlap), max(size - 8, 0)), 0)
    step = max(size - overlap_size, 1)
    chunks: list[str] = []
    start = 0
    while start < len(content) and len(chunks) < max(max_chunks, 1):
        end = min(start + size, len(content))
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(content):
            break
        start += step
    return chunks or [content]


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _rewrite_chunk_with_llm_coref(
    chunk_text: str,
    *,
    context_summary: str,
    anchor_canonical: str,
) -> tuple[str, dict[str, Any]]:
    if not settings.graph_coref_llm_enabled:
        return chunk_text, {"enabled": False, "applied": False, "reason": "llm_disabled"}
    lightrag_model = str(settings.lightrag_llm_model or "").strip()
    lightrag_base_url = str(settings.lightrag_llm_base_url or "").strip()
    lightrag_api_key = str(settings.lightrag_llm_api_key or "").strip()
    if not (lightrag_model and lightrag_base_url and lightrag_api_key):
        return chunk_text, {
            "enabled": True,
            "applied": False,
            "reason": "missing_lightrag_llm_config",
        }

    model = str(settings.graph_coref_llm_model or lightrag_model).strip()
    endpoint = lightrag_base_url.rstrip("/") + "/chat/completions"
    system_prompt = (
        "你是小说图谱抽取前的文本预处理器。"
        "任务是把代词（他/她/它/那家伙等）在有上下文依据时还原为明确实体。"
        "必须保持事实不变，不新增事件，不润色文风。"
        "若不确定则保持原文。只输出 JSON。"
    )
    payload = {
        "anchor": anchor_canonical,
        "context_summary": context_summary,
        "chunk_text": chunk_text,
        "output_schema": {
            "rewritten_text": "string",
            "applied": "boolean",
            "confidence": "number(0-1)",
        },
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": 0,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {lightrag_api_key}"}
    timeout = httpx.Timeout(float(settings.graph_coref_llm_timeout_seconds))
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        return chunk_text, {
            "enabled": True,
            "applied": False,
            "reason": "llm_error",
            "error": str(exc),
            "model": model,
        }

    content = str(data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
    parsed = _extract_json_object(content)
    if not parsed:
        return chunk_text, {
            "enabled": True,
            "applied": False,
            "reason": "llm_invalid_json",
            "model": model,
        }

    rewritten = str(parsed.get("rewritten_text") or "").strip()
    applied = bool(parsed.get("applied")) and bool(rewritten) and rewritten != chunk_text
    confidence_raw = parsed.get("confidence")
    confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else 0.0
    if confidence < 0.65:
        return chunk_text, {
            "enabled": True,
            "applied": False,
            "reason": "llm_low_confidence",
            "confidence": confidence,
            "model": model,
        }
    if not rewritten:
        return chunk_text, {
            "enabled": True,
            "applied": False,
            "reason": "llm_empty",
            "model": model,
        }
    return rewritten, {
        "enabled": True,
        "applied": applied,
        "reason": "llm_applied" if applied else "llm_no_change",
        "confidence": confidence,
        "model": model,
    }


def _extract_inherited_entities(
    text: str,
    alias_hints: list[dict[str, str]],
    *,
    limit: int = 6,
) -> list[str]:
    tail = str(text or "")[-360:]
    hits = _scan_alias_hints_in_text(tail, alias_hints, limit=limit * 2)
    entities: list[str] = []
    seen: set[str] = set()
    for item in hits:
        canonical = str(item.get("canonical") or "").strip()
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        entities.append(canonical)
        if len(entities) >= max(limit, 1):
            break
    return entities


def _build_inheritance_summary(anchor_canonical: str, inherited_entities: list[str]) -> str:
    entities = [item for item in inherited_entities if item]
    if anchor_canonical and anchor_canonical not in entities:
        entities = [anchor_canonical, *entities]
    if not entities:
        return ""
    return "上文实体继承: " + "、".join(entities[:6])


def _build_graph_extraction_segments(
    text: str,
    *,
    action_type: str,
    anchor: str | None,
    alias_map: dict[str, str],
    alias_hint_pool: list[dict[str, str]],
) -> tuple[list[str], dict[str, Any]]:
    # Phase C baseline: keep graph extraction on deterministic alias normalization only.
    # Scheme 2 (overlap/entity-inheritance/pronoun-coref rewrite) is intentionally disabled.
    content = str(text or "")
    alias_hits = _scan_alias_hints_in_text(content, alias_hint_pool, limit=24)
    segment = _inject_alias_hints_into_graph_text(content, alias_hits)
    return [segment], {
        "mode": "scheme1_alias_only",
        "overlap_enabled": False,
        "chunk_count": 1,
        "chunk_size": len(content),
        "chunk_overlap": 0,
        "max_chunks": 1,
        "inheritance_used_chunks": 0,
        "llm_applied_chunks": 0,
        "llm_failed_chunks": 0,
        "rule_applied_chunks": 0,
        "alias_hint_count": len(alias_hits),
        "alias_hint_pool_size": len(alias_hint_pool),
    }


def _is_entity_like_anchor(value: str) -> bool:
    name = str(value or "").strip()
    if len(name) < 2 or len(name) > 24:
        return False
    if re.fullmatch(r"card-\d+", name.lower()):
        return False
    lowered = name.lower()
    banned_tokens = (
        "设定",
        "世界观",
        "规则",
        "剧情",
        "章节",
        "chapter",
        "config",
        "系统",
    )
    return not any(token in lowered for token in banned_tokens)


def _resolve_anchor_canonical(anchor: str | None, alias_map: dict[str, str]) -> str:
    raw = str(anchor or "").strip()
    if not raw:
        return ""
    normalized = _normalize_graph_entity_token(raw)
    return str(alias_map.get(normalized) or raw).strip()


def _replace_entity_pronouns(
    text: str,
    *,
    canonical: str,
    max_replacements: int,
) -> tuple[str, int]:
    if not text or not canonical or max_replacements <= 0:
        return text, 0

    replaced = 0
    output = text

    phrase_tokens = ("那家伙", "这人", "那人", "此人", "对方")
    for token in phrase_tokens:
        while token in output and replaced < max_replacements:
            output = output.replace(token, canonical, 1)
            replaced += 1
        if replaced >= max_replacements:
            break

    if replaced >= max_replacements:
        return output, replaced

    single_pattern = re.compile(r"(^|[，。！？；：,\s])([他她它])(?!们)")

    def single_replacer(match: re.Match[str]) -> str:
        nonlocal replaced
        if replaced >= max_replacements:
            return match.group(0)
        replaced += 1
        return f"{match.group(1)}{canonical}"

    output = single_pattern.sub(single_replacer, output)
    return output, replaced


def _apply_graph_pronoun_coref_preprocess(
    text: str,
    *,
    action_type: str,
    anchor: str | None,
    alias_map: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    enabled = bool(settings.graph_coref_preprocess_enabled)
    if not enabled:
        return text, {
            "enabled": False,
            "applied": False,
            "reason": "disabled",
            "replacements": 0,
            "canonical": "",
        }
    if action_type not in {"card.create", "card.update"}:
        return text, {
            "enabled": True,
            "applied": False,
            "reason": "action_filtered",
            "replacements": 0,
            "canonical": "",
        }

    canonical = _resolve_anchor_canonical(anchor, alias_map)
    if not _is_entity_like_anchor(canonical):
        return text, {
            "enabled": True,
            "applied": False,
            "reason": "anchor_not_entity_like",
            "replacements": 0,
            "canonical": canonical,
        }

    anchor_text = str(anchor or "").strip()
    if canonical and canonical not in text and anchor_text and anchor_text not in text:
        return text, {
            "enabled": True,
            "applied": False,
            "reason": "anchor_not_in_text",
            "replacements": 0,
            "canonical": canonical,
        }

    replaced_text, replacements = _replace_entity_pronouns(
        text,
        canonical=canonical,
        max_replacements=max(int(settings.graph_coref_max_replacements), 1),
    )
    return replaced_text, {
        "enabled": True,
        "applied": replacements > 0,
        "reason": "applied" if replacements > 0 else "no_pronoun_match",
        "replacements": replacements,
        "canonical": canonical,
    }


def _resolve_entity_aliases_for_candidates(
    candidates: list[dict],
    alias_map: dict[str, str],
) -> tuple[list[dict], dict[str, Any]]:
    if not candidates:
        return [], {
            "map_size": len(alias_map),
            "aligned_count": 0,
            "unresolved_count": 0,
            "collapsed_count": 0,
            "samples": [],
        }

    resolved_candidates: list[dict] = []
    aligned_count = 0
    unresolved_count = 0
    samples: list[dict[str, str]] = []

    def resolve_entity(raw_value: Any) -> tuple[str, str, bool]:
        original = str(raw_value or "").strip()
        normalized = _normalize_graph_entity_token(original)
        if not original:
            return "", normalized, False
        resolved = alias_map.get(normalized, original)
        return resolved, normalized, resolved != original

    for idx, item in enumerate(candidates, start=1):
        source, source_norm, source_changed = resolve_entity(item.get("source_entity"))
        target, target_norm, target_changed = resolve_entity(item.get("target_entity"))
        if source_changed:
            aligned_count += 1
        elif source_norm and source_norm not in alias_map:
            unresolved_count += 1
        if target_changed:
            aligned_count += 1
        elif target_norm and target_norm not in alias_map:
            unresolved_count += 1

        if (source_changed or target_changed) and len(samples) < 8:
            samples.append(
                {
                    "source_before": str(item.get("source_entity") or ""),
                    "source_after": source,
                    "target_before": str(item.get("target_entity") or ""),
                    "target_after": target,
                }
            )

        confidence_raw = item.get("confidence")
        confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else None
        candidate = make_graph_candidate(
            source,
            str(item.get("relation") or "RELATES_TO"),
            target,
            evidence=str(item.get("evidence") or ""),
            origin=str(item.get("origin") or "unknown"),
            confidence=confidence,
            item_id=int(item.get("id") or idx),
        )
        if candidate:
            resolved_candidates.append(candidate)

    deduped_candidates = merge_graph_candidates(resolved_candidates, [], limit=24)
    return deduped_candidates, {
        "map_size": len(alias_map),
        "aligned_count": aligned_count,
        "unresolved_count": unresolved_count,
        "collapsed_count": max(0, len(resolved_candidates) - len(deduped_candidates)),
        "samples": samples,
    }


def _extract_rule_graph_candidates(source_entity: str, content_obj: dict[str, object]) -> list[dict]:
    candidates: list[dict] = []
    if not source_entity.strip():
        return candidates

    next_id = 1
    for raw_key, raw_value in content_obj.items():
        key = str(raw_key).strip().lower()
        if key not in _GRAPH_RELATION_FIELD_MAP:
            continue
        relation = _GRAPH_RELATION_FIELD_MAP[key]
        targets = _split_targets(raw_value)
        if not targets:
            continue
        for target in targets:
            candidate = make_graph_candidate(
                source_entity,
                relation,
                target,
                evidence=f"{source_entity} {raw_key}: {_to_text(raw_value)}",
                origin="rule",
                item_id=next_id,
            )
            if candidate:
                candidates.append(candidate)
                next_id += 1
    return candidates[:24]


def _build_graph_extraction_text(action_type: str, payload: dict, project_id: int) -> tuple[str, str | None, list[dict]]:
    anchor_hint = str(payload.get("_graph_anchor") or "").strip()
    if action_type == "card.create":
        title = anchor_hint or str(payload.get("title") or "").strip()
        content = payload.get("content")
        content_obj = content if isinstance(content, dict) else {}
        text = f"[project:{project_id}] card.create\n标题: {title}\n内容: {json.dumps(content_obj, ensure_ascii=False)}"
        return text, title or None, _extract_rule_graph_candidates(title, content_obj)

    if action_type == "card.update":
        title = str(payload.get("title") or "").strip()
        content = payload.get("content")
        content_obj = content if isinstance(content, dict) else {}
        anchor = anchor_hint or title or None
        if not anchor:
            card_id = payload.get("card_id")
            if isinstance(card_id, int):
                anchor = f"card-{card_id}"
        text = f"[project:{project_id}] card.update\n锚点: {anchor or ''}\n内容: {json.dumps(content_obj, ensure_ascii=False)}"
        if anchor and content_obj:
            return text, anchor, _extract_rule_graph_candidates(anchor, content_obj)
        return text, anchor, []

    if action_type == "setting.upsert":
        key = str(payload.get("key") or "").strip()
        value = payload.get("value") or payload.get("content")
        value_obj = value if isinstance(value, dict) else {}
        text = f"[project:{project_id}] setting.upsert\nkey: {key}\nvalue: {json.dumps(value_obj, ensure_ascii=False)}"
        source_entity = anchor_hint or key.replace("设定", "").strip() or key
        return text, source_entity or None, _extract_rule_graph_candidates(source_entity, value_obj)

    return "", None, []


def _sync_graph_for_action(
    db: Session,
    action_id: int,
    *,
    project_id: int,
    action_type: str,
    payload: dict,
) -> tuple[dict | None, list[str]]:
    text, anchor, rule_candidates = _build_graph_extraction_text(action_type, payload, project_id)
    if not text:
        return None, []
    graph_current_chapter = 0
    try:
        graph_current_chapter = int(payload.get("_graph_current_chapter") or 0)
    except Exception:
        graph_current_chapter = 0

    projection_deleted = 0
    projection_sources: list[str] = []
    for key in ("_graph_anchor", "_graph_anchor_before"):
        value = str(payload.get(key) or "").strip()
        if value and value not in projection_sources:
            projection_sources.append(value)
    if anchor and anchor not in projection_sources:
        projection_sources.append(anchor)
    if action_type in {"setting.upsert", "card.create", "card.update"} and projection_sources:
        projection_deleted = delete_neo4j_graph_facts_by_sources(
            project_id,
            projection_sources,
            current_chapter=graph_current_chapter if graph_current_chapter > 0 else None,
        )

    alias_map = _build_project_entity_alias_map(db, project_id)
    alias_prompt_hints_pool = _build_project_alias_prompt_hints(db, project_id, limit=64)
    graph_segments, segment_meta = _build_graph_extraction_segments(
        text,
        action_type=action_type,
        anchor=anchor,
        alias_map=alias_map,
        alias_hint_pool=alias_prompt_hints_pool,
    )
    lightrag_candidates_raw: list[dict[str, Any]] = []
    for segment in graph_segments:
        segment_candidates = fetch_lightrag_graph_candidates(segment, anchor=anchor, limit=24)
        if segment_candidates:
            lightrag_candidates_raw.extend(segment_candidates)
        if len(lightrag_candidates_raw) >= 128:
            break
    lightrag_candidates = merge_graph_candidates(lightrag_candidates_raw, [], limit=64)
    merged_candidates = merge_graph_candidates(lightrag_candidates, rule_candidates, limit=24)
    resolved_candidates, alias_meta = _resolve_entity_aliases_for_candidates(merged_candidates, alias_map)
    merged_candidates = resolved_candidates

    if not merged_candidates:
        return (
            {
                "status": "no_facts",
                "extractor": "none",
                "lightrag_count": len(lightrag_candidates),
                "rule_count": len(rule_candidates),
                "merged_count": 0,
                "written_count": 0,
                "projection_deleted": projection_deleted,
                "projection_mode": "source_replace" if projection_sources else "none",
                "alias_map_size": int(alias_meta.get("map_size", 0)),
                "alias_aligned_count": int(alias_meta.get("aligned_count", 0)),
                "alias_unresolved_count": int(alias_meta.get("unresolved_count", 0)),
                "alias_collapsed_count": int(alias_meta.get("collapsed_count", 0)),
                "alias_samples": alias_meta.get("samples", []),
                "alias_hint_count": int(segment_meta.get("alias_hint_count", 0)),
                "alias_hint_pool_size": len(alias_prompt_hints_pool),
                "coref_enabled": False,
                "coref_applied": False,
                "coref_reason": "disabled_scheme2",
                "coref_canonical": _resolve_anchor_canonical(anchor, alias_map),
                "coref_replacements": 0,
                "coref_segment_meta": segment_meta,
            },
            [],
        )

    extractor = "lightrag+rule" if lightrag_candidates else "rule_fallback"
    source_ref = f"chat_action:{action_id}"
    fact_keys = upsert_neo4j_graph_facts(
        project_id,
        merged_candidates,
        state="candidate",
        source_ref=source_ref,
        current_chapter=graph_current_chapter if graph_current_chapter > 0 else None,
    )
    return (
        {
            "status": "synced" if fact_keys else "queued_or_disabled",
            "extractor": extractor,
            "lightrag_count": len(lightrag_candidates),
            "rule_count": len(rule_candidates),
            "merged_count": len(merged_candidates),
            "written_count": len(fact_keys),
            "projection_deleted": projection_deleted,
            "projection_mode": "source_replace" if projection_sources else "none",
            "alias_map_size": int(alias_meta.get("map_size", 0)),
            "alias_aligned_count": int(alias_meta.get("aligned_count", 0)),
            "alias_unresolved_count": int(alias_meta.get("unresolved_count", 0)),
            "alias_collapsed_count": int(alias_meta.get("collapsed_count", 0)),
            "alias_samples": alias_meta.get("samples", []),
            "alias_hint_count": int(segment_meta.get("alias_hint_count", 0)),
            "alias_hint_pool_size": len(alias_prompt_hints_pool),
            "coref_enabled": False,
            "coref_applied": False,
            "coref_reason": "disabled_scheme2",
            "coref_canonical": _resolve_anchor_canonical(anchor, alias_map),
            "coref_replacements": 0,
            "coref_segment_meta": segment_meta,
            "source_ref": source_ref,
            "current_chapter_index": graph_current_chapter if graph_current_chapter > 0 else None,
            "facts_preview": [
                {
                    "source": fact["source_entity"],
                    "relation": fact["relation"],
                    "target": fact["target_entity"],
                    "origin": fact.get("origin", "unknown"),
                }
                for fact in merged_candidates[:8]
            ],
        },
        fact_keys,
    )


def process_graph_sync_for_action(
    db: Session,
    action: ChatAction,
    *,
    project_id: int,
    action_type: str,
    payload: dict,
    operator_id: str,
    mutation_id: str = "",
    expected_version: int = 0,
    job_idempotency_key: str = "",
    sync_mode: str = "async",
) -> tuple[dict | None, list[str]]:
    db.expire_all()
    latest_action = db.get(ChatAction, action.id)
    if not latest_action:
        return None, []

    current_meta = _graph_sync_meta(latest_action)
    current_mutation_id = str(current_meta.get("mutation_id") or "")
    is_stale, stale_reason = _is_graph_job_stale(
        latest_action,
        mutation_id=mutation_id,
        expected_version=expected_version,
    )
    if is_stale:
        create_action_audit_log(
            db=db,
            action_id=latest_action.id,
            event_type="graph_skipped",
            operator_id=operator_id,
            event_payload={
                "reason": stale_reason or "stale_before_write",
                "mutation_id": mutation_id or current_mutation_id,
                "expected_version": expected_version,
                "metric": "graph_skipped_stale",
            },
        )
        return None, []

    current_status = str(current_meta.get("status") or "")
    if current_status in {"synced", "no_facts"} and (not mutation_id or mutation_id == current_mutation_id):
        return current_meta, []

    graph_sync, fact_keys = _sync_graph_for_action(
        db,
        latest_action.id,
        project_id=project_id,
        action_type=action_type,
        payload=payload,
    )

    db.expire_all()
    post_action = db.get(ChatAction, latest_action.id)
    if not post_action:
        if fact_keys:
            delete_neo4j_graph_facts(project_id, fact_keys)
        return graph_sync, []

    post_meta = _graph_sync_meta(post_action)
    post_mutation_id = str(post_meta.get("mutation_id") or "")
    post_expected_raw = post_meta.get("expected_version")
    post_expected_version = int(post_expected_raw) if isinstance(post_expected_raw, int) else 0
    post_stale, post_stale_reason = _is_graph_job_stale(
        post_action,
        mutation_id=mutation_id,
        expected_version=expected_version,
    )
    if post_stale:
        deleted = delete_neo4j_graph_facts(project_id, fact_keys) if fact_keys else 0
        create_action_audit_log(
            db=db,
            action_id=post_action.id,
            event_type="graph_compensated",
            operator_id=operator_id,
            event_payload={
                "reason": post_stale_reason or "stale_or_undone_after_write",
                "mutation_id": mutation_id or post_mutation_id,
                "expected_version": expected_version or post_expected_version,
                "requested_delete": len(fact_keys),
                "deleted": deleted,
                "metric": "graph_compensated",
            },
        )
        return (
            {
                "status": "compensated",
                "mutation_id": mutation_id or post_mutation_id,
                "expected_version": expected_version or post_expected_version,
                "mode": sync_mode,
                "written_count": len(fact_keys),
                "compensated_delete_count": deleted,
            },
            [],
        )

    db.expire_all()
    write_action = db.get(ChatAction, latest_action.id)
    if not write_action:
        deleted = delete_neo4j_graph_facts(project_id, fact_keys) if fact_keys else 0
        return (
            {
                "status": "compensated",
                "mutation_id": mutation_id or post_mutation_id,
                "expected_version": expected_version or post_expected_version,
                "mode": sync_mode,
                "written_count": len(fact_keys),
                "compensated_delete_count": deleted,
            },
            [],
        )

    write_meta = _graph_sync_meta(write_action)
    write_mutation_id = str(write_meta.get("mutation_id") or "")
    write_expected_raw = write_meta.get("expected_version")
    write_expected_version = int(write_expected_raw) if isinstance(write_expected_raw, int) else 0
    write_stale, write_stale_reason = _is_graph_job_stale(
        write_action,
        mutation_id=mutation_id,
        expected_version=expected_version,
    )
    if write_stale:
        deleted = delete_neo4j_graph_facts(project_id, fact_keys) if fact_keys else 0
        create_action_audit_log(
            db=db,
            action_id=write_action.id,
            event_type="graph_compensated",
            operator_id=operator_id,
            event_payload={
                "reason": write_stale_reason or "stale_before_commit",
                "mutation_id": mutation_id or write_mutation_id,
                "expected_version": expected_version or write_expected_version,
                "requested_delete": len(fact_keys),
                "deleted": deleted,
                "metric": "graph_compensated",
            },
        )
        return (
            {
                "status": "compensated",
                "mutation_id": mutation_id or write_mutation_id,
                "expected_version": expected_version or write_expected_version,
                "mode": sync_mode,
                "written_count": len(fact_keys),
                "compensated_delete_count": deleted,
            },
            [],
        )

    mutation_id_final = mutation_id or write_mutation_id or post_mutation_id
    expected_version_final = expected_version or write_expected_version or post_expected_version
    if graph_sync:
        write_action.apply_result = {
            **(write_action.apply_result or {}),
            "graph_sync": {
                **graph_sync,
                "mode": sync_mode,
                "mutation_id": mutation_id_final,
                "expected_version": expected_version_final,
                "job_idempotency_key": job_idempotency_key or str(write_meta.get("job_idempotency_key") or ""),
            },
        }
    if fact_keys:
        existing_fact_keys_raw = (write_action.undo_payload or {}).get("graph_fact_keys")
        existing_fact_keys = (
            [str(item).strip() for item in existing_fact_keys_raw if str(item).strip()]
            if isinstance(existing_fact_keys_raw, list)
            else []
        )
        merged_fact_keys = list(dict.fromkeys([*existing_fact_keys, *fact_keys]))
        write_action.undo_payload = {**(write_action.undo_payload or {}), "graph_fact_keys": merged_fact_keys}

    db.add(write_action)
    db.commit()
    db.refresh(write_action)

    status = str((graph_sync or {}).get("status") or "")
    event_type = "graph_synced" if status == "synced" else ("graph_skipped" if status == "no_facts" else "graph_degraded")
    create_action_audit_log(
        db=db,
        action_id=write_action.id,
        event_type=event_type,
        operator_id=operator_id,
        event_payload={
            "status": status or "unknown",
            "fact_count": len(fact_keys),
            "source": "graph_sync_pipeline",
            "mode": sync_mode,
            "mutation_id": mutation_id_final,
            "expected_version": expected_version_final,
            "job_idempotency_key": job_idempotency_key,
            "provenance": (write_action.apply_result or {}).get("provenance", {}),
            "metric": "graph_synced"
            if event_type == "graph_synced"
            else ("graph_no_facts" if event_type == "graph_skipped" else "graph_degraded"),
        },
    )
    if (
        status == "synced"
        and project_id > 0
        and settings.entity_merge_scan_enabled
        and settings.entity_merge_scan_auto_enqueue
    ):
        interval_seconds = max(int(settings.entity_merge_scan_enqueue_interval_seconds), 30)
        bucket = int(time.time() // interval_seconds)
        scan_job_key = f"entity-merge-scan:{project_id}:{bucket}"
        scan_queued = enqueue_entity_merge_scan_job(
            project_id,
            operator_id=operator_id or "system-entity-merge",
            reason="graph_sync_followup",
            idempotency_key=scan_job_key,
            attempt=0,
            db=db,
        )
        if scan_queued:
            create_action_audit_log(
                db=db,
                action_id=write_action.id,
                event_type="entity_merge_scan_queued",
                operator_id=operator_id,
                event_payload={
                    "project_id": project_id,
                    "queue": settings.entity_merge_scan_queue_name,
                    "idempotency_key": scan_job_key,
                    "reason": "graph_sync_followup",
                },
            )
    return graph_sync, fact_keys


def apply_action_effects(db: Session, action: ChatAction) -> ChatAction:
    project_id = _project_id_for_action(db, action)
    payload_raw = action.payload or {}
    provenance_raw = payload_raw.get("_provenance") if isinstance(payload_raw.get("_provenance"), dict) else {}
    payload = {key: value for key, value in payload_raw.items() if key != "_provenance"}
    graph_current_chapter = 0
    try:
        graph_current_chapter = int(provenance_raw.get("current_chapter_index") or 0)
    except Exception:
        graph_current_chapter = 0
    if graph_current_chapter > 0:
        payload["_graph_current_chapter"] = graph_current_chapter
    atype = action.action_type

    if atype == "setting.upsert":
        key = _setting_key_from_payload(payload)
        graph_anchor = key.replace("设定", "").strip() or key
        payload["_graph_anchor"] = graph_anchor
        value = payload.get("value", payload.get("content"))
        aliases_in_payload = "aliases" in payload
        aliases_from_payload = _normalize_aliases_payload(payload.get("aliases"))
        if value is None:
            raise ValueError("setting.upsert requires value/content")
        if not isinstance(value, dict):
            raise ValueError("setting value/content must be object")

        stmt = select(SettingEntry).where(SettingEntry.project_id == project_id, SettingEntry.key == key)
        existing = db.exec(stmt).first()
        if existing:
            next_aliases = (
                aliases_from_payload
                if aliases_in_payload
                else _normalize_aliases_payload(getattr(existing, "aliases", []))
            )
            if not next_aliases:
                next_aliases = _normalize_aliases_payload(_extract_aliases_from_content(value))
            before = {
                "exists": True,
                "value": copy.deepcopy(existing.value),
                "aliases": copy.deepcopy(existing.aliases or []),
            }
            existing.value = value
            existing.aliases = next_aliases
            existing.updated_at = _utc_now()
            db.add(existing)
        else:
            next_aliases = aliases_from_payload or _normalize_aliases_payload(_extract_aliases_from_content(value))
            before = {"exists": False}
            db.add(SettingEntry(project_id=project_id, key=key, value=value, aliases=next_aliases))

        action.apply_result = {"project_id": project_id, "key": key, "value": value, "aliases": next_aliases}
        action.undo_payload = {
            "kind": "setting.upsert",
            "project_id": project_id,
            "key": key,
            "before": before,
        }

    elif atype == "setting.delete":
        key = _setting_key_from_payload(payload)
        stmt = select(SettingEntry).where(SettingEntry.project_id == project_id, SettingEntry.key == key)
        existing = db.exec(stmt).first()
        if existing:
            before = {
                "exists": True,
                "value": copy.deepcopy(existing.value),
                "aliases": copy.deepcopy(existing.aliases or []),
            }
            db.delete(existing)
            deleted = True
        else:
            before = {"exists": False}
            deleted = False

        action.apply_result = {"project_id": project_id, "key": key, "deleted": deleted}
        action.undo_payload = {
            "kind": "setting.delete",
            "project_id": project_id,
            "key": key,
            "before": before,
        }

    elif atype == "card.create":
        title = payload.get("title") or "未命名卡片"
        payload["_graph_anchor"] = str(title)
        content = payload.get("content") or {}
        aliases = _normalize_aliases_payload(payload.get("aliases"))
        if not isinstance(content, dict):
            raise ValueError("card.create content must be object")
        if not aliases:
            aliases = _normalize_aliases_payload(_extract_aliases_from_content(content))

        card = StoryCard(project_id=project_id, title=title, content=content, aliases=aliases)
        db.add(card)
        db.flush()
        action.apply_result = {
            "project_id": project_id,
            "card_id": card.id,
            "title": card.title,
            "aliases": aliases,
        }
        action.undo_payload = {"kind": "card.create", "project_id": project_id, "card_id": card.id}

    elif atype == "card.update":
        card_id = payload.get("card_id")
        if not isinstance(card_id, int):
            raise ValueError("card.update requires integer card_id")

        card = db.get(StoryCard, card_id)
        if not card or card.project_id != project_id:
            raise ValueError("card not found in project")

        before = {
            "title": card.title,
            "content": copy.deepcopy(card.content),
            "aliases": copy.deepcopy(card.aliases or []),
        }
        if "title" in payload and isinstance(payload["title"], str):
            card.title = payload["title"]
        if "content" in payload:
            if not isinstance(payload["content"], dict):
                raise ValueError("card.update content must be object")
            merge = bool(payload.get("merge", True))
            card.content = {**(card.content or {}), **payload["content"]} if merge else payload["content"]
        if "aliases" in payload:
            card.aliases = _normalize_aliases_payload(payload.get("aliases"))
        payload["_graph_anchor"] = str(card.title or "")
        payload["_graph_anchor_before"] = str(before.get("title") or "")
        card.updated_at = _utc_now()
        db.add(card)

        action.apply_result = {
            "project_id": project_id,
            "card_id": card.id,
            "title": card.title,
            "aliases": copy.deepcopy(card.aliases or []),
        }
        action.undo_payload = {"kind": "card.update", "project_id": project_id, "card_id": card.id, "before": before}

    elif is_entity_merge_action_type(atype):
        target_card_id = payload.get("target_card_id")
        if not isinstance(target_card_id, int):
            target_card_id = payload.get("canonical_card_id")
        if not isinstance(target_card_id, int):
            target_card_id = payload.get("card_id")
        if not isinstance(target_card_id, int):
            raise ValueError("entity.merge requires integer target_card_id")

        card = db.get(StoryCard, target_card_id)
        if not card or card.project_id != project_id:
            raise ValueError("target card not found in project")

        incoming_aliases = _collect_entity_merge_aliases(payload)
        canonical_token = _normalize_graph_entity_token(str(card.title or ""))
        filtered_aliases = [
            alias for alias in incoming_aliases if _normalize_graph_entity_token(alias) != canonical_token
        ]
        if not filtered_aliases:
            raise ValueError("entity.merge requires at least one alias candidate")

        before = {
            "title": card.title,
            "aliases": copy.deepcopy(card.aliases or []),
        }
        merged_aliases = _normalize_aliases_payload([*(card.aliases or []), *filtered_aliases])
        payload["_graph_anchor"] = str(card.title or "")
        payload["_graph_anchor_before"] = str(card.title or "")
        card.aliases = merged_aliases
        card.updated_at = _utc_now()
        db.add(card)

        action.apply_result = {
            "project_id": project_id,
            "card_id": card.id,
            "title": card.title,
            "aliases": copy.deepcopy(card.aliases or []),
            "merge_aliases_added": filtered_aliases,
            "merge_mode": "aliases_only_manual",
        }
        action.undo_payload = {
            "kind": "entity.merge.aliases",
            "project_id": project_id,
            "card_id": card.id,
            "before": before,
        }

    elif atype == "graph.confirm_candidates":
        source_ref = str(payload.get("source_ref") or "").strip()
        fact_keys_raw = payload.get("fact_keys")
        fact_keys = (
            [str(item).strip() for item in fact_keys_raw if str(item).strip()]
            if isinstance(fact_keys_raw, list)
            else []
        )
        if not source_ref and not fact_keys:
            raise ValueError("graph.confirm_candidates requires source_ref or fact_keys")

        min_confidence_raw = payload.get("min_confidence")
        min_confidence: float | None = None
        if min_confidence_raw is not None:
            try:
                min_confidence = float(min_confidence_raw)
            except Exception:
                raise ValueError("graph.confirm_candidates min_confidence must be number")
            if min_confidence < 0.0 or min_confidence > 1.0:
                raise ValueError("graph.confirm_candidates min_confidence must be between 0 and 1")

        limit_raw = payload.get("limit", 200)
        try:
            limit = max(min(int(limit_raw), 1000), 1)
        except Exception:
            raise ValueError("graph.confirm_candidates limit must be integer")

        promoted_fact_keys = promote_neo4j_candidate_facts(
            project_id,
            fact_keys=fact_keys,
            source_ref=source_ref,
            min_confidence=min_confidence,
            limit=limit,
            current_chapter=graph_current_chapter if graph_current_chapter > 0 else None,
        )
        action.apply_result = {
            "project_id": project_id,
            "source_ref": source_ref or None,
            "requested_fact_keys": fact_keys,
            "requested_min_confidence": min_confidence,
            "limit": limit,
            "promoted_count": len(promoted_fact_keys),
            "promoted_fact_keys": promoted_fact_keys,
        }
        action.undo_payload = {
            "kind": "graph.confirm_candidates",
            "project_id": project_id,
            "promoted_fact_keys": promoted_fact_keys,
        }

    else:
        raise ValueError(f"unsupported action_type: {atype}")

    mutation_version = _bump_project_mutation_version(db, project_id)
    mutation_id = f"m-{project_id}-{mutation_version}-{uuid4().hex[:8]}"
    graph_job_idempotency_key = f"graph-sync:{action.id}:{mutation_id}"
    lifecycle_job_idempotency_key = f"index-lifecycle:{action.id}:{mutation_id}"
    action_provenance = {
        "source_action_id": action.id,
        "operator_id": action.operator_id,
        "source": provenance_raw.get("source") or "unknown",
        "current_chapter_index": graph_current_chapter if graph_current_chapter > 0 else None,
        "resolver_order": provenance_raw.get("resolver_order"),
        "providers": provenance_raw.get("providers", {}),
        "rag_route": provenance_raw.get("rag_route", {}),
        "quality_gate": provenance_raw.get("quality_gate", {}),
        "evidence_summary": provenance_raw.get("evidence_summary", {}),
        "evidence_refs": provenance_raw.get("evidence_refs", {}),
        "mutation_id": mutation_id,
        "expected_version": mutation_version,
    }

    action.status = "applied"
    action.applied_at = _utc_now()
    action.apply_result = {
        **(action.apply_result or {}),
        "provenance": action_provenance,
    }
    action.undo_payload = {
        **(action.undo_payload or {}),
        "provenance": action_provenance,
    }
    if atype in {"setting.upsert", "card.create", "card.update"}:
        action.apply_result = {
            **(action.apply_result or {}),
            "graph_sync": {
                "status": "pending_queue",
                "mode": "pending",
                "mutation_id": mutation_id,
                "expected_version": mutation_version,
                "job_idempotency_key": graph_job_idempotency_key,
            },
        }
    if atype == "setting.delete" and settings.index_lifecycle_enabled:
        action.apply_result = {
            **(action.apply_result or {}),
            "index_lifecycle": {
                "status": "pending_queue",
                "mode": "pending",
                "reason": "setting_delete",
                "mutation_id": mutation_id,
                "expected_version": mutation_version,
                "job_idempotency_key": lifecycle_job_idempotency_key,
            },
        }
    db.add(action)
    db.flush()

    if atype in {"setting.upsert", "card.create", "card.update"}:
        if settings.graph_sync_async_enabled:
            queued = enqueue_graph_sync_job(
                action.id,
                project_id=project_id,
                action_type=atype,
                payload=payload,
                operator_id=action.operator_id,
                mutation_id=mutation_id,
                expected_version=mutation_version,
                idempotency_key=graph_job_idempotency_key,
                attempt=0,
                db=db,
            )
            if queued:
                action.apply_result = {
                    **(action.apply_result or {}),
                    "graph_sync": {
                        "status": "queued",
                        "mode": "async",
                        "queue": settings.graph_sync_queue_name,
                        "mutation_id": mutation_id,
                        "expected_version": mutation_version,
                        "job_idempotency_key": graph_job_idempotency_key,
                    },
                }
                db.add(action)
                db.commit()
                db.refresh(action)
                create_action_audit_log(
                    db=db,
                    action_id=action.id,
                    event_type="graph_queued",
                    operator_id=action.operator_id,
                    event_payload={
                        "queue": settings.graph_sync_queue_name,
                        "mode": "async",
                        "mutation_id": mutation_id,
                        "expected_version": mutation_version,
                        "job_idempotency_key": graph_job_idempotency_key,
                    },
                )
            else:
                graph_sync, _ = process_graph_sync_for_action(
                    db=db,
                    action=action,
                    project_id=project_id,
                    action_type=atype,
                    payload=payload,
                    operator_id=action.operator_id,
                    mutation_id=mutation_id,
                    expected_version=mutation_version,
                    job_idempotency_key=graph_job_idempotency_key,
                    sync_mode="sync_fallback",
                )
                if graph_sync:
                    action.apply_result = {
                        **(action.apply_result or {}),
                        "graph_sync": {
                            **graph_sync,
                            "mode": "sync_fallback",
                            "mutation_id": mutation_id,
                            "expected_version": mutation_version,
                            "job_idempotency_key": graph_job_idempotency_key,
                        },
                    }
                    db.add(action)
                    db.commit()
                    db.refresh(action)
        else:
            process_graph_sync_for_action(
                db=db,
                action=action,
                project_id=project_id,
                action_type=atype,
                payload=payload,
                operator_id=action.operator_id,
                mutation_id=mutation_id,
                expected_version=mutation_version,
                job_idempotency_key=graph_job_idempotency_key,
                sync_mode="sync_inline",
            )

    if atype == "setting.delete" and settings.index_lifecycle_enabled:
        queued = enqueue_index_lifecycle_job(
            project_id=project_id,
            operator_id=action.operator_id,
            reason="setting_delete",
            action_id=action.id,
            mutation_id=mutation_id,
            expected_version=mutation_version,
            idempotency_key=lifecycle_job_idempotency_key,
            lifecycle_slot="default",
            attempt=0,
            db=db,
        )
        if queued:
            action.apply_result = {
                **(action.apply_result or {}),
                "index_lifecycle": {
                    "status": "queued",
                    "mode": "async",
                    "queue": settings.index_lifecycle_queue_name,
                    "reason": "setting_delete",
                    "mutation_id": mutation_id,
                    "expected_version": mutation_version,
                    "job_idempotency_key": lifecycle_job_idempotency_key,
                },
            }
            db.add(action)
            db.commit()
            db.refresh(action)
            create_action_audit_log(
                db=db,
                action_id=action.id,
                event_type="index_lifecycle_queued",
                operator_id=action.operator_id,
                event_payload={
                    "queue": settings.index_lifecycle_queue_name,
                    "mode": "async",
                    "reason": "setting_delete",
                    "mutation_id": mutation_id,
                    "expected_version": mutation_version,
                    "job_idempotency_key": lifecycle_job_idempotency_key,
                },
            )
        else:
            lifecycle_result = process_index_lifecycle_rebuild(
                db=db,
                project_id=project_id,
                reason="setting_delete_sync_fallback",
                lifecycle_id=mutation_id,
            )
            action.apply_result = {
                **(action.apply_result or {}),
                "index_lifecycle": {
                    "status": "completed",
                    "mode": "sync_fallback",
                    "reason": "setting_delete",
                    "mutation_id": mutation_id,
                    "expected_version": mutation_version,
                    "job_idempotency_key": lifecycle_job_idempotency_key,
                    "result": lifecycle_result,
                },
            }
            db.add(action)
            db.commit()
            db.refresh(action)
            create_action_audit_log(
                db=db,
                action_id=action.id,
                event_type="index_lifecycle_done",
                operator_id=action.operator_id,
                event_payload={
                    "mode": "sync_fallback",
                    "reason": "setting_delete",
                    "mutation_id": mutation_id,
                    "expected_version": mutation_version,
                    "job_idempotency_key": lifecycle_job_idempotency_key,
                    "result": lifecycle_result,
                },
            )
    db.add(action)
    db.commit()
    db.refresh(action)
    return action


def undo_action_effects(db: Session, action: ChatAction) -> ChatAction:
    undo_payload = action.undo_payload or {}
    kind = undo_payload.get("kind")
    project_id = _project_id_for_action(db, action)
    provenance_meta = undo_payload.get("provenance") if isinstance(undo_payload.get("provenance"), dict) else {}
    undo_chapter_index = 0
    try:
        undo_chapter_index = int(provenance_meta.get("current_chapter_index") or 0)
    except Exception:
        undo_chapter_index = 0
    compensation_version = _bump_project_mutation_version(db, project_id)
    compensation_mutation_id = f"undo-{project_id}-{compensation_version}-{uuid4().hex[:8]}"

    if kind == "setting.upsert":
        key = undo_payload.get("key")
        before = undo_payload.get("before", {})
        stmt = select(SettingEntry).where(SettingEntry.project_id == project_id, SettingEntry.key == key)
        existing = db.exec(stmt).first()
        if before.get("exists"):
            value = before.get("value", {})
            aliases = _normalize_aliases_payload(before.get("aliases"))
            if existing:
                existing.value = value
                existing.aliases = aliases
                existing.updated_at = _utc_now()
                db.add(existing)
            else:
                db.add(SettingEntry(project_id=project_id, key=key, value=value, aliases=aliases))
        else:
            if existing:
                db.delete(existing)

    elif kind == "setting.delete":
        key = undo_payload.get("key")
        before = undo_payload.get("before", {})
        stmt = select(SettingEntry).where(SettingEntry.project_id == project_id, SettingEntry.key == key)
        existing = db.exec(stmt).first()
        if before.get("exists"):
            value = before.get("value", {})
            aliases = _normalize_aliases_payload(before.get("aliases"))
            if existing:
                existing.value = value
                existing.aliases = aliases
                existing.updated_at = _utc_now()
                db.add(existing)
            else:
                db.add(SettingEntry(project_id=project_id, key=key, value=value, aliases=aliases))

    elif kind == "card.create":
        card_id = undo_payload.get("card_id")
        card = db.get(StoryCard, card_id)
        if card and card.project_id == project_id:
            db.delete(card)

    elif kind == "card.update":
        card_id = undo_payload.get("card_id")
        before = undo_payload.get("before", {})
        card = db.get(StoryCard, card_id)
        if card and card.project_id == project_id:
            card.title = before.get("title", card.title)
            card.content = before.get("content", card.content)
            card.aliases = _normalize_aliases_payload(before.get("aliases"))
            card.updated_at = _utc_now()
            db.add(card)

    elif kind == "entity.merge.aliases":
        card_id = undo_payload.get("card_id")
        before = undo_payload.get("before", {})
        card = db.get(StoryCard, card_id)
        if card and card.project_id == project_id:
            card.aliases = _normalize_aliases_payload(before.get("aliases"))
            card.updated_at = _utc_now()
            db.add(card)

    elif kind == "graph.confirm_candidates":
        promoted_fact_keys_raw = undo_payload.get("promoted_fact_keys")
        promoted_fact_keys = (
            [str(item).strip() for item in promoted_fact_keys_raw if str(item).strip()]
            if isinstance(promoted_fact_keys_raw, list)
            else []
        )
        reverted = (
            update_neo4j_graph_fact_state(
                project_id,
                promoted_fact_keys,
                to_state="candidate",
                from_state="confirmed",
                current_chapter=undo_chapter_index if undo_chapter_index > 0 else None,
            )
            if promoted_fact_keys
            else 0
        )
        action.apply_result = {
            **(action.apply_result or {}),
            "graph_confirm_undo": {
                "requested": len(promoted_fact_keys),
                "reverted": reverted,
                "compensation_mutation_id": compensation_mutation_id,
                "compensation_version": compensation_version,
            },
        }

    else:
        raise ValueError("undo payload invalid")

    graph_fact_keys_raw = undo_payload.get("graph_fact_keys")
    graph_fact_keys = (
        [str(item).strip() for item in graph_fact_keys_raw if str(item).strip()]
        if isinstance(graph_fact_keys_raw, list)
        else []
    )
    lifecycle_compensation_needed = kind == "setting.delete" and settings.index_lifecycle_enabled
    lifecycle_compensation_idempotency_key = f"index-lifecycle:undo:{action.id}:{compensation_mutation_id}"
    base_apply_result = action.apply_result if isinstance(action.apply_result, dict) else {}
    graph_sync_meta = base_apply_result.get("graph_sync")
    graph_sync_final = (
        {
            **graph_sync_meta,
            "status": "canceled",
            "canceled_by": compensation_mutation_id,
            "compensation_version": compensation_version,
        }
        if isinstance(graph_sync_meta, dict)
        else None
    )
    lifecycle_meta = _index_lifecycle_meta(action)
    lifecycle_default_final = (
        {
            **lifecycle_meta,
            "status": "canceled",
            "canceled_by": compensation_mutation_id,
            "compensation_version": compensation_version,
        }
        if kind == "setting.delete" and isinstance(lifecycle_meta, dict)
        else None
    )
    lifecycle_compensation_pending = (
        {
            "status": "pending_queue",
            "mode": "pending",
            "reason": "undo_setting_delete",
            "mutation_id": compensation_mutation_id,
            "expected_version": compensation_version,
            "job_idempotency_key": lifecycle_compensation_idempotency_key,
        }
        if lifecycle_compensation_needed
        else None
    )
    if graph_fact_keys:
        deleted = delete_neo4j_graph_facts(
            project_id,
            graph_fact_keys,
            current_chapter=undo_chapter_index if undo_chapter_index > 0 else None,
        )
        next_apply_result = {
            **base_apply_result,
            "graph_undo": {
                "requested": len(graph_fact_keys),
                "deleted": deleted,
                "compensation_mutation_id": compensation_mutation_id,
                "compensation_version": compensation_version,
            },
        }
        if graph_sync_final:
            next_apply_result["graph_sync"] = graph_sync_final
        if lifecycle_default_final:
            next_apply_result["index_lifecycle"] = lifecycle_default_final
        if lifecycle_compensation_pending:
            next_apply_result["index_lifecycle_compensation"] = lifecycle_compensation_pending
        action.apply_result = next_apply_result
    else:
        next_apply_result = {
            **base_apply_result,
            "graph_undo": {
                "requested": 0,
                "deleted": 0,
                "compensation_mutation_id": compensation_mutation_id,
                "compensation_version": compensation_version,
            },
        }
        if graph_sync_final:
            next_apply_result["graph_sync"] = graph_sync_final
        if lifecycle_default_final:
            next_apply_result["index_lifecycle"] = lifecycle_default_final
        if lifecycle_compensation_pending:
            next_apply_result["index_lifecycle_compensation"] = lifecycle_compensation_pending
        action.apply_result = next_apply_result

    action.status = "undone"
    action.undone_at = _utc_now()
    db.add(action)
    db.flush()

    if lifecycle_compensation_needed:
        queued = enqueue_index_lifecycle_job(
            project_id=project_id,
            operator_id=action.operator_id,
            reason="undo_setting_delete",
            action_id=action.id,
            mutation_id=compensation_mutation_id,
            expected_version=compensation_version,
            idempotency_key=lifecycle_compensation_idempotency_key,
            lifecycle_slot="compensation",
            attempt=0,
            db=db,
        )
        if queued:
            action.apply_result = {
                **(action.apply_result or {}),
                "index_lifecycle_compensation": {
                    "status": "queued",
                    "mode": "async",
                    "queue": settings.index_lifecycle_queue_name,
                    "reason": "undo_setting_delete",
                    "mutation_id": compensation_mutation_id,
                    "expected_version": compensation_version,
                    "job_idempotency_key": lifecycle_compensation_idempotency_key,
                },
            }
            db.add(action)
            db.commit()
            db.refresh(action)
            create_action_audit_log(
                db=db,
                action_id=action.id,
                event_type="index_lifecycle_queued",
                operator_id=action.operator_id,
                event_payload={
                    "queue": settings.index_lifecycle_queue_name,
                    "mode": "async",
                    "reason": "undo_setting_delete",
                    "mutation_id": compensation_mutation_id,
                    "expected_version": compensation_version,
                    "job_idempotency_key": lifecycle_compensation_idempotency_key,
                },
            )
        else:
            lifecycle_result = process_index_lifecycle_rebuild(
                db=db,
                project_id=project_id,
                reason="undo_setting_delete_sync_fallback",
                lifecycle_id=compensation_mutation_id,
            )
            action.apply_result = {
                **(action.apply_result or {}),
                "index_lifecycle_compensation": {
                    "status": "completed",
                    "mode": "sync_fallback",
                    "reason": "undo_setting_delete",
                    "mutation_id": compensation_mutation_id,
                    "expected_version": compensation_version,
                    "job_idempotency_key": lifecycle_compensation_idempotency_key,
                    "result": lifecycle_result,
                },
            }
            db.add(action)
            db.commit()
            db.refresh(action)
            create_action_audit_log(
                db=db,
                action_id=action.id,
                event_type="index_lifecycle_done",
                operator_id=action.operator_id,
                event_payload={
                    "mode": "sync_fallback",
                    "reason": "undo_setting_delete",
                    "mutation_id": compensation_mutation_id,
                    "expected_version": compensation_version,
                    "job_idempotency_key": lifecycle_compensation_idempotency_key,
                    "result": lifecycle_result,
                },
            )
    db.add(action)
    db.commit()
    db.refresh(action)
    return action


def _is_internal_setting_key(key: str) -> bool:
    raw = str(key or "").strip().lower()
    if not raw:
        return False
    return any(raw.startswith(prefix) for prefix in _INTERNAL_SETTING_PREFIXES)


def _normalize_model_profile_id(profile_id: str) -> str:
    normalized = str(profile_id or "").strip().lower()
    if not normalized:
        raise ValueError("model profile id is required")
    if normalized == "active":
        raise ValueError("model profile id 'active' is reserved")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", normalized):
        raise ValueError("model profile id must match [a-z0-9][a-z0-9_-]{0,63}")
    return normalized


def _slugify_profile_id(text: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", str(text or "").strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-_")
    if not slug:
        return "profile"
    return slug[:64]


def _normalize_model_profile_provider(provider: str | None) -> str:
    raw = str(provider or "").strip().lower()
    if not raw:
        raise ValueError("provider is required")
    normalized = _MODEL_PROFILE_PROVIDER_ALIASES.get(raw)
    if not normalized:
        raise ValueError("provider must be one of openai_compatible/deepseek/claude/gemini")
    return normalized


def _normalize_optional_text(value: str | None, *, max_len: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:max_len]


def _mask_secret(value: str | None) -> str | None:
    token = str(value or "").strip()
    if not token:
        return None
    if len(token) <= 8:
        return token[:2] + "***"
    return token[:4] + "..." + token[-3:]


def _model_profile_key(profile_id: str) -> str:
    return f"{_MODEL_PROFILE_PREFIX}{profile_id}"


def _get_model_profile_row(db: Session, project_id: int, profile_id: str) -> SettingEntry | None:
    key = _model_profile_key(profile_id)
    stmt = select(SettingEntry).where(SettingEntry.project_id == project_id, SettingEntry.key == key)
    return db.exec(stmt).first()


def _list_model_profile_rows(db: Session, project_id: int) -> list[SettingEntry]:
    stmt = (
        select(SettingEntry)
        .where(
            SettingEntry.project_id == project_id,
            SettingEntry.key.like(f"{_MODEL_PROFILE_PREFIX}%"),
            SettingEntry.key != _MODEL_PROFILE_ACTIVE_KEY,
        )
        .order_by(SettingEntry.id.asc())
    )
    return db.exec(stmt).all()


def _ensure_unique_model_profile_id(db: Session, project_id: int, base_id: str) -> str:
    candidate = _normalize_model_profile_id(base_id)
    suffix = 1
    while _get_model_profile_row(db, project_id, candidate) is not None:
        suffix += 1
        candidate = _normalize_model_profile_id(f"{base_id[:56]}-{suffix}")
    return candidate


def _extract_model_profile_id_from_key(key: str) -> str:
    if not str(key).startswith(_MODEL_PROFILE_PREFIX):
        raise ValueError("invalid model profile key")
    return _normalize_model_profile_id(str(key)[len(_MODEL_PROFILE_PREFIX) :])


def _get_active_model_profile_id(db: Session, project_id: int) -> str | None:
    stmt = select(SettingEntry).where(
        SettingEntry.project_id == project_id,
        SettingEntry.key == _MODEL_PROFILE_ACTIVE_KEY,
    )
    row = db.exec(stmt).first()
    if not row or not isinstance(row.value, dict):
        return None
    profile_id_raw = row.value.get("profile_id")
    if profile_id_raw is None:
        return None
    try:
        return _normalize_model_profile_id(str(profile_id_raw))
    except ValueError:
        return None


def _set_active_model_profile_id(db: Session, project_id: int, profile_id: str | None) -> None:
    stmt = select(SettingEntry).where(
        SettingEntry.project_id == project_id,
        SettingEntry.key == _MODEL_PROFILE_ACTIVE_KEY,
    )
    row = db.exec(stmt).first()
    if not profile_id:
        if row:
            db.delete(row)
        return
    payload = {"profile_id": profile_id}
    if row:
        row.value = payload
        row.updated_at = _utc_now()
        db.add(row)
        return
    db.add(
        SettingEntry(
            project_id=project_id,
            key=_MODEL_PROFILE_ACTIVE_KEY,
            value=payload,
            aliases=["llm-model-profile-active"],
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
    )


def _model_profile_read_dict(row: SettingEntry, *, active_profile_id: str | None) -> dict[str, Any]:
    value = row.value if isinstance(row.value, dict) else {}
    profile_id = _extract_model_profile_id_from_key(str(row.key))
    api_key = str(value.get("api_key", "") or "").strip()
    provider_raw = str(value.get("provider") or "openai_compatible").strip().lower()
    provider = _MODEL_PROFILE_PROVIDER_ALIASES.get(provider_raw, "openai_compatible")
    name = _normalize_optional_text(value.get("name"), max_len=128) if isinstance(value, dict) else None
    base_url = _normalize_optional_text(value.get("base_url"), max_len=512) if isinstance(value, dict) else None
    model = _normalize_optional_text(value.get("model"), max_len=128) if isinstance(value, dict) else None
    return {
        "profile_id": profile_id,
        "name": name,
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "has_api_key": bool(api_key),
        "api_key_masked": _mask_secret(api_key),
        "is_active": profile_id == active_profile_id,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def list_model_profiles(db: Session, project_id: int) -> list[dict[str, Any]]:
    active_profile_id = _get_active_model_profile_id(db, project_id)
    rows = _list_model_profile_rows(db, project_id)
    result = [_model_profile_read_dict(row, active_profile_id=active_profile_id) for row in rows]
    result.sort(key=lambda item: (0 if item.get("is_active") else 1, str(item.get("profile_id") or "")))
    return result


def create_model_profile(
    db: Session,
    *,
    project_id: int,
    operator_id: str,
    profile_id: str | None,
    name: str | None,
    provider: str,
    base_url: str | None,
    api_key: str | None,
    model: str | None,
) -> dict[str, Any]:
    provider_norm = _normalize_model_profile_provider(provider)
    name_norm = _normalize_optional_text(name, max_len=128)
    base_url_norm = _normalize_optional_text(base_url, max_len=512)
    model_norm = _normalize_optional_text(model, max_len=128)
    api_key_norm = _normalize_optional_text(api_key, max_len=512)

    if profile_id:
        profile_id_norm = _normalize_model_profile_id(profile_id)
        if _get_model_profile_row(db, project_id, profile_id_norm) is not None:
            raise ValueError("model profile already exists")
    else:
        seed = name_norm or f"{provider_norm}-{model_norm or 'default'}"
        profile_id_norm = _ensure_unique_model_profile_id(db, project_id, _slugify_profile_id(seed))

    payload = {
        "name": name_norm,
        "provider": provider_norm,
        "base_url": base_url_norm,
        "api_key": api_key_norm or "",
        "model": model_norm,
        "updated_by": str(operator_id or "system")[:128],
        "updated_at": _utc_now().isoformat(),
    }
    row = SettingEntry(
        project_id=project_id,
        key=_model_profile_key(profile_id_norm),
        value=payload,
        aliases=["llm-model-profile"],
        created_at=_utc_now(),
        updated_at=_utc_now(),
    )
    db.add(row)

    if not _get_active_model_profile_id(db, project_id):
        _set_active_model_profile_id(db, project_id, profile_id_norm)

    db.commit()
    db.refresh(row)
    return _model_profile_read_dict(row, active_profile_id=_get_active_model_profile_id(db, project_id))


def update_model_profile(
    db: Session,
    *,
    project_id: int,
    profile_id: str,
    operator_id: str,
    name: str | None,
    provider: str | None,
    base_url: str | None,
    api_key: str | None,
    api_key_supplied: bool,
    model: str | None,
) -> dict[str, Any]:
    profile_id_norm = _normalize_model_profile_id(profile_id)
    row = _get_model_profile_row(db, project_id, profile_id_norm)
    if not row:
        raise ValueError("model profile not found")

    existing = row.value if isinstance(row.value, dict) else {}
    next_provider = (
        _normalize_model_profile_provider(provider)
        if provider is not None
        else _normalize_model_profile_provider(existing.get("provider"))
    )
    next_name = _normalize_optional_text(name, max_len=128) if name is not None else _normalize_optional_text(existing.get("name"), max_len=128)
    next_base_url = (
        _normalize_optional_text(base_url, max_len=512)
        if base_url is not None
        else _normalize_optional_text(existing.get("base_url"), max_len=512)
    )
    next_model = (
        _normalize_optional_text(model, max_len=128)
        if model is not None
        else _normalize_optional_text(existing.get("model"), max_len=128)
    )
    if api_key_supplied:
        next_api_key = _normalize_optional_text(api_key, max_len=512) or ""
    else:
        next_api_key = str(existing.get("api_key", "") or "").strip()[:512]

    row.value = {
        "name": next_name,
        "provider": next_provider,
        "base_url": next_base_url,
        "api_key": next_api_key,
        "model": next_model,
        "updated_by": str(operator_id or "system")[:128],
        "updated_at": _utc_now().isoformat(),
    }
    row.updated_at = _utc_now()
    db.add(row)
    db.commit()
    db.refresh(row)
    return _model_profile_read_dict(row, active_profile_id=_get_active_model_profile_id(db, project_id))


def delete_model_profile(
    db: Session,
    *,
    project_id: int,
    profile_id: str,
) -> str:
    profile_id_norm = _normalize_model_profile_id(profile_id)
    row = _get_model_profile_row(db, project_id, profile_id_norm)
    if not row:
        raise ValueError("model profile not found")
    active_profile_id = _get_active_model_profile_id(db, project_id)
    db.delete(row)
    if active_profile_id == profile_id_norm:
        remaining = _list_model_profile_rows(db, project_id)
        next_active = None
        for item in remaining:
            try:
                candidate = _extract_model_profile_id_from_key(str(item.key))
            except ValueError:
                continue
            if candidate != profile_id_norm:
                next_active = candidate
                break
        _set_active_model_profile_id(db, project_id, next_active)
    db.commit()
    return profile_id_norm


def activate_model_profile(
    db: Session,
    *,
    project_id: int,
    profile_id: str,
) -> dict[str, Any]:
    profile_id_norm = _normalize_model_profile_id(profile_id)
    row = _get_model_profile_row(db, project_id, profile_id_norm)
    if not row:
        raise ValueError("model profile not found")
    _set_active_model_profile_id(db, project_id, profile_id_norm)
    db.commit()
    db.refresh(row)
    return _model_profile_read_dict(row, active_profile_id=profile_id_norm)


def resolve_model_profile_runtime(
    db: Session,
    *,
    project_id: int,
    profile_id: str | None = None,
) -> dict[str, Any] | None:
    resolved_profile_id = _normalize_model_profile_id(profile_id) if profile_id else _get_active_model_profile_id(db, project_id)
    if not resolved_profile_id:
        return None
    row = _get_model_profile_row(db, project_id, resolved_profile_id)
    if not row:
        if profile_id:
            raise ValueError("model profile not found")
        _set_active_model_profile_id(db, project_id, None)
        db.commit()
        return None
    value = row.value if isinstance(row.value, dict) else {}
    provider = _MODEL_PROFILE_PROVIDER_ALIASES.get(
        str(value.get("provider") or "").strip().lower(),
        "openai_compatible",
    )
    return {
        "profile_id": resolved_profile_id,
        "name": _normalize_optional_text(value.get("name"), max_len=128),
        "provider": provider,
        "base_url": _normalize_optional_text(value.get("base_url"), max_len=512),
        "api_key": _normalize_optional_text(value.get("api_key"), max_len=512),
        "model": _normalize_optional_text(value.get("model"), max_len=128),
    }


def list_settings(
    db: Session,
    project_id: int,
    *,
    limit: int | None = None,
    include_internal: bool = False,
) -> Iterable[SettingEntry]:
    stmt = select(SettingEntry).where(SettingEntry.project_id == project_id)
    if not include_internal:
        for prefix in _INTERNAL_SETTING_PREFIXES:
            stmt = stmt.where(~SettingEntry.key.like(f"{prefix}%"))
    stmt = stmt.order_by(SettingEntry.id.asc())
    if limit is not None:
        stmt = stmt.limit(max(int(limit), 1))
    return db.exec(stmt).all()


def list_cards(
    db: Session,
    project_id: int,
    *,
    limit: int | None = None,
) -> Iterable[StoryCard]:
    stmt = select(StoryCard).where(StoryCard.project_id == project_id).order_by(StoryCard.id.asc())
    if limit is not None:
        stmt = stmt.limit(max(int(limit), 1))
    return db.exec(stmt).all()


def _normalize_prompt_template_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("prompt template name is required")
    return cleaned[:128]


def _normalize_prompt_text(value: str | None, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _normalize_prompt_guard_mode(value: str | None) -> str:
    mode = str(value or "warn").strip().lower()
    if mode in {"off", "warn", "block", "monitor"}:
        return mode
    return "warn"


def _prompt_template_guard_report(system_prompt: str, user_prompt_prefix: str) -> dict[str, Any]:
    corpus = f"{system_prompt}\n{user_prompt_prefix}".lower()
    risky_terms = [term.strip().lower() for term in settings.prompt_template_guard_terms if str(term).strip()]
    term_hits = [term for term in risky_terms if term in corpus]
    pattern_rules: list[tuple[str, str, float]] = [
        (
            "instruction_override",
            r"(ignore|disregard|bypass|override)\s+(all\s+)?(previous|above|prior|system|rules?)",
            0.34,
        ),
        (
            "instruction_override_zh",
            r"(忽略|无视|绕过|覆盖).{0,8}(规则|指令|系统|上文|提示)",
            0.34,
        ),
        (
            "policy_exfiltration",
            r"(reveal|show|print|leak|expose).{0,30}(system prompt|developer|policy|secret|api\s*key)",
            0.4,
        ),
        (
            "policy_exfiltration_zh",
            r"(泄露|输出|显示|暴露).{0,30}(系统提示|开发者消息|策略|密钥|token)",
            0.4,
        ),
        (
            "role_escalation",
            r"(you are now|act as|pretend to be|现在你是|你将扮演)",
            0.22,
        ),
    ]
    pattern_hits: list[str] = []
    pattern_score = 0.0
    for name, pattern, weight in pattern_rules:
        if re.search(pattern, corpus, flags=re.IGNORECASE):
            pattern_hits.append(name)
            pattern_score += weight

    term_score = min(len(term_hits) * 0.12, 0.56)
    max_terms = max(int(settings.prompt_template_guard_max_risk_terms), 1)
    term_density_bonus = 0.18 if len(term_hits) >= max_terms else 0.0
    hybrid_bonus = 0.08 if term_hits and pattern_hits else 0.0
    risk_score = min(1.0, term_score + pattern_score + term_density_bonus + hybrid_bonus)

    mode = _normalize_prompt_guard_mode(settings.prompt_template_guard_mode)
    warn_threshold = max(min(float(settings.prompt_template_guard_warn_score), 1.0), 0.0)
    block_threshold = max(min(float(settings.prompt_template_guard_block_score), 1.0), warn_threshold + 0.05)
    if mode == "off":
        action = "allow"
    elif mode == "block" and risk_score >= block_threshold:
        action = "block"
    elif risk_score >= warn_threshold:
        action = "warn"
    else:
        action = "allow"

    return {
        "mode": mode,
        "action": action,
        "risk_score": round(risk_score, 4),
        "warn_threshold": round(warn_threshold, 4),
        "block_threshold": round(block_threshold, 4),
        "term_hits": term_hits[:8],
        "pattern_hits": pattern_hits[:8],
        "term_count": len(term_hits),
    }


def _validate_prompt_template_security(system_prompt: str, user_prompt_prefix: str) -> None:
    if not settings.prompt_template_guard_enabled:
        return
    if not f"{system_prompt}\n{user_prompt_prefix}".strip():
        return
    report = _prompt_template_guard_report(system_prompt, user_prompt_prefix)
    if report.get("action") == "block":
        terms = ", ".join(report.get("term_hits", [])[:4]) or "pattern-only"
        patterns = ", ".join(report.get("pattern_hits", [])[:3]) or "none"
        raise ValueError(
            "prompt template blocked by security guard: "
            f"score={report.get('risk_score')} terms={terms} patterns={patterns}"
        )
    if report.get("action") == "warn":
        _LOGGER.warning(
            "prompt template guard warning score=%s mode=%s terms=%s patterns=%s",
            report.get("risk_score"),
            report.get("mode"),
            ",".join(report.get("term_hits", [])[:6]),
            ",".join(report.get("pattern_hits", [])[:4]),
        )


def _normalize_prompt_setting_keys(value: list[str] | None) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for raw in value:
        key = str(raw or "").strip()
        if not key:
            continue
        if key in normalized:
            continue
        normalized.append(key[:191])
    return normalized[:200]


def _normalize_prompt_card_ids(value: list[int] | None) -> list[int]:
    if not isinstance(value, list):
        return []
    normalized: list[int] = []
    for raw in value:
        try:
            card_id = int(raw)
        except Exception:
            continue
        if card_id <= 0 or card_id in normalized:
            continue
        normalized.append(card_id)
    return normalized[:200]


def list_prompt_templates(db: Session, project_id: int) -> Iterable[PromptTemplate]:
    stmt = select(PromptTemplate).where(PromptTemplate.project_id == project_id).order_by(PromptTemplate.id.asc())
    return db.exec(stmt).all()


def get_prompt_template(db: Session, project_id: int, template_id: int) -> PromptTemplate | None:
    stmt = select(PromptTemplate).where(
        PromptTemplate.project_id == project_id,
        PromptTemplate.id == template_id,
    )
    return db.exec(stmt).first()


def _prompt_template_name_conflict(
    db: Session,
    *,
    project_id: int,
    name: str,
    exclude_template_id: int | None = None,
) -> bool:
    stmt = select(PromptTemplate).where(
        PromptTemplate.project_id == project_id,
        PromptTemplate.name == name,
    )
    row = db.exec(stmt).first()
    if not row:
        return False
    if exclude_template_id is not None and int(getattr(row, "id", 0) or 0) == int(exclude_template_id):
        return False
    return True


def _next_prompt_template_revision_version(db: Session, template_id: int) -> int:
    stmt = select(PromptTemplateRevision.version).where(PromptTemplateRevision.template_id == template_id)
    rows = db.exec(stmt).all()
    if not rows:
        return 1
    return max(int(item) for item in rows) + 1


def _append_prompt_template_revision(
    db: Session,
    *,
    template: PromptTemplate,
    operator_id: str,
    source: str,
) -> PromptTemplateRevision:
    if template.id is None:
        raise ValueError("prompt template id missing")
    revision = PromptTemplateRevision(
        template_id=template.id,
        project_id=template.project_id,
        version=_next_prompt_template_revision_version(db, template.id),
        name=template.name,
        system_prompt=template.system_prompt,
        user_prompt_prefix=template.user_prompt_prefix,
        knowledge_setting_keys=_normalize_prompt_setting_keys(template.knowledge_setting_keys),
        knowledge_card_ids=_normalize_prompt_card_ids(template.knowledge_card_ids),
        operator_id=(operator_id or "system").strip() or "system",
        source=source[:32],
        created_at=_utc_now(),
    )
    db.add(revision)
    return revision


def list_prompt_template_revisions(
    db: Session,
    *,
    project_id: int,
    template_id: int,
    limit: int = 20,
) -> Iterable[PromptTemplateRevision]:
    template = get_prompt_template(db, project_id, template_id)
    if not template:
        raise ValueError("prompt template not found")
    stmt = (
        select(PromptTemplateRevision)
        .where(
            PromptTemplateRevision.project_id == project_id,
            PromptTemplateRevision.template_id == template_id,
        )
        .order_by(PromptTemplateRevision.version.desc())
        .limit(limit)
    )
    return db.exec(stmt).all()


def create_prompt_template(
    db: Session,
    *,
    project_id: int,
    name: str,
    system_prompt: str,
    user_prompt_prefix: str,
    knowledge_setting_keys: list[str],
    knowledge_card_ids: list[int],
    operator_id: str,
) -> PromptTemplate:
    name_norm = _normalize_prompt_template_name(name)
    if _prompt_template_name_conflict(db, project_id=project_id, name=name_norm):
        raise ValueError("prompt template name already exists")
    normalized_system_prompt = _normalize_prompt_text(system_prompt, max_chars=40000)
    normalized_user_prefix = _normalize_prompt_text(user_prompt_prefix, max_chars=20000)
    _validate_prompt_template_security(normalized_system_prompt, normalized_user_prefix)

    row = PromptTemplate(
        project_id=project_id,
        name=name_norm,
        system_prompt=normalized_system_prompt,
        user_prompt_prefix=normalized_user_prefix,
        knowledge_setting_keys=_normalize_prompt_setting_keys(knowledge_setting_keys),
        knowledge_card_ids=_normalize_prompt_card_ids(knowledge_card_ids),
        created_at=_utc_now(),
        updated_at=_utc_now(),
    )
    db.add(row)
    db.flush()
    _append_prompt_template_revision(
        db,
        template=row,
        operator_id=operator_id,
        source="create",
    )
    db.commit()
    db.refresh(row)
    return row


def update_prompt_template(
    db: Session,
    *,
    project_id: int,
    template_id: int,
    name: str,
    system_prompt: str,
    user_prompt_prefix: str,
    knowledge_setting_keys: list[str],
    knowledge_card_ids: list[int],
    operator_id: str,
) -> PromptTemplate:
    row = get_prompt_template(db, project_id, template_id)
    if not row:
        raise ValueError("prompt template not found")
    name_norm = _normalize_prompt_template_name(name)
    if _prompt_template_name_conflict(
        db,
        project_id=project_id,
        name=name_norm,
        exclude_template_id=int(getattr(row, "id", 0) or 0),
    ):
        raise ValueError("prompt template name already exists")
    normalized_system_prompt = _normalize_prompt_text(system_prompt, max_chars=40000)
    normalized_user_prefix = _normalize_prompt_text(user_prompt_prefix, max_chars=20000)
    _validate_prompt_template_security(normalized_system_prompt, normalized_user_prefix)

    row.name = name_norm
    row.system_prompt = normalized_system_prompt
    row.user_prompt_prefix = normalized_user_prefix
    row.knowledge_setting_keys = _normalize_prompt_setting_keys(knowledge_setting_keys)
    row.knowledge_card_ids = _normalize_prompt_card_ids(knowledge_card_ids)
    row.updated_at = _utc_now()
    db.add(row)
    _append_prompt_template_revision(
        db,
        template=row,
        operator_id=operator_id,
        source="save",
    )
    db.commit()
    db.refresh(row)
    return row


def rollback_prompt_template(
    db: Session,
    *,
    project_id: int,
    template_id: int,
    target_version: int,
    operator_id: str,
) -> PromptTemplate:
    row = get_prompt_template(db, project_id, template_id)
    if not row:
        raise ValueError("prompt template not found")
    stmt = select(PromptTemplateRevision).where(
        PromptTemplateRevision.project_id == project_id,
        PromptTemplateRevision.template_id == template_id,
        PromptTemplateRevision.version == target_version,
    )
    target = db.exec(stmt).first()
    if not target:
        raise ValueError(f"target_version {target_version} not found")

    normalized_system_prompt = _normalize_prompt_text(target.system_prompt, max_chars=40000)
    normalized_user_prefix = _normalize_prompt_text(target.user_prompt_prefix, max_chars=20000)
    _validate_prompt_template_security(normalized_system_prompt, normalized_user_prefix)

    row.name = target.name
    row.system_prompt = normalized_system_prompt
    row.user_prompt_prefix = normalized_user_prefix
    row.knowledge_setting_keys = _normalize_prompt_setting_keys(target.knowledge_setting_keys)
    row.knowledge_card_ids = _normalize_prompt_card_ids(target.knowledge_card_ids)
    row.updated_at = _utc_now()
    db.add(row)
    _append_prompt_template_revision(
        db,
        template=row,
        operator_id=operator_id,
        source="rollback",
    )
    db.commit()
    db.refresh(row)
    return row


def delete_prompt_template(db: Session, project_id: int, template_id: int) -> int:
    row = get_prompt_template(db, project_id, template_id)
    if not row:
        raise ValueError("prompt template not found")
    deleted_id = int(getattr(row, "id", 0) or 0)
    rev_stmt = select(PromptTemplateRevision).where(
        PromptTemplateRevision.project_id == project_id,
        PromptTemplateRevision.template_id == template_id,
    )
    revisions = db.exec(rev_stmt).all()
    for revision in revisions:
        db.delete(revision)
    db.delete(row)
    db.commit()
    return deleted_id


def _default_volume_title(volume_index: int) -> str:
    return f"第{volume_index}卷"


def _normalize_volume_title(title: str | None, volume_index: int) -> str:
    cleaned = (title or "").strip()
    if not cleaned:
        return _default_volume_title(volume_index)
    return cleaned[:255]


def _normalize_volume_outline(outline: str | None) -> str:
    text = str(outline or "").strip()
    if len(text) <= 200000:
        return text
    return text[:200000]


def _next_project_volume_index(db: Session, project_id: int) -> int:
    stmt = select(ProjectVolume.volume_index).where(ProjectVolume.project_id == project_id)
    rows = db.exec(stmt).all()
    if not rows:
        return 1
    return max(int(item) for item in rows) + 1


def _insert_project_volume(
    db: Session,
    *,
    project_id: int,
    volume_index: int,
    title: str,
    outline: str,
) -> ProjectVolume:
    row = ProjectVolume(
        project_id=project_id,
        volume_index=volume_index,
        title=title,
        outline=outline,
        created_at=_utc_now(),
        updated_at=_utc_now(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _ordered_project_volumes(db: Session, project_id: int) -> list[ProjectVolume]:
    stmt = select(ProjectVolume).where(ProjectVolume.project_id == project_id).order_by(ProjectVolume.volume_index.asc())
    return db.exec(stmt).all()


def _reindex_project_volumes(db: Session, project_id: int) -> None:
    rows = _ordered_project_volumes(db, project_id)
    for idx, row in enumerate(rows, start=1):
        if int(row.volume_index) == idx:
            continue
        row.volume_index = idx
        row.title = _normalize_volume_title(row.title, idx)
        row.updated_at = _utc_now()
        db.add(row)


def list_project_volumes(db: Session, project_id: int) -> Iterable[ProjectVolume]:
    rows = _ordered_project_volumes(db, project_id)
    if rows:
        return rows
    volume = _insert_project_volume(
        db,
        project_id=project_id,
        volume_index=1,
        title=_default_volume_title(1),
        outline="",
    )
    return [volume]


def get_project_volume(db: Session, project_id: int, volume_id: int) -> ProjectVolume | None:
    stmt = select(ProjectVolume).where(
        ProjectVolume.project_id == project_id,
        ProjectVolume.id == volume_id,
    )
    return db.exec(stmt).first()


def create_project_volume(
    db: Session,
    *,
    project_id: int,
    title: str | None,
    outline: str | None,
) -> ProjectVolume:
    volume_index = _next_project_volume_index(db, project_id)
    return _insert_project_volume(
        db,
        project_id=project_id,
        volume_index=volume_index,
        title=_normalize_volume_title(title, volume_index),
        outline=_normalize_volume_outline(outline),
    )


def update_project_volume(
    db: Session,
    *,
    project_id: int,
    volume_id: int,
    title: str,
    outline: str,
) -> ProjectVolume:
    row = get_project_volume(db, project_id, volume_id)
    if not row:
        raise ValueError("volume not found")
    row.title = _normalize_volume_title(title, int(row.volume_index))
    row.outline = _normalize_volume_outline(outline)
    row.updated_at = _utc_now()
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def delete_project_volume(
    db: Session,
    *,
    project_id: int,
    volume_id: int,
) -> tuple[int, int]:
    rows = _ordered_project_volumes(db, project_id)
    if not rows:
        raise ValueError("volume not found")
    target_pos = next((idx for idx, item in enumerate(rows) if int(item.id or 0) == int(volume_id)), -1)
    if target_pos < 0:
        raise ValueError("volume not found")
    if len(rows) <= 1:
        raise ValueError("至少保留一个卷")

    target = rows[target_pos]
    fallback = rows[0] if target_pos > 0 else rows[1]
    if target.id is None or fallback.id is None:
        raise ValueError("invalid volume id")

    chapter_stmt = select(ProjectChapter).where(
        ProjectChapter.project_id == project_id,
        ProjectChapter.volume_id == target.id,
    )
    chapters = db.exec(chapter_stmt).all()
    for chapter in chapters:
        chapter.volume_id = fallback.id
        db.add(chapter)

    db.delete(target)
    db.flush()
    _reindex_project_volumes(db, project_id)
    db.commit()
    return int(target.id), int(fallback.id)


def _fallback_consolidated_facts(chapters: list[ProjectChapter], *, max_facts: int) -> list[str]:
    facts: list[str] = []
    for chapter in chapters:
        title = str(getattr(chapter, "title", "") or "").strip() or f"Chapter {getattr(chapter, 'chapter_index', '')}"
        content = str(getattr(chapter, "content", "") or "").strip()
        summary = content[:220] if content else "（无正文）"
        facts.append(f"{title}: {summary}")
        if len(facts) >= max_facts:
            break
    return facts


def _call_volume_memory_consolidation_llm(
    *,
    volume_title: str,
    volume_outline: str,
    chapters: list[ProjectChapter],
    max_facts: int,
) -> tuple[list[str], str]:
    model = str(settings.lightrag_llm_model or "").strip()
    base_url = str(settings.lightrag_llm_base_url or "").strip()
    api_key = str(settings.lightrag_llm_api_key or "").strip()
    if not settings.memory_consolidation_enabled:
        return _fallback_consolidated_facts(chapters, max_facts=max_facts), "disabled"
    if not (model and base_url and api_key):
        return _fallback_consolidated_facts(chapters, max_facts=max_facts), "fallback_missing_lightrag_llm"

    endpoint = base_url.rstrip("/") + "/chat/completions"
    chapter_payload = [
        {
            "chapter_index": int(getattr(item, "chapter_index", 0) or 0),
            "title": str(getattr(item, "title", "") or ""),
            "content_preview": str(getattr(item, "content", "") or "")[: settings.memory_consolidation_preview_chars],
        }
        for item in chapters
    ]
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是小说记忆固化器。请把一整卷内容提炼为高密度、可检索、可验证的事实。"
                    "输出 JSON，格式: {\"facts\":[\"...\"],\"notes\":\"...\"}。"
                    "facts 必须是陈述句，禁止编造。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "volume_title": volume_title,
                        "volume_outline": volume_outline[:1000],
                        "chapters": chapter_payload,
                        "max_facts": max_facts,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        timeout = httpx.Timeout(float(settings.memory_consolidation_llm_timeout_seconds))
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint, json=body, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return _fallback_consolidated_facts(chapters, max_facts=max_facts), "fallback_llm_error"

    content = str(payload.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
    parsed = _extract_json_object(content)
    if not parsed:
        return _fallback_consolidated_facts(chapters, max_facts=max_facts), "fallback_invalid_json"
    raw_facts = parsed.get("facts")
    if not isinstance(raw_facts, list):
        return _fallback_consolidated_facts(chapters, max_facts=max_facts), "fallback_missing_facts"
    facts: list[str] = []
    for item in raw_facts:
        text = str(item or "").strip()
        if not text:
            continue
        if text not in facts:
            facts.append(text[:260])
        if len(facts) >= max_facts:
            break
    if not facts:
        return _fallback_consolidated_facts(chapters, max_facts=max_facts), "fallback_empty_facts"
    return facts, "llm"


def consolidate_volume_memory(
    db: Session,
    *,
    project_id: int,
    volume_id: int,
    operator_id: str,
    force: bool = False,
) -> dict[str, Any]:
    volume = get_project_volume(db, project_id, volume_id)
    if not volume:
        raise ValueError("volume not found")
    chapter_stmt = (
        select(ProjectChapter)
        .where(ProjectChapter.project_id == project_id, ProjectChapter.volume_id == volume_id)
        .order_by(ProjectChapter.chapter_index.asc())
    )
    chapters = db.exec(chapter_stmt).all()
    if not chapters:
        raise ValueError("volume has no chapters")

    max_facts = max(int(settings.memory_consolidation_max_facts), 3)
    facts, source = _call_volume_memory_consolidation_llm(
        volume_title=str(getattr(volume, "title", "") or ""),
        volume_outline=str(getattr(volume, "outline", "") or ""),
        chapters=chapters,
        max_facts=max_facts,
    )
    key_prefix = str(settings.memory_semantic_key_prefix or "memory.semantic.volume.").strip()
    stored_key = f"{key_prefix}{int(getattr(volume, 'volume_index', 0) or 0)}"
    value = {
        "volume_id": int(getattr(volume, "id", 0) or 0),
        "volume_index": int(getattr(volume, "volume_index", 0) or 0),
        "volume_title": str(getattr(volume, "title", "") or ""),
        "facts": facts,
        "chapters_count": len(chapters),
        "chapter_ids": [int(getattr(item, "id", 0) or 0) for item in chapters],
        "archive_policy": "soft_archive_low_priority",
        "force": bool(force),
        "generated_by": str(operator_id or "system"),
        "generated_at": _utc_now().isoformat(),
        "source": source,
    }
    row_stmt = select(SettingEntry).where(SettingEntry.project_id == project_id, SettingEntry.key == stored_key)
    existing = db.exec(row_stmt).first()
    if existing:
        if not force and isinstance(existing.value, dict):
            already = existing.value.get("facts")
            if isinstance(already, list) and already:
                source = "skipped_existing"
                facts = [str(item) for item in already if str(item).strip()][:max_facts]
                value = {**existing.value, "source": source, "generated_at": _utc_now().isoformat()}
            else:
                existing.value = value
                existing.updated_at = _utc_now()
                db.add(existing)
        else:
            existing.value = value
            existing.updated_at = _utc_now()
            db.add(existing)
    else:
        db.add(
            SettingEntry(
                project_id=project_id,
                key=stored_key,
                value=value,
                aliases=[f"volume-{int(getattr(volume, 'volume_index', 0) or 0)}", "semantic-memory"],
            )
        )
    db.commit()
    return {
        "project_id": project_id,
        "volume_id": int(getattr(volume, "id", 0) or 0),
        "volume_index": int(getattr(volume, "volume_index", 0) or 0),
        "chapters_count": len(chapters),
        "stored_key": stored_key,
        "fact_count": len(facts),
        "source": source,
    }


def _ensure_default_project_volume(db: Session, project_id: int) -> ProjectVolume:
    volumes = list_project_volumes(db, project_id)
    first = next(iter(volumes), None)
    if not first:
        raise ValueError("volume not found")
    return first


def _resolve_chapter_volume_id(db: Session, project_id: int, volume_id: int | None) -> int:
    default_volume = _ensure_default_project_volume(db, project_id)
    if volume_id is None:
        if default_volume.id is None:
            raise ValueError("default volume id missing")
        return int(default_volume.id)

    row = get_project_volume(db, project_id, int(volume_id))
    if not row or row.id is None:
        raise ValueError("volume not found")
    return int(row.id)


def _default_chapter_title(chapter_index: int) -> str:
    return f"第{chapter_index}章"


def _normalize_chapter_title(title: str | None, chapter_index: int) -> str:
    cleaned = (title or "").strip()
    if not cleaned:
        return _default_chapter_title(chapter_index)
    return cleaned[:255]


def _next_project_chapter_index(db: Session, project_id: int) -> int:
    stmt = select(ProjectChapter.chapter_index).where(ProjectChapter.project_id == project_id)
    rows = db.exec(stmt).all()
    if not rows:
        return 1
    return max(int(row) for row in rows) + 1


def _insert_project_chapter(
    db: Session,
    *,
    project_id: int,
    volume_id: int,
    chapter_index: int,
    title: str,
    content: str,
    operator_id: str,
    source: str,
) -> ProjectChapter:
    chapter = ProjectChapter(
        project_id=project_id,
        volume_id=volume_id,
        chapter_index=chapter_index,
        title=title,
        content=content,
        version=1,
        created_at=_utc_now(),
        updated_at=_utc_now(),
    )
    db.add(chapter)
    db.flush()
    if chapter.id is None:
        raise ValueError("project chapter id missing")
    revision = ProjectChapterRevision(
        chapter_id=chapter.id,
        project_id=project_id,
        version=chapter.version,
        title=title,
        content=content,
        operator_id=operator_id,
        source=source,
        created_at=_utc_now(),
    )
    db.add(revision)
    db.commit()
    db.refresh(chapter)
    return chapter


def list_project_chapters(db: Session, project_id: int) -> Iterable[ProjectChapter]:
    default_volume_id = _resolve_chapter_volume_id(db, project_id, None)
    stmt = select(ProjectChapter).where(ProjectChapter.project_id == project_id).order_by(ProjectChapter.chapter_index.asc())
    rows = db.exec(stmt).all()
    if rows:
        patched = False
        for row in rows:
            if row.volume_id is not None:
                continue
            row.volume_id = default_volume_id
            db.add(row)
            patched = True
        if patched:
            db.commit()
            rows = db.exec(stmt).all()
        return rows

    chapter = _insert_project_chapter(
        db,
        project_id=project_id,
        volume_id=default_volume_id,
        chapter_index=1,
        title=_default_chapter_title(1),
        content="",
        operator_id="system",
        source="create",
    )
    return [chapter]


def create_project_chapter(
    db: Session,
    *,
    project_id: int,
    operator_id: str,
    title: str | None = None,
    volume_id: int | None = None,
) -> ProjectChapter:
    chapter_index = _next_project_chapter_index(db, project_id)
    chapter_title = _normalize_chapter_title(title, chapter_index)
    resolved_volume_id = _resolve_chapter_volume_id(db, project_id, volume_id)
    return _insert_project_chapter(
        db,
        project_id=project_id,
        volume_id=resolved_volume_id,
        chapter_index=chapter_index,
        title=chapter_title,
        content="",
        operator_id=operator_id,
        source="create",
    )


def get_project_chapter(db: Session, project_id: int, chapter_id: int) -> ProjectChapter | None:
    stmt = select(ProjectChapter).where(
        ProjectChapter.id == chapter_id,
        ProjectChapter.project_id == project_id,
    )
    return db.exec(stmt).first()


def save_project_chapter(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    title: str,
    content: str,
    volume_id: int | None,
    operator_id: str,
    expected_version: int | None = None,
) -> ProjectChapter:
    chapter = get_project_chapter(db, project_id, chapter_id)
    if not chapter:
        raise ValueError("chapter not found")
    if expected_version is not None and int(expected_version) != int(chapter.version):
        raise DraftVersionConflictError(
            f"chapter version conflict: expected {expected_version}, current {chapter.version}"
        )

    title_norm = _normalize_chapter_title(title, int(chapter.chapter_index))
    resolved_volume_id = _resolve_chapter_volume_id(db, project_id, volume_id)
    if title_norm == chapter.title and content == chapter.content and int(chapter.volume_id or 0) == resolved_volume_id:
        return chapter
    if chapter.id is None:
        raise ValueError("project chapter id missing")

    next_version = int(chapter.version) + 1
    chapter.title = title_norm
    chapter.content = content
    chapter.volume_id = resolved_volume_id
    chapter.version = next_version
    chapter.updated_at = _utc_now()
    db.add(chapter)

    revision = ProjectChapterRevision(
        chapter_id=chapter.id,
        project_id=project_id,
        version=next_version,
        title=title_norm,
        content=content,
        operator_id=operator_id,
        source="save",
        created_at=_utc_now(),
    )
    db.add(revision)
    db.commit()
    db.refresh(chapter)
    return chapter


def list_project_chapter_revisions(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    limit: int = 20,
) -> Iterable[ProjectChapterRevision]:
    chapter = get_project_chapter(db, project_id, chapter_id)
    if not chapter:
        raise ValueError("chapter not found")
    stmt = (
        select(ProjectChapterRevision)
        .where(
            ProjectChapterRevision.project_id == project_id,
            ProjectChapterRevision.chapter_id == chapter_id,
        )
        .order_by(ProjectChapterRevision.version.desc())
        .limit(limit)
    )
    return db.exec(stmt).all()


def _summarize_action_for_revision(action: ChatAction) -> str:
    payload = action.payload if isinstance(action.payload, dict) else {}
    action_type = str(action.action_type or "").strip() or "unknown"
    if action_type == "setting.upsert":
        key = str(payload.get("key") or "未命名设定").strip() or "未命名设定"
        return f"更新设定：{key}"
    if action_type == "setting.delete":
        key = str(payload.get("key") or "未命名设定").strip() or "未命名设定"
        return f"删除设定：{key}"
    if action_type == "card.create":
        title = str(payload.get("title") or "未命名卡片").strip() or "未命名卡片"
        return f"新建卡片：{title}"
    if action_type == "card.update":
        title = str(payload.get("title") or payload.get("card_title") or "卡片更新").strip() or "卡片更新"
        return f"更新卡片：{title}"
    if is_entity_merge_action_type(action_type):
        source = str(payload.get("source_entity") or payload.get("alias") or "候选别名").strip() or "候选别名"
        target = str(payload.get("target_title") or payload.get("canonical_name") or "目标实体").strip() or "目标实体"
        return f"别名归一化：{source} -> {target}"
    return f"应用动作：{action_type}"


def _default_revision_semantic_summary(revision: ProjectChapterRevision) -> list[str]:
    source = str(revision.source or "").strip().lower()
    if source == "create":
        return ["创建章节初始版本。"]
    if source == "rollback":
        return ["执行了正文版本回滚。"]
    if source == "save":
        return ["手动保存正文改动。"]
    return [f"版本来源：{revision.source}"]


def list_project_chapter_revisions_with_semantic(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    limit: int = 20,
) -> list[dict[str, Any]]:
    revisions = list(list_project_chapter_revisions(db, project_id=project_id, chapter_id=chapter_id, limit=limit))
    if not revisions:
        return []

    newest_at = revisions[0].created_at
    if newest_at is None:
        newest_at = _utc_now()

    action_stmt = (
        select(ChatAction)
        .join(ChatSession, ChatAction.session_id == ChatSession.id)
        .where(
            ChatSession.project_id == project_id,
            ChatAction.status == "applied",
            ChatAction.applied_at.is_not(None),
            ChatAction.applied_at <= newest_at,
        )
        .order_by(ChatAction.applied_at.desc())
        .limit(320)
    )
    action_rows = db.exec(action_stmt).all()

    results: list[dict[str, Any]] = []
    for idx, revision in enumerate(revisions):
        window_upper = revision.created_at or newest_at
        next_revision = revisions[idx + 1] if idx + 1 < len(revisions) else None
        window_lower = next_revision.created_at if next_revision else None

        semantic_lines: list[str] = []
        for action in action_rows:
            applied_at = action.applied_at
            if applied_at is None:
                continue
            if applied_at > window_upper:
                continue
            if window_lower is not None and applied_at <= window_lower:
                continue
            semantic_lines.append(_summarize_action_for_revision(action))
            if len(semantic_lines) >= 3:
                break
        if not semantic_lines:
            semantic_lines = _default_revision_semantic_summary(revision)

        results.append(
            {
                "id": int(revision.id or 0),
                "chapter_id": int(revision.chapter_id),
                "project_id": int(revision.project_id),
                "version": int(revision.version),
                "title": str(revision.title),
                "content": str(revision.content),
                "operator_id": str(revision.operator_id),
                "source": str(revision.source),
                "semantic_summary": semantic_lines,
                "created_at": revision.created_at,
            }
        )
    return results


def rollback_project_chapter(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    target_version: int,
    operator_id: str,
) -> ProjectChapter:
    chapter = get_project_chapter(db, project_id, chapter_id)
    if not chapter:
        raise ValueError("chapter not found")

    target_stmt = select(ProjectChapterRevision).where(
        ProjectChapterRevision.project_id == project_id,
        ProjectChapterRevision.chapter_id == chapter_id,
        ProjectChapterRevision.version == target_version,
    )
    target = db.exec(target_stmt).first()
    if not target:
        raise ValueError(f"target_version {target_version} not found")
    if target.title == chapter.title and target.content == chapter.content:
        raise ValueError("chapter already matches target version content")
    if chapter.id is None:
        raise ValueError("project chapter id missing")

    next_version = int(chapter.version) + 1
    chapter.title = target.title
    chapter.content = target.content
    chapter.version = next_version
    chapter.updated_at = _utc_now()
    db.add(chapter)

    revision = ProjectChapterRevision(
        chapter_id=chapter.id,
        project_id=project_id,
        version=next_version,
        title=chapter.title,
        content=chapter.content,
        operator_id=operator_id,
        source="rollback",
        created_at=_utc_now(),
    )
    db.add(revision)
    db.commit()
    db.refresh(chapter)
    return chapter


def _ordered_project_chapters(db: Session, project_id: int) -> list[ProjectChapter]:
    stmt = select(ProjectChapter).where(ProjectChapter.project_id == project_id).order_by(ProjectChapter.chapter_index.asc())
    return db.exec(stmt).all()


def _reindex_project_chapters(db: Session, project_id: int) -> None:
    rows = _ordered_project_chapters(db, project_id)
    for idx, chapter in enumerate(rows, start=1):
        if int(chapter.chapter_index) == idx:
            continue
        chapter.chapter_index = idx
        db.add(chapter)


def move_project_chapter(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    direction: str,
) -> ProjectChapter:
    rows = _ordered_project_chapters(db, project_id)
    if not rows:
        raise ValueError("chapter not found")
    current_pos = next((idx for idx, item in enumerate(rows) if item.id == chapter_id), -1)
    if current_pos < 0:
        raise ValueError("chapter not found")
    if direction not in {"up", "down"}:
        raise ValueError("direction must be up or down")

    if direction == "up":
        if current_pos == 0:
            return rows[current_pos]
        swap_pos = current_pos - 1
    else:
        if current_pos >= len(rows) - 1:
            return rows[current_pos]
        swap_pos = current_pos + 1

    current = rows[current_pos]
    target = rows[swap_pos]
    current_index = int(current.chapter_index)
    target_index = int(target.chapter_index)

    # Avoid unique(project_id, chapter_index) collision during swap.
    current.chapter_index = -1
    db.add(current)
    db.flush()
    target.chapter_index = current_index
    db.add(target)
    db.flush()
    current.chapter_index = target_index
    db.add(current)
    db.commit()
    db.refresh(current)
    return current


def _delete_chapter_with_revisions(db: Session, project_id: int, chapter_id: int) -> None:
    rev_stmt = select(ProjectChapterRevision).where(
        ProjectChapterRevision.project_id == project_id,
        ProjectChapterRevision.chapter_id == chapter_id,
    )
    revisions = db.exec(rev_stmt).all()
    for item in revisions:
        db.delete(item)

    chapter = get_project_chapter(db, project_id, chapter_id)
    if chapter:
        db.delete(chapter)


def delete_project_chapter(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    operator_id: str,
) -> tuple[int, int | None]:
    rows = _ordered_project_chapters(db, project_id)
    if not rows:
        raise ValueError("chapter not found")
    current_pos = next((idx for idx, item in enumerate(rows) if item.id == chapter_id), -1)
    if current_pos < 0:
        raise ValueError("chapter not found")

    deleted_chapter_id = chapter_id
    if len(rows) == 1:
        _delete_chapter_with_revisions(db, project_id, chapter_id)
        db.commit()
        created = _insert_project_chapter(
            db,
            project_id=project_id,
            volume_id=_resolve_chapter_volume_id(db, project_id, None),
            chapter_index=1,
            title=_default_chapter_title(1),
            content="",
            operator_id=operator_id,
            source="recreate_after_delete_last",
        )
        return deleted_chapter_id, created.id

    _delete_chapter_with_revisions(db, project_id, chapter_id)
    db.flush()
    _reindex_project_chapters(db, project_id)
    db.commit()

    remaining = _ordered_project_chapters(db, project_id)
    if not remaining:
        return deleted_chapter_id, None

    if current_pos < len(remaining):
        return deleted_chapter_id, remaining[current_pos].id
    return deleted_chapter_id, remaining[-1].id


def reorder_project_chapters(
    db: Session,
    *,
    project_id: int,
    ordered_ids: list[int],
) -> list[ProjectChapter]:
    rows = _ordered_project_chapters(db, project_id)
    if not rows:
        raise ValueError("chapter not found")

    if not ordered_ids:
        raise ValueError("ordered_ids required")
    existing_ids = [int(item.id or 0) for item in rows]
    if 0 in existing_ids:
        raise ValueError("invalid chapter id")
    if set(existing_ids) != set(int(item) for item in ordered_ids):
        raise ValueError("ordered_ids must contain all chapter ids exactly once")

    id_to_chapter = {int(item.id): item for item in rows if item.id is not None}
    if len(id_to_chapter) != len(existing_ids):
        raise ValueError("invalid chapter map")

    # Two-phase assign to avoid unique(project_id, chapter_index) conflicts.
    temp_base = -(len(ordered_ids) + 10)
    for idx, chapter_id in enumerate(ordered_ids):
        chapter = id_to_chapter.get(int(chapter_id))
        if not chapter:
            raise ValueError(f"chapter {chapter_id} not found")
        chapter.chapter_index = temp_base - idx
        db.add(chapter)
    db.flush()

    for idx, chapter_id in enumerate(ordered_ids, start=1):
        chapter = id_to_chapter[int(chapter_id)]
        chapter.chapter_index = idx
        db.add(chapter)
    db.commit()

    return _ordered_project_chapters(db, project_id)


def get_scene_beat(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    beat_id: int,
) -> ChapterSceneBeat | None:
    stmt = select(ChapterSceneBeat).where(
        ChapterSceneBeat.project_id == project_id,
        ChapterSceneBeat.chapter_id == chapter_id,
        ChapterSceneBeat.id == beat_id,
    )
    return db.exec(stmt).first()


def list_scene_beats(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
) -> Iterable[ChapterSceneBeat]:
    chapter = get_project_chapter(db, project_id, chapter_id)
    if not chapter:
        raise ValueError("chapter not found")
    stmt = (
        select(ChapterSceneBeat)
        .where(
            ChapterSceneBeat.project_id == project_id,
            ChapterSceneBeat.chapter_id == chapter_id,
        )
        .order_by(ChapterSceneBeat.beat_index.asc())
    )
    return db.exec(stmt).all()


def _next_scene_beat_index(db: Session, project_id: int, chapter_id: int) -> int:
    stmt = select(ChapterSceneBeat.beat_index).where(
        ChapterSceneBeat.project_id == project_id,
        ChapterSceneBeat.chapter_id == chapter_id,
    )
    rows = db.exec(stmt).all()
    if not rows:
        return 1
    return max(int(item) for item in rows) + 1


def _normalize_scene_beat_status(status: str | None) -> str:
    raw = str(status or "pending").strip().lower()
    if raw == "done":
        return "done"
    return "pending"


def create_scene_beat(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    content: str,
    status: str,
) -> ChapterSceneBeat:
    chapter = get_project_chapter(db, project_id, chapter_id)
    if not chapter:
        raise ValueError("chapter not found")
    beat = ChapterSceneBeat(
        project_id=project_id,
        chapter_id=chapter_id,
        beat_index=_next_scene_beat_index(db, project_id, chapter_id),
        content=str(content or "").strip()[:20000],
        status=_normalize_scene_beat_status(status),
        created_at=_utc_now(),
        updated_at=_utc_now(),
    )
    db.add(beat)
    db.commit()
    db.refresh(beat)
    return beat


def update_scene_beat(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    beat_id: int,
    content: str,
    status: str,
) -> ChapterSceneBeat:
    beat = get_scene_beat(db, project_id=project_id, chapter_id=chapter_id, beat_id=beat_id)
    if not beat:
        raise ValueError("scene beat not found")
    beat.content = str(content or "").strip()[:20000]
    beat.status = _normalize_scene_beat_status(status)
    beat.updated_at = _utc_now()
    db.add(beat)
    db.commit()
    db.refresh(beat)
    return beat


def _reindex_scene_beats(db: Session, project_id: int, chapter_id: int) -> None:
    stmt = (
        select(ChapterSceneBeat)
        .where(
            ChapterSceneBeat.project_id == project_id,
            ChapterSceneBeat.chapter_id == chapter_id,
        )
        .order_by(ChapterSceneBeat.beat_index.asc())
    )
    rows = db.exec(stmt).all()
    for idx, row in enumerate(rows, start=1):
        if int(row.beat_index) == idx:
            continue
        row.beat_index = idx
        row.updated_at = _utc_now()
        db.add(row)


def delete_scene_beat(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    beat_id: int,
) -> int:
    beat = get_scene_beat(db, project_id=project_id, chapter_id=chapter_id, beat_id=beat_id)
    if not beat:
        raise ValueError("scene beat not found")
    deleted_id = int(beat.id or 0)
    db.delete(beat)
    db.flush()
    _reindex_scene_beats(db, project_id, chapter_id)
    db.commit()
    return deleted_id


def _validate_chapter_in_project(db: Session, project_id: int, chapter_id: int | None) -> int | None:
    if chapter_id is None:
        return None
    chapter = get_project_chapter(db, project_id, int(chapter_id))
    if not chapter:
        raise ValueError(f"chapter {chapter_id} not found")
    if chapter.id is None:
        raise ValueError("chapter id missing")
    return int(chapter.id)


def _normalize_foreshadow_status(status: str | None) -> str:
    raw = str(status or "open").strip().lower()
    if raw == "resolved":
        return "resolved"
    return "open"


def list_foreshadowing_cards(
    db: Session,
    *,
    project_id: int,
    status: str | None = None,
) -> Iterable[ForeshadowingCard]:
    stmt = select(ForeshadowingCard).where(ForeshadowingCard.project_id == project_id)
    status_norm = _normalize_foreshadow_status(status) if status else None
    if status_norm:
        stmt = stmt.where(ForeshadowingCard.status == status_norm)
    stmt = stmt.order_by(ForeshadowingCard.id.asc())
    return db.exec(stmt).all()


def get_foreshadowing_card(db: Session, *, project_id: int, card_id: int) -> ForeshadowingCard | None:
    stmt = select(ForeshadowingCard).where(
        ForeshadowingCard.project_id == project_id,
        ForeshadowingCard.id == card_id,
    )
    return db.exec(stmt).first()


def create_foreshadowing_card(
    db: Session,
    *,
    project_id: int,
    title: str,
    description: str,
    planted_in_chapter_id: int | None,
    source_action_id: int | None,
) -> ForeshadowingCard:
    planted_id = _validate_chapter_in_project(db, project_id, planted_in_chapter_id)
    row = ForeshadowingCard(
        project_id=project_id,
        title=(title or "").strip()[:255] or "未命名伏笔",
        description=str(description or "").strip()[:50000],
        status="open",
        planted_in_chapter_id=planted_id,
        resolved_in_chapter_id=None,
        source_action_id=(int(source_action_id) if source_action_id else None),
        created_at=_utc_now(),
        updated_at=_utc_now(),
        resolved_at=None,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def update_foreshadowing_card(
    db: Session,
    *,
    project_id: int,
    card_id: int,
    title: str,
    description: str,
    status: str,
    planted_in_chapter_id: int | None,
    resolved_in_chapter_id: int | None,
) -> ForeshadowingCard:
    row = get_foreshadowing_card(db, project_id=project_id, card_id=card_id)
    if not row:
        raise ValueError("foreshadow card not found")

    planted_id = _validate_chapter_in_project(db, project_id, planted_in_chapter_id)
    resolved_id = _validate_chapter_in_project(db, project_id, resolved_in_chapter_id)
    status_norm = _normalize_foreshadow_status(status)
    if status_norm == "open":
        resolved_id = None

    row.title = (title or "").strip()[:255] or row.title
    row.description = str(description or "").strip()[:50000]
    row.status = status_norm
    row.planted_in_chapter_id = planted_id
    row.resolved_in_chapter_id = resolved_id
    row.updated_at = _utc_now()
    row.resolved_at = _utc_now() if status_norm == "resolved" else None
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def delete_foreshadowing_card(
    db: Session,
    *,
    project_id: int,
    card_id: int,
) -> int:
    row = get_foreshadowing_card(db, project_id=project_id, card_id=card_id)
    if not row:
        raise ValueError("foreshadow card not found")
    deleted_id = int(row.id or 0)
    db.delete(row)
    db.commit()
    return deleted_id


def list_overdue_foreshadowing_cards(
    db: Session,
    *,
    project_id: int,
    current_chapter_id: int | None,
    chapter_gap: int = 50,
    limit: int = 8,
) -> list[ForeshadowingCard]:
    if current_chapter_id is None:
        return []
    chapter = get_project_chapter(db, project_id, int(current_chapter_id))
    if not chapter:
        return []
    current_index = int(chapter.chapter_index)
    if current_index <= 0:
        return []

    rows = list_foreshadowing_cards(db, project_id=project_id, status="open")
    if not rows:
        return []
    chapter_stmt = select(ProjectChapter).where(ProjectChapter.project_id == project_id)
    chapter_rows = db.exec(chapter_stmt).all()
    chapter_index_map = {int(item.id): int(item.chapter_index) for item in chapter_rows if item.id is not None}
    overdue: list[ForeshadowingCard] = []
    for item in rows:
        planted_id = int(item.planted_in_chapter_id or 0)
        planted_index = chapter_index_map.get(planted_id)
        if planted_index is None:
            continue
        if current_index - planted_index < max(int(chapter_gap), 1):
            continue
        overdue.append(item)
    overdue.sort(
        key=lambda item: (
            chapter_index_map.get(int(item.planted_in_chapter_id or 0), 0),
            int(item.id or 0),
        )
    )
    return overdue[: max(int(limit), 1)]
