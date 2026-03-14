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

from pydantic import BaseModel, Field
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
from app.services.graph_mutation_registry import (
    mark_pending_graph_mutation_canceled,
    mark_pending_graph_mutation_status,
    upsert_pending_graph_mutation,
)
from app.services.index_lifecycle_queue import enqueue_index_lifecycle_job
from app.services.entity_merge_queue import enqueue_entity_merge_scan_job
from app.services.index_lifecycle_service import process_index_lifecycle_rebuild
from app.services.llm_provider import generate_structured_sync
from app.services.retrieval_adapters import (
    delete_neo4j_graph_facts,
    delete_neo4j_graph_facts_by_sources,
    fetch_neo4j_entity_profiles,
    fetch_neo4j_graph_timeline_snapshot,
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
_MODEL_PROFILE_ALLOWED_PROVIDERS: frozenset[str] = frozenset(
    {"openai_compatible", "deepseek", "claude", "gemini"}
)


class GraphCorefRewriteOutput(BaseModel):
    rewritten_text: str = Field(default="", max_length=4000)
    applied: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VolumeMemoryConsolidationOutput(BaseModel):
    facts: list[str] = Field(default_factory=list, max_length=32)
    notes: str = Field(default="", max_length=400)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DraftVersionConflictError(ValueError):
    pass


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


def _is_internal_setting_key(key: str) -> bool:
    raw = str(key or "").strip().lower()
    if not raw:
        return False
    return any(raw.startswith(prefix) for prefix in _INTERNAL_SETTING_PREFIXES)


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

