from typing import Any
from collections import deque
import copy
import json
import re
import time

from sqlmodel import Session, select

from app.core.config import settings

# ---------------------------------------------------------------------------
# Graph coreference defaults (previously exposed as env vars).
# Only consumed by this module — all disabled by default.
# ---------------------------------------------------------------------------
_COREF_PREPROCESS_ENABLED = False
_COREF_MAX_REPLACEMENTS = 12
_COREF_OVERLAP_ENABLED = False
_COREF_CHUNK_SIZE = 420
_COREF_CHUNK_OVERLAP = 120
_COREF_MAX_CHUNKS = 6
_COREF_LLM_ENABLED = False
_COREF_LLM_TIMEOUT_SECONDS = 12

from app.models.chat import ChatAction, ChatSession
from app.models.content import StoryCard
from app.services.graph_job_queue import enqueue_graph_sync_job
from app.services.graph_mutation_registry import (
    mark_pending_graph_mutation_canceled,
    mark_pending_graph_mutation_status,
    upsert_pending_graph_mutation,
)
from app.services.entity_merge_queue import enqueue_entity_merge_scan_job
from app.services.llm_provider import generate_structured_sync
from app.services.retrieval_adapters import (
    delete_neo4j_graph_facts,
    delete_neo4j_graph_facts_by_sources,
    fetch_neo4j_graph_timeline_snapshot,
    fetch_lightrag_graph_candidates,
    make_graph_candidate,
    merge_graph_candidates,
    upsert_neo4j_graph_facts,
)
from app.services.chat_service._common import (
    _LOGGER,
    _normalize_graph_entity_token,
    _normalize_aliases_payload,
    _to_text,
    _split_targets,
    _GRAPH_RELATION_FIELD_MAP,
    _setting_key_from_payload,
    GraphCorefRewriteOutput,
)
from app.services.chat_service.actions import create_action_audit_log, is_entity_merge_action_type
from app.services.chat_service.mutations import (
    _graph_sync_meta,
    _action_graph_identifiers,
    _is_graph_job_stale,
)
from app.services.chat_service.entity_merge import (
    _build_project_entity_alias_map,
    _build_project_alias_prompt_hints,
)


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


def _build_lightrag_runtime_config(model: str, base_url: str, api_key: str) -> dict[str, str]:
    return {
        "provider": "openai_compatible",
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }


def _rewrite_chunk_with_llm_coref(
    chunk_text: str,
    *,
    context_summary: str,
    anchor_canonical: str,
) -> tuple[str, dict[str, Any]]:
    if not _COREF_LLM_ENABLED:
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

    model = str(settings.llm_model or lightrag_model).strip()
    system_prompt = (
        "你是小说图谱抽取前的文本预处理器。"
        "任务是把代词（他/她/它/那家伙等）在有上下文依据时还原为明确实体。"
        "必须保持事实不变，不新增事件，不润色文风。"
        "若不确定则保持原文。只输出符合 schema 的 JSON。"
    )
    payload = {
        "anchor": anchor_canonical,
        "context_summary": context_summary,
        "chunk_text": chunk_text,
    }
    try:
        structured = generate_structured_sync(
            json.dumps(payload, ensure_ascii=False),
            output_model=GraphCorefRewriteOutput,
            schema_name="graph_coref_rewrite_output",
            context={"system": system_prompt, "raw_prompt": True},
            runtime_config=_build_lightrag_runtime_config(model, lightrag_base_url, lightrag_api_key),
            temperature_override=0.0,
        )
    except Exception as exc:
        return chunk_text, {
            "enabled": True,
            "applied": False,
            "reason": "llm_error",
            "error": str(exc),
            "model": model,
        }

    parsed = structured.parsed
    if not parsed:
        return chunk_text, {
            "enabled": True,
            "applied": False,
            "reason": "llm_invalid_json",
            "model": model,
        }

    rewritten = str(getattr(parsed, "rewritten_text", "") or "").strip()
    applied = bool(getattr(parsed, "applied", False)) and bool(rewritten) and rewritten != chunk_text
    confidence = float(getattr(parsed, "confidence", 0.0) or 0.0)
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
    enabled = bool(_COREF_PREPROCESS_ENABLED)
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
        max_replacements=max(int(_COREF_MAX_REPLACEMENTS), 1),
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


def _graph_preview_edge_key(source: str, relation: str, target: str) -> str:
    source_norm = _normalize_graph_entity_token(source)
    target_norm = _normalize_graph_entity_token(target)
    relation_norm = str(relation or "").strip().upper()
    if not source_norm or not target_norm or not relation_norm:
        return ""
    return f"{source_norm}|{relation_norm}|{target_norm}"


def _merge_graph_preview_node(
    node_map: dict[str, dict[str, Any]],
    *,
    label: str,
    change: str,
    role: str,
    current_labels: dict[str, str],
) -> None:
    normalized = _normalize_graph_entity_token(label)
    if not normalized:
        return

    current_label = current_labels.get(normalized, "")
    display_label = str(label or current_label or normalized).strip() or normalized
    next_item = {
        "id": display_label,
        "label": display_label,
        "change": change,
        "role": role,
        "in_current_graph": bool(current_label),
    }
    existing = node_map.get(normalized)
    if existing is None:
        node_map[normalized] = next_item
        return

    severity = {"delete": 4, "update": 3, "create": 2, "touch": 1}
    existing_change = str(existing.get("change") or "touch")
    next_change = str(change or "touch")
    should_replace = severity.get(next_change, 0) > severity.get(existing_change, 0)
    if str(existing.get("role") or "") != "anchor" and role == "anchor":
        should_replace = True
    if should_replace:
        node_map[normalized] = next_item
        return

    existing["in_current_graph"] = bool(existing.get("in_current_graph")) or bool(current_label)
    if not existing.get("label") and display_label:
        existing["label"] = display_label
        existing["id"] = display_label


def build_action_blast_radius_preview(
    db: Session,
    *,
    project_id: int,
    action_type: str,
    payload: dict[str, Any],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preview = {
        "source": "none",
        "action_type": str(action_type or ""),
        "chapter_index": None,
        "nodes": [],
        "edges": [],
        "summary": {
            "nodes": {"create": 0, "update": 0, "delete": 0, "touch": 0},
            "edges": {"add": 0, "update": 0, "delete": 0},
        },
        "notes": [],
    }
    if not isinstance(payload, dict):
        preview["notes"].append("payload 无效，无法推导图谱影响。")
        return preview

    action_payload = copy.deepcopy(payload)
    provenance_payload = provenance if isinstance(provenance, dict) else {}
    graph_current_chapter = 0
    try:
        graph_current_chapter = int(provenance_payload.get("current_chapter_index") or 0)
    except Exception:
        graph_current_chapter = 0
    if graph_current_chapter > 0:
        action_payload["_graph_current_chapter"] = graph_current_chapter
        preview["chapter_index"] = graph_current_chapter

    normalized_action_type = str(action_type or "").strip()
    if normalized_action_type == "setting.upsert":
        key = _setting_key_from_payload(action_payload)
        graph_anchor = key.replace("设定", "").strip() or key
        action_payload["_graph_anchor"] = graph_anchor
    elif normalized_action_type == "card.create":
        title = str(action_payload.get("title") or "未命名卡片").strip() or "未命名卡片"
        action_payload["_graph_anchor"] = title
    elif normalized_action_type == "card.update":
        card_id = action_payload.get("card_id")
        card = db.get(StoryCard, card_id) if isinstance(card_id, int) else None
        current_title = str(card.title or "").strip() if card else ""
        next_title = str(action_payload.get("title") or current_title).strip()
        if next_title:
            action_payload["_graph_anchor"] = next_title
        if current_title:
            action_payload["_graph_anchor_before"] = current_title
    elif normalized_action_type == "setting.delete":
        preview["notes"].append("此动作不会直接改写图谱关系，仅影响设定与索引生命周期。")
        return preview
    elif is_entity_merge_action_type(normalized_action_type):
        target_card_id = action_payload.get("target_card_id")
        if not isinstance(target_card_id, int):
            target_card_id = action_payload.get("canonical_card_id")
        if not isinstance(target_card_id, int):
            target_card_id = action_payload.get("card_id")
        card = db.get(StoryCard, target_card_id) if isinstance(target_card_id, int) else None
        if card:
            node_label = str(card.title or "").strip() or f"card-{target_card_id}"
            current_labels = {_normalize_graph_entity_token(node_label): node_label}
            node_map: dict[str, dict[str, Any]] = {}
            _merge_graph_preview_node(
                node_map,
                label=node_label,
                change="update",
                role="anchor",
                current_labels=current_labels,
            )
            preview["source"] = "alias_preview"
            preview["nodes"] = list(node_map.values())
            preview["summary"]["nodes"]["update"] = len(preview["nodes"])
        preview["notes"].append("此动作只改 aliases 归一化，不会立即增删图谱边。")
        return preview

    text, anchor, rule_candidates = _build_graph_extraction_text(normalized_action_type, action_payload, project_id)
    projection_sources: list[str] = []
    for key in ("_graph_anchor", "_graph_anchor_before"):
        value = str(action_payload.get(key) or "").strip()
        if value and value not in projection_sources:
            projection_sources.append(value)
    if anchor and anchor not in projection_sources:
        projection_sources.append(anchor)

    current_snapshot = fetch_neo4j_graph_timeline_snapshot(
        project_id,
        current_chapter=graph_current_chapter if graph_current_chapter > 0 else None,
        limit=260,
    )
    current_nodes_raw = current_snapshot.get("nodes") if isinstance(current_snapshot, dict) else []
    current_edges_raw = current_snapshot.get("edges") if isinstance(current_snapshot, dict) else []
    current_labels: dict[str, str] = {}
    if isinstance(current_nodes_raw, list):
        for item in current_nodes_raw:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("id") or "").strip()
            normalized = _normalize_graph_entity_token(label)
            if normalized and normalized not in current_labels:
                current_labels[normalized] = label

    alias_map = _build_project_entity_alias_map(db, project_id)
    resolved_candidates, _ = _resolve_entity_aliases_for_candidates(
        merge_graph_candidates(rule_candidates, [], limit=24),
        alias_map,
    )

    projection_source_norms = {
        _normalize_graph_entity_token(item)
        for item in projection_sources
        if _normalize_graph_entity_token(item)
    }
    existing_edges: dict[str, dict[str, Any]] = {}
    if isinstance(current_edges_raw, list):
        for item in current_edges_raw:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip()
            target = str(item.get("target") or "").strip()
            relation = str(item.get("relation") or "").strip().upper()
            source_norm = _normalize_graph_entity_token(source)
            if projection_source_norms and source_norm not in projection_source_norms:
                continue
            edge_key = _graph_preview_edge_key(source, relation, target)
            if not edge_key or edge_key in existing_edges:
                continue
            existing_edges[edge_key] = {
                "key": edge_key,
                "source": source,
                "target": target,
                "relation": relation,
                "change": "delete",
                "in_current_graph": True,
            }

    candidate_edges: dict[str, dict[str, Any]] = {}
    for item in resolved_candidates:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source_entity") or "").strip()
        target = str(item.get("target_entity") or "").strip()
        relation = str(item.get("relation") or "").strip().upper()
        edge_key = _graph_preview_edge_key(source, relation, target)
        if not edge_key or edge_key in candidate_edges:
            continue
        candidate_edges[edge_key] = {
            "key": edge_key,
            "source": source,
            "target": target,
            "relation": relation,
            "change": "add",
            "in_current_graph": edge_key in existing_edges,
        }

    node_map: dict[str, dict[str, Any]] = {}
    anchor_norms = {
        _normalize_graph_entity_token(item)
        for item in projection_sources
        if _normalize_graph_entity_token(item)
    }
    for item in projection_sources:
        normalized = _normalize_graph_entity_token(item)
        if not normalized:
            continue
        current_exists = normalized in current_labels
        anchor_change = (
            "create"
            if normalized_action_type == "card.create" and not current_exists
            else "update"
        )
        _merge_graph_preview_node(
            node_map,
            label=item,
            change=anchor_change,
            role="anchor",
            current_labels=current_labels,
        )

    edge_items: list[dict[str, Any]] = []
    for edge_key in sorted(set(existing_edges) | set(candidate_edges)):
        edge_item = candidate_edges.get(edge_key) or existing_edges.get(edge_key) or {}
        if edge_key in existing_edges and edge_key in candidate_edges:
            next_edge = {**candidate_edges[edge_key], "change": "update", "in_current_graph": True}
        elif edge_key in candidate_edges:
            next_edge = candidate_edges[edge_key]
        else:
            next_edge = existing_edges[edge_key]
        edge_items.append(next_edge)

        source_norm = _normalize_graph_entity_token(str(edge_item.get("source") or ""))
        target_norm = _normalize_graph_entity_token(str(edge_item.get("target") or ""))
        source_change = "update" if source_norm in anchor_norms else "touch"
        target_current_exists = target_norm in current_labels
        if next_edge["change"] == "add":
            target_change = "touch" if target_current_exists else "create"
        elif next_edge["change"] == "update":
            target_change = "touch" if target_current_exists else "create"
        else:
            target_change = "touch"
        _merge_graph_preview_node(
            node_map,
            label=str(edge_item.get("source") or ""),
            change=source_change,
            role="anchor" if source_norm in anchor_norms else "related",
            current_labels=current_labels,
        )
        _merge_graph_preview_node(
            node_map,
            label=str(edge_item.get("target") or ""),
            change=target_change,
            role="related",
            current_labels=current_labels,
        )

    preview["source"] = "rule_preview" if text or resolved_candidates or projection_sources else "none"
    preview["nodes"] = sorted(
        node_map.values(),
        key=lambda item: (
            0 if str(item.get("role") or "") == "anchor" else 1,
            0
            if str(item.get("change") or "") == "delete"
            else (
                1
                if str(item.get("change") or "") == "update"
                else (2 if str(item.get("change") or "") == "create" else 3)
            ),
            str(item.get("label") or ""),
        ),
    )
    preview["edges"] = edge_items

    node_summary = {"create": 0, "update": 0, "delete": 0, "touch": 0}
    for item in preview["nodes"]:
        change = str(item.get("change") or "touch")
        if change in node_summary:
            node_summary[change] += 1
    edge_summary = {"add": 0, "update": 0, "delete": 0}
    for item in preview["edges"]:
        change = str(item.get("change") or "add")
        if change in edge_summary:
            edge_summary[change] += 1
    preview["summary"] = {"nodes": node_summary, "edges": edge_summary}
    if not preview["edges"]:
        preview["notes"].append("当前动作未推导出可视化关系边，仅影响锚点节点。")
    if not current_labels:
        preview["notes"].append("当前章节未加载到已确认图谱，绿色项为待写入预览。")
    return preview


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
        if mutation_id:
            mark_pending_graph_mutation_status(
                db,
                mutation_id=mutation_id,
                status="skipped",
                cancel_reason="action_missing",
            )
        return None, []

    current_meta = _graph_sync_meta(latest_action)
    current_mutation_id = str(current_meta.get("mutation_id") or "")
    is_stale, stale_reason = _is_graph_job_stale(
        latest_action,
        mutation_id=mutation_id,
        expected_version=expected_version,
    )
    if is_stale:
        if mutation_id or current_mutation_id:
            mark_pending_graph_mutation_status(
                db,
                mutation_id=mutation_id or current_mutation_id,
                status="skipped",
                cancel_reason=stale_reason or "stale_before_write",
            )
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
        if mutation_id or current_mutation_id:
            mark_pending_graph_mutation_status(
                db,
                mutation_id=mutation_id or current_mutation_id,
                status=current_status,
            )
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
            if mutation_id:
                mark_pending_graph_mutation_status(
                    db,
                    mutation_id=mutation_id,
                    status="compensated",
                    cancel_reason="action_missing_after_write",
                )
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
        if mutation_id or post_mutation_id:
            mark_pending_graph_mutation_status(
                db,
                mutation_id=mutation_id or post_mutation_id,
                status="compensated",
                cancel_reason=post_stale_reason or "stale_or_undone_after_write",
            )
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
        if mutation_id or post_mutation_id:
            mark_pending_graph_mutation_status(
                db,
                mutation_id=mutation_id or post_mutation_id,
                status="compensated",
                cancel_reason="action_missing_before_commit",
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
        if mutation_id or write_mutation_id:
            mark_pending_graph_mutation_status(
                db,
                mutation_id=mutation_id or write_mutation_id,
                status="compensated",
                cancel_reason=write_stale_reason or "stale_before_commit",
            )
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
    if mutation_id_final:
        mark_pending_graph_mutation_status(
            db,
            mutation_id=mutation_id_final,
            status=status or "unknown",
        )
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
                    "queue": "entity_merge_scan_jobs",
                    "idempotency_key": scan_job_key,
                    "reason": "graph_sync_followup",
                },
            )
    return graph_sync, fact_keys

