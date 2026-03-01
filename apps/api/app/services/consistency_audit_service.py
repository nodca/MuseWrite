import json
import re
from datetime import date, datetime, timezone
from typing import Any
from uuid import uuid4

import httpx
from sqlmodel import Session, select

from app.core.config import settings
from app.models.content import ForeshadowingCard, ProjectChapter, SettingEntry
from app.services.retrieval_adapters import fetch_neo4j_graph_facts

_CONSISTENCY_AUDIT_REPORT_PREFIX = "consistency.audit.report."


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _truncate_text(value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_reason(value: str | None) -> str:
    reason = str(value or "consistency_audit").strip().lower()
    if not reason:
        return "consistency_audit"
    return re.sub(r"[^a-z0-9_.-]+", "_", reason)[:64]


def _extract_terms(text: str, limit: int = 10) -> list[str]:
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text or "")
    terms: list[str] = []
    for token in tokens:
        term = str(token).strip()
        if not term:
            continue
        lowered = term.lower()
        if lowered in terms:
            continue
        terms.append(lowered)
        if len(terms) >= limit:
            break
    return terms


def _normalize_issue_type(value: str) -> str:
    token = str(value or "").strip().lower()
    if token in {"temporal", "timeline", "timeline_conflict", "temporal_conflict"}:
        return "temporal_conflict"
    if token in {"foreshadow", "foreshadow_gap", "foreshadow_overdue", "stale_foreshadow"}:
        return "foreshadow_overdue"
    if token in {"continuity", "continuity_risk", "context_noise"}:
        return "continuity_risk"
    return "continuity_risk"


def _normalize_severity(value: str) -> str:
    token = str(value or "").strip().lower()
    if token in {"critical", "high", "medium", "low"}:
        return token
    return "medium"


def _normalize_issue(
    item: dict[str, Any],
    *,
    chapter_id: int | None = None,
    chapter_index: int | None = None,
) -> dict[str, Any] | None:
    issue_type = _normalize_issue_type(str(item.get("type") or item.get("issue_type") or ""))
    title = str(item.get("title") or "").strip()
    detail = str(item.get("detail") or item.get("description") or "").strip()
    if not title:
        if issue_type == "temporal_conflict":
            title = "章节与时序事实存在冲突"
        elif issue_type == "foreshadow_overdue":
            title = "伏笔长期未收束"
        else:
            title = "上下文一致性风险"
    if not detail:
        detail = title
    evidence = item.get("evidence")
    evidence_payload = evidence if isinstance(evidence, dict) else {}
    suggestion = str(item.get("suggestion") or "").strip()
    normalized = {
        "type": issue_type,
        "severity": _normalize_severity(str(item.get("severity") or "")),
        "title": title[:120],
        "detail": detail[:500],
        "chapter_id": int(chapter_id) if isinstance(chapter_id, int) and chapter_id > 0 else None,
        "chapter_index": int(chapter_index) if isinstance(chapter_index, int) and chapter_index > 0 else None,
        "evidence": evidence_payload,
        "suggestion": suggestion[:260] if suggestion else None,
    }
    return normalized


def _issue_identity(item: dict[str, Any]) -> str:
    return "|".join(
        [
            str(item.get("type") or ""),
            str(item.get("chapter_id") or ""),
            str(item.get("chapter_index") or ""),
            str(item.get("title") or ""),
            str(item.get("detail") or ""),
        ]
    ).lower()


def _merge_issues(items: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        identity = _issue_identity(item)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def consistency_audit_report_prefix() -> str:
    return _CONSISTENCY_AUDIT_REPORT_PREFIX


def list_project_ids_with_chapters(db: Session, *, limit: int = 200) -> list[int]:
    size = max(int(limit), 1)
    stmt = (
        select(ProjectChapter.project_id)
        .group_by(ProjectChapter.project_id)
        .order_by(ProjectChapter.project_id.asc())
        .limit(size)
    )
    rows = db.exec(stmt).all()
    project_ids: list[int] = []
    for value in rows:
        try:
            project_ids.append(int(value))
        except Exception:
            continue
    return project_ids


def latest_project_chapter_update_at(db: Session, project_id: int) -> datetime | None:
    if project_id <= 0:
        return None
    stmt = (
        select(ProjectChapter.updated_at)
        .where(ProjectChapter.project_id == project_id)
        .order_by(ProjectChapter.updated_at.desc(), ProjectChapter.id.desc())
        .limit(1)
    )
    raw = db.exec(stmt).first()
    return _coerce_datetime(raw)


def latest_consistency_audit_timestamp(db: Session, project_id: int) -> datetime | None:
    if project_id <= 0:
        return None
    stmt = (
        select(SettingEntry)
        .where(
            SettingEntry.project_id == project_id,
            SettingEntry.key.like(f"{_CONSISTENCY_AUDIT_REPORT_PREFIX}%"),
        )
        .order_by(SettingEntry.updated_at.desc(), SettingEntry.id.desc())
        .limit(12)
    )
    rows = db.exec(stmt).all()
    latest: datetime | None = None
    for row in rows:
        generated_at = None
        if isinstance(row.value, dict):
            generated_at = _coerce_datetime(row.value.get("generated_at"))
        generated_at = generated_at or _coerce_datetime(row.updated_at)
        if generated_at is None:
            continue
        if latest is None or generated_at > latest:
            latest = generated_at
    return latest


def has_consistency_audit_report_on_date(db: Session, project_id: int, target_date_utc: date) -> bool:
    if project_id <= 0:
        return False
    stmt = (
        select(SettingEntry)
        .where(
            SettingEntry.project_id == project_id,
            SettingEntry.key.like(f"{_CONSISTENCY_AUDIT_REPORT_PREFIX}%"),
        )
        .order_by(SettingEntry.updated_at.desc(), SettingEntry.id.desc())
        .limit(30)
    )
    rows = db.exec(stmt).all()
    for row in rows:
        generated_at = None
        if isinstance(row.value, dict):
            generated_at = _coerce_datetime(row.value.get("generated_at"))
        generated_at = generated_at or _coerce_datetime(row.updated_at)
        if not generated_at:
            continue
        if generated_at.astimezone(timezone.utc).date() == target_date_utc:
            return True
    return False


def _chapter_id_to_index_map(db: Session, project_id: int) -> dict[int, int]:
    stmt = select(ProjectChapter).where(ProjectChapter.project_id == project_id)
    rows = db.exec(stmt).all()
    mapping: dict[int, int] = {}
    for row in rows:
        chapter_id = int(getattr(row, "id", 0) or 0)
        chapter_index = int(getattr(row, "chapter_index", 0) or 0)
        if chapter_id <= 0 or chapter_index <= 0:
            continue
        mapping[chapter_id] = chapter_index
    return mapping


def _build_overdue_foreshadow_items(
    db: Session,
    *,
    project_id: int,
    current_chapter_index: int,
    limit: int,
) -> list[dict[str, Any]]:
    if current_chapter_index <= 0:
        return []
    gap = max(int(settings.consistency_audit_foreshadow_gap), 1)
    chapter_index_map = _chapter_id_to_index_map(db, project_id)
    stmt = (
        select(ForeshadowingCard)
        .where(
            ForeshadowingCard.project_id == project_id,
            ForeshadowingCard.status == "open",
        )
        .order_by(ForeshadowingCard.updated_at.asc(), ForeshadowingCard.id.asc())
    )
    rows = db.exec(stmt).all()
    issues: list[dict[str, Any]] = []
    for card in rows:
        planted_chapter_id = int(getattr(card, "planted_in_chapter_id", 0) or 0)
        planted_index = chapter_index_map.get(planted_chapter_id)
        if not isinstance(planted_index, int) or planted_index <= 0:
            continue
        stale_chapters = current_chapter_index - planted_index
        if stale_chapters < gap:
            continue
        title = str(getattr(card, "title", "") or "").strip() or "未命名伏笔"
        description = str(getattr(card, "description", "") or "").strip()
        issue = _normalize_issue(
            {
                "type": "foreshadow_overdue",
                "severity": "medium" if stale_chapters < (gap * 2) else "high",
                "title": f"伏笔“{title}”长期未收束",
                "detail": (
                    f"该伏笔埋于第{planted_index}章，距当前第{current_chapter_index}章已过去 {stale_chapters} 章。"
                    f"建议在后续章节安排回收或转化。"
                ),
                "evidence": {
                    "foreshadow_id": int(getattr(card, "id", 0) or 0),
                    "planted_chapter_id": planted_chapter_id,
                    "planted_chapter_index": planted_index,
                    "current_chapter_index": current_chapter_index,
                    "stale_chapters": stale_chapters,
                    "description_preview": _truncate_text(description, 180),
                },
                "suggestion": "在最近 1-2 章安排一次与该伏笔相关的场景推进。",
            },
            chapter_id=planted_chapter_id,
            chapter_index=planted_index,
        )
        if issue:
            issues.append(issue)
        if len(issues) >= limit:
            break
    return issues


def _extract_graph_entities(fact_text: str) -> list[str]:
    pattern = re.compile(r"\s*(.+?)\s*-\[[^\]]+\]->\s*(.+?)(?:\s*\{.*)?$")
    matched = pattern.match(str(fact_text or "").strip())
    if not matched:
        return []
    source = matched.group(1).strip()
    target = matched.group(2).strip()
    values = [source, target]
    entities: list[str] = []
    for value in values:
        if not value or value in entities:
            continue
        entities.append(value)
    return entities


def _build_temporal_heuristic_items(
    *,
    chapter_id: int,
    chapter_index: int,
    chapter_text: str,
    graph_facts: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    text = str(chapter_text or "")
    if not text:
        return []

    intact_keywords = ("完好无损", "毫发无损", "崭新如初", "没有伤痕", "毫无裂痕")
    broken_keywords = ("折断", "断裂", "损坏", "破碎", "已死", "阵亡", "失踪", "遗失")
    if not any(keyword in text for keyword in intact_keywords):
        return []

    issues: list[dict[str, Any]] = []
    for fact in graph_facts:
        fact_text = str(fact.get("fact") or "").strip()
        if not fact_text:
            continue
        if not any(keyword in fact_text for keyword in broken_keywords):
            continue
        entities = _extract_graph_entities(fact_text)
        if entities and not any(entity in text for entity in entities):
            continue

        issue = _normalize_issue(
            {
                "type": "temporal_conflict",
                "severity": "high",
                "title": "章节描述与图谱状态可能冲突",
                "detail": (
                    "正文出现“完好/无损”描述，但当前章节有效图谱包含“折断/损坏/失踪”事实。"
                    "建议核查该实体状态是否需要修正文案或回溯设定。"
                ),
                "evidence": {
                    "fact": _truncate_text(fact_text, 220),
                    "entities": entities[:4],
                },
                "suggestion": "确认当前章时间线后，统一实体状态描述并补一条过渡说明。",
            },
            chapter_id=chapter_id,
            chapter_index=chapter_index,
        )
        if issue:
            issues.append(issue)
        if len(issues) >= limit:
            break
    return issues


def _call_consistency_judge_llm(
    *,
    chapter_id: int,
    chapter_index: int,
    chapter_title: str,
    chapter_preview: str,
    graph_facts: list[dict[str, Any]],
    max_items: int,
) -> list[dict[str, Any]]:
    if not settings.consistency_audit_llm_enabled:
        return []
    model = str(settings.lightrag_llm_model or "").strip()
    base_url = str(settings.lightrag_llm_base_url or "").strip()
    api_key = str(settings.lightrag_llm_api_key or "").strip()
    if not (model and base_url and api_key):
        return []

    graph_preview = [
        {
            "fact": _truncate_text(str(item.get("fact") or ""), 180),
            "confidence": item.get("confidence"),
        }
        for item in graph_facts[:8]
        if isinstance(item, dict)
    ]
    endpoint = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是小说一致性审计器。请识别当前章节与图谱事实的冲突风险。"
                    "仅输出 JSON: "
                    "{\"issues\":[{\"type\":\"temporal_conflict|continuity_risk|foreshadow_overdue\","
                    "\"severity\":\"high|medium|low\","
                    "\"title\":\"...\",\"detail\":\"...\","
                    "\"evidence\":{\"fact\":\"...\"},"
                    "\"suggestion\":\"...\"}]}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "chapter_id": chapter_id,
                        "chapter_index": chapter_index,
                        "chapter_title": chapter_title,
                        "chapter_preview": chapter_preview,
                        "graph_facts": graph_preview,
                        "max_items": max_items,
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
        timeout = httpx.Timeout(float(settings.consistency_audit_llm_timeout_seconds))
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint, json=body, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return []

    content = str(payload.get("choices", [{}])[0].get("message", {}).get("content", "") or "")
    parsed = _extract_json_object(content)
    if not parsed:
        return []
    raw_issues = parsed.get("issues")
    if not isinstance(raw_issues, list):
        return []

    issues: list[dict[str, Any]] = []
    for item in raw_issues:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_issue(item, chapter_id=chapter_id, chapter_index=chapter_index)
        if not normalized:
            continue
        issues.append(normalized)
        if len(issues) >= max_items:
            break
    return issues


def _next_report_key(now: datetime, report_id: str) -> str:
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_id = str(report_id or "").replace("-", "")[:8] or uuid4().hex[:8]
    return f"{_CONSISTENCY_AUDIT_REPORT_PREFIX}{stamp}.{short_id}"


def _row_to_report_dict(row: SettingEntry) -> dict[str, Any]:
    value = row.value if isinstance(row.value, dict) else {}
    generated_at = _coerce_datetime(value.get("generated_at")) or _coerce_datetime(row.updated_at) or _utc_now()
    return {
        "report_id": str(value.get("report_id") or str(row.key).replace(_CONSISTENCY_AUDIT_REPORT_PREFIX, "")),
        "project_id": int(value.get("project_id") or row.project_id),
        "reason": str(value.get("reason") or "consistency_audit"),
        "trigger_source": str(value.get("trigger_source") or "manual"),
        "status": str(value.get("status") or "ok"),
        "summary": value.get("summary") if isinstance(value.get("summary"), dict) else {},
        "items": value.get("items") if isinstance(value.get("items"), list) else [],
        "generated_at": generated_at,
        "generated_by": str(value.get("generated_by") or "system"),
        "stored_key": str(row.key),
    }


def _save_report(
    db: Session,
    *,
    project_id: int,
    report_id: str,
    reason: str,
    trigger_source: str,
    status: str,
    summary: dict[str, Any],
    items: list[dict[str, Any]],
    generated_by: str,
    generated_at: datetime,
) -> dict[str, Any]:
    stored_key = _next_report_key(generated_at, report_id)
    payload = {
        "report_id": report_id,
        "project_id": int(project_id),
        "reason": reason,
        "trigger_source": trigger_source,
        "status": status,
        "summary": summary,
        "items": items,
        "generated_by": str(generated_by or "system")[:128],
        "generated_at": generated_at.astimezone(timezone.utc).isoformat(),
        "source": "consistency_audit_worker",
    }
    row = SettingEntry(
        project_id=project_id,
        key=stored_key,
        value=payload,
        aliases=["consistency-audit", "nightly-audit"],
        created_at=generated_at,
        updated_at=generated_at,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    report = _row_to_report_dict(row)
    report["stored_key"] = stored_key
    return report


def run_consistency_audit(
    db: Session,
    *,
    project_id: int,
    operator_id: str,
    reason: str = "consistency_audit",
    trigger_source: str = "manual",
    force: bool = False,
    max_chapters: int | None = None,
) -> dict[str, Any]:
    if project_id <= 0:
        raise ValueError("project_id is required")

    now = _utc_now()
    reason_norm = _normalize_reason(reason)
    trigger_norm = _normalize_reason(trigger_source)
    chapter_limit = int(max_chapters) if isinstance(max_chapters, int) and max_chapters > 0 else int(settings.consistency_audit_max_chapters)
    chapter_limit = max(chapter_limit, 1)
    max_items = max(int(settings.consistency_audit_max_items), 1)
    chapter_preview_chars = max(int(settings.consistency_audit_chapter_preview_chars), 300)
    graph_limit = max(int(settings.consistency_audit_graph_facts_limit), 1)

    chapter_stmt = (
        select(ProjectChapter)
        .where(ProjectChapter.project_id == project_id)
        .order_by(ProjectChapter.chapter_index.desc(), ProjectChapter.id.desc())
        .limit(chapter_limit)
    )
    chapter_rows_desc = db.exec(chapter_stmt).all()
    chapter_rows = list(reversed(chapter_rows_desc))

    if not chapter_rows:
        report_id = f"audit-{uuid4().hex[:12]}"
        summary = {
            "chapters_scanned": [],
            "issues": 0,
            "temporal_conflicts": 0,
            "foreshadow_overdue": 0,
            "continuity_risks": 0,
            "force": bool(force),
        }
        return _save_report(
            db,
            project_id=project_id,
            report_id=report_id,
            reason=reason_norm,
            trigger_source=trigger_norm,
            status="ok",
            summary=summary,
            items=[],
            generated_by=operator_id or "system",
            generated_at=now,
        )

    collected: list[dict[str, Any]] = []
    chapter_indexes: list[int] = []
    latest_chapter_index = 0
    llm_item_budget = max(1, min(4, max_items // max(len(chapter_rows), 1)))

    for chapter in chapter_rows:
        chapter_id = int(getattr(chapter, "id", 0) or 0)
        chapter_index = int(getattr(chapter, "chapter_index", 0) or 0)
        chapter_title = str(getattr(chapter, "title", "") or "").strip() or f"第{chapter_index}章"
        chapter_content = str(getattr(chapter, "content", "") or "")
        chapter_preview = _truncate_text(chapter_content, chapter_preview_chars)
        if chapter_index > 0:
            chapter_indexes.append(chapter_index)
            latest_chapter_index = max(latest_chapter_index, chapter_index)

        terms = _extract_terms(f"{chapter_title}\n{chapter_preview}", limit=10)
        graph_facts = fetch_neo4j_graph_facts(
            project_id,
            terms,
            limit=graph_limit,
            current_chapter=chapter_index if chapter_index > 0 else None,
        )
        heuristic_items = _build_temporal_heuristic_items(
            chapter_id=chapter_id,
            chapter_index=chapter_index,
            chapter_text=chapter_content,
            graph_facts=graph_facts,
            limit=2,
        )
        llm_items = _call_consistency_judge_llm(
            chapter_id=chapter_id,
            chapter_index=chapter_index,
            chapter_title=chapter_title,
            chapter_preview=chapter_preview,
            graph_facts=graph_facts,
            max_items=llm_item_budget,
        )
        collected.extend(heuristic_items)
        collected.extend(llm_items)

    overdue_items = _build_overdue_foreshadow_items(
        db,
        project_id=project_id,
        current_chapter_index=latest_chapter_index,
        limit=max_items,
    )
    collected.extend(overdue_items)
    merged_items = _merge_issues(collected, limit=max_items)

    temporal_conflicts = len([item for item in merged_items if str(item.get("type")) == "temporal_conflict"])
    foreshadow_overdue = len([item for item in merged_items if str(item.get("type")) == "foreshadow_overdue"])
    continuity_risks = len([item for item in merged_items if str(item.get("type")) == "continuity_risk"])
    summary = {
        "chapters_scanned": chapter_indexes,
        "issues": len(merged_items),
        "temporal_conflicts": temporal_conflicts,
        "foreshadow_overdue": foreshadow_overdue,
        "continuity_risks": continuity_risks,
        "force": bool(force),
    }
    status = "warning" if merged_items else "ok"
    report_id = f"audit-{uuid4().hex[:12]}"
    return _save_report(
        db,
        project_id=project_id,
        report_id=report_id,
        reason=reason_norm,
        trigger_source=trigger_norm,
        status=status,
        summary=summary,
        items=merged_items,
        generated_by=operator_id or "system",
        generated_at=now,
    )


def list_consistency_audit_reports(
    db: Session,
    *,
    project_id: int,
    limit: int = 20,
) -> list[dict[str, Any]]:
    size = max(int(limit), 1)
    stmt = (
        select(SettingEntry)
        .where(
            SettingEntry.project_id == project_id,
            SettingEntry.key.like(f"{_CONSISTENCY_AUDIT_REPORT_PREFIX}%"),
        )
        .order_by(SettingEntry.updated_at.desc(), SettingEntry.id.desc())
        .limit(size)
    )
    rows = db.exec(stmt).all()
    return [_row_to_report_dict(row) for row in rows]

