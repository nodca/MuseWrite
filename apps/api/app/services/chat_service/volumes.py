from typing import Any, Iterable
import json

from sqlmodel import Session, select

from app.core.config import settings
from app.models.content import ProjectChapter, ProjectVolume, SettingEntry
from app.services.chat_service._common import _utc_now, VolumeMemoryConsolidationOutput
from app.services.llm_provider import generate_structured_sync


def _build_lightrag_runtime_config(model: str, base_url: str, api_key: str) -> dict[str, str]:
    return {
        "provider": "openai_compatible",
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }


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

    chapter_payload = [
        {
            "chapter_index": int(getattr(item, "chapter_index", 0) or 0),
            "title": str(getattr(item, "title", "") or ""),
            "content_preview": str(getattr(item, "content", "") or "")[: settings.memory_consolidation_preview_chars],
        }
        for item in chapters
    ]
    try:
        structured = generate_structured_sync(
            json.dumps(
                {
                    "volume_title": volume_title,
                    "volume_outline": volume_outline[:1000],
                    "chapters": chapter_payload,
                    "max_facts": max_facts,
                },
                ensure_ascii=False,
            ),
            output_model=VolumeMemoryConsolidationOutput,
            schema_name="volume_memory_consolidation_output",
            context={
                "system": (
                    "你是小说记忆固化器。请把一整卷内容提炼为高密度、可检索、可验证的事实。"
                    "facts 必须是陈述句，禁止编造。"
                    "只输出符合 schema 的 JSON。"
                ),
                "raw_prompt": True,
            },
            runtime_config=_build_lightrag_runtime_config(model, base_url, api_key),
            temperature_override=0.0,
        )
    except Exception:
        return _fallback_consolidated_facts(chapters, max_facts=max_facts), "fallback_llm_error"

    parsed = structured.parsed
    if not parsed:
        return _fallback_consolidated_facts(chapters, max_facts=max_facts), "fallback_invalid_json"
    raw_facts = getattr(parsed, "facts", [])
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

