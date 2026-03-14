import time
from typing import Any

from sqlmodel import Session

from app.core.config import settings
from app.services.chat_service import list_settings, list_cards
from app.services.context_compiler._types import ContextPack, SettingSnapshot, CardSnapshot
from app.services.context_compiler._state import _CONTEXT_PACK_CACHE, _CONTEXT_PACK_LOCK
from app.services.context_compiler._utils import _serialize


def _to_setting_snapshot(row: Any, project_id: int) -> SettingSnapshot:
    raw_value = getattr(row, "value", {})
    raw_aliases = getattr(row, "aliases", [])
    return SettingSnapshot(
        id=int(getattr(row, "id")),
        project_id=project_id,
        key=str(getattr(row, "key") or ""),
        value=raw_value,
        aliases=[str(item).strip() for item in raw_aliases if str(item).strip()] if isinstance(raw_aliases, list) else [],
        value_text=_serialize(raw_value),
        updated_at=getattr(row, "updated_at", None),
    )


def _to_card_snapshot(row: Any, project_id: int) -> CardSnapshot:
    raw_content = getattr(row, "content", {})
    raw_aliases = getattr(row, "aliases", [])
    return CardSnapshot(
        id=int(getattr(row, "id")),
        project_id=project_id,
        title=str(getattr(row, "title") or ""),
        content=raw_content,
        aliases=[str(item).strip() for item in raw_aliases if str(item).strip()] if isinstance(raw_aliases, list) else [],
        content_text=_serialize(raw_content),
        updated_at=getattr(row, "updated_at", None),
    )


def _load_context_pack(db: Session, project_id: int, *, force_refresh: bool = False) -> tuple[list[Any], list[Any], dict[str, Any]]:
    settings_limit = max(int(settings.context_pack_max_settings), 1)
    cards_limit = max(int(settings.context_pack_max_cards), 1)
    if not settings.context_pack_enabled:
        settings_rows = [
            _to_setting_snapshot(row, project_id)
            for row in list_settings(db, project_id, limit=settings_limit)
        ]
        cards_rows = [
            _to_card_snapshot(row, project_id)
            for row in list_cards(db, project_id, limit=cards_limit)
        ]
        return settings_rows, cards_rows, {"enabled": False, "source": "disabled", "age_ms": 0}

    now = time.time()
    ttl = max(int(settings.context_pack_ttl_seconds), 0)
    with _CONTEXT_PACK_LOCK:
        cached = _CONTEXT_PACK_CACHE.get(project_id)
        if cached and not force_refresh and (now - cached.generated_at) <= ttl:
            return (
                cached.settings_rows,
                cached.cards_rows,
                {
                    "enabled": True,
                    "source": "cache_hit",
                    "age_ms": int((now - cached.generated_at) * 1000),
                },
            )

    settings_rows = [
        _to_setting_snapshot(row, project_id)
        for row in list_settings(db, project_id, limit=settings_limit)
    ]
    cards_rows = [
        _to_card_snapshot(row, project_id)
        for row in list_cards(db, project_id, limit=cards_limit)
    ]
    pack = ContextPack(
        project_id=project_id,
        generated_at=now,
        settings_rows=settings_rows,
        cards_rows=cards_rows,
    )
    with _CONTEXT_PACK_LOCK:
        _CONTEXT_PACK_CACHE[project_id] = pack
        if len(_CONTEXT_PACK_CACHE) > 128:
            oldest_project = min(_CONTEXT_PACK_CACHE.items(), key=lambda item: item[1].generated_at)[0]
            _CONTEXT_PACK_CACHE.pop(oldest_project, None)

    return settings_rows, cards_rows, {"enabled": True, "source": "cache_miss", "age_ms": 0}


def preheat_context_pack(db: Session, project_id: int) -> dict[str, Any]:
    settings_rows, cards_rows, _ = _load_context_pack(db, project_id, force_refresh=True)
    return {
        "project_id": project_id,
        "settings_count": len(settings_rows),
        "cards_count": len(cards_rows),
        "ttl_seconds": max(int(settings.context_pack_ttl_seconds), 0),
    }


def _normalize_reference_project_ids(
    value: list[int] | None,
    *,
    current_project_id: int,
    max_items: int = 5,
) -> list[int]:
    if not isinstance(value, list):
        return []
    normalized: list[int] = []
    for raw in value:
        try:
            project_id = int(raw)
        except Exception:
            continue
        if project_id <= 0 or project_id == current_project_id:
            continue
        if project_id in normalized:
            continue
        normalized.append(project_id)
        if len(normalized) >= max(max_items, 1):
            break
    return normalized


def _load_reference_context(
    db: Session,
    reference_project_ids: list[int],
) -> tuple[list[SettingSnapshot], list[CardSnapshot], list[dict[str, int]]]:
    settings_rows: list[SettingSnapshot] = []
    cards_rows: list[CardSnapshot] = []
    project_meta: list[dict[str, int]] = []
    settings_limit = max(int(settings.context_pack_max_settings), 1)
    cards_limit = max(int(settings.context_pack_max_cards), 1)
    for ref_project_id in reference_project_ids:
        ref_settings = [
            _to_setting_snapshot(row, ref_project_id)
            for row in list_settings(db, ref_project_id, limit=settings_limit)
        ]
        ref_cards = [
            _to_card_snapshot(row, ref_project_id)
            for row in list_cards(db, ref_project_id, limit=cards_limit)
        ]
        settings_rows.extend(ref_settings)
        cards_rows.extend(ref_cards)
        project_meta.append(
            {
                "project_id": ref_project_id,
                "settings": len(ref_settings),
                "cards": len(ref_cards),
            }
        )
    return settings_rows, cards_rows, project_meta

