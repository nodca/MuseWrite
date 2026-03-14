from typing import Any
import re

from sqlmodel import Session, select

from app.models.content import SettingEntry
from app.services.chat_service._common import (
    _utc_now,
    _MODEL_PROFILE_PREFIX,
    _MODEL_PROFILE_ACTIVE_KEY,
    _MODEL_PROFILE_ALLOWED_PROVIDERS,
    _normalize_optional_text,
    _mask_secret,
)


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
    if raw not in _MODEL_PROFILE_ALLOWED_PROVIDERS:
        raise ValueError("provider must be one of openai_compatible/deepseek/claude/gemini")
    return raw


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
    provider = (
        provider_raw
        if provider_raw in _MODEL_PROFILE_ALLOWED_PROVIDERS
        else "openai_compatible"
    )
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
    provider_raw = str(value.get("provider") or "").strip().lower()
    provider = (
        provider_raw
        if provider_raw in _MODEL_PROFILE_ALLOWED_PROVIDERS
        else "openai_compatible"
    )
    return {
        "profile_id": resolved_profile_id,
        "name": _normalize_optional_text(value.get("name"), max_len=128),
        "provider": provider,
        "base_url": _normalize_optional_text(value.get("base_url"), max_len=512),
        "api_key": _normalize_optional_text(value.get("api_key"), max_len=512),
        "model": _normalize_optional_text(value.get("model"), max_len=128),
    }

