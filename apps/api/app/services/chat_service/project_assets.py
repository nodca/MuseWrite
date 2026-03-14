from typing import Iterable

from sqlmodel import Session, select

from app.models.content import SettingEntry, StoryCard
from app.services.chat_service._common import _INTERNAL_SETTING_PREFIXES


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

