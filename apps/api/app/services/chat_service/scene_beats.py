from typing import Any, Iterable

from sqlmodel import Session, select

from app.models.content import ChapterSceneBeat, ForeshadowingCard, ProjectChapter
from app.services.book_memory.consolidation_queue import enqueue_book_memory_consolidation_job
from app.services.chat_service._common import _utc_now
from app.services.chat_service.chapters import get_project_chapter


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
    operator_id: str = "system",
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
    enqueue_book_memory_consolidation_job(
        project_id=project_id,
        chapter_id=chapter_id,
        scene_beat_id=int(beat.id or 0) or None,
        operator_id=operator_id,
        reason="scene_beat_created",
        idempotency_key=f"book-memory:scene:{project_id}:{chapter_id}:{int(beat.id or 0)}:{int(beat.updated_at.timestamp())}",
        db=db,
    )
    db.commit()
    return beat


def update_scene_beat(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    beat_id: int,
    content: str,
    status: str,
    operator_id: str = "system",
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
    enqueue_book_memory_consolidation_job(
        project_id=project_id,
        chapter_id=chapter_id,
        scene_beat_id=beat_id,
        operator_id=operator_id,
        reason="scene_beat_updated",
        idempotency_key=f"book-memory:scene:{project_id}:{chapter_id}:{beat_id}:{int(beat.updated_at.timestamp())}",
        db=db,
    )
    db.commit()
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
    enqueue_book_memory_consolidation_job(
        project_id=project_id,
        chapter_id=chapter_id,
        operator_id="system",
        reason="scene_beat_deleted",
        idempotency_key=f"book-memory:scene-delete:{project_id}:{chapter_id}:{deleted_id}",
        db=db,
    )
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
