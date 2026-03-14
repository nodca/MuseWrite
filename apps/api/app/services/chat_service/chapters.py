from typing import Any, Iterable

from sqlmodel import Session, select

from app.models.chat import ChatAction, ChatSession
from app.models.content import ProjectChapter, ProjectChapterRevision
from app.services.book_memory.consolidation_queue import enqueue_book_memory_consolidation_job
from app.services.chat_service._common import _utc_now, DraftVersionConflictError
from app.services.chat_service.actions import is_entity_merge_action_type
from app.services.chat_service.volumes import _ensure_default_project_volume, get_project_volume


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
    chapter_id = int(chapter.id or 0)
    if chapter_id <= 0:
        raise ValueError("project chapter id missing")
    enqueue_book_memory_consolidation_job(
        project_id=project_id,
        chapter_id=chapter_id,
        operator_id=operator_id,
        reason="chapter_created" if source == "create" else "chapter_saved",
        idempotency_key=f"book-memory:chapter:{project_id}:{chapter_id}:v{int(chapter.version)}",
        db=db,
    )
    db.commit()
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
    enqueue_book_memory_consolidation_job(
        project_id=project_id,
        chapter_id=chapter_id,
        operator_id=operator_id,
        reason="chapter_saved",
        idempotency_key=f"book-memory:chapter:{project_id}:{chapter_id}:v{next_version}",
        db=db,
    )
    db.commit()
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
    enqueue_book_memory_consolidation_job(
        project_id=project_id,
        chapter_id=chapter_id,
        operator_id=operator_id,
        reason="chapter_rollback",
        idempotency_key=f"book-memory:chapter:{project_id}:{chapter_id}:v{next_version}",
        db=db,
    )
    db.commit()
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
