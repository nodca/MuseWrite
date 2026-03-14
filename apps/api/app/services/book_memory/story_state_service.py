from sqlmodel import Session, select

from app.models.book_memory import StoryStateSnapshot


def get_story_state_snapshot(
    db: Session,
    *,
    project_id: int,
    chapter_id: int | None = None,
    scene_beat_id: int | None = None,
) -> StoryStateSnapshot | None:
    stmt = select(StoryStateSnapshot).where(StoryStateSnapshot.project_id == project_id)
    if chapter_id is None:
        stmt = stmt.where(StoryStateSnapshot.chapter_id.is_(None))
    else:
        stmt = stmt.where(StoryStateSnapshot.chapter_id == chapter_id)
    if scene_beat_id is None:
        stmt = stmt.where(StoryStateSnapshot.scene_beat_id.is_(None))
    else:
        stmt = stmt.where(StoryStateSnapshot.scene_beat_id == scene_beat_id)
    stmt = stmt.order_by(StoryStateSnapshot.updated_at.desc(), StoryStateSnapshot.id.desc())
    return db.exec(stmt).first()


def upsert_story_state_snapshot(
    db: Session,
    *,
    project_id: int,
    volume_id: int | None = None,
    chapter_id: int | None = None,
    scene_beat_id: int | None = None,
    chapter_goal: str = "",
    active_characters: list[str] | None = None,
    current_location: str = "",
    active_conflicts: list[str] | None = None,
    open_questions: list[str] | None = None,
    source_refs: list[dict] | None = None,
) -> StoryStateSnapshot:
    snapshot = get_story_state_snapshot(
        db,
        project_id=project_id,
        chapter_id=chapter_id,
        scene_beat_id=scene_beat_id,
    )
    if snapshot is None:
        snapshot = StoryStateSnapshot(
            project_id=project_id,
            volume_id=volume_id,
            chapter_id=chapter_id,
            scene_beat_id=scene_beat_id,
        )
    snapshot.volume_id = volume_id
    snapshot.chapter_id = chapter_id
    snapshot.scene_beat_id = scene_beat_id
    snapshot.chapter_goal = str(chapter_goal or "").strip()
    snapshot.active_characters = list(active_characters or [])
    snapshot.current_location = str(current_location or "").strip()
    snapshot.active_conflicts = list(active_conflicts or [])
    snapshot.open_questions = list(open_questions or [])
    snapshot.source_refs = list(source_refs or [])
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    return snapshot
