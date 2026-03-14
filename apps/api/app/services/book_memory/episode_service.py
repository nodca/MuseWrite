from sqlmodel import Session, select

from app.models.book_memory import StoryEpisode


def list_story_episodes(
    db: Session,
    *,
    project_id: int,
    chapter_id: int | None = None,
    limit: int | None = None,
) -> list[StoryEpisode]:
    stmt = select(StoryEpisode).where(StoryEpisode.project_id == project_id)
    if chapter_id is not None:
        stmt = stmt.where(StoryEpisode.chapter_id == chapter_id)
    stmt = stmt.order_by(StoryEpisode.chapter_id.asc(), StoryEpisode.episode_index.asc(), StoryEpisode.id.asc())
    if limit is not None:
        stmt = stmt.limit(max(int(limit), 1))
    return db.exec(stmt).all()


def create_story_episode(
    db: Session,
    *,
    project_id: int,
    episode_index: int,
    title: str,
    summary: str,
    chapter_id: int | None = None,
    scene_beat_id: int | None = None,
    event_type: str = "scene",
    participants: list[str] | None = None,
    location: str = "",
    visibility: str = "public",
    importance: int = 50,
    source_text_ref: dict | None = None,
) -> StoryEpisode:
    episode = StoryEpisode(
        project_id=project_id,
        chapter_id=chapter_id,
        scene_beat_id=scene_beat_id,
        episode_index=int(episode_index),
        title=str(title or "").strip() or "未命名事件",
        summary=str(summary or "").strip(),
        event_type=str(event_type or "scene").strip() or "scene",
        participants=list(participants or []),
        location=str(location or "").strip(),
        visibility=str(visibility or "public").strip() or "public",
        importance=int(importance),
        source_text_ref=dict(source_text_ref or {}),
    )
    db.add(episode)
    db.commit()
    db.refresh(episode)
    return episode
