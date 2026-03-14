from sqlmodel import Session, select

from app.models.book_memory import CharacterKnowledgeState


def list_character_knowledge_states(
    db: Session,
    *,
    project_id: int,
    character_profile_id: int | None = None,
    gained_at_chapter: int | None = None,
) -> list[CharacterKnowledgeState]:
    stmt = select(CharacterKnowledgeState).where(CharacterKnowledgeState.project_id == project_id)
    if character_profile_id is not None:
        stmt = stmt.where(CharacterKnowledgeState.character_profile_id == character_profile_id)
    if gained_at_chapter is not None:
        stmt = stmt.where(CharacterKnowledgeState.gained_at_chapter == gained_at_chapter)
    stmt = stmt.order_by(
        CharacterKnowledgeState.gained_at_chapter.asc(),
        CharacterKnowledgeState.character_profile_id.asc(),
        CharacterKnowledgeState.id.asc(),
    )
    return db.exec(stmt).all()
