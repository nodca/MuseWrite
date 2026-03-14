from typing import Iterable

from sqlmodel import Session, select

from app.models.chat import ActionAuditLog, ChatAction, ChatMessage, ChatSession
from app.services.chat_service._common import _utc_now


def build_session_title(user_text: str, limit: int = 24) -> str:
    title = user_text.strip().replace("\n", " ")
    if len(title) <= limit:
        return title or "新对话"
    return title[:limit].rstrip() + "..."


def create_session(db: Session, project_id: int, user_id: str, title: str) -> ChatSession:
    session = ChatSession(project_id=project_id, user_id=user_id, title=title, updated_at=_utc_now())
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session_by_id(db: Session, session_id: int) -> ChatSession | None:
    return db.get(ChatSession, session_id)


def list_project_sessions(
    db: Session,
    *,
    project_id: int,
    user_id: str,
    limit: int = 24,
) -> Iterable[ChatSession]:
    size = max(min(int(limit), 100), 1)
    stmt = (
        select(ChatSession)
        .where(
            ChatSession.project_id == project_id,
            ChatSession.user_id == user_id,
        )
        .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        .limit(size)
    )
    return db.exec(stmt).all()


def update_session_title(
    db: Session,
    *,
    session_id: int,
    title: str,
) -> ChatSession:
    session = db.get(ChatSession, session_id)
    if not session:
        raise ValueError("chat session not found")
    normalized_title = str(title or "").strip()
    if not normalized_title:
        raise ValueError("title is required")
    session.title = normalized_title[:255]
    session.updated_at = _utc_now()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def delete_session_with_children(
    db: Session,
    *,
    session_id: int,
) -> int:
    session = db.get(ChatSession, session_id)
    if not session:
        raise ValueError("chat session not found")

    action_rows = db.exec(select(ChatAction).where(ChatAction.session_id == session_id)).all()
    for action in action_rows:
        log_rows = db.exec(select(ActionAuditLog).where(ActionAuditLog.action_id == int(action.id or 0))).all()
        for log in log_rows:
            db.delete(log)
        db.delete(action)

    message_rows = db.exec(select(ChatMessage).where(ChatMessage.session_id == session_id)).all()
    for message in message_rows:
        db.delete(message)

    deleted_session_id = int(session.id or 0)
    db.delete(session)
    db.commit()
    return deleted_session_id
