from typing import Any, Iterable

from sqlmodel import Session, select

from app.core.database import engine
from app.models.chat import ChatMessage, ChatSession
from app.services.chat_service._common import _utc_now


def append_message(
    db: Session,
    session_id: int,
    role: str,
    content: str,
    model: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> ChatMessage:
    msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        model=model,
        provenance=provenance or {},
    )
    db.add(msg)

    session = db.get(ChatSession, session_id)
    if session:
        session.updated_at = _utc_now()
        db.add(session)

    db.commit()
    db.refresh(msg)

    return msg


def update_message_content(
    message_id: int,
    content: str,
    *,
    db: Session | None = None,
) -> None:
    def _write(target_db: Session) -> None:
        msg = target_db.get(ChatMessage, message_id)
        if not msg:
            return
        msg.content = content
        session = target_db.get(ChatSession, msg.session_id)
        if session:
            session.updated_at = _utc_now()
            target_db.add(session)
        target_db.add(msg)
        target_db.commit()

    if db is not None:
        _write(db)
        return

    with Session(engine) as managed_db:
        _write(managed_db)


def update_message_provenance(
    message_id: int,
    provenance: dict[str, Any],
    *,
    db: Session | None = None,
) -> None:
    def _write(target_db: Session) -> None:
        msg = target_db.get(ChatMessage, message_id)
        if not msg:
            return
        msg.provenance = provenance if isinstance(provenance, dict) else {}
        session = target_db.get(ChatSession, msg.session_id)
        if session:
            session.updated_at = _utc_now()
            target_db.add(session)
        target_db.add(msg)
        target_db.commit()

    if db is not None:
        _write(db)
        return

    with Session(engine) as managed_db:
        _write(managed_db)


def list_messages(db: Session, session_id: int, *, limit: int | None = None) -> Iterable[ChatMessage]:
    base_stmt = select(ChatMessage).where(ChatMessage.session_id == session_id)
    if limit is not None:
        size = max(int(limit), 1)
        rows = db.exec(base_stmt.order_by(ChatMessage.id.desc()).limit(size)).all()
        rows.reverse()
        return rows
    return db.exec(base_stmt.order_by(ChatMessage.id.asc())).all()
