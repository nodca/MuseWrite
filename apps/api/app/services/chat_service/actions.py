from typing import Any, Iterable

from sqlmodel import Session, select

from app.models.chat import ActionAuditLog, ChatAction
from app.services.chat_service._common import _utc_now


def create_action(
    db: Session,
    session_id: int,
    action_type: str,
    payload: dict,
    operator_id: str,
    idempotency_key: str,
) -> ChatAction:
    existing = get_action_by_idempotency_key(db, session_id, idempotency_key)
    if existing:
        return existing

    action = ChatAction(
        session_id=session_id,
        action_type=action_type,
        payload=payload,
        operator_id=operator_id,
        idempotency_key=idempotency_key,
        status="proposed",
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    return action


def is_entity_merge_action_type(action_type: str) -> bool:
    raw = str(action_type or "").strip().lower()
    if not raw:
        return False
    normalized = raw.replace("_", ".").replace("-", ".")
    return normalized.startswith("entity.merge") or normalized.startswith("graph.entity.merge")


def is_manual_merge_operator(operator_id: str) -> bool:
    raw = str(operator_id or "").strip().lower()
    if not raw:
        return False
    blocked_tokens = ("system", "worker", "assistant", "auto", "daemon", "scheduler", "bot")
    return not any(
        raw == token
        or raw.startswith(token)
        or raw.startswith(f"{token}_")
        or raw.startswith(f"{token}-")
        for token in blocked_tokens
    )


def get_action_by_id(db: Session, action_id: int) -> ChatAction | None:
    return db.get(ChatAction, action_id)


def get_action_by_idempotency_key(db: Session, session_id: int, idempotency_key: str) -> ChatAction | None:
    stmt = select(ChatAction).where(
        ChatAction.session_id == session_id, ChatAction.idempotency_key == idempotency_key
    )
    return db.exec(stmt).first()


def list_actions(db: Session, session_id: int) -> Iterable[ChatAction]:
    stmt = select(ChatAction).where(ChatAction.session_id == session_id).order_by(ChatAction.id.asc())
    return db.exec(stmt).all()


def list_action_logs(db: Session, action_id: int) -> Iterable[ActionAuditLog]:
    stmt = select(ActionAuditLog).where(ActionAuditLog.action_id == action_id).order_by(ActionAuditLog.id.asc())
    return db.exec(stmt).all()


def create_action_audit_log(
    db: Session,
    action_id: int,
    event_type: str,
    operator_id: str,
    event_payload: dict | None = None,
) -> ActionAuditLog:
    log = ActionAuditLog(
        action_id=action_id,
        event_type=event_type,
        operator_id=operator_id,
        event_payload=event_payload or {},
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def set_action_status(
    db: Session,
    action: ChatAction,
    status: str,
    *,
    set_applied_at: bool = False,
    set_undone_at: bool = False,
) -> ChatAction:
    action.status = status
    if set_applied_at:
        action.applied_at = _utc_now()
    if set_undone_at:
        action.undone_at = _utc_now()
    db.add(action)
    db.commit()
    db.refresh(action)
    return action




def apply_action_effects(db: Session, action: ChatAction) -> ChatAction:
    from app.services.chat_action_effects import apply_action_effects as _impl
    return _impl(db, action)


def undo_action_effects(db: Session, action: ChatAction) -> ChatAction:
    from app.services.chat_action_effects import undo_action_effects as _impl
    return _impl(db, action)
