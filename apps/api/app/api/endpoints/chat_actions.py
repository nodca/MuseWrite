from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.core.auth import AuthPrincipal, get_current_principal
from app.core.database import get_session
from app.schemas.chat import (
    ActionAuditLogRead,
    ChatActionCreateRequest,
    ChatActionDecisionRequest,
    ChatActionRead,
    ChatMessageRead,
)
from app.services.chat_service import (
    apply_action_effects,
    create_action,
    create_action_audit_log,
    is_entity_merge_action_type,
    is_manual_merge_operator,
    list_action_logs,
    list_actions,
    list_messages,
    set_action_status,
    undo_action_effects,
)
from .chat_helpers import (
    action_provenance_from_payload as _action_provenance_from_payload,
    ensure_action_session_access as _ensure_action_access,
    ensure_session_member_access as _ensure_session_access,
)

router = APIRouter()


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageRead])
def session_messages(
    session_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_session_access(db, session_id, principal)
    return list_messages(db, session_id)


@router.get("/sessions/{session_id}/actions", response_model=list[ChatActionRead])
def session_actions(
    session_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_session_access(db, session_id, principal)
    return list_actions(db, session_id)


@router.get("/actions/{action_id}/logs", response_model=list[ActionAuditLogRead])
def action_logs(
    action_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_action_access(db, action_id, principal)
    return list_action_logs(db, action_id)


@router.post("/sessions/{session_id}/actions", response_model=ChatActionRead)
def create_session_action(
    session_id: int,
    payload: ChatActionCreateRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_session_access(db, session_id, principal)
    if is_entity_merge_action_type(payload.action_type):
        if bool(payload.payload.get("auto_apply")):
            raise HTTPException(status_code=400, detail="entity merge does not support auto_apply")

    action = create_action(
        db=db,
        session_id=session_id,
        action_type=payload.action_type,
        payload=payload.payload,
        operator_id=principal.user_id,
        idempotency_key=payload.idempotency_key,
    )
    if action.status == "proposed":
        manual_provenance = payload.payload.get("_provenance") if isinstance(payload.payload, dict) else {}
        create_action_audit_log(
            db=db,
            action_id=action.id,
            event_type="proposed",
            operator_id=principal.user_id,
            event_payload={
                "source": "manual_api",
                "payload_keys": sorted(payload.payload.keys()) if isinstance(payload.payload, dict) else [],
                "provenance": manual_provenance if isinstance(manual_provenance, dict) else {},
            },
        )
    return action


@router.post("/actions/{action_id}/apply", response_model=ChatActionRead)
def apply_action(
    action_id: int,
    payload: ChatActionDecisionRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    action = _ensure_action_access(db, action_id, principal)
    if action.status == "applied":
        return action
    if action.status != "proposed":
        raise HTTPException(status_code=409, detail="action is not in proposed state")
    if is_entity_merge_action_type(action.action_type):
        if not bool(payload.event_payload.get("manual_confirmed")):
            raise HTTPException(status_code=400, detail="entity merge requires manual_confirmed=true")
        if not is_manual_merge_operator(principal.user_id):
            raise HTTPException(status_code=403, detail="entity merge can only be applied by a human operator")
    if action.action_type == "graph.confirm_candidates":
        if not bool(payload.event_payload.get("manual_confirmed")):
            raise HTTPException(status_code=400, detail="graph candidate confirmation requires manual_confirmed=true")
        if not is_manual_merge_operator(principal.user_id):
            raise HTTPException(
                status_code=403,
                detail="graph candidate confirmation can only be applied by a human operator",
            )

    create_action_audit_log(
        db=db,
        action_id=action.id,
        event_type="apply_requested",
        operator_id=principal.user_id,
        event_payload={**payload.event_payload, "provenance": _action_provenance_from_payload(action)},
    )

    try:
        action = apply_action_effects(db, action)
    except ValueError as exc:
        db.rollback()
        action = _ensure_action_access(db, action_id, principal)
        action = set_action_status(db, action, "failed")
        create_action_audit_log(
            db=db,
            action_id=action.id,
            event_type="failed",
            operator_id=principal.user_id,
            event_payload={
                "error": str(exc),
                "provenance": _action_provenance_from_payload(action),
            },
        )
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="action apply failed")

    applied_provenance = {}
    if isinstance(action.apply_result, dict):
        raw_applied_provenance = action.apply_result.get("provenance")
        if isinstance(raw_applied_provenance, dict):
            applied_provenance = raw_applied_provenance
    create_action_audit_log(
        db=db,
        action_id=action.id,
        event_type="applied",
        operator_id=principal.user_id,
        event_payload={**payload.event_payload, "provenance": applied_provenance},
    )
    return action


@router.post("/actions/{action_id}/reject", response_model=ChatActionRead)
def reject_action(
    action_id: int,
    payload: ChatActionDecisionRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    action = _ensure_action_access(db, action_id, principal)
    if action.status == "rejected":
        return action
    if action.status != "proposed":
        raise HTTPException(status_code=409, detail="action is not in proposed state")

    action = set_action_status(db, action, "rejected")
    create_action_audit_log(
        db=db,
        action_id=action.id,
        event_type="rejected",
        operator_id=principal.user_id,
        event_payload={**payload.event_payload, "provenance": _action_provenance_from_payload(action)},
    )
    return action


@router.post("/actions/{action_id}/undo", response_model=ChatActionRead)
def undo_action(
    action_id: int,
    payload: ChatActionDecisionRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    action = _ensure_action_access(db, action_id, principal)
    if action.status == "undone":
        return action
    if action.status != "applied":
        raise HTTPException(status_code=409, detail="only applied action can be undone")

    try:
        action = undo_action_effects(db, action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    create_action_audit_log(
        db=db,
        action_id=action.id,
        event_type="undone",
        operator_id=principal.user_id,
        event_payload={
            **payload.event_payload,
            "provenance": (
                action.apply_result.get("provenance", {}) if isinstance(action.apply_result, dict) else {}
            ),
        },
    )
    return action
