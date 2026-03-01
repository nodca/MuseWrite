from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from app.core.auth import AuthPrincipal, get_current_principal
from app.core.database import get_session
from app.schemas.chat import ChatSessionDeleteResult, ChatSessionRead, ChatSessionUpdateRequest
from app.services.chat_service import (
    delete_session_with_children,
    list_project_sessions,
    update_session_title,
)
from .chat_helpers import (
    ensure_project_scope_access as _ensure_project_access,
    ensure_session_member_access as _ensure_session_access,
)

router = APIRouter()


@router.get("/projects/{project_id}/sessions", response_model=list[ChatSessionRead])
def project_sessions(
    project_id: int,
    limit: int = Query(default=24, ge=1, le=100),
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    return list_project_sessions(
        db,
        project_id=project_id,
        user_id=principal.user_id,
        limit=limit,
    )


@router.put("/projects/{project_id}/sessions/{session_id}", response_model=ChatSessionRead)
def rename_project_session(
    project_id: int,
    session_id: int,
    payload: ChatSessionUpdateRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    _ensure_session_access(db, session_id, principal, expected_project_id=project_id)
    try:
        return update_session_title(
            db,
            session_id=session_id,
            title=payload.title,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/projects/{project_id}/sessions/{session_id}", response_model=ChatSessionDeleteResult)
def remove_project_session(
    project_id: int,
    session_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    _ensure_session_access(db, session_id, principal, expected_project_id=project_id)
    try:
        deleted_session_id = delete_session_with_children(
            db,
            session_id=session_id,
        )
        return ChatSessionDeleteResult(deleted_session_id=deleted_session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
