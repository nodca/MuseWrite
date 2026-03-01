from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from app.core.auth import AuthPrincipal, get_current_principal
from app.core.database import get_session
from app.schemas.chat import (
    IndexLifecycleDeadLetterRead,
    IndexLifecycleReplayRequest,
    IndexLifecycleReplayResult,
)
from app.services.index_lifecycle_queue import (
    peek_index_lifecycle_dead_letters,
    pop_index_lifecycle_dead_letters,
)
from .chat_helpers import (
    ensure_project_scope_access as _ensure_project_access,
    filter_project_dead_letters as _filter_project_dead_letters,
    replay_dead_letters as _replay_dead_letters,
)

router = APIRouter()


@router.get("/index-lifecycle/dead-letters", response_model=list[IndexLifecycleDeadLetterRead])
def index_lifecycle_dead_letters(
    project_id: int | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    if project_id is None:
        raise HTTPException(status_code=400, detail="project_id is required")
    _ensure_project_access(project_id, principal)
    rows = peek_index_lifecycle_dead_letters(limit=limit)
    return _filter_project_dead_letters(
        rows,
        project_id=project_id,
        fallback_operator_id=principal.user_id,
    )


@router.post("/index-lifecycle/dead-letters/replay", response_model=IndexLifecycleReplayResult)
def replay_index_lifecycle_dead_letters(
    payload: IndexLifecycleReplayRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    if payload.project_id is None:
        raise HTTPException(status_code=400, detail="project_id is required")
    _ensure_project_access(payload.project_id, principal)
    replay_request_id = f"replay-{uuid4().hex[:12]}"
    dead_letters = pop_index_lifecycle_dead_letters(limit=payload.limit, project_id=payload.project_id)
    counters = _replay_dead_letters(
        dead_letters=dead_letters,
        db=db,
        replay_request_id=replay_request_id,
        principal_user_id=principal.user_id,
    )

    return IndexLifecycleReplayResult(
        requested=int(payload.limit),
        project_id=payload.project_id,
        replayed=int(counters.get("replayed", 0)),
        requeue_failed=int(counters.get("requeue_failed", 0)),
        skipped_invalid=int(counters.get("skipped_invalid", 0)),
        replay_request_id=replay_request_id,
    )
