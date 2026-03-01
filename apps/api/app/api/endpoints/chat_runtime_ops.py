from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from app.core.auth import AuthPrincipal, get_current_principal
from app.core.database import get_session
from app.schemas.chat import (
    ConsistencyAuditReportRead,
    ConsistencyAuditRunRequest,
    ConsistencyAuditRunResponse,
    GraphTimelineSnapshotRead,
    ModelProfileDeleteResult,
    ModelProfileRead,
    ModelProfileUpsertRequest,
)
from app.services.chat_service import (
    activate_model_profile,
    create_model_profile,
    delete_model_profile,
    list_model_profiles,
    list_project_chapters,
    update_model_profile,
)
from app.services.consistency_audit_queue import enqueue_consistency_audit_job
from app.services.consistency_audit_service import (
    list_consistency_audit_reports,
    run_consistency_audit,
)
from app.services.retrieval_adapters import fetch_neo4j_graph_timeline_snapshot
from .chat_helpers import ensure_project_scope_access as _ensure_project_access

router = APIRouter()


@router.get("/projects/{project_id}/model-profiles", response_model=list[ModelProfileRead])
def project_model_profiles(
    project_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    return list_model_profiles(db, project_id)


@router.post("/projects/{project_id}/model-profiles", response_model=ModelProfileRead)
def create_project_model_profile(
    project_id: int,
    payload: ModelProfileUpsertRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return create_model_profile(
            db,
            project_id=project_id,
            operator_id=principal.user_id,
            profile_id=payload.profile_id,
            name=payload.name,
            provider=payload.provider or "openai_compatible",
            base_url=payload.base_url,
            api_key=payload.api_key,
            model=payload.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/projects/{project_id}/model-profiles/{profile_id}", response_model=ModelProfileRead)
def save_project_model_profile(
    project_id: int,
    profile_id: str,
    payload: ModelProfileUpsertRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return update_model_profile(
            db,
            project_id=project_id,
            profile_id=profile_id,
            operator_id=principal.user_id,
            name=payload.name,
            provider=payload.provider,
            base_url=payload.base_url,
            api_key=payload.api_key,
            api_key_supplied=("api_key" in payload.model_fields_set),
            model=payload.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/projects/{project_id}/model-profiles/{profile_id}/activate", response_model=ModelProfileRead)
def activate_project_model_profile(
    project_id: int,
    profile_id: str,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return activate_model_profile(
            db,
            project_id=project_id,
            profile_id=profile_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/projects/{project_id}/model-profiles/{profile_id}", response_model=ModelProfileDeleteResult)
def remove_project_model_profile(
    project_id: int,
    profile_id: str,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        deleted_profile_id = delete_model_profile(
            db,
            project_id=project_id,
            profile_id=profile_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ModelProfileDeleteResult(deleted_profile_id=deleted_profile_id)


@router.get("/projects/{project_id}/consistency-audits", response_model=list[ConsistencyAuditReportRead])
def project_consistency_audits(
    project_id: int,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    return list_consistency_audit_reports(db, project_id=project_id, limit=limit)


@router.post("/projects/{project_id}/consistency-audits/run", response_model=ConsistencyAuditRunResponse)
def run_project_consistency_audit(
    project_id: int,
    payload: ConsistencyAuditRunRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    reason = str(payload.reason or "manual").strip() or "manual"
    run_mode = str(payload.run_mode or "async").strip().lower()
    max_chapters = int(payload.max_chapters) if isinstance(payload.max_chapters, int) and payload.max_chapters > 0 else None
    if run_mode == "sync":
        report = run_consistency_audit(
            db,
            project_id=project_id,
            operator_id=principal.user_id,
            reason=reason,
            trigger_source="manual_sync",
            force=bool(payload.force),
            max_chapters=max_chapters,
        )
        return ConsistencyAuditRunResponse(
            project_id=project_id,
            queued=False,
            run_mode="sync",
            reason=reason,
            trigger_source="manual_sync",
            report=ConsistencyAuditReportRead.model_validate(report),
        )

    idempotency_key = f"consistency-manual-{project_id}-{uuid4().hex[:10]}"
    queued = enqueue_consistency_audit_job(
        project_id,
        operator_id=principal.user_id,
        reason=reason,
        trigger_source="manual_async",
        idempotency_key=idempotency_key,
        force=bool(payload.force),
        max_chapters=max_chapters,
        db=db,
    )
    db.commit()
    return ConsistencyAuditRunResponse(
        project_id=project_id,
        queued=bool(queued),
        run_mode="async",
        reason=reason,
        trigger_source="manual_async",
        idempotency_key=idempotency_key,
        report=None,
    )


@router.get("/projects/{project_id}/graph-timeline", response_model=GraphTimelineSnapshotRead)
def project_graph_timeline(
    project_id: int,
    chapter_index: int = Query(default=0, ge=0, le=100000),
    limit: int = Query(default=240, ge=20, le=1200),
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    resolved_chapter_index = int(chapter_index)
    if resolved_chapter_index <= 0:
        chapters = list_project_chapters(db, project_id)
        resolved_chapter_index = max((int(item.chapter_index or 0) for item in chapters), default=0)
    snapshot = fetch_neo4j_graph_timeline_snapshot(
        project_id,
        current_chapter=resolved_chapter_index if resolved_chapter_index > 0 else None,
        limit=limit,
    )
    snapshot["chapter_index"] = resolved_chapter_index
    return GraphTimelineSnapshotRead.model_validate(snapshot)
