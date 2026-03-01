from fastapi import APIRouter, Depends, Query
from sqlmodel import Session

from app.core.auth import AuthPrincipal, get_current_principal
from app.core.database import get_session
from app.schemas.chat import (
    EntityMergeScanRequest,
    EntityMergeScanResult,
    GraphCandidateBatchReviewRequest,
    GraphCandidateBatchReviewResponse,
    GraphCandidateListResponse,
)
from app.services.retrieval_adapters import list_neo4j_graph_candidates
from .chat_helpers import (
    ensure_project_scope_access as _ensure_project_access,
    execute_entity_merge_scan as _execute_entity_merge_scan,
    review_graph_candidate_batch as _review_graph_candidate_batch,
)

router = APIRouter()


@router.get("/projects/{project_id}/graph-candidates", response_model=GraphCandidateListResponse)
def project_graph_candidates(
    project_id: int,
    page: int = Query(default=1, ge=1, le=100000),
    page_size: int = Query(default=50, ge=1, le=200),
    keyword: str | None = Query(default=None, max_length=128),
    source_ref: str | None = Query(default=None, max_length=512),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    chapter_index: int | None = Query(default=None, ge=1),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    items, total = list_neo4j_graph_candidates(
        project_id,
        keyword=str(keyword or "").strip(),
        source_ref=str(source_ref or "").strip(),
        min_confidence=min_confidence,
        page=page,
        page_size=page_size,
        current_chapter=chapter_index,
    )
    return GraphCandidateListResponse(
        project_id=project_id,
        page=page,
        page_size=page_size,
        total=max(int(total), 0),
        items=items,
    )


@router.post("/projects/{project_id}/graph-candidates/review", response_model=GraphCandidateBatchReviewResponse)
def review_project_graph_candidates(
    project_id: int,
    payload: GraphCandidateBatchReviewRequest,
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    chapter_index = int(payload.chapter_index or 0) if payload.chapter_index else None
    result = _review_graph_candidate_batch(
        project_id=project_id,
        decision=payload.decision,
        fact_keys=payload.fact_keys,
        manual_confirmed=bool(payload.manual_confirmed),
        chapter_index=chapter_index,
        operator_id=principal.user_id,
    )
    return GraphCandidateBatchReviewResponse(
        project_id=project_id,
        decision=str(result.get("decision") or "confirm"),
        requested_count=int(result.get("requested_count") or 0),
        reviewed_count=int(result.get("reviewed_count") or 0),
        fact_keys=list(result.get("fact_keys") or []),
    )


@router.post("/projects/{project_id}/entity-merge/scan", response_model=EntityMergeScanResult)
def scan_entity_merge_candidates(
    project_id: int,
    payload: EntityMergeScanRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    outcome = _execute_entity_merge_scan(
        project_id=project_id,
        run_mode=payload.run_mode,
        max_proposals=payload.max_proposals,
        operator_id=principal.user_id,
        db=db,
    )
    return EntityMergeScanResult(
        project_id=project_id,
        run_mode=str(outcome.get("run_mode") or "sync"),
        queued=bool(outcome.get("queued")),
        result=dict(outcome.get("result") or {}),
    )
