from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from app.core.auth import AuthPrincipal, get_current_principal
from app.core.database import get_session
from app.schemas.chat import (
    ForeshadowingCardCreateRequest,
    ForeshadowingCardDeleteResult,
    ForeshadowingCardRead,
    ForeshadowingCardUpdateRequest,
    ProjectChapterCreateRequest,
    ProjectChapterDeleteRequest,
    ProjectChapterDeleteResult,
    ProjectChapterMoveRequest,
    ProjectChapterRead,
    ProjectChapterReorderRequest,
    ProjectChapterRevisionRead,
    ProjectChapterRollbackRequest,
    ProjectChapterSaveRequest,
    ProjectVolumeCreateRequest,
    ProjectVolumeDeleteResult,
    ProjectVolumeRead,
    ProjectVolumeUpdateRequest,
    SceneBeatCreateRequest,
    SceneBeatDeleteResult,
    SceneBeatRead,
    SceneBeatUpdateRequest,
    VolumeMemoryConsolidationRequest,
    VolumeMemoryConsolidationResponse,
)
from app.services.chat_service import (
    DraftVersionConflictError,
    consolidate_volume_memory,
    create_foreshadowing_card,
    create_project_chapter,
    create_project_volume,
    create_scene_beat,
    delete_foreshadowing_card,
    delete_project_chapter,
    delete_project_volume,
    delete_scene_beat,
    get_project_chapter,
    list_foreshadowing_cards,
    list_overdue_foreshadowing_cards,
    list_project_chapter_revisions_with_semantic,
    list_project_chapters,
    list_project_volumes,
    list_scene_beats,
    move_project_chapter,
    reorder_project_chapters,
    rollback_project_chapter,
    save_project_chapter,
    update_foreshadowing_card,
    update_project_volume,
    update_scene_beat,
)
from .chat_helpers import ensure_project_scope_access as _ensure_project_access

router = APIRouter()


@router.get("/projects/{project_id}/volumes", response_model=list[ProjectVolumeRead])
def project_volumes(
    project_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    return list_project_volumes(db, project_id)


@router.post("/projects/{project_id}/volumes", response_model=ProjectVolumeRead)
def create_volume(
    project_id: int,
    payload: ProjectVolumeCreateRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    return create_project_volume(
        db,
        project_id=project_id,
        title=payload.title,
        outline=payload.outline,
    )


@router.put("/projects/{project_id}/volumes/{volume_id}", response_model=ProjectVolumeRead)
def save_volume(
    project_id: int,
    volume_id: int,
    payload: ProjectVolumeUpdateRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return update_project_volume(
            db,
            project_id=project_id,
            volume_id=volume_id,
            title=payload.title,
            outline=payload.outline,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/projects/{project_id}/volumes/{volume_id}", response_model=ProjectVolumeDeleteResult)
def remove_volume(
    project_id: int,
    volume_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        deleted_volume_id, fallback_volume_id = delete_project_volume(
            db,
            project_id=project_id,
            volume_id=volume_id,
        )
        return ProjectVolumeDeleteResult(
            deleted_volume_id=deleted_volume_id,
            fallback_volume_id=fallback_volume_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post(
    "/projects/{project_id}/volumes/{volume_id}/memory/consolidate",
    response_model=VolumeMemoryConsolidationResponse,
)
def consolidate_volume_semantic_memory(
    project_id: int,
    volume_id: int,
    payload: VolumeMemoryConsolidationRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return consolidate_volume_memory(
            db,
            project_id=project_id,
            volume_id=volume_id,
            operator_id=principal.user_id,
            force=bool(payload.force),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/projects/{project_id}/chapters", response_model=list[ProjectChapterRead])
def project_chapters(
    project_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    return list_project_chapters(db, project_id)


@router.post("/projects/{project_id}/chapters", response_model=ProjectChapterRead)
def create_chapter(
    project_id: int,
    payload: ProjectChapterCreateRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    return create_project_chapter(
        db,
        project_id=project_id,
        operator_id=principal.user_id,
        title=payload.title,
        volume_id=payload.volume_id,
    )


@router.post("/projects/{project_id}/chapters/reorder", response_model=list[ProjectChapterRead])
def reorder_chapters(
    project_id: int,
    payload: ProjectChapterReorderRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return reorder_project_chapters(
            db,
            project_id=project_id,
            ordered_ids=payload.ordered_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/projects/{project_id}/chapters/{chapter_id}", response_model=ProjectChapterRead)
def project_chapter(
    project_id: int,
    chapter_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    chapter = get_project_chapter(db, project_id, chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="chapter not found")
    return chapter


@router.put("/projects/{project_id}/chapters/{chapter_id}", response_model=ProjectChapterRead)
def save_chapter(
    project_id: int,
    chapter_id: int,
    payload: ProjectChapterSaveRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return save_project_chapter(
            db,
            project_id=project_id,
            chapter_id=chapter_id,
            title=payload.title,
            content=payload.content,
            volume_id=payload.volume_id,
            operator_id=principal.user_id,
            expected_version=payload.expected_version,
        )
    except DraftVersionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/projects/{project_id}/chapters/{chapter_id}/move", response_model=ProjectChapterRead)
def move_chapter(
    project_id: int,
    chapter_id: int,
    payload: ProjectChapterMoveRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return move_project_chapter(
            db,
            project_id=project_id,
            chapter_id=chapter_id,
            direction=payload.direction,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get(
    "/projects/{project_id}/chapters/{chapter_id}/revisions",
    response_model=list[ProjectChapterRevisionRead],
)
def project_chapter_revisions(
    project_id: int,
    chapter_id: int,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return list_project_chapter_revisions_with_semantic(
            db,
            project_id=project_id,
            chapter_id=chapter_id,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/projects/{project_id}/chapters/{chapter_id}/rollback", response_model=ProjectChapterRead)
def rollback_chapter(
    project_id: int,
    chapter_id: int,
    payload: ProjectChapterRollbackRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return rollback_project_chapter(
            db,
            project_id=project_id,
            chapter_id=chapter_id,
            target_version=payload.target_version,
            operator_id=principal.user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/projects/{project_id}/chapters/{chapter_id}", response_model=ProjectChapterDeleteResult)
def delete_chapter(
    project_id: int,
    chapter_id: int,
    payload: ProjectChapterDeleteRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        deleted_chapter_id, active_chapter_id = delete_project_chapter(
            db,
            project_id=project_id,
            chapter_id=chapter_id,
            operator_id=principal.user_id,
        )
        return ProjectChapterDeleteResult(
            deleted_chapter_id=deleted_chapter_id,
            active_chapter_id=active_chapter_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/projects/{project_id}/chapters/{chapter_id}/scene-beats", response_model=list[SceneBeatRead])
def chapter_scene_beats(
    project_id: int,
    chapter_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return list_scene_beats(db, project_id=project_id, chapter_id=chapter_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/projects/{project_id}/chapters/{chapter_id}/scene-beats", response_model=SceneBeatRead)
def create_chapter_scene_beat(
    project_id: int,
    chapter_id: int,
    payload: SceneBeatCreateRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return create_scene_beat(
            db,
            project_id=project_id,
            chapter_id=chapter_id,
            content=payload.content,
            status=payload.status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/projects/{project_id}/chapters/{chapter_id}/scene-beats/{beat_id}", response_model=SceneBeatRead)
def save_chapter_scene_beat(
    project_id: int,
    chapter_id: int,
    beat_id: int,
    payload: SceneBeatUpdateRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return update_scene_beat(
            db,
            project_id=project_id,
            chapter_id=chapter_id,
            beat_id=beat_id,
            content=payload.content,
            status=payload.status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete(
    "/projects/{project_id}/chapters/{chapter_id}/scene-beats/{beat_id}",
    response_model=SceneBeatDeleteResult,
)
def remove_chapter_scene_beat(
    project_id: int,
    chapter_id: int,
    beat_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        deleted_beat_id = delete_scene_beat(
            db,
            project_id=project_id,
            chapter_id=chapter_id,
            beat_id=beat_id,
        )
        return SceneBeatDeleteResult(deleted_beat_id=deleted_beat_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/projects/{project_id}/foreshadowing-cards", response_model=list[ForeshadowingCardRead])
def project_foreshadowing_cards(
    project_id: int,
    status: str | None = Query(default=None),
    overdue_for_chapter_id: int | None = Query(default=None, ge=1),
    chapter_gap: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    if overdue_for_chapter_id is not None:
        return list_overdue_foreshadowing_cards(
            db,
            project_id=project_id,
            current_chapter_id=overdue_for_chapter_id,
            chapter_gap=chapter_gap,
        )
    return list_foreshadowing_cards(db, project_id=project_id, status=status)


@router.post("/projects/{project_id}/foreshadowing-cards", response_model=ForeshadowingCardRead)
def create_project_foreshadowing_card(
    project_id: int,
    payload: ForeshadowingCardCreateRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return create_foreshadowing_card(
            db,
            project_id=project_id,
            title=payload.title,
            description=payload.description,
            planted_in_chapter_id=payload.planted_in_chapter_id,
            source_action_id=payload.source_action_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/projects/{project_id}/foreshadowing-cards/{card_id}", response_model=ForeshadowingCardRead)
def save_project_foreshadowing_card(
    project_id: int,
    card_id: int,
    payload: ForeshadowingCardUpdateRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return update_foreshadowing_card(
            db,
            project_id=project_id,
            card_id=card_id,
            title=payload.title,
            description=payload.description,
            status=payload.status,
            planted_in_chapter_id=payload.planted_in_chapter_id,
            resolved_in_chapter_id=payload.resolved_in_chapter_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/projects/{project_id}/foreshadowing-cards/{card_id}", response_model=ForeshadowingCardDeleteResult)
def remove_project_foreshadowing_card(
    project_id: int,
    card_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        deleted_foreshadow_id = delete_foreshadowing_card(
            db,
            project_id=project_id,
            card_id=card_id,
        )
        return ForeshadowingCardDeleteResult(deleted_foreshadow_id=deleted_foreshadow_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
