from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from app.core.auth import AuthPrincipal, get_current_principal
from app.core.database import get_session
from app.schemas.chat import (
    PromptTemplateRead,
    PromptTemplateRevisionRead,
    PromptTemplateRollbackRequest,
    PromptTemplateUpsertRequest,
    SettingEntryRead,
    StoryCardRead,
)
from app.services.chat_service import (
    create_prompt_template,
    delete_prompt_template,
    list_cards,
    list_prompt_template_revisions,
    list_prompt_templates,
    list_settings,
    rollback_prompt_template,
    update_prompt_template,
)
from .chat_helpers import ensure_project_scope_access as _ensure_project_access

router = APIRouter()


@router.get("/projects/{project_id}/settings", response_model=list[SettingEntryRead])
def project_settings(
    project_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    return list_settings(db, project_id)


@router.get("/projects/{project_id}/cards", response_model=list[StoryCardRead])
def project_cards(
    project_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    return list_cards(db, project_id)


@router.get("/projects/{project_id}/prompt-templates", response_model=list[PromptTemplateRead])
def project_prompt_templates(
    project_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    return list_prompt_templates(db, project_id)


@router.post("/projects/{project_id}/prompt-templates", response_model=PromptTemplateRead)
def create_project_prompt_template(
    project_id: int,
    payload: PromptTemplateUpsertRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return create_prompt_template(
            db,
            project_id=project_id,
            name=payload.name,
            system_prompt=payload.system_prompt,
            user_prompt_prefix=payload.user_prompt_prefix,
            knowledge_setting_keys=payload.knowledge_setting_keys,
            knowledge_card_ids=payload.knowledge_card_ids,
            operator_id=principal.user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/projects/{project_id}/prompt-templates/{template_id}", response_model=PromptTemplateRead)
def save_project_prompt_template(
    project_id: int,
    template_id: int,
    payload: PromptTemplateUpsertRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return update_prompt_template(
            db,
            project_id=project_id,
            template_id=template_id,
            name=payload.name,
            system_prompt=payload.system_prompt,
            user_prompt_prefix=payload.user_prompt_prefix,
            knowledge_setting_keys=payload.knowledge_setting_keys,
            knowledge_card_ids=payload.knowledge_card_ids,
            operator_id=principal.user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get(
    "/projects/{project_id}/prompt-templates/{template_id}/revisions",
    response_model=list[PromptTemplateRevisionRead],
)
def project_prompt_template_revisions(
    project_id: int,
    template_id: int,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return list_prompt_template_revisions(
            db,
            project_id=project_id,
            template_id=template_id,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post(
    "/projects/{project_id}/prompt-templates/{template_id}/rollback",
    response_model=PromptTemplateRead,
)
def rollback_project_prompt_template(
    project_id: int,
    template_id: int,
    payload: PromptTemplateRollbackRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return rollback_prompt_template(
            db,
            project_id=project_id,
            template_id=template_id,
            target_version=payload.target_version,
            operator_id=principal.user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/projects/{project_id}/prompt-templates/{template_id}")
def remove_project_prompt_template(
    project_id: int,
    template_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        deleted_id = delete_prompt_template(db, project_id=project_id, template_id=template_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"deleted_template_id": deleted_id}
