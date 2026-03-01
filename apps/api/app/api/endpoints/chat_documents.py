from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.core.auth import AuthPrincipal, get_current_principal
from app.core.database import get_session
from app.schemas.chat import (
    LightRAGDeleteDocumentsRequest,
    LightRAGInsertTextRequest,
    LightRAGListDocumentsRequest,
)
from app.services.context_compiler import preheat_context_pack
from app.services.lightrag_documents import (
    delete_documents,
    get_pipeline_status,
    insert_text_document,
    list_project_documents,
)
from .chat_helpers import ensure_project_scope_access as _ensure_project_access

router = APIRouter()


@router.post("/projects/{project_id}/context-pack/preheat")
def preheat_project_context_pack(
    project_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    return preheat_context_pack(db, project_id)


@router.post("/projects/{project_id}/documents/text")
def lightrag_insert_project_text_document(
    project_id: int,
    payload: LightRAGInsertTextRequest,
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return insert_text_document(
            project_id=project_id,
            text=payload.text,
            file_source=payload.file_source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/projects/{project_id}/documents/paginated")
def lightrag_list_project_documents(
    project_id: int,
    payload: LightRAGListDocumentsRequest,
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return list_project_documents(
            project_id=project_id,
            page=payload.page,
            page_size=payload.page_size,
            status_filter=payload.status_filter,
            sort_field=payload.sort_field,
            sort_direction=payload.sort_direction,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.delete("/projects/{project_id}/documents")
def lightrag_delete_project_documents(
    project_id: int,
    payload: LightRAGDeleteDocumentsRequest,
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    # project_id is kept in route for API consistency with project-scoped panel operations.
    # Native deletion is delegated to LightRAG documents API.
    _ = project_id
    try:
        return delete_documents(
            doc_ids=payload.doc_ids,
            delete_file=payload.delete_file,
            delete_llm_cache=payload.delete_llm_cache,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/projects/{project_id}/documents/pipeline-status")
def lightrag_documents_pipeline_status(
    project_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    _ = project_id
    try:
        return get_pipeline_status()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
