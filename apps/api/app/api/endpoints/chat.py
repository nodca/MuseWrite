from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.core.auth import (
    AuthPrincipal,
    filter_accessible_project_ids,
    get_current_principal,
)
from app.core.database import get_session
from app.schemas.chat import ChatStreamRequest
from app.services.chat_service import (
    append_message,
    build_session_title,
    create_session,
    resolve_model_profile_runtime,
    update_message_content,
)
from app.services.context_compiler import compile_context_bundle
from app.services.llm_provider import generate_chat, generate_tot_brainstorm
from app.services.telemetry import emit_chat_trace
from .chat_helpers import (
    build_action_provenance as _build_action_provenance,
    create_proposed_actions as _create_proposed_actions,
    enforce_quality_gate as _enforce_quality_gate,
    ensure_project_scope_access as _ensure_project_access,
    ensure_session_member_access as _ensure_session_access,
)
from .chat_stream_pipeline import stream_chat_events
from .chat_ghost_text import router as ghost_text_router
from .chat_documents import router as documents_router
from .chat_graph_ops import router as graph_ops_router
from .chat_index_lifecycle import router as index_lifecycle_router
from .chat_actions import router as actions_router
from .chat_sessions import router as sessions_router
from .chat_project_assets import router as project_assets_router
from .chat_story_workspace import router as story_workspace_router
from .chat_runtime_ops import router as runtime_ops_router

router = APIRouter(prefix="/chat", tags=["chat"])
router.include_router(actions_router)
router.include_router(sessions_router)
router.include_router(documents_router)
router.include_router(graph_ops_router)
router.include_router(index_lifecycle_router)
router.include_router(project_assets_router)
router.include_router(story_workspace_router)
router.include_router(runtime_ops_router)
router.include_router(ghost_text_router)


@router.post("/stream")
async def chat_stream(
    payload: ChatStreamRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(payload.project_id, principal)
    try:
        runtime_model_profile = resolve_model_profile_runtime(
            db,
            project_id=payload.project_id,
            profile_id=payload.model_profile_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    safe_reference_project_ids = filter_accessible_project_ids(
        principal.user_id,
        payload.reference_project_ids,
    )
    if payload.session_id is None:
        session = create_session(
            db=db,
            project_id=payload.project_id,
            user_id=principal.user_id,
            title=build_session_title(payload.content),
        )
    else:
        session = _ensure_session_access(
            db,
            payload.session_id,
            principal,
            expected_project_id=payload.project_id,
        )

    append_message(
        db=db,
        session_id=session.id,
        role="user",
        content=payload.content,
        model=payload.model,
    )
    compiled_bundle = compile_context_bundle(
        db,
        session_id=session.id,
        project_id=payload.project_id,
        chapter_id=payload.chapter_id,
        scene_beat_id=payload.scene_beat_id,
        prompt_template_id=payload.prompt_template_id,
        user_input=payload.content,
        pov_mode=payload.pov_mode,
        pov_anchor=payload.pov_anchor,
        rag_mode_override=payload.rag_mode,
        deterministic_first=payload.deterministic_first,
        thinking_enabled=payload.thinking_enabled,
        reference_project_ids=safe_reference_project_ids,
        context_window_profile=payload.context_window_profile,
        budget_mode=payload.budget_mode,
        current_location=payload.current_location,
        temperature_profile=payload.temperature_profile,
    )
    assistant_msg = append_message(
        db=db,
        session_id=session.id,
        role="assistant",
        content="",
        model=payload.model,
    )

    return StreamingResponse(
        stream_chat_events(
            payload=payload,
            db=db,
            principal=principal,
            session=session,
            assistant_msg=assistant_msg,
            compiled_bundle=compiled_bundle,
            runtime_model_profile=runtime_model_profile,
            generate_chat_fn=generate_chat,
            generate_tot_brainstorm_fn=generate_tot_brainstorm,
            enforce_quality_gate_fn=_enforce_quality_gate,
            create_proposed_actions_fn=_create_proposed_actions,
            build_action_provenance_fn=_build_action_provenance,
            emit_chat_trace_fn=emit_chat_trace,
            update_message_content_fn=update_message_content,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

