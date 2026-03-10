import logging

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlmodel import Session

from app.core.auth import AuthPrincipal, get_current_principal
from app.core.database import get_session
from app.schemas.chat import RewriteRequest, RewriteResponse
from app.services.chat_service import resolve_model_profile_runtime
from app.services.llm_provider import generate_chat
from .chat_helpers import (
    build_ghost_user_input as _build_ghost_user_input,
    ensure_project_scope_access as _ensure_project_access,
    normalize_ghost_suggestion as _normalize_ghost_suggestion,
)


router = APIRouter(prefix="/writing", tags=["writing"])
logger = logging.getLogger(__name__)


@router.post("/rewrite", response_model=RewriteResponse)
async def rewrite_text(
    payload: RewriteRequest,
    response: Response,
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

    prompt = _build_ghost_user_input(
        payload.mode,
        source_text=payload.text,
    )
    try:
        generation = await generate_chat(
            prompt,
            context={
                "runtime_options": {
                    "source": "rewrite_shim",
                    "mode": payload.mode,
                    "deprecated_endpoint": "/api/writing/rewrite",
                }
            },
            model_override=payload.model,
            thinking_enabled=False,
            temperature_profile=payload.temperature_profile or "chat",
            temperature_override=payload.temperature_override,
            runtime_config=runtime_model_profile,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"rewrite failed: {exc}")

    response.headers["X-Deprecated-Endpoint"] = "/api/writing/rewrite"
    logger.warning(
        "deprecated_rewrite_endpoint_called project_id=%s mode=%s",
        payload.project_id,
        payload.mode,
    )
    return RewriteResponse(
        result=_normalize_ghost_suggestion(generation.assistant_text, max_length=5000),
        usage={
            **(generation.usage or {}),
            "source": "rewrite_shim",
            "ghost_mode": payload.mode,
        },
    )

