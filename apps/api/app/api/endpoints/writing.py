from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.core.auth import AuthPrincipal, get_current_principal
from app.core.database import get_session
from app.schemas.chat import RewriteRequest, RewriteResponse
from app.services.chat_service import resolve_model_profile_runtime
from app.services.llm_provider import generate_chat
from .chat_helpers import ensure_project_scope_access as _ensure_project_access


router = APIRouter(prefix="/writing", tags=["writing"])


def _build_rewrite_prompt(mode: str, text: str) -> str:
    normalized = str(text or "").strip()
    if mode == "expand":
        instruction = (
            "请基于这段正文进行扩写，保持同一人称与语气，不改变既有剧情事实、时间线与人设，"
            "不引入越权设定与外部资料。"
            "只输出扩写后的完整正文，不要解释，不要标题，不要列点。"
        )
    else:
        instruction = (
            "请在不改变剧情事实、时间线与人设的前提下润色这段正文，提升语言节奏、画面与情绪表达，"
            "但不要新增事件。"
            "只输出润色后的完整正文，不要解释，不要标题，不要列点。"
        )
    return f"{instruction}\n\n<正文>\n{normalized}\n</正文>"


@router.post("/rewrite", response_model=RewriteResponse)
async def rewrite_text(
    payload: RewriteRequest,
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

    prompt = _build_rewrite_prompt(payload.mode, payload.text)
    try:
        generation = await generate_chat(
            prompt,
            context={"runtime_options": {"source": "rewrite", "mode": payload.mode}},
            model_override=payload.model,
            thinking_enabled=False,
            temperature_profile=payload.temperature_profile or "chat",
            temperature_override=payload.temperature_override,
            runtime_config=runtime_model_profile,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"rewrite failed: {exc}")

    return RewriteResponse(
        result=str(generation.assistant_text or "").strip(),
        usage=generation.usage or {},
    )

