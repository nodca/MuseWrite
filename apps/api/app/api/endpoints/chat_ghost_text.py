from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.core.auth import AuthPrincipal, get_current_principal
from app.core.database import get_session
from app.schemas.chat import GhostTextResponse, GhostTextRewriteRequest
from app.services.chat_service import (
    get_project_chapter,
    get_prompt_template,
    get_project_volume,
    list_scene_beats,
    resolve_model_profile_runtime,
)
from app.services.llm_provider import generate_chat
from .chat_helpers import (
    build_ghost_user_input as _build_ghost_user_input,
    ensure_project_scope_access as _ensure_project_access,
    normalize_ghost_suggestion as _normalize_ghost_suggestion,
)

router = APIRouter()


@router.post("/ghost-text/polish", response_model=GhostTextResponse)
async def generate_ghost_polish(
    payload: GhostTextRewriteRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    return await _generate_ghost_rewrite("polish", payload, db, principal)


@router.post("/ghost-text/expand", response_model=GhostTextResponse)
async def generate_ghost_expand(
    payload: GhostTextRewriteRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    return await _generate_ghost_rewrite("expand", payload, db, principal)


async def _generate_ghost_rewrite(
    rewrite_mode: Literal["polish", "expand"],
    payload: GhostTextRewriteRequest,
    db: Session,
    principal: AuthPrincipal,
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

    source_text = str(payload.text or "").strip()
    if not source_text:
        raise HTTPException(status_code=400, detail="text is required")

    template = None
    if payload.prompt_template_id is not None:
        template = get_prompt_template(db, payload.project_id, payload.prompt_template_id)

    use_template_style = bool(payload.style_guard and template is not None)
    template_system_prompt = (
        str(getattr(template, "system_prompt", "") or "").strip()[:1800] if use_template_style else ""
    )
    template_user_prompt_prefix = (
        str(getattr(template, "user_prompt_prefix", "") or "").strip()[:600] if use_template_style else ""
    )
    template_name = str(getattr(template, "name", "") or "").strip() if template is not None else ""
    prompt_workshop_meta: dict[str, Any] = {
        "enabled": bool(template is not None),
        "reason": (
            "ghost_style_guard_enabled"
            if use_template_style
            else (
                "style_guard_disabled"
                if template is not None
                else ("template_not_found" if payload.prompt_template_id is not None else "no_template_requested")
            )
        ),
        "requested_template_id": payload.prompt_template_id,
        "template_id": int(getattr(template, "id", 0)) if template is not None else None,
        "template_name": template_name or None,
        "injected_settings": 0,
        "injected_cards": 0,
        "ghost_context_mode": "light",
    }
    outline_hint_parts: list[str] = []
    normalized_chapter_goal = str(payload.chapter_goal or "").strip()
    if len(normalized_chapter_goal) > 220:
        normalized_chapter_goal = normalized_chapter_goal[:220]
    normalized_active_roles: list[str] = []
    seen_roles: set[str] = set()
    for item in payload.active_roles or []:
        role = str(item or "").strip()
        if not role:
            continue
        role_key = role.lower()
        if role_key in seen_roles:
            continue
        seen_roles.add(role_key)
        normalized_active_roles.append(role[:24])
        if len(normalized_active_roles) >= 8:
            break

    story_outline: dict[str, Any] = {
        "volume": None,
        "scene_beat": None,
        "chapter_goal": normalized_chapter_goal or None,
        "active_roles": normalized_active_roles,
    }
    outline_hint_parts: list[str] = []
    if normalized_chapter_goal:
        outline_hint_parts.append(f"当前章节目标：{normalized_chapter_goal}")
    if normalized_active_roles:
        outline_hint_parts.append(f"活跃角色：{'、'.join(normalized_active_roles)}")

    chapter_for_context = None
    if payload.chapter_id is not None:
        chapter = get_project_chapter(db, payload.project_id, payload.chapter_id)
        if chapter is not None:
            chapter_for_context = chapter
            volume_id = int(getattr(chapter, "volume_id", 0) or 0)
            if volume_id > 0:
                volume = get_project_volume(db, payload.project_id, volume_id)
                if volume is not None:
                    volume_outline = str(getattr(volume, "outline", "") or "").strip()
                    story_outline["volume"] = {
                        "id": int(getattr(volume, "id", 0) or 0),
                        "volume_index": int(getattr(volume, "volume_index", 0) or 0),
                        "title": str(getattr(volume, "title", "") or ""),
                        "outline_preview": volume_outline[:420],
                    }
                    outline_hint_parts.append(
                        f"当前卷：{story_outline['volume']['title']}（卷纲：{volume_outline[:220] or '未填写'}）"
                    )

            beats = list(list_scene_beats(db, project_id=payload.project_id, chapter_id=payload.chapter_id))
            active_beat = None
            if payload.scene_beat_id is not None:
                active_beat = next((item for item in beats if int(getattr(item, "id", 0) or 0) == int(payload.scene_beat_id)), None)
            if active_beat is None and beats:
                active_beat = next((item for item in beats if str(getattr(item, "status", "")) == "pending"), None)
            if active_beat is None and beats:
                active_beat = beats[0]
            if active_beat is not None:
                active_idx = next(
                    (idx for idx, item in enumerate(beats) if int(getattr(item, "id", 0) or 0) == int(getattr(active_beat, "id", 0) or 0)),
                    0,
                )
                prev_beat = beats[active_idx - 1] if active_idx > 0 else None
                next_beat = beats[active_idx + 1] if active_idx + 1 < len(beats) else None
                story_outline["scene_beat"] = {
                    "active": {
                        "id": int(getattr(active_beat, "id", 0) or 0),
                        "beat_index": int(getattr(active_beat, "beat_index", 0) or 0),
                        "content": str(getattr(active_beat, "content", "") or "")[:420],
                        "status": str(getattr(active_beat, "status", "") or "pending"),
                    },
                    "previous": (
                        {
                            "id": int(getattr(prev_beat, "id", 0) or 0),
                            "content": str(getattr(prev_beat, "content", "") or "")[:180],
                        }
                        if prev_beat is not None
                        else None
                    ),
                    "next": (
                        {
                            "id": int(getattr(next_beat, "id", 0) or 0),
                            "content": str(getattr(next_beat, "content", "") or "")[:180],
                        }
                        if next_beat is not None
                        else None
                    ),
                    "total": len(beats),
                }
                outline_hint_parts.append(f"当前节拍：{story_outline['scene_beat']['active']['content']}")
                if story_outline["scene_beat"]["previous"]:
                    outline_hint_parts.append(
                        f"上一节拍：{story_outline['scene_beat']['previous']['content']}"
                    )
                if story_outline["scene_beat"]["next"]:
                    outline_hint_parts.append(
                        f"下一节拍：{story_outline['scene_beat']['next']['content']}"
                    )

    model_context = {
        "runtime_options": {"thinking_enabled": False, "scene_beat_id": payload.scene_beat_id},
        "current_chapter": (
            {
                "id": int(getattr(chapter_for_context, "id", 0) or 0),
                "volume_id": int(getattr(chapter_for_context, "volume_id", 0) or 0) or None,
                "chapter_index": int(getattr(chapter_for_context, "chapter_index", 0) or 0),
                "title": str(getattr(chapter_for_context, "title", "") or ""),
            }
            if chapter_for_context is not None
            else None
        ),
        "story_outline": story_outline,
        "prompt_workshop": {
            "template": (
                {
                    "id": int(getattr(template, "id", 0)),
                    "name": template_name,
                    "system_prompt": template_system_prompt,
                    "user_prompt_prefix": template_user_prompt_prefix,
                }
                if use_template_style
                else None
            ),
            "knowledge_injection": {"settings": [], "cards": []},
            "meta": prompt_workshop_meta,
        },
    }

    try:
        generation = await generate_chat(
            _build_ghost_user_input(
                rewrite_mode,
                source_text=source_text,
                style_prefix=template_user_prompt_prefix,
                outline_hint="\n".join(outline_hint_parts),
            ),
            context={
                **model_context,
                "runtime_options": {
                    "thinking_enabled": False,
                    "scene_beat_id": payload.scene_beat_id,
                    "source": "ghost_text",
                    "mode": rewrite_mode,
                },
            },
            model_override=payload.model,
            thinking_enabled=False,
            temperature_profile=payload.temperature_profile or "chat",
            temperature_override=payload.temperature_override,
            runtime_config=runtime_model_profile,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ghost generation failed: {exc}")

    suggestion = _normalize_ghost_suggestion(
        generation.assistant_text,
        max_length=5000,
    )
    usage = {
        **(generation.usage or {}),
        "ghost_context_mode": "light",
        "ghost_mode": rewrite_mode,
        "style_guard": bool(payload.style_guard),
        "prompt_template_hit": bool(template is not None),
    }
    if isinstance(runtime_model_profile, dict):
        usage["model_profile_id"] = runtime_model_profile.get("profile_id")
        usage["model_profile_provider"] = runtime_model_profile.get("provider")
    return GhostTextResponse(
        suggestion=suggestion,
        usage=usage,
    )
