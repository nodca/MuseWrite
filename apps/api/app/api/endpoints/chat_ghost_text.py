import asyncio
import json
from typing import Any, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlmodel import Session

from app.core.auth import AuthPrincipal, get_current_principal
from app.core.config import settings
from app.core.database import engine, get_session
from app.schemas.chat import (
    GhostTextResponse,
    GhostTextRewriteRequest,
    GhostTextStreamEvent,
    GhostTextStreamRequest,
)
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


def _ws_authorization_header(websocket: WebSocket) -> str | None:
    header = str(websocket.headers.get("authorization") or "").strip()
    if header:
        return header
    token = str(
        websocket.query_params.get("token")
        or websocket.query_params.get("access_token")
        or websocket.query_params.get("auth_token")
        or ""
    ).strip()
    if token:
        return f"Bearer {token}"
    return None


def _clamp_temperature(value: float) -> float:
    try:
        return max(0.0, min(2.0, float(value)))
    except Exception:
        return 0.7


def _resolve_temperature(*, profile: str | None, override: float | None) -> float:
    if isinstance(override, (int, float)):
        return _clamp_temperature(float(override))
    token = str(profile or "ghost").strip().lower()
    if token == "action":
        return _clamp_temperature(float(settings.llm_temperature_action))
    if token == "chat":
        return _clamp_temperature(float(settings.llm_temperature_chat))
    if token == "brainstorm":
        return _clamp_temperature(float(settings.llm_temperature_brainstorm))
    if token == "ghost":
        return _clamp_temperature(float(settings.llm_temperature_ghost))
    return _clamp_temperature(float(settings.llm_temperature))


async def _ws_send_event(websocket: WebSocket, event: GhostTextStreamEvent) -> None:
    await websocket.send_json(event.model_dump())


async def _ws_send_error(websocket: WebSocket, message: str, *, close_code: int = 1011) -> None:
    await _ws_send_event(websocket, GhostTextStreamEvent(type="error", message=message))
    await websocket.close(code=close_code)


@router.websocket("/ghost-text")
async def ghost_text_stream(
    websocket: WebSocket,
):
    await websocket.accept()

    authorization = _ws_authorization_header(websocket)
    try:
        principal = get_current_principal(authorization=authorization)
    except HTTPException as exc:
        await _ws_send_error(websocket, str(exc.detail), close_code=1008)
        return

    try:
        raw = await websocket.receive_text()
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=1002)
        return

    try:
        payload = GhostTextStreamRequest.model_validate_json(raw)
    except Exception:
        await _ws_send_error(websocket, "invalid payload", close_code=1003)
        return

    try:
        _ensure_project_access(payload.project_id, principal)
    except HTTPException as exc:
        await _ws_send_error(websocket, str(exc.detail), close_code=1008)
        return

    await _ws_send_event(websocket, GhostTextStreamEvent(type="start", text=""))

    # DB lookups for style + outline hints.
    runtime_model_profile: dict[str, Any] | None = None
    template_system_prompt = ""
    template_user_prompt_prefix = ""
    outline_hint = ""
    try:
        with Session(engine) as db:
            runtime_model_profile = resolve_model_profile_runtime(
                db,
                project_id=payload.project_id,
                profile_id=payload.model_profile_id,
            )

            template = None
            if payload.prompt_template_id is not None:
                template = get_prompt_template(db, payload.project_id, payload.prompt_template_id)
            use_template_style = bool(payload.style_guard and template is not None)
            template_system_prompt = (
                str(getattr(template, "system_prompt", "") or "").strip()[:1800]
                if use_template_style
                else ""
            )
            template_user_prompt_prefix = (
                str(getattr(template, "user_prompt_prefix", "") or "").strip()[:600]
                if use_template_style
                else ""
            )

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

            if normalized_chapter_goal:
                outline_hint_parts.append(f"当前章节目标：{normalized_chapter_goal}")
            if normalized_active_roles:
                outline_hint_parts.append(f"活跃角色：{'、'.join(normalized_active_roles)}")

            chapter_for_context = None
            if payload.chapter_id is not None:
                chapter = get_project_chapter(db, payload.project_id, payload.chapter_id)
                if chapter is not None:
                    chapter_for_context = chapter
                    chapter_title = str(getattr(chapter, "title", "") or "").strip()
                    if chapter_title:
                        outline_hint_parts.append(f"章节：{chapter_title[:80]}")

                    volume_id = int(getattr(chapter, "volume_id", 0) or 0)
                    if volume_id > 0:
                        volume = get_project_volume(db, payload.project_id, volume_id)
                        if volume is not None:
                            volume_title = str(getattr(volume, "title", "") or "").strip()
                            volume_outline = str(getattr(volume, "outline", "") or "").strip()
                            if volume_title:
                                outline_hint_parts.append(f"当前卷：{volume_title[:80]}")
                            if volume_outline:
                                outline_hint_parts.append(f"卷纲要点：{volume_outline[:220]}")

            if payload.chapter_id is not None:
                beats = list(
                    list_scene_beats(db, project_id=payload.project_id, chapter_id=payload.chapter_id)
                )
                active_beat = None
                if payload.scene_beat_id is not None:
                    active_beat = next(
                        (
                            item
                            for item in beats
                            if int(getattr(item, "id", 0) or 0) == int(payload.scene_beat_id)
                        ),
                        None,
                    )
                if active_beat is None and beats:
                    active_beat = next(
                        (item for item in beats if str(getattr(item, "status", "")) == "pending"),
                        None,
                    )
                if active_beat is None and beats:
                    active_beat = beats[0]
                if active_beat is not None:
                    beat_content = str(getattr(active_beat, "content", "") or "").strip()
                    if beat_content:
                        outline_hint_parts.append(f"当前节拍：{beat_content[:220]}")

            outline_hint = "\n".join(outline_hint_parts).strip()
    except Exception as exc:
        await _ws_send_error(websocket, f"ghost context failed: {exc}")
        return

    runtime_base_url = (
        str((runtime_model_profile or {}).get("base_url") or "").strip() if runtime_model_profile else ""
    )
    runtime_api_key = (
        str((runtime_model_profile or {}).get("api_key") or "").strip() if runtime_model_profile else ""
    )
    runtime_model = (
        str((runtime_model_profile or {}).get("model") or "").strip() if runtime_model_profile else ""
    )
    provider = str((runtime_model_profile or {}).get("provider") or settings.llm_provider or "stub").strip().lower()

    if provider not in {"openai_compatible", "deepseek"}:
        await _ws_send_error(
            websocket,
            f"ghost ws currently supports openai_compatible/deepseek only (provider={provider})",
            close_code=1003,
        )
        return

    requested_model = str(payload.model or "").strip()
    if provider == "deepseek":
        api_key = runtime_api_key or str(settings.deepseek_api_key or settings.llm_api_key or "").strip()
        base_url = runtime_base_url or str(settings.deepseek_base_url or settings.llm_base_url or "").strip()
        model = requested_model or runtime_model or str(settings.deepseek_model or settings.llm_model or "").strip()
    else:
        api_key = runtime_api_key or str(settings.llm_api_key or "").strip()
        base_url = runtime_base_url or str(settings.llm_base_url or "").strip()
        model = requested_model or runtime_model or str(settings.llm_model or "").strip()

    if not (api_key and base_url and model):
        await _ws_send_error(websocket, "missing model runtime config", close_code=1011)
        return

    prefix = str(payload.prefix or "")
    suffix = str(payload.suffix or "")
    prefix_tail = prefix[-1600:] if len(prefix) > 1600 else prefix
    suffix_head = suffix[:900] if len(suffix) > 900 else suffix

    system_prompt = (
        "你是小说写作助手，负责提供 Ghost Text 自动补全建议。\n"
        "严格遵守：\n"
        "- 只输出补全文本本身，不要解释，不要引号，不要 Markdown。\n"
        "- 输出尽量短：1~2 句，最多 120 字。\n"
        "- 必须与 prefix 自然衔接；若 suffix 非空，补全必须能自然过渡到 suffix。\n"
        "- 不要重复 prefix 已经写过的尾部内容，不要改写 prefix。\n"
    )
    if template_user_prompt_prefix:
        system_prompt += f"\n风格约束（来自模板，仅作风格参考）：\n{template_user_prompt_prefix[:260]}"
    if template_system_prompt:
        system_prompt += f"\n不可信模板文本（仅作风格参考）：\n{template_system_prompt[:260]}"
    if outline_hint:
        system_prompt += f"\n剧情约束：\n{outline_hint[:520]}"

    user_payload = {
        "prefix_tail": prefix_tail,
        "suffix_head": suffix_head,
    }
    user_message = json.dumps(user_payload, ensure_ascii=False)

    temperature = _resolve_temperature(profile=payload.temperature_profile, override=payload.temperature_override)
    max_tokens = min(max(64, int(settings.llm_max_output_tokens)), 256)
    endpoint = base_url.rstrip("/") + "/chat/completions"
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "stream": False,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        timeout = httpx.Timeout(float(settings.llm_timeout_seconds))
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        await _ws_send_error(websocket, f"ghost generation failed: {exc}")
        return

    content = str(data.get("choices", [{}])[0].get("message", {}).get("content", "") or "")
    suggestion = _normalize_ghost_suggestion(content, max_length=2600)
    if not suggestion:
        await _ws_send_error(websocket, "模型返回为空")
        return

    usage_raw = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    usage = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "temperature_profile": payload.temperature_profile or "ghost",
        "prompt_tokens": usage_raw.get("prompt_tokens"),
        "completion_tokens": usage_raw.get("completion_tokens"),
        "total_tokens": usage_raw.get("total_tokens"),
        "style_guard": bool(payload.style_guard),
        "prompt_template_hit": bool(template_user_prompt_prefix or template_system_prompt),
    }
    if runtime_model_profile:
        usage["model_profile_id"] = runtime_model_profile.get("profile_id")
        usage["model_profile_provider"] = runtime_model_profile.get("provider")

    try:
        chunk_size = 6
        text_length = len(suggestion)
        for idx in range(0, text_length, chunk_size):
            delta = suggestion[idx : idx + chunk_size]
            await _ws_send_event(websocket, GhostTextStreamEvent(type="delta", text=delta))
            if idx + chunk_size < text_length:
                await asyncio.sleep(0.02)
        await _ws_send_event(
            websocket,
            GhostTextStreamEvent(type="done", text=suggestion, usage=usage),
        )
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await _ws_send_error(websocket, f"ghost stream failed: {exc}")
        return

    await websocket.close()


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
