import asyncio
from typing import Any, AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.core.auth import (
    AuthPrincipal,
    filter_accessible_project_ids,
    get_current_principal,
)
from app.core.config import settings
from app.core.database import get_session
from app.core.sse import sse_event
from app.schemas.chat import (
    GhostTextRequest,
    GhostTextResponse,
    ChatStreamRequest,
)
from app.services.chat_service import (
    append_message,
    build_session_title,
    create_session,
    get_project_chapter,
    get_prompt_template,
    get_project_volume,
    list_scene_beats,
    resolve_model_profile_runtime,
    update_message_content,
)
from app.services.context_compiler import compile_context_bundle
from app.services.llm_provider import ChatGenerationResult, generate_chat, generate_tot_brainstorm
from app.services.telemetry import ChatTracePayload, emit_chat_trace
from .chat_helpers import (
    build_action_provenance as _build_action_provenance,
    build_ghost_user_input as _build_ghost_user_input,
    create_proposed_actions as _create_proposed_actions,
    enforce_quality_gate as _enforce_quality_gate,
    ensure_project_scope_access as _ensure_project_access,
    ensure_session_member_access as _ensure_session_access,
    normalize_ghost_suggestion as _normalize_ghost_suggestion,
)
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

    async def event_gen() -> AsyncGenerator[str, None]:
        chunks: list[str] = []
        proposed_action_ids: list[int] = []
        generation: ChatGenerationResult
        model_input = payload.content
        tot_meta: dict[str, Any] = {"enabled": False, "triggered": False}
        trace_seq = 0

        def _trace_event(
            message: str,
            *,
            stage: str,
            status: str = "info",
            scope: str = "pipeline",
            step: int | None = None,
            total: int | None = None,
            meta: dict[str, Any] | None = None,
        ) -> str:
            nonlocal trace_seq
            trace_seq += 1
            payload_data: dict[str, Any] = {
                "type": "trace",
                "seq": trace_seq,
                "scope": scope,
                "stage": stage,
                "status": status,
                "message": str(message or "").strip(),
            }
            if isinstance(step, int) and step > 0:
                payload_data["step"] = step
            if isinstance(total, int) and total > 0:
                payload_data["total"] = total
            if isinstance(meta, dict) and meta:
                payload_data["meta"] = meta
            return sse_event(payload_data)

        policy_obj = compiled_bundle.evidence_event.get("policy") if isinstance(compiled_bundle.evidence_event, dict) else {}
        retrieval_runtime = (
            policy_obj.get("retrieval_runtime")
            if isinstance(policy_obj, dict) and isinstance(policy_obj.get("retrieval_runtime"), dict)
            else {}
        )
        reflective_meta = (
            retrieval_runtime.get("self_reflective")
            if isinstance(retrieval_runtime, dict) and isinstance(retrieval_runtime.get("self_reflective"), dict)
            else {}
        )
        if reflective_meta:
            triggered = bool(reflective_meta.get("triggered"))
            issues = (
                reflective_meta.get("issues")
                if isinstance(reflective_meta.get("issues"), list)
                else []
            )
            query_count = int(reflective_meta.get("query_count", 0) or 0)
            reason = str(reflective_meta.get("trigger_reason") or "").strip() or "none"
            status = "ok" if triggered else "skip"
            message = (
                f"Self-Reflective{'已触发' if triggered else '未触发'}，"
                f"原因：{reason}，补检索查询：{query_count} 条。"
            )
            yield _trace_event(
                message,
                stage="self_reflective",
                status=status,
                scope="retrieval",
                meta={
                    "triggered": triggered,
                    "issues": issues[:6],
                    "query_count": query_count,
                    "source": reflective_meta.get("source"),
                },
            )

        if settings.tot_enabled and str(payload.temperature_profile or "").strip().lower() == "brainstorm":
            tot_meta = {"enabled": True, "triggered": True, "source": "brainstorm_profile"}
            yield _trace_event(
                "[1/3] 检索时序图谱与记忆快照，准备剧情分支推演。",
                stage="tot_prepare",
                status="running",
                scope="tot",
                step=1,
                total=3,
            )
            try:
                tot_result = await generate_tot_brainstorm(
                    payload.content,
                    context=compiled_bundle.model_context,
                    model_override=payload.model,
                    runtime_config=runtime_model_profile,
                )
                branches = [item for item in tot_result.branches if isinstance(item, dict)]
                if branches:
                    risk_marked = len(
                        [
                            item
                            for item in branches
                            if str(item.get("consistency_risk") or "").strip()
                        ]
                    )
                    yield _trace_event(
                        (
                            f"[2/3] 已生成 {len(branches)} 条候选分支，"
                            f"其中 {risk_marked} 条带一致性风险提示。"
                        ),
                        stage="tot_branches",
                        status="ok",
                        scope="tot",
                        step=2,
                        total=3,
                        meta={
                            "branches_count": len(branches),
                            "risk_marked": risk_marked,
                        },
                    )
                    lines = []
                    for item in branches[: max(int(settings.tot_max_branches), 1)]:
                        branch_id = str(item.get("id") or "")
                        title = str(item.get("title") or "")
                        hypothesis = str(item.get("hypothesis") or "")
                        rationale = str(item.get("rationale") or "")
                        lines.append(f"{branch_id} {title}: {hypothesis}（依据: {rationale}）")
                    model_input = (
                        payload.content
                        + "\n\n[ToT 推演候选]\n"
                        + "\n".join(lines)
                        + "\n请基于候选分支输出你推荐的剧情推进建议。"
                    )
                    tot_meta = {
                        **tot_meta,
                        "provider": str((tot_result.usage or {}).get("provider") or "unknown"),
                        "branches": branches,
                        "recommended": tot_result.recommended,
                        "rationale": tot_result.rationale,
                        "usage": tot_result.usage,
                    }
                    yield _trace_event(
                        (
                            f"[3/3] 推荐分支：{str(tot_result.recommended or branches[0].get('id') or 'B1')}，"
                            "已注入主模型生成。"
                        ),
                        stage="tot_recommend",
                        status="ok",
                        scope="tot",
                        step=3,
                        total=3,
                        meta={
                            "recommended": tot_result.recommended,
                            "provider": str((tot_result.usage or {}).get("provider") or ""),
                        },
                    )
                else:
                    tot_meta = {**tot_meta, "reason": "empty_branches"}
                    yield _trace_event(
                        "[2/3] ToT 未生成有效分支，回退常规生成。",
                        stage="tot_branches",
                        status="warning",
                        scope="tot",
                        step=2,
                        total=3,
                    )
            except Exception as exc:
                tot_meta = {**tot_meta, "reason": "tot_error", "error": str(exc)}
                yield _trace_event(
                    f"[2/3] ToT 推演失败，已回退常规生成：{str(exc)[:120]}",
                    stage="tot_error",
                    status="error",
                    scope="tot",
                    step=2,
                    total=3,
                )

        if isinstance(compiled_bundle.model_context, dict):
            runtime_options = compiled_bundle.model_context.get("runtime_options")
            if not isinstance(runtime_options, dict):
                runtime_options = {}
            runtime_options["tot"] = tot_meta
            if isinstance(runtime_model_profile, dict):
                runtime_options["model_profile_id"] = runtime_model_profile.get("profile_id")
                runtime_options["model_provider"] = runtime_model_profile.get("provider")
            compiled_bundle.model_context["runtime_options"] = runtime_options
        if isinstance(compiled_bundle.evidence_event, dict):
            policy = compiled_bundle.evidence_event.get("policy")
            if not isinstance(policy, dict):
                policy = {}
            policy["tot"] = {
                "enabled": bool(tot_meta.get("enabled")),
                "triggered": bool(tot_meta.get("triggered")),
                "recommended": tot_meta.get("recommended"),
                "branches_count": len(tot_meta.get("branches", [])) if isinstance(tot_meta.get("branches"), list) else 0,
                "source": tot_meta.get("source"),
            }
            compiled_bundle.evidence_event["policy"] = policy

        try:
            generation = await generate_chat(
                model_input,
                context=compiled_bundle.model_context,
                model_override=payload.model,
                thinking_enabled=payload.thinking_enabled,
                temperature_profile=payload.temperature_profile or "action",
                temperature_override=payload.temperature_override,
                runtime_config=runtime_model_profile,
            )
        except Exception as exc:
            generation = ChatGenerationResult(
                assistant_text=f"模型调用失败：{exc}",
                proposed_actions=[],
                usage={"provider": "error"},
            )
        generation.usage = {
            **(generation.usage or {}),
            "tot": {
                "enabled": bool(tot_meta.get("enabled")),
                "triggered": bool(tot_meta.get("triggered")),
                "branches_count": len(tot_meta.get("branches", [])) if isinstance(tot_meta.get("branches"), list) else 0,
                "provider": tot_meta.get("provider"),
                "recommended": tot_meta.get("recommended"),
            },
        }
        if isinstance(runtime_model_profile, dict):
            generation.usage["model_profile_id"] = runtime_model_profile.get("profile_id")
            generation.usage["model_profile_provider"] = runtime_model_profile.get("provider")
        generation = _enforce_quality_gate(generation, compiled_bundle.model_context)

        if generation.proposed_actions:
            proposed_action_ids = _create_proposed_actions(
                db=db,
                session_id=session.id,
                user_id=principal.user_id,
                proposed_actions=generation.proposed_actions,
                provenance=_build_action_provenance(compiled_bundle.model_context),
            )

        emit_chat_trace(
            ChatTracePayload(
                project_id=payload.project_id,
                session_id=session.id,
                user_id=principal.user_id,
                model=payload.model,
                user_input=payload.content,
                assistant_text=generation.assistant_text or "",
                usage=generation.usage or {},
                proposed_actions_count=len(proposed_action_ids),
                evidence_policy=compiled_bundle.evidence_event.get("policy", {}),
                evidence_summary=compiled_bundle.evidence_event.get("summary", {}),
                error=(
                    str((generation.usage or {}).get("error"))
                    if isinstance(generation.usage, dict) and (generation.usage or {}).get("error") is not None
                    else None
                ),
            )
        )

        yield sse_event(
            {
                "type": "meta",
                "session_id": session.id,
                "assistant_message_id": assistant_msg.id,
                "proposed_action_ids": proposed_action_ids,
            }
        )
        yield sse_event(compiled_bundle.evidence_event)

        try:
            chunk_size = 8
            text = generation.assistant_text or ""
            text_length = len(text)
            for idx in range(0, len(text), chunk_size):
                delta = text[idx : idx + chunk_size]
                chunks.append(delta)
                yield sse_event({"type": "delta", "text": delta})
                if idx + chunk_size < text_length:
                    await asyncio.sleep(0.02)

            final_text = "".join(chunks)
            update_message_content(assistant_msg.id, final_text, db=db)
            yield sse_event(
                {
                    "type": "done",
                    "assistant_message_id": assistant_msg.id,
                    "usage": generation.usage
                    or {"input_chars": len(payload.content), "output_chars": len(final_text)},
                }
            )
        except asyncio.CancelledError:
            update_message_content(assistant_msg.id, "".join(chunks), db=db)
            raise
        except Exception as exc:
            update_message_content(assistant_msg.id, "".join(chunks), db=db)
            yield sse_event({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/ghost-text", response_model=GhostTextResponse)
async def generate_ghost_text(
    payload: GhostTextRequest,
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
    notes = [
        "Ghost Text 使用轻上下文（prefix_text + 可选模板风格约束）。",
        "默认不调用 DSL/GRAPH/RAG，优先保证下一句响应速度。",
    ]
    if payload.prompt_template_id is not None and template is None:
        notes.append(f"prompt_template_id={payload.prompt_template_id} 未命中，已忽略模板风格约束。")
    if template is not None and not payload.style_guard:
        notes.append("style_guard=false，已跳过模板风格约束。")

    story_outline: dict[str, Any] = {"volume": None, "scene_beat": None}
    outline_context_meta: dict[str, Any] = {
        "enabled": False,
        "reason": "chapter_not_requested",
        "requested_chapter_id": payload.chapter_id,
        "requested_scene_beat_id": payload.scene_beat_id,
        "selected_scene_beat_id": None,
    }
    outline_hint_parts: list[str] = []
    chapter_for_context = None
    if payload.chapter_id is not None:
        chapter = get_project_chapter(db, payload.project_id, payload.chapter_id)
        if chapter is None:
            outline_context_meta["reason"] = "chapter_not_found"
            notes.append(f"chapter_id={payload.chapter_id} 未命中，已忽略卷纲/Beat 注入。")
        else:
            chapter_for_context = chapter
            outline_context_meta["reason"] = "ok"
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
                    outline_context_meta["enabled"] = True
                    outline_hint_parts.append(
                        f"当前卷：{story_outline['volume']['title']}（卷纲：{volume_outline[:220] or '未填写'}）"
                    )
                else:
                    notes.append(f"volume_id={volume_id} 未命中，已忽略卷纲注入。")
                    outline_context_meta["reason"] = "volume_not_found"
            else:
                outline_context_meta["reason"] = "volume_not_bound"

            beats = list(list_scene_beats(db, project_id=payload.project_id, chapter_id=payload.chapter_id))
            active_beat = None
            if payload.scene_beat_id is not None:
                active_beat = next((item for item in beats if int(getattr(item, "id", 0) or 0) == int(payload.scene_beat_id)), None)
                if active_beat is None:
                    notes.append(f"scene_beat_id={payload.scene_beat_id} 未命中，已退回章节默认 Beat。")
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
                outline_context_meta["selected_scene_beat_id"] = int(getattr(active_beat, "id", 0) or 0)
                outline_context_meta["enabled"] = True
                outline_hint_parts.append(f"当前节拍：{story_outline['scene_beat']['active']['content']}")
                if story_outline["scene_beat"]["previous"]:
                    outline_hint_parts.append(
                        f"上一节拍：{story_outline['scene_beat']['previous']['content']}"
                    )
                if story_outline["scene_beat"]["next"]:
                    outline_hint_parts.append(
                        f"下一节拍：{story_outline['scene_beat']['next']['content']}"
                    )
            elif beats:
                outline_context_meta["reason"] = "no_active_scene_beat"
            else:
                if outline_context_meta["reason"] == "ok":
                    outline_context_meta["reason"] = "ok_no_scene_beats"

    quality_gate = {
        "degraded": False,
        "degrade_reasons": [],
        "citation_required": False,
        "citation_count": 0,
        "reranker_expected": False,
        "reranker_effective": False,
    }
    providers = {
        "dsl": "disabled_for_ghost_light_mode",
        "graph": "disabled_for_ghost_light_mode",
        "rag": "disabled_for_ghost_light_mode",
    }
    evidence_policy = {
        "mode": "ghost",
        "anchor": None,
        "notes": notes,
        "resolver_order": "PREFIX > TEMPLATE_STYLE",
        "ranking_dimensions": "local_continuation + style_consistency",
        "providers": providers,
        "rag_route": {"mode": "disabled", "reason": "ghost_light_context"},
        "rag_short_circuit": {"enabled": True, "reason": "ghost_light_context"},
        "prompt_workshop": prompt_workshop_meta,
        "runtime_options": {"thinking_enabled": False},
        "outline_context": outline_context_meta,
        "quality_gate": quality_gate,
        "ghost_context_mode": "light",
    }
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
        "evidence": {
            "resolver_order": ["PREFIX", "TEMPLATE_STYLE"],
            "ranking_dimensions": ["local_continuation", "style_consistency"],
            "providers": providers,
            "rag_route": {"mode": "disabled", "reason": "ghost_light_context"},
            "rag_short_circuit": {"enabled": True, "reason": "ghost_light_context"},
            "prompt_workshop": prompt_workshop_meta,
            "outline_context": outline_context_meta,
            "quality_gate": quality_gate,
            "dsl_hits": [],
            "graph_facts": [],
            "semantic_hits": [],
            "ghost_context_mode": "light",
        },
    }

    try:
        generation = await generate_chat(
            _build_ghost_user_input(
                payload.prefix_text,
                style_prefix=template_user_prompt_prefix,
                outline_hint="\n".join(outline_hint_parts),
            ),
            context=model_context,
            model_override=payload.model,
            thinking_enabled=False,
            temperature_profile=payload.temperature_profile or "ghost",
            temperature_override=payload.temperature_override,
            runtime_config=runtime_model_profile,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ghost generation failed: {exc}")

    suggestion = _normalize_ghost_suggestion(generation.assistant_text)
    usage = {
        **(generation.usage or {}),
        "ghost_context_mode": "light",
        "style_guard": bool(payload.style_guard),
        "prompt_template_hit": bool(template is not None),
    }
    if isinstance(runtime_model_profile, dict):
        usage["model_profile_id"] = runtime_model_profile.get("profile_id")
        usage["model_profile_provider"] = runtime_model_profile.get("provider")
    return GhostTextResponse(
        suggestion=suggestion,
        usage=usage,
        evidence_policy=evidence_policy,
    )




