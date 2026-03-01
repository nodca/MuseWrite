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
    ActionAuditLogRead,
    ChatActionCreateRequest,
    ChatActionDecisionRequest,
    ChatActionRead,
    ChatMessageRead,
    ChatSessionDeleteResult,
    ChatSessionRead,
    ChatSessionUpdateRequest,
    EntityMergeScanRequest,
    EntityMergeScanResult,
    ConsistencyAuditReportRead,
    ConsistencyAuditRunRequest,
    ConsistencyAuditRunResponse,
    GraphTimelineSnapshotRead,
    ForeshadowingCardCreateRequest,
    ForeshadowingCardDeleteResult,
    ForeshadowingCardRead,
    ForeshadowingCardUpdateRequest,
    IndexLifecycleDeadLetterRead,
    IndexLifecycleReplayRequest,
    IndexLifecycleReplayResult,
    ModelProfileDeleteResult,
    ModelProfileRead,
    ModelProfileUpsertRequest,
    GhostTextRequest,
    GhostTextResponse,
    GraphCandidateBatchReviewRequest,
    GraphCandidateBatchReviewResponse,
    GraphCandidateListResponse,
    LightRAGDeleteDocumentsRequest,
    LightRAGInsertTextRequest,
    LightRAGListDocumentsRequest,
    PromptTemplateRead,
    PromptTemplateRevisionRead,
    PromptTemplateRollbackRequest,
    PromptTemplateUpsertRequest,
    ProjectChapterCreateRequest,
    ProjectChapterDeleteRequest,
    ProjectChapterDeleteResult,
    ProjectChapterMoveRequest,
    ProjectChapterReorderRequest,
    ProjectChapterRead,
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
    SettingEntryRead,
    StoryCardRead,
    ChatStreamRequest,
    VolumeMemoryConsolidationRequest,
    VolumeMemoryConsolidationResponse,
)
from app.services.chat_service import (
    create_foreshadowing_card,
    DraftVersionConflictError,
    append_message,
    apply_action_effects,
    build_session_title,
    create_action,
    create_action_audit_log,
    delete_project_chapter,
    create_project_chapter,
    create_session,
    get_project_chapter,
    get_prompt_template,
    list_cards,
    list_project_sessions,
    list_actions,
    list_action_logs,
    list_prompt_template_revisions,
    list_prompt_templates,
    list_project_chapters,
    list_messages,
    list_settings,
    rollback_project_chapter,
    rollback_prompt_template,
    update_prompt_template,
    move_project_chapter,
    reorder_project_chapters,
    save_project_chapter,
    create_prompt_template,
    create_project_volume,
    create_scene_beat,
    delete_session_with_children,
    delete_foreshadowing_card,
    delete_prompt_template,
    delete_project_volume,
    delete_scene_beat,
    is_entity_merge_action_type,
    is_manual_merge_operator,
    run_entity_merge_scan,
    get_project_volume,
    set_action_status,
    undo_action_effects,
    list_foreshadowing_cards,
    list_overdue_foreshadowing_cards,
    list_project_chapter_revisions_with_semantic,
    list_project_volumes,
    list_scene_beats,
    list_model_profiles,
    update_message_content,
    update_foreshadowing_card,
    update_project_volume,
    update_scene_beat,
    update_session_title,
    consolidate_volume_memory,
    create_model_profile,
    update_model_profile,
    delete_model_profile,
    activate_model_profile,
    resolve_model_profile_runtime,
)
from app.services.context_compiler import compile_context_bundle
from app.services.context_compiler import preheat_context_pack
from app.services.consistency_audit_queue import enqueue_consistency_audit_job
from app.services.consistency_audit_service import (
    list_consistency_audit_reports,
    run_consistency_audit,
)
from app.services.entity_merge_queue import enqueue_entity_merge_scan_job
from app.services.index_lifecycle_queue import (
    peek_index_lifecycle_dead_letters,
    pop_index_lifecycle_dead_letters,
)
from app.services.lightrag_documents import (
    delete_documents,
    get_pipeline_status,
    insert_text_document,
    list_project_documents,
)
from app.services.llm_provider import ChatGenerationResult, generate_chat, generate_tot_brainstorm
from app.services.retrieval_adapters import (
    fetch_neo4j_graph_timeline_snapshot,
    list_neo4j_graph_candidates,
)
from app.services.telemetry import ChatTracePayload, emit_chat_trace
from .chat_helpers import (
    action_provenance_from_payload as _action_provenance_from_payload,
    build_action_provenance as _build_action_provenance,
    build_ghost_user_input as _build_ghost_user_input,
    create_proposed_actions as _create_proposed_actions,
    enforce_quality_gate as _enforce_quality_gate,
    ensure_action_session_access as _ensure_action_access,
    ensure_project_scope_access as _ensure_project_access,
    ensure_session_member_access as _ensure_session_access,
    filter_project_dead_letters as _filter_project_dead_letters,
    normalize_ghost_suggestion as _normalize_ghost_suggestion,
    replay_dead_letters as _replay_dead_letters,
    review_graph_candidate_batch as _review_graph_candidate_batch,
)

router = APIRouter(prefix="/chat", tags=["chat"])


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


@router.get("/projects/{project_id}/sessions", response_model=list[ChatSessionRead])
def project_sessions(
    project_id: int,
    limit: int = Query(default=24, ge=1, le=100),
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    return list_project_sessions(
        db,
        project_id=project_id,
        user_id=principal.user_id,
        limit=limit,
    )


@router.put("/projects/{project_id}/sessions/{session_id}", response_model=ChatSessionRead)
def rename_project_session(
    project_id: int,
    session_id: int,
    payload: ChatSessionUpdateRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    _ensure_session_access(db, session_id, principal, expected_project_id=project_id)
    try:
        return update_session_title(
            db,
            session_id=session_id,
            title=payload.title,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/projects/{project_id}/sessions/{session_id}", response_model=ChatSessionDeleteResult)
def remove_project_session(
    project_id: int,
    session_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    _ensure_session_access(db, session_id, principal, expected_project_id=project_id)
    try:
        deleted_session_id = delete_session_with_children(
            db,
            session_id=session_id,
        )
        return ChatSessionDeleteResult(deleted_session_id=deleted_session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageRead])
def session_messages(
    session_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_session_access(db, session_id, principal)
    return list_messages(db, session_id)


@router.get("/sessions/{session_id}/actions", response_model=list[ChatActionRead])
def session_actions(
    session_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_session_access(db, session_id, principal)
    return list_actions(db, session_id)


@router.get("/actions/{action_id}/logs", response_model=list[ActionAuditLogRead])
def action_logs(
    action_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_action_access(db, action_id, principal)
    return list_action_logs(db, action_id)


@router.post("/sessions/{session_id}/actions", response_model=ChatActionRead)
def create_session_action(
    session_id: int,
    payload: ChatActionCreateRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_session_access(db, session_id, principal)
    if is_entity_merge_action_type(payload.action_type):
        if bool(payload.payload.get("auto_apply")):
            raise HTTPException(status_code=400, detail="entity merge does not support auto_apply")

    action = create_action(
        db=db,
        session_id=session_id,
        action_type=payload.action_type,
        payload=payload.payload,
        operator_id=principal.user_id,
        idempotency_key=payload.idempotency_key,
    )
    if action.status == "proposed":
        manual_provenance = payload.payload.get("_provenance") if isinstance(payload.payload, dict) else {}
        create_action_audit_log(
            db=db,
            action_id=action.id,
            event_type="proposed",
            operator_id=principal.user_id,
            event_payload={
                "source": "manual_api",
                "payload_keys": sorted(payload.payload.keys()) if isinstance(payload.payload, dict) else [],
                "provenance": manual_provenance if isinstance(manual_provenance, dict) else {},
            },
        )
    return action


@router.post("/actions/{action_id}/apply", response_model=ChatActionRead)
def apply_action(
    action_id: int,
    payload: ChatActionDecisionRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    action = _ensure_action_access(db, action_id, principal)
    if action.status == "applied":
        return action
    if action.status != "proposed":
        raise HTTPException(status_code=409, detail="action is not in proposed state")
    if is_entity_merge_action_type(action.action_type):
        if not bool(payload.event_payload.get("manual_confirmed")):
            raise HTTPException(status_code=400, detail="entity merge requires manual_confirmed=true")
        if not is_manual_merge_operator(principal.user_id):
            raise HTTPException(status_code=403, detail="entity merge can only be applied by a human operator")
    if action.action_type == "graph.confirm_candidates":
        if not bool(payload.event_payload.get("manual_confirmed")):
            raise HTTPException(status_code=400, detail="graph candidate confirmation requires manual_confirmed=true")
        if not is_manual_merge_operator(principal.user_id):
            raise HTTPException(
                status_code=403,
                detail="graph candidate confirmation can only be applied by a human operator",
            )

    create_action_audit_log(
        db=db,
        action_id=action.id,
        event_type="apply_requested",
        operator_id=principal.user_id,
        event_payload={**payload.event_payload, "provenance": _action_provenance_from_payload(action)},
    )

    try:
        action = apply_action_effects(db, action)
    except ValueError as exc:
        db.rollback()
        action = _ensure_action_access(db, action_id, principal)
        action = set_action_status(db, action, "failed")
        create_action_audit_log(
            db=db,
            action_id=action.id,
            event_type="failed",
            operator_id=principal.user_id,
            event_payload={
                "error": str(exc),
                "provenance": _action_provenance_from_payload(action),
            },
        )
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="action apply failed")

    applied_provenance = {}
    if isinstance(action.apply_result, dict):
        raw_applied_provenance = action.apply_result.get("provenance")
        if isinstance(raw_applied_provenance, dict):
            applied_provenance = raw_applied_provenance
    create_action_audit_log(
        db=db,
        action_id=action.id,
        event_type="applied",
        operator_id=principal.user_id,
        event_payload={**payload.event_payload, "provenance": applied_provenance},
    )
    return action


@router.post("/actions/{action_id}/reject", response_model=ChatActionRead)
def reject_action(
    action_id: int,
    payload: ChatActionDecisionRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    action = _ensure_action_access(db, action_id, principal)
    if action.status == "rejected":
        return action
    if action.status != "proposed":
        raise HTTPException(status_code=409, detail="action is not in proposed state")

    action = set_action_status(db, action, "rejected")
    create_action_audit_log(
        db=db,
        action_id=action.id,
        event_type="rejected",
        operator_id=principal.user_id,
        event_payload={**payload.event_payload, "provenance": _action_provenance_from_payload(action)},
    )
    return action


@router.post("/actions/{action_id}/undo", response_model=ChatActionRead)
def undo_action(
    action_id: int,
    payload: ChatActionDecisionRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    action = _ensure_action_access(db, action_id, principal)
    if action.status == "undone":
        return action
    if action.status != "applied":
        raise HTTPException(status_code=409, detail="only applied action can be undone")

    try:
        action = undo_action_effects(db, action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    create_action_audit_log(
        db=db,
        action_id=action.id,
        event_type="undone",
        operator_id=principal.user_id,
        event_payload={
            **payload.event_payload,
            "provenance": (
                action.apply_result.get("provenance", {}) if isinstance(action.apply_result, dict) else {}
            ),
        },
    )
    return action


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
    if payload.run_mode == "async":
        manual_job_key = f"entity-merge-manual:{project_id}:{int(uuid4().int % 10_000_000)}"
        queued = enqueue_entity_merge_scan_job(
            project_id,
            operator_id=principal.user_id,
            reason="manual_scan_request",
            idempotency_key=manual_job_key,
            attempt=0,
            db=db,
        )
        return EntityMergeScanResult(
            project_id=project_id,
            run_mode="async",
            queued=queued,
            result={
                "status": "queued" if queued else "deduped_or_skipped",
                "idempotency_key": manual_job_key,
            },
        )

    result = run_entity_merge_scan(
        db,
        project_id=project_id,
        operator_id=principal.user_id,
        max_proposals=payload.max_proposals,
        source="manual_scan_api",
    )
    return EntityMergeScanResult(
        project_id=project_id,
        run_mode="sync",
        queued=False,
        result=result,
    )


@router.get("/index-lifecycle/dead-letters", response_model=list[IndexLifecycleDeadLetterRead])
def index_lifecycle_dead_letters(
    project_id: int | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    if project_id is None:
        raise HTTPException(status_code=400, detail="project_id is required")
    _ensure_project_access(project_id, principal)
    rows = peek_index_lifecycle_dead_letters(limit=limit)
    return _filter_project_dead_letters(
        rows,
        project_id=project_id,
        fallback_operator_id=principal.user_id,
    )


@router.post("/index-lifecycle/dead-letters/replay", response_model=IndexLifecycleReplayResult)
def replay_index_lifecycle_dead_letters(
    payload: IndexLifecycleReplayRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    if payload.project_id is None:
        raise HTTPException(status_code=400, detail="project_id is required")
    _ensure_project_access(payload.project_id, principal)
    replay_request_id = f"replay-{uuid4().hex[:12]}"
    dead_letters = pop_index_lifecycle_dead_letters(limit=payload.limit, project_id=payload.project_id)
    counters = _replay_dead_letters(
        dead_letters=dead_letters,
        db=db,
        replay_request_id=replay_request_id,
        principal_user_id=principal.user_id,
    )

    return IndexLifecycleReplayResult(
        requested=int(payload.limit),
        project_id=payload.project_id,
        replayed=int(counters.get("replayed", 0)),
        requeue_failed=int(counters.get("requeue_failed", 0)),
        skipped_invalid=int(counters.get("skipped_invalid", 0)),
        replay_request_id=replay_request_id,
    )


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


@router.get("/projects/{project_id}/model-profiles", response_model=list[ModelProfileRead])
def project_model_profiles(
    project_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    return list_model_profiles(db, project_id)


@router.post("/projects/{project_id}/model-profiles", response_model=ModelProfileRead)
def create_project_model_profile(
    project_id: int,
    payload: ModelProfileUpsertRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return create_model_profile(
            db,
            project_id=project_id,
            operator_id=principal.user_id,
            profile_id=payload.profile_id,
            name=payload.name,
            provider=payload.provider or "openai_compatible",
            base_url=payload.base_url,
            api_key=payload.api_key,
            model=payload.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/projects/{project_id}/model-profiles/{profile_id}", response_model=ModelProfileRead)
def save_project_model_profile(
    project_id: int,
    profile_id: str,
    payload: ModelProfileUpsertRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return update_model_profile(
            db,
            project_id=project_id,
            profile_id=profile_id,
            operator_id=principal.user_id,
            name=payload.name,
            provider=payload.provider,
            base_url=payload.base_url,
            api_key=payload.api_key,
            api_key_supplied=(
                "api_key" in getattr(payload, "model_fields_set", set())
                or "api_key" in getattr(payload, "__fields_set__", set())
            ),
            model=payload.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/projects/{project_id}/model-profiles/{profile_id}/activate", response_model=ModelProfileRead)
def activate_project_model_profile(
    project_id: int,
    profile_id: str,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        return activate_model_profile(
            db,
            project_id=project_id,
            profile_id=profile_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/projects/{project_id}/model-profiles/{profile_id}", response_model=ModelProfileDeleteResult)
def remove_project_model_profile(
    project_id: int,
    profile_id: str,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    try:
        deleted_profile_id = delete_model_profile(
            db,
            project_id=project_id,
            profile_id=profile_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ModelProfileDeleteResult(deleted_profile_id=deleted_profile_id)


@router.get("/projects/{project_id}/consistency-audits", response_model=list[ConsistencyAuditReportRead])
def project_consistency_audits(
    project_id: int,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    return list_consistency_audit_reports(db, project_id=project_id, limit=limit)


@router.post("/projects/{project_id}/consistency-audits/run", response_model=ConsistencyAuditRunResponse)
def run_project_consistency_audit(
    project_id: int,
    payload: ConsistencyAuditRunRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    reason = str(payload.reason or "manual").strip() or "manual"
    run_mode = str(payload.run_mode or "async").strip().lower()
    max_chapters = int(payload.max_chapters) if isinstance(payload.max_chapters, int) and payload.max_chapters > 0 else None
    if run_mode == "sync":
        report = run_consistency_audit(
            db,
            project_id=project_id,
            operator_id=principal.user_id,
            reason=reason,
            trigger_source="manual_sync",
            force=bool(payload.force),
            max_chapters=max_chapters,
        )
        return ConsistencyAuditRunResponse(
            project_id=project_id,
            queued=False,
            run_mode="sync",
            reason=reason,
            trigger_source="manual_sync",
            report=ConsistencyAuditReportRead.model_validate(report),
        )

    idempotency_key = f"consistency-manual-{project_id}-{uuid4().hex[:10]}"
    queued = enqueue_consistency_audit_job(
        project_id,
        operator_id=principal.user_id,
        reason=reason,
        trigger_source="manual_async",
        idempotency_key=idempotency_key,
        force=bool(payload.force),
        max_chapters=max_chapters,
        db=db,
    )
    db.commit()
    return ConsistencyAuditRunResponse(
        project_id=project_id,
        queued=bool(queued),
        run_mode="async",
        reason=reason,
        trigger_source="manual_async",
        idempotency_key=idempotency_key,
        report=None,
    )


@router.get("/projects/{project_id}/graph-timeline", response_model=GraphTimelineSnapshotRead)
def project_graph_timeline(
    project_id: int,
    chapter_index: int = Query(default=0, ge=0, le=100000),
    limit: int = Query(default=240, ge=20, le=1200),
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project_access(project_id, principal)
    resolved_chapter_index = int(chapter_index)
    if resolved_chapter_index <= 0:
        chapters = list_project_chapters(db, project_id)
        resolved_chapter_index = max((int(item.chapter_index or 0) for item in chapters), default=0)
    snapshot = fetch_neo4j_graph_timeline_snapshot(
        project_id,
        current_chapter=resolved_chapter_index if resolved_chapter_index > 0 else None,
        limit=limit,
    )
    snapshot["chapter_index"] = resolved_chapter_index
    return GraphTimelineSnapshotRead.model_validate(snapshot)


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
