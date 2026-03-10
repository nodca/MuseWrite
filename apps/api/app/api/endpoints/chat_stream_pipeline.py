import asyncio
from copy import deepcopy
from typing import Any, AsyncGenerator, Awaitable, Callable

from app.core.config import settings
from app.core.sse import sse_event
from app.services.llm_provider import ChatGenerationResult
from app.services.telemetry import ChatTracePayload


async def stream_chat_events(
    *,
    payload: Any,
    db: Any,
    principal: Any,
    session: Any,
    assistant_msg: Any,
    compiled_bundle: Any,
    runtime_model_profile: dict[str, Any] | None,
    generate_chat_fn: Callable[..., Awaitable[ChatGenerationResult]],
    generate_tot_brainstorm_fn: Callable[..., Awaitable[Any]],
    enforce_quality_gate_fn: Callable[[ChatGenerationResult, dict[str, Any] | None], ChatGenerationResult],
    create_proposed_actions_fn: Callable[..., list[int]],
    build_action_provenance_fn: Callable[[dict[str, Any] | None], dict[str, Any]],
    emit_chat_trace_fn: Callable[[ChatTracePayload], None],
    update_message_content_fn: Callable[..., Any],
    update_message_provenance_fn: Callable[..., Any],
) -> AsyncGenerator[str, None]:
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
            tot_result = await generate_tot_brainstorm_fn(
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
        generation = await generate_chat_fn(
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
    generation = enforce_quality_gate_fn(generation, compiled_bundle.model_context)

    if generation.proposed_actions:
        proposed_action_ids = create_proposed_actions_fn(
            db=db,
            session_id=session.id,
            user_id=principal.user_id,
            proposed_actions=generation.proposed_actions,
            provenance=build_action_provenance_fn(compiled_bundle.model_context),
        )

    emit_chat_trace_fn(
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
    update_message_provenance_fn(
        assistant_msg.id,
        {
            "context_xray": {
                "version": 1,
                "evidence": deepcopy(compiled_bundle.evidence_event)
                if isinstance(compiled_bundle.evidence_event, dict)
                else None,
            }
        },
        db=db,
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
        update_message_content_fn(assistant_msg.id, final_text, db=db)
        yield sse_event(
            {
                "type": "done",
                "assistant_message_id": assistant_msg.id,
                "usage": generation.usage
                or {"input_chars": len(payload.content), "output_chars": len(final_text)},
            }
        )
    except asyncio.CancelledError:
        update_message_content_fn(assistant_msg.id, "".join(chunks), db=db)
        raise
    except Exception as exc:
        update_message_content_fn(assistant_msg.id, "".join(chunks), db=db)
        yield sse_event({"type": "error", "message": str(exc)})
