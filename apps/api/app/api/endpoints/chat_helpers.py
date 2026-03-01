from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from sqlmodel import Session

from app.core.auth import AuthPrincipal, ensure_project_access
from app.core.config import settings
from app.services.chat_service import (
    create_action,
    create_action_audit_log,
    get_action_by_id,
    get_session_by_id,
    is_manual_merge_operator,
    run_entity_merge_scan,
)
from app.services.entity_merge_queue import enqueue_entity_merge_scan_job
from app.services.index_lifecycle_queue import (
    enqueue_index_lifecycle_job,
    push_index_lifecycle_dead_letter,
)
from app.services.llm_provider import ChatGenerationResult
from app.services.retrieval_adapters import (
    promote_neo4j_candidate_facts,
    update_neo4j_graph_fact_state,
)


def enforce_quality_gate(generation: ChatGenerationResult, model_context: dict) -> ChatGenerationResult:
    evidence = model_context.get("evidence") if isinstance(model_context, dict) else None
    quality_gate = evidence.get("quality_gate") if isinstance(evidence, dict) else None
    if not isinstance(quality_gate, dict):
        return generation
    degraded = bool(quality_gate.get("degraded"))
    reasons = quality_gate.get("degrade_reasons")
    reason_text = ",".join(str(item) for item in reasons if str(item).strip()) if isinstance(reasons, list) else ""
    generation.usage = {
        **(generation.usage or {}),
        "quality_gate": "degraded" if degraded else "ok",
        "quality_reasons": reason_text,
        "citation_count": quality_gate.get("citation_count"),
        "citation_required": quality_gate.get("citation_required"),
    }
    if degraded and settings.quality_gate_enforce:
        if settings.citation_block_actions:
            generation.proposed_actions = []
        if generation.assistant_text:
            generation.assistant_text = f"[quality-warning:{reason_text or 'unknown'}]\n" + generation.assistant_text
    return generation


def create_proposed_actions(
    db: Session,
    session_id: int,
    user_id: str,
    proposed_actions: list[dict],
    provenance: dict[str, Any] | None = None,
) -> list[int]:
    provenance_payload = provenance or {}
    action_ids: list[int] = []
    for item in proposed_actions:
        action_type = item.get("action_type")
        payload = item.get("payload")
        if not isinstance(action_type, str) or not isinstance(payload, dict):
            continue
        action_payload = dict(payload)
        if provenance_payload:
            action_payload["_provenance"] = provenance_payload
        action = create_action(
            db=db,
            session_id=session_id,
            action_type=action_type,
            payload=action_payload,
            operator_id=user_id,
            idempotency_key=f"chat-{session_id}-{uuid4().hex[:12]}",
        )
        create_action_audit_log(
            db=db,
            action_id=action.id,
            event_type="proposed",
            operator_id=user_id,
            event_payload={
                "source": "model_structured_output",
                "payload_keys": sorted(action_payload.keys()),
                "provenance": provenance_payload,
            },
        )
        action_ids.append(action.id)
    return action_ids


def build_action_provenance(model_context: dict[str, Any] | None) -> dict[str, Any]:
    evidence = model_context.get("evidence") if isinstance(model_context, dict) else {}
    runtime_options = model_context.get("runtime_options") if isinstance(model_context, dict) else {}
    current_chapter = model_context.get("current_chapter") if isinstance(model_context, dict) else {}
    if not isinstance(evidence, dict):
        evidence = {}
    if not isinstance(runtime_options, dict):
        runtime_options = {}
    if not isinstance(current_chapter, dict):
        current_chapter = {}
    quality_gate = evidence.get("quality_gate") if isinstance(evidence.get("quality_gate"), dict) else {}
    dsl_hits = evidence.get("dsl_hits") if isinstance(evidence.get("dsl_hits"), list) else []
    graph_facts = evidence.get("graph_facts") if isinstance(evidence.get("graph_facts"), list) else []
    semantic_hits = evidence.get("semantic_hits") if isinstance(evidence.get("semantic_hits"), list) else []
    dsl_refs = [
        {
            "id": item.get("id"),
            "title": item.get("title"),
            "project_id": item.get("project_id"),
        }
        for item in dsl_hits[:4]
        if isinstance(item, dict)
    ]
    graph_refs = [
        {
            "id": item.get("id"),
            "fact": item.get("fact"),
            "project_id": item.get("project_id"),
        }
        for item in graph_facts[:4]
        if isinstance(item, dict)
    ]
    rag_refs = [
        {
            "id": item.get("id"),
            "title": item.get("title"),
            "citation_source": (
                item.get("citation", {}).get("source")
                if isinstance(item.get("citation"), dict)
                else None
            ),
            "citation_chunk": (
                item.get("citation", {}).get("chunk")
                if isinstance(item.get("citation"), dict)
                else None
            ),
            "project_id": item.get("project_id"),
        }
        for item in semantic_hits[:4]
        if isinstance(item, dict)
    ]
    return {
        "source": "model_structured_output",
        "resolver_order": evidence.get("resolver_order", ["DSL", "GRAPH", "RAG"]),
        "providers": evidence.get("providers", {}),
        "rag_route": evidence.get("rag_route", {}),
        "quality_gate": {
            "degraded": bool(quality_gate.get("degraded")),
            "degrade_reasons": quality_gate.get("degrade_reasons", []),
        },
        "current_chapter_id": current_chapter.get("id"),
        "current_chapter_index": (
            runtime_options.get("current_chapter_index")
            if runtime_options.get("current_chapter_index") is not None
            else current_chapter.get("chapter_index")
        ),
        "evidence_summary": {
            "dsl": len(dsl_hits),
            "graph": len(graph_facts),
            "rag": len(semantic_hits),
        },
        "evidence_refs": {
            "dsl": dsl_refs,
            "graph": graph_refs,
            "rag": rag_refs,
        },
    }


def action_provenance_from_payload(action: Any) -> dict[str, Any]:
    payload = action.payload if hasattr(action, "payload") else {}
    if not isinstance(payload, dict):
        return {}
    provenance = payload.get("_provenance")
    return provenance if isinstance(provenance, dict) else {}


def ensure_session_member_access(
    db: Session,
    session_id: int,
    principal: AuthPrincipal,
    *,
    expected_project_id: int | None = None,
):
    session = get_session_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="chat session not found")
    if expected_project_id is not None and int(session.project_id) != int(expected_project_id):
        raise HTTPException(status_code=400, detail="project_id mismatch")
    ensure_project_access(principal.user_id, int(session.project_id))
    if str(session.user_id) != str(principal.user_id):
        raise HTTPException(status_code=403, detail="session access denied")
    return session


def ensure_action_session_access(db: Session, action_id: int, principal: AuthPrincipal):
    action = get_action_by_id(db, action_id)
    if not action:
        raise HTTPException(status_code=404, detail="action not found")
    ensure_session_member_access(db, int(action.session_id), principal)
    return action


def ensure_project_scope_access(project_id: int, principal: AuthPrincipal) -> None:
    ensure_project_access(principal.user_id, int(project_id))


def normalize_index_lifecycle_dead_letter_row(
    row: Any,
    *,
    fallback_operator_id: str,
) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None

    def _as_int(value: Any, *, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _as_optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    return {
        "project_id": _as_int(row.get("project_id", 0)),
        "operator_id": str(row.get("operator_id") or fallback_operator_id),
        "reason": str(row.get("reason") or "unspecified"),
        "action_id": _as_int(row.get("action_id", 0)),
        "mutation_id": str(row.get("mutation_id") or ""),
        "expected_version": _as_int(row.get("expected_version", 0)),
        "idempotency_key": str(row.get("idempotency_key") or ""),
        "lifecycle_slot": str(row.get("lifecycle_slot") or "default"),
        "attempt": _as_int(row.get("attempt", 0)),
        "queued_at": _as_optional_int(row.get("queued_at")),
        "dead_letter_at": _as_optional_int(row.get("dead_letter_at")),
        "error": str(row.get("error") or ""),
    }


def build_ghost_user_input(prefix_text: str, *, style_prefix: str = "", outline_hint: str = "") -> str:
    compact_prefix = (prefix_text or "").strip()
    if len(compact_prefix) > 1800:
        compact_prefix = compact_prefix[-1800:]
    style_note = ""
    style_hint = (style_prefix or "").strip()
    if style_hint:
        style_note = f"\n风格约束（来自模板）：\n{style_hint[:260]}"
    outline_note = ""
    outline_text = (outline_hint or "").strip()
    if outline_text:
        outline_note = f"\n剧情节拍约束：\n{outline_text[:520]}"
    return (
        "你正在执行 Ghost Text 续写任务。请只输出“下一小段可直接接在正文后”的中文文本，"
        "不要解释，不要使用引号，不要分点。\n"
        "要求：20~80字，保持当前人称与语气，不引入越权设定。\n"
        f"正文前缀：\n{compact_prefix}{style_note}{outline_note}"
    )


def normalize_ghost_suggestion(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1].strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    merged = " ".join(lines)
    if len(merged) > 200:
        merged = merged[:200].rstrip()
    return merged


def review_graph_candidate_batch(
    *,
    project_id: int,
    decision: str | None,
    fact_keys: list[Any],
    manual_confirmed: bool,
    chapter_index: int | None,
    operator_id: str,
) -> dict[str, Any]:
    if not bool(manual_confirmed):
        raise HTTPException(status_code=400, detail="graph candidate review requires manual_confirmed=true")
    if not is_manual_merge_operator(operator_id):
        raise HTTPException(status_code=403, detail="graph candidate review can only be applied by a human operator")

    normalized_fact_keys = list(dict.fromkeys([str(item).strip() for item in fact_keys if str(item).strip()]))
    if not normalized_fact_keys:
        raise HTTPException(status_code=400, detail="fact_keys is required")

    normalized_decision = str(decision or "confirm").strip().lower()
    reviewed_count = 0
    reviewed_fact_keys: list[str] = []
    if normalized_decision == "confirm":
        reviewed_fact_keys = promote_neo4j_candidate_facts(
            project_id,
            fact_keys=normalized_fact_keys,
            source_ref="",
            min_confidence=None,
            limit=max(len(normalized_fact_keys), 1),
            current_chapter=chapter_index,
        )
        reviewed_count = len(reviewed_fact_keys)
    else:
        reviewed_count = update_neo4j_graph_fact_state(
            project_id,
            normalized_fact_keys,
            to_state="rejected",
            from_state="candidate",
            current_chapter=chapter_index,
        )
        reviewed_fact_keys = normalized_fact_keys

    return {
        "decision": normalized_decision,
        "requested_count": len(normalized_fact_keys),
        "reviewed_count": max(int(reviewed_count), 0),
        "fact_keys": reviewed_fact_keys,
    }


def execute_entity_merge_scan(
    *,
    project_id: int,
    run_mode: str,
    max_proposals: int,
    operator_id: str,
    db: Session,
) -> dict[str, Any]:
    normalized_mode = str(run_mode or "sync").strip().lower()
    if normalized_mode == "async":
        manual_job_key = f"entity-merge-manual:{project_id}:{int(uuid4().int % 10_000_000)}"
        queued = enqueue_entity_merge_scan_job(
            project_id,
            operator_id=operator_id,
            reason="manual_scan_request",
            idempotency_key=manual_job_key,
            attempt=0,
            db=db,
        )
        return {
            "run_mode": "async",
            "queued": bool(queued),
            "result": {
                "status": "queued" if queued else "deduped_or_skipped",
                "idempotency_key": manual_job_key,
            },
        }

    result = run_entity_merge_scan(
        db,
        project_id=project_id,
        operator_id=operator_id,
        max_proposals=max_proposals,
        source="manual_scan_api",
    )
    return {
        "run_mode": "sync",
        "queued": False,
        "result": result,
    }


def filter_project_dead_letters(
    rows: list[Any],
    *,
    project_id: int,
    fallback_operator_id: str,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        normalized_row = normalize_index_lifecycle_dead_letter_row(
            row,
            fallback_operator_id=fallback_operator_id,
        )
        if not isinstance(normalized_row, dict):
            continue
        if int(normalized_row.get("project_id", 0)) != int(project_id):
            continue
        filtered.append(normalized_row)
    return filtered


def replay_dead_letters(
    *,
    dead_letters: list[Any],
    db: Session,
    replay_request_id: str,
    principal_user_id: str,
) -> dict[str, int]:
    replayed = 0
    requeue_failed = 0
    skipped_invalid = 0

    for item in dead_letters:
        normalized_item = normalize_index_lifecycle_dead_letter_row(
            item,
            fallback_operator_id=principal_user_id,
        )
        if not isinstance(normalized_item, dict):
            skipped_invalid += 1
            continue
        project_id = int(normalized_item.get("project_id", 0))
        action_id = int(normalized_item.get("action_id", 0))
        operator_id = str(normalized_item.get("operator_id") or principal_user_id)
        reason = str(normalized_item.get("reason") or "unspecified")
        mutation_id = str(normalized_item.get("mutation_id") or "")
        expected_version = int(normalized_item.get("expected_version", 0))
        idempotency_key = str(normalized_item.get("idempotency_key") or "")
        lifecycle_slot = str(normalized_item.get("lifecycle_slot") or "default")
        if project_id <= 0:
            skipped_invalid += 1
            continue

        queued = enqueue_index_lifecycle_job(
            project_id=project_id,
            operator_id=operator_id,
            reason=reason,
            action_id=action_id,
            mutation_id=mutation_id,
            expected_version=expected_version,
            idempotency_key=idempotency_key,
            lifecycle_slot=lifecycle_slot,
            attempt=0,
            db=db,
        )
        if queued:
            replayed += 1
        else:
            requeue_failed += 1
            push_index_lifecycle_dead_letter(normalized_item, "replay_enqueue_failed")

        if action_id > 0:
            action = get_action_by_id(db, action_id)
            if action:
                create_action_audit_log(
                    db=db,
                    action_id=action.id,
                    event_type="index_lifecycle_replayed",
                    operator_id=principal_user_id,
                    event_payload={
                        "replay_request_id": replay_request_id,
                        "queued": queued,
                        "reason": reason,
                        "mutation_id": mutation_id,
                        "expected_version": expected_version,
                        "job_idempotency_key": idempotency_key,
                        "lifecycle_slot": lifecycle_slot,
                        "dead_letter_error": normalized_item.get("error"),
                        "dead_letter_at": normalized_item.get("dead_letter_at"),
                    },
                )

    return {
        "replayed": replayed,
        "requeue_failed": requeue_failed,
        "skipped_invalid": skipped_invalid,
    }
