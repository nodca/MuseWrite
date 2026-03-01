from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from app.core.config import settings

_CLIENT_LOCK = Lock()
_LANGFUSE_CLIENT: Any | None = None
_LANGFUSE_INIT_ERROR: str | None = None


@dataclass
class ChatTracePayload:
    project_id: int
    session_id: int
    user_id: str
    model: str | None
    user_input: str
    assistant_text: str
    usage: dict[str, Any]
    proposed_actions_count: int
    evidence_policy: dict[str, Any]
    evidence_summary: dict[str, Any]
    error: str | None = None


def _get_langfuse_client() -> Any | None:
    global _LANGFUSE_CLIENT
    global _LANGFUSE_INIT_ERROR
    if not settings.langfuse_enabled:
        return None
    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT
    if _LANGFUSE_INIT_ERROR is not None:
        return None

    with _CLIENT_LOCK:
        if _LANGFUSE_CLIENT is not None:
            return _LANGFUSE_CLIENT
        if _LANGFUSE_INIT_ERROR is not None:
            return None
        try:
            from langfuse import Langfuse  # type: ignore

            _LANGFUSE_CLIENT = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            return _LANGFUSE_CLIENT
        except Exception as exc:  # pragma: no cover - optional dependency/network
            _LANGFUSE_INIT_ERROR = str(exc)
            return None


def emit_chat_trace(payload: ChatTracePayload) -> None:
    client = _get_langfuse_client()
    if client is None:
        return

    metadata = {
        "project_id": payload.project_id,
        "session_id": payload.session_id,
        "resolver_order": payload.evidence_policy.get("resolver_order"),
        "ranking_dimensions": payload.evidence_policy.get("ranking_dimensions"),
        "pov_mode": payload.evidence_policy.get("mode"),
        "pov_anchor": payload.evidence_policy.get("anchor"),
        "providers": payload.evidence_policy.get("providers", {}),
        "rag_route": payload.evidence_policy.get("rag_route", {}),
        "rag_short_circuit": payload.evidence_policy.get("rag_short_circuit", {}),
        "quality_gate": payload.evidence_policy.get("quality_gate", {}),
        "context_pack": payload.evidence_policy.get("context_pack", {}),
        "evidence_summary": payload.evidence_summary,
        "proposed_actions_count": payload.proposed_actions_count,
        "error": payload.error,
    }
    trace_input = {
        "user_input": payload.user_input,
        "model": payload.model,
    }
    trace_output = {
        "assistant_text": payload.assistant_text,
        "usage": payload.usage,
    }
    try:
        trace = client.trace(
            name="chat_stream",
            user_id=payload.user_id,
            session_id=str(payload.session_id),
            input=trace_input,
            output=trace_output,
            metadata=metadata,
        )
        if trace is not None and hasattr(trace, "generation"):
            trace.generation(
                name="chat_completion",
                model=payload.model or "",
                input=payload.user_input,
                output=payload.assistant_text,
                metadata=metadata,
                usage=payload.usage,
            )
        if hasattr(client, "flush"):
            client.flush()
    except Exception:
        return
