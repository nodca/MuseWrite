from typing import Any

from sqlmodel import Session

from app.core.config import settings
from app.services.base_queue import BaseQueue

def _queue_name() -> str:
    return settings.graph_sync_queue_name.strip()


_GRAPH_QUEUE = BaseQueue(
    queue_name_getter=_queue_name,
    max_retries_getter=lambda: settings.graph_sync_max_retries,
)


def enqueue_graph_sync_job(
    action_id: int,
    *,
    project_id: int,
    action_type: str,
    payload: dict[str, Any],
    operator_id: str,
    mutation_id: str = "",
    expected_version: int = 0,
    idempotency_key: str = "",
    attempt: int = 0,
    db: Session | None = None,
) -> bool:
    message = {
        "action_id": int(action_id),
        "project_id": int(project_id),
        "action_type": str(action_type or ""),
        "payload": payload if isinstance(payload, dict) else {},
        "operator_id": str(operator_id or "system"),
        "mutation_id": str(mutation_id or ""),
        "expected_version": int(expected_version),
        "idempotency_key": str(idempotency_key or ""),
        "attempt": int(attempt),
    }
    return _GRAPH_QUEUE.enqueue(
        payload=message,
        project_id=int(project_id),
        action_id=int(action_id),
        operator_id=str(operator_id or "system"),
        reason="graph_sync",
        mutation_id=str(mutation_id or ""),
        expected_version=int(expected_version),
        idempotency_key=str(idempotency_key or ""),
        lifecycle_slot="default",
        attempt=int(attempt),
        db=db,
    )


def dequeue_graph_sync_job(timeout_seconds: int, *, worker_id: str = "worker") -> dict[str, Any] | None:
    return _GRAPH_QUEUE.dequeue(timeout_seconds, worker_id=worker_id)


def complete_graph_sync_job(job: dict[str, Any], *, final_status: str = "done", error: str = "") -> bool:
    return _GRAPH_QUEUE.complete(job, final_status=final_status, error=error)


def retry_graph_sync_job(
    job: dict[str, Any],
    *,
    next_attempt: int,
    delay_seconds: int,
    error: str = "",
) -> bool:
    return _GRAPH_QUEUE.retry(
        job,
        next_attempt=next_attempt,
        delay_seconds=max(int(delay_seconds), 0),
        error=error,
    )


def fail_graph_sync_job(job: dict[str, Any], *, error: str = "") -> bool:
    return _GRAPH_QUEUE.fail(job, error=error, failed_status="failed")
