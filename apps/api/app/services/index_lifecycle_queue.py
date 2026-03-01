from typing import Any

from sqlmodel import Session

from app.core.config import settings
from app.services.base_queue import BaseQueue, DeadLetterQueue

def _queue_name() -> str:
    return settings.index_lifecycle_queue_name.strip()


def _dead_letter_queue_name() -> str:
    return settings.index_lifecycle_dead_letter_queue_name.strip()


_INDEX_LIFECYCLE_QUEUE = BaseQueue(
    queue_name_getter=_queue_name,
    max_retries_getter=lambda: settings.index_lifecycle_max_retries,
)

_INDEX_LIFECYCLE_DEAD_LETTER_QUEUE = DeadLetterQueue(queue_name_getter=_dead_letter_queue_name)


def enqueue_index_lifecycle_job(
    *,
    project_id: int,
    operator_id: str,
    reason: str,
    action_id: int = 0,
    mutation_id: str = "",
    expected_version: int = 0,
    idempotency_key: str = "",
    lifecycle_slot: str = "default",
    attempt: int = 0,
    db: Session | None = None,
) -> bool:
    message = {
        "project_id": int(project_id),
        "operator_id": str(operator_id or "system"),
        "reason": str(reason or "unspecified"),
        "action_id": int(action_id),
        "mutation_id": str(mutation_id or ""),
        "expected_version": int(expected_version),
        "idempotency_key": str(idempotency_key or ""),
        "lifecycle_slot": str(lifecycle_slot or "default"),
        "attempt": int(attempt),
    }
    return _INDEX_LIFECYCLE_QUEUE.enqueue(
        payload=message,
        project_id=int(project_id),
        action_id=int(action_id),
        operator_id=str(operator_id or "system"),
        reason=str(reason or "unspecified"),
        mutation_id=str(mutation_id or ""),
        expected_version=int(expected_version),
        idempotency_key=str(idempotency_key or ""),
        lifecycle_slot=str(lifecycle_slot or "default"),
        attempt=int(attempt),
        db=db,
    )


def dequeue_index_lifecycle_job(timeout_seconds: int, *, worker_id: str = "worker") -> dict[str, Any] | None:
    return _INDEX_LIFECYCLE_QUEUE.dequeue(timeout_seconds, worker_id=worker_id)


def complete_index_lifecycle_job(job: dict[str, Any], *, final_status: str = "done", error: str = "") -> bool:
    return _INDEX_LIFECYCLE_QUEUE.complete(job, final_status=final_status, error=error)


def retry_index_lifecycle_job(
    job: dict[str, Any],
    *,
    next_attempt: int,
    delay_seconds: int,
    error: str = "",
) -> bool:
    return _INDEX_LIFECYCLE_QUEUE.retry(
        job,
        next_attempt=max(int(next_attempt), 0),
        delay_seconds=max(int(delay_seconds), 0),
        error=error,
    )


def fail_index_lifecycle_job(job: dict[str, Any], *, error: str = "") -> bool:
    return _INDEX_LIFECYCLE_QUEUE.fail(job, error=error, failed_status="failed")


def push_index_lifecycle_dead_letter(job: dict[str, Any], error: str) -> bool:
    return _INDEX_LIFECYCLE_DEAD_LETTER_QUEUE.push(job, error)


def peek_index_lifecycle_dead_letters(limit: int = 50) -> list[dict[str, Any]]:
    return _INDEX_LIFECYCLE_DEAD_LETTER_QUEUE.peek(limit=limit)


def pop_index_lifecycle_dead_letters(
    *,
    limit: int = 20,
    project_id: int | None = None,
) -> list[dict[str, Any]]:
    return _INDEX_LIFECYCLE_DEAD_LETTER_QUEUE.pop(limit=limit, project_id=project_id)
