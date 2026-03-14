from typing import Any

from sqlmodel import Session, select

from app.models.chat import AsyncJob
from app.services.base_queue import BaseQueue

_QUEUE = "book_memory_consolidation_jobs"


def _queue_name() -> str:
    return _QUEUE


_BOOK_MEMORY_CONSOLIDATION_QUEUE = BaseQueue(
    queue_name_getter=_queue_name,
    max_retries_getter=lambda: 2,
)


def _has_pending_job(db: Session, *, queue_name: str, idempotency_key: str) -> bool:
    if not idempotency_key:
        return False
    stmt = (
        select(AsyncJob)
        .where(
            AsyncJob.queue_name == queue_name,
            AsyncJob.idempotency_key == idempotency_key,
            AsyncJob.status.in_(("queued", "processing")),
        )
        .order_by(AsyncJob.id.desc())
        .limit(1)
    )
    return db.exec(stmt).first() is not None


def enqueue_book_memory_consolidation_job(
    *,
    project_id: int,
    chapter_id: int | None = None,
    scene_beat_id: int | None = None,
    operator_id: str = "system",
    reason: str = "book_memory_consolidation",
    action_id: int = 0,
    idempotency_key: str = "",
    attempt: int = 0,
    db: Session | None = None,
) -> bool:
    queue_name = _BOOK_MEMORY_CONSOLIDATION_QUEUE.queue_name
    normalized_idempotency_key = str(idempotency_key or "").strip()
    if db is not None and _has_pending_job(db, queue_name=queue_name, idempotency_key=normalized_idempotency_key):
        return False

    payload: dict[str, Any] = {
        "project_id": int(project_id),
        "chapter_id": int(chapter_id) if chapter_id else None,
        "scene_beat_id": int(scene_beat_id) if scene_beat_id else None,
        "operator_id": str(operator_id or "system"),
        "reason": str(reason or "book_memory_consolidation"),
        "idempotency_key": normalized_idempotency_key,
        "attempt": int(attempt),
    }
    return _BOOK_MEMORY_CONSOLIDATION_QUEUE.enqueue(
        payload=payload,
        project_id=int(project_id),
        action_id=int(action_id),
        operator_id=str(operator_id or "system"),
        reason=str(reason or "book_memory_consolidation"),
        mutation_id="",
        expected_version=0,
        idempotency_key=normalized_idempotency_key,
        lifecycle_slot="default",
        attempt=int(attempt),
        db=db,
    )


def dequeue_book_memory_consolidation_job(timeout_seconds: int, *, worker_id: str = "worker") -> dict[str, Any] | None:
    return _BOOK_MEMORY_CONSOLIDATION_QUEUE.dequeue(timeout_seconds, worker_id=worker_id)


def complete_book_memory_consolidation_job(job: dict[str, Any], *, final_status: str = "done", error: str = "") -> bool:
    return _BOOK_MEMORY_CONSOLIDATION_QUEUE.complete(job, final_status=final_status, error=error)


def retry_book_memory_consolidation_job(
    job: dict[str, Any],
    *,
    next_attempt: int,
    delay_seconds: int,
    error: str = "",
) -> bool:
    return _BOOK_MEMORY_CONSOLIDATION_QUEUE.retry(
        job,
        next_attempt=max(int(next_attempt), 0),
        delay_seconds=max(int(delay_seconds), 0),
        error=error,
    )


def fail_book_memory_consolidation_job(job: dict[str, Any], *, error: str = "") -> bool:
    return _BOOK_MEMORY_CONSOLIDATION_QUEUE.fail(job, error=error, failed_status="failed")
