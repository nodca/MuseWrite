from typing import Any

from sqlmodel import Session, select

from app.core.config import settings
from app.models.chat import AsyncJob
from app.services.base_queue import BaseQueue

def _queue_name() -> str:
    return settings.consistency_audit_queue_name.strip()


_CONSISTENCY_AUDIT_QUEUE = BaseQueue(
    queue_name_getter=_queue_name,
    max_retries_getter=lambda: settings.consistency_audit_max_retries,
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
    existing = db.exec(stmt).first()
    return existing is not None


def enqueue_consistency_audit_job(
    project_id: int,
    *,
    operator_id: str = "system",
    reason: str = "consistency_audit",
    trigger_source: str = "manual",
    idempotency_key: str = "",
    force: bool = False,
    max_chapters: int | None = None,
    attempt: int = 0,
    db: Session | None = None,
) -> bool:
    queue_name = _CONSISTENCY_AUDIT_QUEUE.queue_name
    normalized_idempotency_key = str(idempotency_key or "").strip()
    if db is not None and _has_pending_job(
        db,
        queue_name=queue_name,
        idempotency_key=normalized_idempotency_key,
    ):
        return False

    payload: dict[str, Any] = {
        "project_id": int(project_id),
        "operator_id": str(operator_id or "system"),
        "reason": str(reason or "consistency_audit"),
        "trigger_source": str(trigger_source or "manual"),
        "idempotency_key": normalized_idempotency_key,
        "force": bool(force),
        "max_chapters": int(max_chapters) if isinstance(max_chapters, int) and max_chapters > 0 else None,
        "attempt": int(attempt),
    }
    return _CONSISTENCY_AUDIT_QUEUE.enqueue(
        payload=payload,
        project_id=int(project_id),
        action_id=0,
        operator_id=str(operator_id or "system"),
        reason=str(reason or "consistency_audit"),
        mutation_id="",
        expected_version=0,
        idempotency_key=normalized_idempotency_key,
        lifecycle_slot="default",
        attempt=int(attempt),
        db=db,
    )


def dequeue_consistency_audit_job(timeout_seconds: int, *, worker_id: str = "worker") -> dict[str, Any] | None:
    return _CONSISTENCY_AUDIT_QUEUE.dequeue(timeout_seconds, worker_id=worker_id)


def complete_consistency_audit_job(job: dict[str, Any], *, final_status: str = "done", error: str = "") -> bool:
    return _CONSISTENCY_AUDIT_QUEUE.complete(job, final_status=final_status, error=error)


def retry_consistency_audit_job(
    job: dict[str, Any],
    *,
    next_attempt: int,
    delay_seconds: int,
    error: str = "",
) -> bool:
    return _CONSISTENCY_AUDIT_QUEUE.retry(
        job,
        next_attempt=max(int(next_attempt), 0),
        delay_seconds=max(int(delay_seconds), 0),
        error=error,
    )


def fail_consistency_audit_job(job: dict[str, Any], *, error: str = "") -> bool:
    return _CONSISTENCY_AUDIT_QUEUE.fail(job, error=error, failed_status="failed")
