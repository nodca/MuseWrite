import time
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from sqlmodel import Session, select

from app.core.config import settings
from app.core.database import engine
from app.models.chat import AsyncJob


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _is_postgres(db: Session) -> bool:
    bind = db.get_bind()
    if bind is None:
        return False
    return str(getattr(bind.dialect, "name", "")).lower().startswith("postgres")


def _recover_stale_processing_jobs(db: Session, queue_name: str, now: datetime) -> None:
    timeout_seconds = max(int(settings.job_processing_timeout_seconds), 15)
    threshold = now - timedelta(seconds=timeout_seconds)
    stmt = select(AsyncJob).where(
        AsyncJob.queue_name == queue_name,
        AsyncJob.status == "processing",
        AsyncJob.locked_at.is_not(None),
        AsyncJob.locked_at < threshold,
    )
    rows = db.exec(stmt).all()
    if not rows:
        return
    for row in rows:
        row.status = "queued"
        row.locked_at = None
        row.lock_token = ""
        row.locked_by = ""
        row.available_at = now
        row.updated_at = now
        row.last_error = (f"{row.last_error}\nrecovered_from_stale_processing").strip()
        db.add(row)
    db.commit()


def _serialize_job_payload(row: AsyncJob) -> dict[str, Any]:
    payload = row.payload if isinstance(row.payload, dict) else {}
    normalized = {**payload}
    normalized["project_id"] = int(row.project_id)
    normalized["action_id"] = int(row.action_id)
    normalized["operator_id"] = str(row.operator_id or "worker")
    normalized["reason"] = str(row.reason or "")
    normalized["mutation_id"] = str(row.mutation_id or "")
    normalized["expected_version"] = int(row.expected_version)
    normalized["idempotency_key"] = str(row.idempotency_key or "")
    normalized["lifecycle_slot"] = str(row.lifecycle_slot or "default")
    normalized["attempt"] = int(row.attempt)
    normalized["max_retries"] = int(row.max_retries)
    normalized["queued_at"] = int(row.created_at.timestamp())
    normalized["_job_id"] = int(row.id or 0)
    normalized["_job_queue_name"] = row.queue_name
    normalized["_job_lock_token"] = row.lock_token
    return normalized


def enqueue_async_job(
    *,
    queue_name: str,
    payload: dict[str, Any],
    project_id: int,
    action_id: int,
    operator_id: str,
    reason: str,
    mutation_id: str,
    expected_version: int,
    idempotency_key: str,
    lifecycle_slot: str = "default",
    attempt: int = 0,
    max_retries: int = 3,
    available_delay_seconds: float = 0.0,
    db: Session | None = None,
) -> bool:
    now = _utc_now()
    available_at = now + timedelta(seconds=max(float(available_delay_seconds), 0.0))
    body = payload if isinstance(payload, dict) else {}
    body = {key: value for key, value in body.items() if not str(key).startswith("_job_")}
    row = AsyncJob(
        queue_name=queue_name,
        project_id=int(project_id),
        action_id=int(action_id),
        operator_id=str(operator_id or "system"),
        reason=str(reason or ""),
        mutation_id=str(mutation_id or ""),
        expected_version=int(expected_version),
        idempotency_key=str(idempotency_key or ""),
        lifecycle_slot=str(lifecycle_slot or "default"),
        payload=body,
        attempt=int(attempt),
        max_retries=max(int(max_retries), 0),
        status="queued",
        available_at=available_at,
        created_at=now,
        updated_at=now,
    )

    if db is not None:
        db.add(row)
        db.flush()
        return True

    try:
        with Session(engine) as managed:
            managed.add(row)
            managed.commit()
        return True
    except Exception:
        return False


def dequeue_async_job(
    queue_name: str,
    timeout_seconds: int,
    *,
    worker_id: str,
) -> dict[str, Any] | None:
    timeout_seconds = max(int(timeout_seconds), 1)
    poll_interval = max(float(settings.job_queue_poll_interval_seconds), 0.05)
    deadline = time.monotonic() + timeout_seconds

    while True:
        now = _utc_now()
        try:
            with Session(engine) as db:
                _recover_stale_processing_jobs(db, queue_name, now)
                stmt = (
                    select(AsyncJob)
                    .where(
                        AsyncJob.queue_name == queue_name,
                        AsyncJob.status == "queued",
                        AsyncJob.available_at <= now,
                    )
                    .order_by(AsyncJob.id.asc())
                    .limit(1)
                )
                if _is_postgres(db):
                    stmt = stmt.with_for_update(skip_locked=True)
                row = db.exec(stmt).first()
                if row is not None:
                    row.status = "processing"
                    row.locked_at = now
                    row.locked_by = str(worker_id or "worker")
                    row.lock_token = uuid4().hex[:32]
                    row.updated_at = now
                    db.add(row)
                    db.commit()
                    db.refresh(row)
                    return _serialize_job_payload(row)
        except Exception:
            pass

        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_interval)


def complete_async_job(job: dict[str, Any], *, final_status: str = "done", error: str = "") -> bool:
    job_id = int(job.get("_job_id", 0))
    lock_token = str(job.get("_job_lock_token") or "")
    if job_id <= 0 or not lock_token:
        return False
    now = _utc_now()
    try:
        with Session(engine) as db:
            row = db.get(AsyncJob, job_id)
            if not row:
                return False
            if row.status != "processing":
                return False
            if str(row.lock_token or "") != lock_token:
                return False
            row.status = str(final_status or "done")
            row.locked_at = None
            row.locked_by = ""
            row.lock_token = ""
            if error:
                row.last_error = str(error)[:3900]
            row.updated_at = now
            db.add(row)
            db.commit()
        return True
    except Exception:
        return False


def reschedule_async_job(
    job: dict[str, Any],
    *,
    next_attempt: int,
    delay_seconds: float,
    error: str = "",
) -> bool:
    job_id = int(job.get("_job_id", 0))
    lock_token = str(job.get("_job_lock_token") or "")
    if job_id <= 0 or not lock_token:
        return False
    now = _utc_now()
    try:
        with Session(engine) as db:
            row = db.get(AsyncJob, job_id)
            if not row:
                return False
            if row.status != "processing":
                return False
            if str(row.lock_token or "") != lock_token:
                return False
            row.status = "queued"
            row.attempt = max(int(next_attempt), 0)
            row.available_at = now + timedelta(seconds=max(float(delay_seconds), 0.0))
            row.locked_at = None
            row.locked_by = ""
            row.lock_token = ""
            if error:
                row.last_error = str(error)[:3900]
            row.updated_at = now
            db.add(row)
            db.commit()
        return True
    except Exception:
        return False


def fail_async_job(job: dict[str, Any], *, error: str = "", failed_status: str = "failed") -> bool:
    job_id = int(job.get("_job_id", 0))
    lock_token = str(job.get("_job_lock_token") or "")
    if job_id <= 0 or not lock_token:
        return False
    now = _utc_now()
    try:
        with Session(engine) as db:
            row = db.get(AsyncJob, job_id)
            if not row:
                return False
            if row.status != "processing":
                return False
            if str(row.lock_token or "") != lock_token:
                return False
            row.status = str(failed_status or "failed")
            row.locked_at = None
            row.locked_by = ""
            row.lock_token = ""
            if error:
                row.last_error = str(error)[:3900]
            row.updated_at = now
            db.add(row)
            db.commit()
        return True
    except Exception:
        return False


def peek_queue_payloads(queue_name: str, *, limit: int = 50) -> list[dict[str, Any]]:
    size = max(int(limit), 1)
    try:
        with Session(engine) as db:
            stmt = (
                select(AsyncJob)
                .where(
                    AsyncJob.queue_name == queue_name,
                    AsyncJob.status == "queued",
                )
                .order_by(AsyncJob.id.desc())
                .limit(size)
            )
            rows = db.exec(stmt).all()
            return [_serialize_job_payload(row) for row in rows]
    except Exception:
        return []


def pop_queue_payloads(
    queue_name: str,
    *,
    limit: int = 20,
    project_id: int | None = None,
) -> list[dict[str, Any]]:
    max_items = max(int(limit), 1)
    try:
        with Session(engine) as db:
            scan_size = max_items if project_id is None else max(max_items * 10, 200)
            stmt = (
                select(AsyncJob)
                .where(
                    AsyncJob.queue_name == queue_name,
                    AsyncJob.status == "queued",
                )
                .order_by(AsyncJob.id.desc())
                .limit(scan_size)
            )
            rows = db.exec(stmt).all()
            popped: list[dict[str, Any]] = []
            for row in rows:
                if len(popped) >= max_items:
                    break
                if project_id is not None and int(row.project_id) != int(project_id):
                    continue
                payload = _serialize_job_payload(row)
                db.delete(row)
                popped.append(payload)
            if popped:
                db.commit()
            return popped
    except Exception:
        return []


def cleanup_async_jobs(
    *,
    statuses: tuple[str, ...] = ("done", "failed"),
    older_than_seconds: int = 7 * 86400,
    limit: int = 200,
) -> int:
    normalized_statuses = tuple(
        str(item).strip().lower()
        for item in statuses
        if str(item).strip()
    )
    if not normalized_statuses:
        return 0

    max_items = max(int(limit), 1)
    threshold = _utc_now() - timedelta(seconds=max(int(older_than_seconds), 0))
    try:
        with Session(engine) as db:
            stmt = (
                select(AsyncJob)
                .where(
                    AsyncJob.status.in_(normalized_statuses),
                    AsyncJob.updated_at < threshold,
                )
                .order_by(AsyncJob.updated_at.asc(), AsyncJob.id.asc())
                .limit(max_items)
            )
            rows = db.exec(stmt).all()
            if not rows:
                return 0
            for row in rows:
                db.delete(row)
            db.commit()
            return len(rows)
    except Exception:
        return 0
