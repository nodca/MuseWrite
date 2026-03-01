from __future__ import annotations

import time
from typing import Any, Callable

from sqlmodel import Session

from app.services.pg_job_queue import (
    complete_async_job,
    dequeue_async_job,
    enqueue_async_job,
    fail_async_job,
    peek_queue_payloads,
    pop_queue_payloads,
    reschedule_async_job,
)


class BaseQueue:
    """Shared queue wrapper preserving existing pg_job_queue semantics."""

    def __init__(
        self,
        *,
        queue_name_getter: Callable[[], str],
        max_retries_getter: Callable[[], int],
    ) -> None:
        self._queue_name_getter = queue_name_getter
        self._max_retries_getter = max_retries_getter

    @property
    def queue_name(self) -> str:
        return self._queue_name_getter().strip()

    @property
    def max_retries(self) -> int:
        return max(int(self._max_retries_getter()), 0)

    def enqueue(
        self,
        *,
        payload: dict[str, Any],
        project_id: int,
        action_id: int,
        operator_id: str,
        reason: str,
        mutation_id: str = "",
        expected_version: int = 0,
        idempotency_key: str = "",
        lifecycle_slot: str = "default",
        attempt: int = 0,
        db: Session | None = None,
        max_retries: int | None = None,
    ) -> bool:
        return enqueue_async_job(
            queue_name=self.queue_name,
            payload=payload,
            project_id=int(project_id),
            action_id=int(action_id),
            operator_id=str(operator_id or "system"),
            reason=str(reason or ""),
            mutation_id=str(mutation_id or ""),
            expected_version=int(expected_version),
            idempotency_key=str(idempotency_key or ""),
            lifecycle_slot=str(lifecycle_slot or "default"),
            attempt=int(attempt),
            max_retries=self.max_retries if max_retries is None else max(int(max_retries), 0),
            db=db,
        )

    def dequeue(self, timeout_seconds: int, *, worker_id: str = "worker") -> dict[str, Any] | None:
        return dequeue_async_job(self.queue_name, timeout_seconds, worker_id=worker_id)

    def complete(self, job: dict[str, Any], *, final_status: str = "done", error: str = "") -> bool:
        return complete_async_job(job, final_status=final_status, error=error)

    def retry(
        self,
        job: dict[str, Any],
        *,
        next_attempt: int,
        delay_seconds: int,
        error: str = "",
    ) -> bool:
        return reschedule_async_job(
            job,
            next_attempt=max(int(next_attempt), 0),
            delay_seconds=max(int(delay_seconds), 0),
            error=error,
        )

    def fail(self, job: dict[str, Any], *, error: str = "", failed_status: str = "failed") -> bool:
        return fail_async_job(job, error=error, failed_status=failed_status)


class DeadLetterQueue:
    """Dead-letter queue helper with same payload schema as source jobs."""

    def __init__(self, *, queue_name_getter: Callable[[], str]) -> None:
        self._queue_name_getter = queue_name_getter

    @property
    def queue_name(self) -> str:
        return self._queue_name_getter().strip()

    def push(self, job: dict[str, Any], error: str) -> bool:
        payload = {
            **(job if isinstance(job, dict) else {}),
            "dead_letter_at": int(time.time()),
            "error": str(error or "unknown"),
        }
        return enqueue_async_job(
            queue_name=self.queue_name,
            payload=payload,
            project_id=int(payload.get("project_id", 0)),
            action_id=int(payload.get("action_id", 0)),
            operator_id=str(payload.get("operator_id") or "system"),
            reason=str(payload.get("reason") or "unspecified"),
            mutation_id=str(payload.get("mutation_id") or ""),
            expected_version=int(payload.get("expected_version", 0)),
            idempotency_key=str(payload.get("idempotency_key") or ""),
            lifecycle_slot=str(payload.get("lifecycle_slot") or "default"),
            attempt=int(payload.get("attempt", 0)),
            max_retries=0,
        )

    def peek(self, limit: int = 50) -> list[dict[str, Any]]:
        return peek_queue_payloads(self.queue_name, limit=limit)

    def pop(self, *, limit: int = 20, project_id: int | None = None) -> list[dict[str, Any]]:
        return pop_queue_payloads(self.queue_name, limit=limit, project_id=project_id)
