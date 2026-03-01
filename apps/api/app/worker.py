import logging
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

from sqlalchemy import text
from sqlmodel import Session, select

from app.core.config import settings
from app.core.database import engine, init_db
from app.models.chat import ChatAction, ProjectMutationVersion
from app.services.chat_service import (
    create_action_audit_log,
    process_graph_sync_for_action,
    run_entity_merge_scan,
)
from app.services.consistency_audit_queue import (
    complete_consistency_audit_job,
    dequeue_consistency_audit_job,
    enqueue_consistency_audit_job,
    fail_consistency_audit_job,
    retry_consistency_audit_job,
)
from app.services.consistency_audit_service import (
    has_consistency_audit_report_on_date,
    latest_consistency_audit_timestamp,
    latest_project_chapter_update_at,
    list_project_ids_with_chapters,
    run_consistency_audit,
)
from app.services.entity_merge_queue import (
    complete_entity_merge_scan_job,
    dequeue_entity_merge_scan_job,
    fail_entity_merge_scan_job,
    retry_entity_merge_scan_job,
)
from app.services.graph_job_queue import (
    complete_graph_sync_job,
    dequeue_graph_sync_job,
    fail_graph_sync_job,
    retry_graph_sync_job,
)
from app.services.index_lifecycle_queue import (
    complete_index_lifecycle_job,
    dequeue_index_lifecycle_job,
    fail_index_lifecycle_job,
    push_index_lifecycle_dead_letter,
    retry_index_lifecycle_job,
)
from app.services.index_lifecycle_service import process_index_lifecycle_rebuild
from app.services.pg_job_queue import cleanup_async_jobs

_LOGGER = logging.getLogger(__name__)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _error_fields(error: str, *, default: str = "unknown") -> dict[str, str]:
    value = str(error or default)
    return {"error": value, "err": value}


def _lifecycle_slot_key(slot: str) -> str:
    return "index_lifecycle_compensation" if slot == "compensation" else "index_lifecycle"


def _action_lifecycle_meta(action: ChatAction, slot: str) -> dict[str, Any]:
    if not isinstance(action.apply_result, dict):
        return {}
    key = _lifecycle_slot_key(slot)
    raw = action.apply_result.get(key)
    return raw if isinstance(raw, dict) else {}


def _current_project_version(db: Session, project_id: int) -> int:
    if project_id <= 0:
        return 0
    stmt = select(ProjectMutationVersion).where(ProjectMutationVersion.project_id == project_id)
    row = db.exec(stmt).first()
    if not row:
        return 0
    return int(row.version)


@contextmanager
def _project_advisory_lock(db: Session, project_id: int) -> Iterator[bool]:
    if project_id <= 0 or not settings.project_advisory_lock_enabled:
        yield True
        return
    bind = db.get_bind()
    dialect_name = str(getattr(getattr(bind, "dialect", None), "name", "")).lower()
    if not dialect_name.startswith("postgres"):
        yield True
        return

    lock_key = int(project_id)
    acquired = False
    try:
        row = db.exec(
            text("SELECT pg_try_advisory_lock(:lock_key)"),
            {"lock_key": lock_key},
        ).first()
        acquired = bool(row[0]) if row is not None and len(row) > 0 else False
        yield acquired
    finally:
        if acquired:
            try:
                db.exec(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": lock_key},
                )
            except Exception:
                pass


def _process_graph_job(job: dict[str, Any]) -> tuple[str, int]:
    action_id = _as_int(job.get("action_id"), 0)
    project_id = _as_int(job.get("project_id"), 0)
    attempt = _as_int(job.get("attempt"), 0)
    mutation_id = str(job.get("mutation_id") or "")
    expected_version = _as_int(job.get("expected_version"), 0)
    job_idempotency_key = str(job.get("idempotency_key") or "")
    if action_id <= 0:
        return "drop_invalid", attempt

    with Session(engine) as db:
        action = db.get(ChatAction, action_id)
        if not action:
            return "drop_missing", attempt

        with _project_advisory_lock(db, project_id) as locked:
            if not locked:
                return "retry_locked", attempt

            if expected_version > 0 and project_id > 0:
                project_version = _current_project_version(db, project_id)
                if project_version > expected_version:
                    create_action_audit_log(
                        db=db,
                        action_id=action.id,
                        event_type="graph_skipped",
                        operator_id=action.operator_id,
                        event_payload={
                            "reason": "project_version_advanced",
                            "job_expected_version": expected_version,
                            "current_project_version": project_version,
                            "job_idempotency_key": job_idempotency_key,
                            "metric": "graph_skipped_stale",
                        },
                    )
                    return "drop_stale", attempt

            if action.status == "undone":
                create_action_audit_log(
                    db=db,
                    action_id=action.id,
                    event_type="graph_skipped",
                    operator_id=action.operator_id,
                    event_payload={
                        "reason": "action_already_undone",
                        "mutation_id": mutation_id,
                        "expected_version": expected_version,
                        "metric": "graph_skipped_stale",
                    },
                )
                return "drop_undone", attempt
            if action.status != "applied":
                return "retry_not_applied", attempt

            sync_status = ""
            current_mutation_id = ""
            if isinstance(action.apply_result, dict):
                graph_sync = action.apply_result.get("graph_sync")
                if isinstance(graph_sync, dict):
                    sync_status = str(graph_sync.get("status") or "")
                    current_mutation_id = str(graph_sync.get("mutation_id") or "")

            if mutation_id and current_mutation_id and mutation_id != current_mutation_id:
                create_action_audit_log(
                    db=db,
                    action_id=action.id,
                    event_type="graph_skipped",
                    operator_id=action.operator_id,
                    event_payload={
                        "reason": "stale_job_mutation",
                        "job_mutation_id": mutation_id,
                        "current_mutation_id": current_mutation_id,
                        "job_idempotency_key": job_idempotency_key,
                        "metric": "graph_skipped_stale",
                    },
                )
                return "drop_stale", attempt

            if sync_status == "synced" and (not mutation_id or mutation_id == current_mutation_id):
                create_action_audit_log(
                    db=db,
                    action_id=action.id,
                    event_type="graph_skipped",
                    operator_id=action.operator_id,
                    event_payload={
                        "reason": "already_synced",
                        "mutation_id": mutation_id or current_mutation_id,
                        "job_idempotency_key": job_idempotency_key,
                        "metric": "graph_skipped_duplicate",
                    },
                )
                return "drop_done", attempt

            process_graph_sync_for_action(
                db=db,
                action=action,
                project_id=project_id,
                action_type=str(job.get("action_type") or action.action_type),
                payload=job.get("payload") if isinstance(job.get("payload"), dict) else (action.payload or {}),
                operator_id=str(job.get("operator_id") or action.operator_id),
                mutation_id=mutation_id,
                expected_version=expected_version,
                job_idempotency_key=job_idempotency_key,
                sync_mode="worker_async",
            )
            return "ok", attempt


def _requeue_graph_job(job: dict[str, Any], attempt: int, error: str = "") -> None:
    max_retries = max(int(settings.graph_sync_max_retries), 0)
    action_id = _as_int(job.get("action_id"), 0)

    def _write_graph_degraded_audit(reason: str) -> None:
        if action_id <= 0:
            return
        with Session(engine) as db:
            action = db.get(ChatAction, action_id)
            if action:
                create_action_audit_log(
                    db=db,
                    action_id=action.id,
                    event_type="graph_degraded",
                    operator_id=action.operator_id,
                    event_payload={
                        "reason": reason,
                        "attempt": attempt,
                        "max_retries": max_retries,
                        **_error_fields(error),
                    },
                )

    if attempt >= max_retries:
        fail_graph_sync_job(job, error=error or "max_retries_exceeded")
        _write_graph_degraded_audit("max_retries_exceeded")
        return

    delay_seconds = max(int(settings.graph_sync_retry_delay_seconds), 0)
    rescheduled = retry_graph_sync_job(
        job,
        next_attempt=attempt + 1,
        delay_seconds=delay_seconds,
        error=error or "retry_scheduled",
    )
    if rescheduled:
        return

    fail_graph_sync_job(job, error=error or "retry_reschedule_failed")
    _write_graph_degraded_audit("requeue_failed")


def _process_index_lifecycle_job(job: dict[str, Any]) -> tuple[str, int, str]:
    attempt = _as_int(job.get("attempt"), 0)
    project_id = _as_int(job.get("project_id"), 0)
    action_id = _as_int(job.get("action_id"), 0)
    reason = str(job.get("reason") or "unspecified")
    mutation_id = str(job.get("mutation_id") or "")
    expected_version = _as_int(job.get("expected_version"), 0)
    idempotency_key = str(job.get("idempotency_key") or "")
    lifecycle_slot = str(job.get("lifecycle_slot") or "default").strip().lower()
    slot = "compensation" if lifecycle_slot == "compensation" else "default"
    key = _lifecycle_slot_key(slot)
    if project_id <= 0:
        return "drop_invalid", attempt, "project_id_invalid"

    with Session(engine) as db:
        action = db.get(ChatAction, action_id) if action_id > 0 else None
        operator_id = str(job.get("operator_id") or (action.operator_id if action else "worker"))

        with _project_advisory_lock(db, project_id) as locked:
            if not locked:
                return "retry_locked", attempt, "project_lock_busy"

            if expected_version > 0:
                current_project_version = _current_project_version(db, project_id)
                if current_project_version > expected_version:
                    if action is not None:
                        create_action_audit_log(
                            db=db,
                            action_id=action.id,
                            event_type="index_lifecycle_skipped",
                            operator_id=operator_id,
                            event_payload={
                                "reason": "project_version_advanced",
                                "job_reason": reason,
                                "job_expected_version": expected_version,
                                "current_project_version": current_project_version,
                                "job_idempotency_key": idempotency_key,
                            },
                        )
                    return "drop_stale", attempt, ""

            if action is not None:
                lifecycle_meta = _action_lifecycle_meta(action, slot)
                current_status = str(lifecycle_meta.get("status") or "")
                current_mutation_id = str(lifecycle_meta.get("mutation_id") or "")
                current_expected_version = _as_int(lifecycle_meta.get("expected_version"), 0)

                if slot == "default" and action.status == "undone":
                    create_action_audit_log(
                        db=db,
                        action_id=action.id,
                        event_type="index_lifecycle_skipped",
                        operator_id=operator_id,
                        event_payload={
                            "reason": "action_already_undone",
                            "job_reason": reason,
                            "job_mutation_id": mutation_id,
                            "expected_version": expected_version,
                            "job_idempotency_key": idempotency_key,
                        },
                    )
                    return "drop_undone", attempt, ""

                if slot == "default" and action.status != "applied":
                    return "retry_not_applied", attempt, ""

                if slot == "compensation" and action.status not in {"undone", "applied"}:
                    return "retry_not_applied", attempt, ""

                if mutation_id and current_mutation_id and mutation_id != current_mutation_id:
                    create_action_audit_log(
                        db=db,
                        action_id=action.id,
                        event_type="index_lifecycle_skipped",
                        operator_id=operator_id,
                        event_payload={
                            "reason": "stale_job_mutation",
                            "job_reason": reason,
                            "job_mutation_id": mutation_id,
                            "current_mutation_id": current_mutation_id,
                            "job_idempotency_key": idempotency_key,
                        },
                    )
                    return "drop_stale", attempt, ""

                if (
                    expected_version > 0
                    and current_expected_version > 0
                    and expected_version != current_expected_version
                ):
                    create_action_audit_log(
                        db=db,
                        action_id=action.id,
                        event_type="index_lifecycle_skipped",
                        operator_id=operator_id,
                        event_payload={
                            "reason": "expected_version_mismatch",
                            "job_reason": reason,
                            "job_expected_version": expected_version,
                            "current_expected_version": current_expected_version,
                            "job_idempotency_key": idempotency_key,
                        },
                    )
                    return "drop_stale", attempt, ""

                if current_status == "completed" and (not mutation_id or mutation_id == current_mutation_id):
                    create_action_audit_log(
                        db=db,
                        action_id=action.id,
                        event_type="index_lifecycle_skipped",
                        operator_id=operator_id,
                        event_payload={
                            "reason": "already_completed",
                            "job_reason": reason,
                            "mutation_id": mutation_id or current_mutation_id,
                            "job_idempotency_key": idempotency_key,
                        },
                    )
                    return "drop_done", attempt, ""

                if current_status == "canceled" and slot == "default":
                    create_action_audit_log(
                        db=db,
                        action_id=action.id,
                        event_type="index_lifecycle_skipped",
                        operator_id=operator_id,
                        event_payload={
                            "reason": "canceled_by_compensation",
                            "job_reason": reason,
                            "mutation_id": mutation_id or current_mutation_id,
                            "job_idempotency_key": idempotency_key,
                        },
                    )
                    return "drop_stale", attempt, ""

            lifecycle_result = process_index_lifecycle_rebuild(
                db=db,
                project_id=project_id,
                reason=reason,
                lifecycle_id=mutation_id or f"lifecycle-{project_id}-{int(time.time())}",
            )

            if action is not None:
                lifecycle_meta = _action_lifecycle_meta(action, slot)
                action.apply_result = {
                    **(action.apply_result or {}),
                    key: {
                        **lifecycle_meta,
                        "status": "completed",
                        "mode": "worker_async",
                        "reason": reason,
                        "mutation_id": mutation_id or str(lifecycle_meta.get("mutation_id") or ""),
                        "expected_version": expected_version
                        or _as_int(lifecycle_meta.get("expected_version"), 0),
                        "job_idempotency_key": idempotency_key
                        or str(lifecycle_meta.get("job_idempotency_key") or ""),
                        "result": lifecycle_result,
                    },
                }
                db.add(action)
                db.commit()
                db.refresh(action)
                create_action_audit_log(
                    db=db,
                    action_id=action.id,
                    event_type="index_lifecycle_done",
                    operator_id=operator_id,
                    event_payload={
                        "mode": "worker_async",
                        "slot": slot,
                        "reason": reason,
                        "mutation_id": mutation_id,
                        "expected_version": expected_version,
                        "job_idempotency_key": idempotency_key,
                        "result": lifecycle_result,
                    },
                )

            return "ok", attempt, ""


def _requeue_index_lifecycle_job(job: dict[str, Any], attempt: int, error: str) -> None:
    max_retries = max(int(settings.index_lifecycle_max_retries), 0)
    action_id = _as_int(job.get("action_id"), 0)
    operator_id = str(job.get("operator_id") or "worker")
    reason = str(job.get("reason") or "unspecified")
    idempotency_key = str(job.get("idempotency_key") or "")

    if attempt >= max_retries:
        fail_index_lifecycle_job(job, error=error or "max_retries_exceeded")
        pushed = push_index_lifecycle_dead_letter(job, error or "max_retries_exceeded")
        if action_id > 0:
            with Session(engine) as db:
                action = db.get(ChatAction, action_id)
                if action:
                    create_action_audit_log(
                        db=db,
                        action_id=action.id,
                        event_type="index_lifecycle_degraded",
                        operator_id=operator_id or action.operator_id,
                        event_payload={
                            "reason": "max_retries_exceeded",
                            "attempt": attempt,
                            "max_retries": max_retries,
                            "dead_letter_pushed": pushed,
                            "job_reason": reason,
                            "job_idempotency_key": idempotency_key,
                            **_error_fields(error),
                        },
                    )
        return

    delay_seconds = max(int(settings.index_lifecycle_retry_delay_seconds), 0)
    rescheduled = retry_index_lifecycle_job(
        job,
        next_attempt=attempt + 1,
        delay_seconds=delay_seconds,
        error=error or "retry_scheduled",
    )
    if rescheduled:
        return

    fail_index_lifecycle_job(job, error=error or "retry_reschedule_failed")
    pushed = push_index_lifecycle_dead_letter(job, error or "requeue_failed")
    if action_id > 0:
        with Session(engine) as db:
            action = db.get(ChatAction, action_id)
            if action:
                create_action_audit_log(
                    db=db,
                    action_id=action.id,
                    event_type="index_lifecycle_degraded",
                    operator_id=operator_id or action.operator_id,
                    event_payload={
                        "reason": "requeue_failed",
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "dead_letter_pushed": pushed,
                        "job_reason": reason,
                        "job_idempotency_key": idempotency_key,
                        **_error_fields(error),
                    },
                )


def _process_entity_merge_scan_job(job: dict[str, Any]) -> tuple[str, int, str]:
    attempt = _as_int(job.get("attempt"), 0)
    project_id = _as_int(job.get("project_id"), 0)
    operator_id = str(job.get("operator_id") or "system-entity-merge")
    reason = str(job.get("reason") or "entity_merge_scan")
    if project_id <= 0:
        return "drop_invalid", attempt, "project_id_invalid"
    if not settings.entity_merge_scan_enabled:
        return "drop_disabled", attempt, ""

    with Session(engine) as db:
        with _project_advisory_lock(db, project_id) as locked:
            if not locked:
                return "retry_locked", attempt, "project_lock_busy"
            result = run_entity_merge_scan(
                db,
                project_id=project_id,
                operator_id=operator_id,
                max_proposals=settings.entity_merge_scan_max_proposals,
                source=f"worker:{reason}",
            )
            return "ok", attempt, str(result.get("status") or "")


def _requeue_entity_merge_scan_job(job: dict[str, Any], attempt: int, error: str) -> None:
    max_retries = max(int(settings.entity_merge_scan_max_retries), 0)
    if attempt >= max_retries:
        fail_entity_merge_scan_job(job, error=error or "max_retries_exceeded")
        return

    delay_seconds = max(int(settings.entity_merge_scan_retry_delay_seconds), 0)
    rescheduled = retry_entity_merge_scan_job(
        job,
        next_attempt=attempt + 1,
        delay_seconds=delay_seconds,
        error=error or "retry_scheduled",
    )
    if rescheduled:
        return
    fail_entity_merge_scan_job(job, error=error or "retry_reschedule_failed")


def _process_consistency_audit_job(job: dict[str, Any]) -> tuple[str, int, str]:
    attempt = _as_int(job.get("attempt"), 0)
    project_id = _as_int(job.get("project_id"), 0)
    operator_id = str(job.get("operator_id") or "system-consistency-audit")
    reason = str(job.get("reason") or "consistency_audit")
    trigger_source = str(job.get("trigger_source") or "worker")
    force = bool(job.get("force"))
    max_chapters_raw = _as_int(job.get("max_chapters"), 0)
    max_chapters = max_chapters_raw if max_chapters_raw > 0 else None
    if project_id <= 0:
        return "drop_invalid", attempt, "project_id_invalid"
    if not settings.consistency_audit_enabled:
        return "drop_disabled", attempt, ""

    with Session(engine) as db:
        with _project_advisory_lock(db, project_id) as locked:
            if not locked:
                return "retry_locked", attempt, "project_lock_busy"
            report = run_consistency_audit(
                db,
                project_id=project_id,
                operator_id=operator_id,
                reason=reason,
                trigger_source=trigger_source,
                force=force,
                max_chapters=max_chapters,
            )
            return "ok", attempt, str(report.get("status") or "ok")


def _requeue_consistency_audit_job(job: dict[str, Any], attempt: int, error: str) -> None:
    max_retries = max(int(settings.consistency_audit_max_retries), 0)
    if attempt >= max_retries:
        fail_consistency_audit_job(job, error=error or "max_retries_exceeded")
        return

    delay_seconds = max(int(settings.consistency_audit_retry_delay_seconds), 0)
    rescheduled = retry_consistency_audit_job(
        job,
        next_attempt=attempt + 1,
        delay_seconds=delay_seconds,
        error=error or "retry_scheduled",
    )
    if rescheduled:
        return
    fail_consistency_audit_job(job, error=error or "retry_reschedule_failed")


def _enqueue_due_consistency_audit_jobs() -> int:
    if not settings.consistency_audit_enabled or not settings.consistency_audit_auto_enqueue:
        return 0

    now = datetime.now(timezone.utc)
    scheduler_limit = max(int(settings.consistency_audit_scheduler_project_scan_limit), 1)
    idle_threshold = now - timedelta(minutes=max(int(settings.consistency_audit_idle_minutes), 1))
    daily_hour_utc = min(max(int(settings.consistency_audit_daily_hour_utc), 0), 23)
    project_ids: list[int] = []
    queued_count = 0

    with Session(engine) as db:
        project_ids = list_project_ids_with_chapters(db, limit=scheduler_limit)
        for project_id in project_ids:
            latest_update_at = latest_project_chapter_update_at(db, project_id)
            if latest_update_at is None:
                continue
            latest_report_at = latest_consistency_audit_timestamp(db, project_id)
            if latest_report_at is not None and latest_report_at >= latest_update_at:
                continue

            due_idle = latest_update_at <= idle_threshold
            due_daily = (
                now.hour >= daily_hour_utc
                and not has_consistency_audit_report_on_date(
                    db,
                    project_id,
                    now.date(),
                )
            )
            if not (due_idle or due_daily):
                continue

            reason = "nightly_idle" if due_idle else "nightly_daily"
            idempotency_key = (
                f"consistency-auto-{project_id}-{int(latest_update_at.timestamp())}-{reason}"
            )
            queued = enqueue_consistency_audit_job(
                project_id,
                operator_id="system-consistency-audit",
                reason=reason,
                trigger_source="worker_scheduler",
                idempotency_key=idempotency_key,
                force=False,
                max_chapters=int(settings.consistency_audit_max_chapters),
                db=db,
            )
            if queued:
                queued_count += 1

        if queued_count > 0:
            db.commit()

    if queued_count > 0:
        _LOGGER.info(
            "worker_consistency_audit_enqueue queued=%s scanned_projects=%s",
            queued_count,
            len(project_ids),
        )
    return queued_count


def run() -> None:
    init_db()
    graph_block_seconds = max(int(settings.graph_sync_worker_block_seconds), 1)
    entity_merge_block_seconds = max(int(settings.entity_merge_scan_worker_block_seconds), 1)
    consistency_audit_block_seconds = max(int(settings.consistency_audit_worker_block_seconds), 1)
    consistency_audit_scheduler_interval = max(float(settings.consistency_audit_scheduler_interval_seconds), 10.0)
    cleanup_interval_seconds = max(float(settings.job_cleanup_interval_seconds), 5.0)
    cleanup_retention_seconds = max(int(settings.job_cleanup_retention_seconds), 0)
    cleanup_batch_size = max(int(settings.job_cleanup_batch_size), 1)
    next_cleanup_at = time.monotonic() + cleanup_interval_seconds
    next_consistency_enqueue_at = time.monotonic() + consistency_audit_scheduler_interval

    while True:
        if settings.job_cleanup_enabled and time.monotonic() >= next_cleanup_at:
            cleanup_started_at = time.perf_counter()
            deleted_jobs = cleanup_async_jobs(
                statuses=("done", "failed"),
                older_than_seconds=cleanup_retention_seconds,
                limit=cleanup_batch_size,
            )
            cleanup_elapsed_ms = int((time.perf_counter() - cleanup_started_at) * 1000)
            _LOGGER.info(
                "worker_async_job_cleanup deleted=%s elapsed_ms=%s retention_seconds=%s batch_size=%s",
                deleted_jobs,
                cleanup_elapsed_ms,
                cleanup_retention_seconds,
                cleanup_batch_size,
            )
            next_cleanup_at = time.monotonic() + cleanup_interval_seconds

        if (
            settings.consistency_audit_enabled
            and settings.consistency_audit_auto_enqueue
            and time.monotonic() >= next_consistency_enqueue_at
        ):
            try:
                _enqueue_due_consistency_audit_jobs()
            except Exception as exc:
                err = str(exc)
                _LOGGER.warning("worker_consistency_audit_enqueue_failed err=%s error=%s", err, err)
            next_consistency_enqueue_at = time.monotonic() + consistency_audit_scheduler_interval

        processed = False

        graph_job = dequeue_graph_sync_job(1, worker_id="graph-worker")
        if graph_job:
            processed = True
            graph_error = ""
            try:
                graph_result, graph_attempt = _process_graph_job(graph_job)
            except Exception as exc:
                graph_result, graph_attempt = "retry_error", _as_int(graph_job.get("attempt"), 0)
                graph_error = str(exc)

            if graph_result in {"retry_not_applied", "retry_error", "retry_locked"}:
                _requeue_graph_job(graph_job, graph_attempt, graph_error or graph_result)
            else:
                complete_graph_sync_job(
                    graph_job,
                    final_status="done",
                    error=graph_error if graph_result == "retry_error" else "",
                )

        lifecycle_job = dequeue_index_lifecycle_job(
            1 if processed else graph_block_seconds,
            worker_id="index-lifecycle-worker",
        )
        if lifecycle_job:
            processed = True
            try:
                lifecycle_result, lifecycle_attempt, lifecycle_error = _process_index_lifecycle_job(lifecycle_job)
            except Exception as exc:
                lifecycle_result = "retry_error"
                lifecycle_attempt = _as_int(lifecycle_job.get("attempt"), 0)
                lifecycle_error = str(exc)

            if lifecycle_result in {"retry_not_applied", "retry_error", "retry_locked"}:
                _requeue_index_lifecycle_job(lifecycle_job, lifecycle_attempt, lifecycle_error)
            else:
                complete_index_lifecycle_job(lifecycle_job, final_status="done", error=lifecycle_error)

        merge_job = dequeue_entity_merge_scan_job(
            1 if processed else entity_merge_block_seconds,
            worker_id="entity-merge-scan-worker",
        )
        if merge_job:
            processed = True
            try:
                merge_result, merge_attempt, merge_error = _process_entity_merge_scan_job(merge_job)
            except Exception as exc:
                merge_result = "retry_error"
                merge_attempt = _as_int(merge_job.get("attempt"), 0)
                merge_error = str(exc)

            if merge_result in {"retry_error", "retry_locked"}:
                _requeue_entity_merge_scan_job(merge_job, merge_attempt, merge_error)
            else:
                complete_entity_merge_scan_job(merge_job, final_status="done", error="")

        consistency_job = dequeue_consistency_audit_job(
            1 if processed else consistency_audit_block_seconds,
            worker_id="consistency-audit-worker",
        )
        if consistency_job:
            processed = True
            try:
                audit_result, audit_attempt, audit_error = _process_consistency_audit_job(consistency_job)
            except Exception as exc:
                audit_result = "retry_error"
                audit_attempt = _as_int(consistency_job.get("attempt"), 0)
                audit_error = str(exc)

            if audit_result in {"retry_error", "retry_locked"}:
                _requeue_consistency_audit_job(consistency_job, audit_attempt, audit_error)
            else:
                complete_consistency_audit_job(consistency_job, final_status="done", error="")

        if not processed:
            continue


if __name__ == "__main__":
    run()
