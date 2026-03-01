import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app import worker as worker_module
from app.core.config import settings
from app.models.chat import ChatAction, ChatSession, ProjectMutationVersion


def _mock_project_lock(acquired: bool):
    @contextmanager
    def _lock(_db, _project_id):
        yield acquired

    return _lock


class WorkerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

        self._snapshot = {
            "engine": worker_module.engine,
            "index_lifecycle_max_retries": settings.index_lifecycle_max_retries,
            "index_lifecycle_retry_delay_seconds": settings.index_lifecycle_retry_delay_seconds,
            "consistency_audit_enabled": settings.consistency_audit_enabled,
            "consistency_audit_auto_enqueue": settings.consistency_audit_auto_enqueue,
            "consistency_audit_scheduler_project_scan_limit": settings.consistency_audit_scheduler_project_scan_limit,
            "consistency_audit_idle_minutes": settings.consistency_audit_idle_minutes,
            "consistency_audit_daily_hour_utc": settings.consistency_audit_daily_hour_utc,
            "consistency_audit_max_chapters": settings.consistency_audit_max_chapters,
        }
        worker_module.engine = self.engine

    def tearDown(self) -> None:
        worker_module.engine = self._snapshot["engine"]
        settings.index_lifecycle_max_retries = self._snapshot["index_lifecycle_max_retries"]
        settings.index_lifecycle_retry_delay_seconds = self._snapshot["index_lifecycle_retry_delay_seconds"]
        settings.consistency_audit_enabled = self._snapshot["consistency_audit_enabled"]
        settings.consistency_audit_auto_enqueue = self._snapshot["consistency_audit_auto_enqueue"]
        settings.consistency_audit_scheduler_project_scan_limit = self._snapshot[
            "consistency_audit_scheduler_project_scan_limit"
        ]
        settings.consistency_audit_idle_minutes = self._snapshot["consistency_audit_idle_minutes"]
        settings.consistency_audit_daily_hour_utc = self._snapshot["consistency_audit_daily_hour_utc"]
        settings.consistency_audit_max_chapters = self._snapshot["consistency_audit_max_chapters"]
        SQLModel.metadata.drop_all(self.engine)
        self.engine.dispose()

    def _create_action(
        self,
        *,
        project_id: int = 1,
        status: str = "applied",
        apply_result: dict | None = None,
        operator_id: str = "tester",
    ) -> int:
        with Session(self.engine) as db:
            session = ChatSession(project_id=project_id, user_id=operator_id, title="worker-test")
            db.add(session)
            db.commit()
            db.refresh(session)

            action = ChatAction(
                session_id=int(session.id or 0),
                action_type="setting.upsert",
                status=status,
                payload={},
                apply_result=apply_result or {},
                idempotency_key=f"worker-{status}-{project_id}-{int(session.id or 0)}",
                operator_id=operator_id,
            )
            db.add(action)
            db.commit()
            db.refresh(action)
            return int(action.id or 0)

    def _set_project_version(self, project_id: int, version: int) -> None:
        with Session(self.engine) as db:
            row = ProjectMutationVersion(project_id=project_id, version=version)
            db.add(row)
            db.commit()

    def test_process_graph_job_returns_drop_invalid_when_action_id_missing(self) -> None:
        result, attempt = worker_module._process_graph_job({"action_id": 0, "attempt": 7})
        self.assertEqual(result, "drop_invalid")
        self.assertEqual(attempt, 7)

    def test_process_graph_job_returns_retry_locked_when_project_lock_busy(self) -> None:
        action_id = self._create_action(project_id=11, status="applied")
        job = {"action_id": action_id, "project_id": 11, "attempt": 1}

        with patch("app.worker._project_advisory_lock", _mock_project_lock(False)):
            result, attempt = worker_module._process_graph_job(job)

        self.assertEqual(result, "retry_locked")
        self.assertEqual(attempt, 1)

    def test_process_graph_job_drops_stale_when_project_version_advanced(self) -> None:
        action_id = self._create_action(project_id=7, status="applied")
        self._set_project_version(project_id=7, version=5)
        job = {
            "action_id": action_id,
            "project_id": 7,
            "expected_version": 2,
            "attempt": 1,
            "idempotency_key": "graph-idem-1",
        }

        with patch("app.worker.create_action_audit_log") as mock_audit, patch(
            "app.worker.process_graph_sync_for_action"
        ) as mock_sync:
            result, attempt = worker_module._process_graph_job(job)

        self.assertEqual(result, "drop_stale")
        self.assertEqual(attempt, 1)
        mock_sync.assert_not_called()
        self.assertEqual(mock_audit.call_count, 1)
        payload = mock_audit.call_args.kwargs.get("event_payload", {})
        self.assertEqual(payload.get("reason"), "project_version_advanced")
        self.assertEqual(int(payload.get("job_expected_version", 0)), 2)
        self.assertEqual(int(payload.get("current_project_version", 0)), 5)

    def test_process_graph_job_drops_undone_action(self) -> None:
        action_id = self._create_action(project_id=8, status="undone")
        job = {
            "action_id": action_id,
            "project_id": 8,
            "attempt": 2,
            "mutation_id": "m-undone",
            "expected_version": 3,
        }

        with patch("app.worker.create_action_audit_log") as mock_audit, patch(
            "app.worker.process_graph_sync_for_action"
        ) as mock_sync:
            result, attempt = worker_module._process_graph_job(job)

        self.assertEqual(result, "drop_undone")
        self.assertEqual(attempt, 2)
        mock_sync.assert_not_called()
        self.assertEqual(mock_audit.call_count, 1)
        payload = mock_audit.call_args.kwargs.get("event_payload", {})
        self.assertEqual(payload.get("reason"), "action_already_undone")

    def test_process_graph_job_retries_when_action_not_applied(self) -> None:
        action_id = self._create_action(project_id=9, status="proposed")
        with patch("app.worker.process_graph_sync_for_action") as mock_sync:
            result, attempt = worker_module._process_graph_job(
                {"action_id": action_id, "project_id": 9, "attempt": 4}
            )

        self.assertEqual(result, "retry_not_applied")
        self.assertEqual(attempt, 4)
        mock_sync.assert_not_called()

    def test_process_graph_job_drops_done_when_already_synced(self) -> None:
        action_id = self._create_action(
            project_id=10,
            status="applied",
            apply_result={"graph_sync": {"status": "synced", "mutation_id": "m-1"}},
        )
        job = {
            "action_id": action_id,
            "project_id": 10,
            "attempt": 1,
            "mutation_id": "m-1",
            "idempotency_key": "idem-done",
        }

        with patch("app.worker.create_action_audit_log") as mock_audit, patch(
            "app.worker.process_graph_sync_for_action"
        ) as mock_sync:
            result, attempt = worker_module._process_graph_job(job)

        self.assertEqual(result, "drop_done")
        self.assertEqual(attempt, 1)
        mock_sync.assert_not_called()
        payload = mock_audit.call_args.kwargs.get("event_payload", {})
        self.assertEqual(payload.get("reason"), "already_synced")
        self.assertEqual(payload.get("mutation_id"), "m-1")

    def test_requeue_index_lifecycle_job_max_retries_marks_failed_and_dead_letters(self) -> None:
        settings.index_lifecycle_max_retries = 2
        action_id = self._create_action(project_id=1, status="applied")
        job = {
            "project_id": 1,
            "action_id": action_id,
            "operator_id": "worker",
            "reason": "unit_retry",
            "idempotency_key": "idx-retry-max",
        }

        with patch("app.worker.fail_index_lifecycle_job") as mock_fail, patch(
            "app.worker.push_index_lifecycle_dead_letter", return_value=True
        ) as mock_push, patch("app.worker.retry_index_lifecycle_job") as mock_retry, patch(
            "app.worker.create_action_audit_log"
        ) as mock_audit:
            worker_module._requeue_index_lifecycle_job(job, attempt=2, error="boom")

        mock_retry.assert_not_called()
        mock_fail.assert_called_once_with(job, error="boom")
        mock_push.assert_called_once_with(job, "boom")
        payload = mock_audit.call_args.kwargs.get("event_payload", {})
        self.assertEqual(payload.get("reason"), "max_retries_exceeded")
        self.assertTrue(bool(payload.get("dead_letter_pushed")))
        self.assertEqual(payload.get("error"), "boom")
        self.assertEqual(payload.get("err"), "boom")

    def test_requeue_index_lifecycle_job_reschedules_when_retry_succeeds(self) -> None:
        settings.index_lifecycle_max_retries = 3
        settings.index_lifecycle_retry_delay_seconds = 4
        job = {"project_id": 2, "action_id": 0, "operator_id": "worker", "reason": "unit"}

        with patch("app.worker.retry_index_lifecycle_job", return_value=True) as mock_retry, patch(
            "app.worker.fail_index_lifecycle_job"
        ) as mock_fail, patch("app.worker.push_index_lifecycle_dead_letter") as mock_push:
            worker_module._requeue_index_lifecycle_job(job, attempt=1, error="transient")

        mock_retry.assert_called_once_with(
            job,
            next_attempt=2,
            delay_seconds=4,
            error="transient",
        )
        mock_fail.assert_not_called()
        mock_push.assert_not_called()

    def test_requeue_index_lifecycle_job_marks_failed_when_reschedule_fails(self) -> None:
        settings.index_lifecycle_max_retries = 5
        action_id = self._create_action(project_id=3, status="applied")
        job = {
            "project_id": 3,
            "action_id": action_id,
            "operator_id": "worker",
            "reason": "unit_requeue_failed",
            "idempotency_key": "idx-requeue-failed",
        }

        with patch("app.worker.retry_index_lifecycle_job", return_value=False), patch(
            "app.worker.fail_index_lifecycle_job"
        ) as mock_fail, patch(
            "app.worker.push_index_lifecycle_dead_letter", return_value=False
        ) as mock_push, patch("app.worker.create_action_audit_log") as mock_audit:
            worker_module._requeue_index_lifecycle_job(job, attempt=1, error="queue_busy")

        mock_fail.assert_called_once_with(job, error="queue_busy")
        mock_push.assert_called_once_with(job, "queue_busy")
        payload = mock_audit.call_args.kwargs.get("event_payload", {})
        self.assertEqual(payload.get("reason"), "requeue_failed")
        self.assertFalse(bool(payload.get("dead_letter_pushed")))
        self.assertEqual(payload.get("error"), "queue_busy")
        self.assertEqual(payload.get("err"), "queue_busy")

    def test_requeue_graph_job_max_retries_marks_failed_and_writes_error_alias(self) -> None:
        action_id = self._create_action(project_id=4, status="applied")
        job = {
            "project_id": 4,
            "action_id": action_id,
        }

        with patch.object(settings, "graph_sync_max_retries", 2), patch(
            "app.worker.fail_graph_sync_job"
        ) as mock_fail, patch("app.worker.retry_graph_sync_job") as mock_retry, patch(
            "app.worker.create_action_audit_log"
        ) as mock_audit:
            worker_module._requeue_graph_job(job, attempt=2, error="graph_boom")

        mock_retry.assert_not_called()
        mock_fail.assert_called_once_with(job, error="graph_boom")
        payload = mock_audit.call_args.kwargs.get("event_payload", {})
        self.assertEqual(payload.get("reason"), "max_retries_exceeded")
        self.assertEqual(payload.get("error"), "graph_boom")
        self.assertEqual(payload.get("err"), "graph_boom")

    def test_requeue_graph_job_marks_failed_when_reschedule_fails(self) -> None:
        action_id = self._create_action(project_id=5, status="applied")
        job = {
            "project_id": 5,
            "action_id": action_id,
        }

        with patch.object(settings, "graph_sync_max_retries", 3), patch.object(
            settings, "graph_sync_retry_delay_seconds", 4
        ), patch("app.worker.retry_graph_sync_job", return_value=False) as mock_retry, patch(
            "app.worker.fail_graph_sync_job"
        ) as mock_fail, patch("app.worker.create_action_audit_log") as mock_audit:
            worker_module._requeue_graph_job(job, attempt=1, error="retry_blocked")

        mock_retry.assert_called_once_with(
            job,
            next_attempt=2,
            delay_seconds=4,
            error="retry_blocked",
        )
        mock_fail.assert_called_once_with(job, error="retry_blocked")
        payload = mock_audit.call_args.kwargs.get("event_payload", {})
        self.assertEqual(payload.get("reason"), "requeue_failed")
        self.assertEqual(payload.get("error"), "retry_blocked")
        self.assertEqual(payload.get("err"), "retry_blocked")

    def test_requeue_entity_merge_job_marks_failed_when_reschedule_fails(self) -> None:
        job = {"project_id": 7}
        with patch.object(settings, "entity_merge_scan_max_retries", 3), patch.object(
            settings, "entity_merge_scan_retry_delay_seconds", 6
        ), patch("app.worker.retry_entity_merge_scan_job", return_value=False) as mock_retry, patch(
            "app.worker.fail_entity_merge_scan_job"
        ) as mock_fail:
            worker_module._requeue_entity_merge_scan_job(job, attempt=1, error="merge_queue_busy")

        mock_retry.assert_called_once_with(
            job,
            next_attempt=2,
            delay_seconds=6,
            error="merge_queue_busy",
        )
        mock_fail.assert_called_once_with(job, error="merge_queue_busy")

    def test_requeue_consistency_job_marks_failed_when_reschedule_fails(self) -> None:
        job = {"project_id": 8}
        with patch.object(settings, "consistency_audit_max_retries", 4), patch.object(
            settings, "consistency_audit_retry_delay_seconds", 5
        ), patch("app.worker.retry_consistency_audit_job", return_value=False) as mock_retry, patch(
            "app.worker.fail_consistency_audit_job"
        ) as mock_fail:
            worker_module._requeue_consistency_audit_job(job, attempt=1, error="audit_retry_blocked")

        mock_retry.assert_called_once_with(
            job,
            next_attempt=2,
            delay_seconds=5,
            error="audit_retry_blocked",
        )
        mock_fail.assert_called_once_with(job, error="audit_retry_blocked")

    def test_process_consistency_audit_job_returns_drop_invalid_for_bad_project_id(self) -> None:
        result, attempt, error = worker_module._process_consistency_audit_job(
            {"project_id": 0, "attempt": 3}
        )
        self.assertEqual(result, "drop_invalid")
        self.assertEqual(attempt, 3)
        self.assertEqual(error, "project_id_invalid")

    def test_process_consistency_audit_job_returns_drop_disabled_when_feature_off(self) -> None:
        settings.consistency_audit_enabled = False
        result, attempt, error = worker_module._process_consistency_audit_job(
            {"project_id": 1, "attempt": 1}
        )
        self.assertEqual(result, "drop_disabled")
        self.assertEqual(attempt, 1)
        self.assertEqual(error, "")

    def test_process_consistency_audit_job_retries_when_project_lock_busy(self) -> None:
        settings.consistency_audit_enabled = True
        with patch("app.worker._project_advisory_lock", _mock_project_lock(False)):
            result, attempt, error = worker_module._process_consistency_audit_job(
                {"project_id": 5, "attempt": 2}
            )

        self.assertEqual(result, "retry_locked")
        self.assertEqual(attempt, 2)
        self.assertEqual(error, "project_lock_busy")

    def test_process_consistency_audit_job_runs_service_and_returns_status(self) -> None:
        settings.consistency_audit_enabled = True
        job = {
            "project_id": 6,
            "attempt": 1,
            "operator_id": "ops-user",
            "reason": "nightly",
            "trigger_source": "worker_scheduler",
            "force": True,
            "max_chapters": "7",
        }

        with patch("app.worker.run_consistency_audit", return_value={"status": "warning"}) as mock_run:
            result, attempt, error = worker_module._process_consistency_audit_job(job)

        self.assertEqual(result, "ok")
        self.assertEqual(attempt, 1)
        self.assertEqual(error, "warning")
        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs.get("project_id"), 6)
        self.assertEqual(kwargs.get("operator_id"), "ops-user")
        self.assertEqual(kwargs.get("reason"), "nightly")
        self.assertEqual(kwargs.get("trigger_source"), "worker_scheduler")
        self.assertTrue(bool(kwargs.get("force")))
        self.assertEqual(kwargs.get("max_chapters"), 7)

    def test_enqueue_due_consistency_audit_jobs_only_queues_due_projects(self) -> None:
        settings.consistency_audit_enabled = True
        settings.consistency_audit_auto_enqueue = True
        settings.consistency_audit_scheduler_project_scan_limit = 20
        settings.consistency_audit_idle_minutes = 30
        settings.consistency_audit_daily_hour_utc = 8
        settings.consistency_audit_max_chapters = 4

        fixed_now = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
        updates = {
            1: fixed_now - timedelta(hours=2),
            2: fixed_now - timedelta(minutes=5),
            3: fixed_now - timedelta(minutes=5),
            4: fixed_now - timedelta(hours=3),
            5: None,
        }
        latest_reports = {
            1: None,
            2: None,
            3: None,
            4: fixed_now - timedelta(hours=2),
            5: None,
        }

        class _FixedDateTime:
            @staticmethod
            def now(tz=None):
                return fixed_now if tz is not None else fixed_now.replace(tzinfo=None)

        def _latest_update(_db, project_id: int):
            return updates.get(project_id)

        def _latest_report(_db, project_id: int):
            return latest_reports.get(project_id)

        def _has_report_on_date(_db, project_id: int, _target_date):
            return project_id == 3

        with patch("app.worker.datetime", _FixedDateTime), patch(
            "app.worker.list_project_ids_with_chapters", return_value=[1, 2, 3, 4, 5]
        ), patch(
            "app.worker.latest_project_chapter_update_at", side_effect=_latest_update
        ), patch(
            "app.worker.latest_consistency_audit_timestamp", side_effect=_latest_report
        ), patch(
            "app.worker.has_consistency_audit_report_on_date", side_effect=_has_report_on_date
        ), patch(
            "app.worker.enqueue_consistency_audit_job", side_effect=lambda project_id, **_kwargs: project_id == 1
        ) as mock_enqueue:
            queued_count = worker_module._enqueue_due_consistency_audit_jobs()

        self.assertEqual(queued_count, 1)
        self.assertEqual(mock_enqueue.call_count, 2)

        first = mock_enqueue.call_args_list[0]
        second = mock_enqueue.call_args_list[1]
        self.assertEqual(first.args[0], 1)
        self.assertEqual(first.kwargs.get("reason"), "nightly_idle")
        self.assertEqual(first.kwargs.get("trigger_source"), "worker_scheduler")
        self.assertEqual(first.kwargs.get("max_chapters"), 4)
        self.assertIn("nightly_idle", str(first.kwargs.get("idempotency_key", "")))

        self.assertEqual(second.args[0], 2)
        self.assertEqual(second.kwargs.get("reason"), "nightly_daily")
        self.assertIn("nightly_daily", str(second.kwargs.get("idempotency_key", "")))


class _BreakLoop(Exception):
    pass


class WorkerRunLoopTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "job_cleanup_enabled": settings.job_cleanup_enabled,
            "job_cleanup_interval_seconds": settings.job_cleanup_interval_seconds,
            "job_cleanup_retention_seconds": settings.job_cleanup_retention_seconds,
            "job_cleanup_batch_size": settings.job_cleanup_batch_size,
            "graph_sync_worker_block_seconds": settings.graph_sync_worker_block_seconds,
            "entity_merge_scan_worker_block_seconds": settings.entity_merge_scan_worker_block_seconds,
            "consistency_audit_worker_block_seconds": settings.consistency_audit_worker_block_seconds,
            "consistency_audit_enabled": settings.consistency_audit_enabled,
            "consistency_audit_auto_enqueue": settings.consistency_audit_auto_enqueue,
            "consistency_audit_scheduler_interval_seconds": settings.consistency_audit_scheduler_interval_seconds,
        }
        settings.job_cleanup_enabled = False
        settings.consistency_audit_enabled = False
        settings.consistency_audit_auto_enqueue = False

    def tearDown(self) -> None:
        settings.job_cleanup_enabled = self._snapshot["job_cleanup_enabled"]
        settings.job_cleanup_interval_seconds = self._snapshot["job_cleanup_interval_seconds"]
        settings.job_cleanup_retention_seconds = self._snapshot["job_cleanup_retention_seconds"]
        settings.job_cleanup_batch_size = self._snapshot["job_cleanup_batch_size"]
        settings.graph_sync_worker_block_seconds = self._snapshot["graph_sync_worker_block_seconds"]
        settings.entity_merge_scan_worker_block_seconds = self._snapshot["entity_merge_scan_worker_block_seconds"]
        settings.consistency_audit_worker_block_seconds = self._snapshot["consistency_audit_worker_block_seconds"]
        settings.consistency_audit_enabled = self._snapshot["consistency_audit_enabled"]
        settings.consistency_audit_auto_enqueue = self._snapshot["consistency_audit_auto_enqueue"]
        settings.consistency_audit_scheduler_interval_seconds = self._snapshot[
            "consistency_audit_scheduler_interval_seconds"
        ]

    def test_run_requeues_graph_retry_result_and_uses_fast_followup_poll(self) -> None:
        graph_job = {"attempt": 3, "project_id": 1}

        with patch("app.worker.init_db"), patch(
            "app.worker.dequeue_graph_sync_job", side_effect=[graph_job, _BreakLoop()]
        ), patch("app.worker._process_graph_job", return_value=("retry_not_applied", 3)), patch(
            "app.worker._requeue_graph_job"
        ) as mock_requeue_graph, patch(
            "app.worker.complete_graph_sync_job"
        ) as mock_complete_graph, patch(
            "app.worker.dequeue_index_lifecycle_job", return_value=None
        ) as mock_dequeue_lifecycle, patch(
            "app.worker.dequeue_entity_merge_scan_job", return_value=None
        ) as mock_dequeue_merge, patch(
            "app.worker.dequeue_consistency_audit_job", return_value=None
        ) as mock_dequeue_audit:
            with self.assertRaises(_BreakLoop):
                worker_module.run()

        mock_requeue_graph.assert_called_once_with(graph_job, 3, "retry_not_applied")
        mock_complete_graph.assert_not_called()
        mock_dequeue_lifecycle.assert_called_once_with(1, worker_id="index-lifecycle-worker")
        mock_dequeue_merge.assert_called_once_with(1, worker_id="entity-merge-scan-worker")
        mock_dequeue_audit.assert_called_once_with(1, worker_id="consistency-audit-worker")

    def test_run_completes_graph_job_when_process_returns_done_state(self) -> None:
        graph_job = {"attempt": 1, "project_id": 2}

        with patch("app.worker.init_db"), patch(
            "app.worker.dequeue_graph_sync_job", side_effect=[graph_job, _BreakLoop()]
        ), patch("app.worker._process_graph_job", return_value=("drop_done", 1)), patch(
            "app.worker._requeue_graph_job"
        ) as mock_requeue_graph, patch(
            "app.worker.complete_graph_sync_job"
        ) as mock_complete_graph, patch(
            "app.worker.dequeue_index_lifecycle_job", return_value=None
        ), patch(
            "app.worker.dequeue_entity_merge_scan_job", return_value=None
        ), patch(
            "app.worker.dequeue_consistency_audit_job", return_value=None
        ):
            with self.assertRaises(_BreakLoop):
                worker_module.run()

        mock_requeue_graph.assert_not_called()
        mock_complete_graph.assert_called_once_with(graph_job, final_status="done", error="")

    def test_run_requeues_graph_job_on_processing_exception(self) -> None:
        graph_job = {"attempt": "7", "project_id": 3}

        with patch("app.worker.init_db"), patch(
            "app.worker.dequeue_graph_sync_job", side_effect=[graph_job, _BreakLoop()]
        ), patch("app.worker._process_graph_job", side_effect=RuntimeError("graph exploded")), patch(
            "app.worker._requeue_graph_job"
        ) as mock_requeue_graph, patch(
            "app.worker.complete_graph_sync_job"
        ) as mock_complete_graph, patch(
            "app.worker.dequeue_index_lifecycle_job", return_value=None
        ), patch(
            "app.worker.dequeue_entity_merge_scan_job", return_value=None
        ), patch(
            "app.worker.dequeue_consistency_audit_job", return_value=None
        ):
            with self.assertRaises(_BreakLoop):
                worker_module.run()

        mock_complete_graph.assert_not_called()
        mock_requeue_graph.assert_called_once_with(graph_job, 7, "graph exploded")

    def test_run_requeues_lifecycle_job_when_locked(self) -> None:
        settings.graph_sync_worker_block_seconds = 9
        lifecycle_job = {"attempt": 1, "project_id": 5}

        with patch("app.worker.init_db"), patch(
            "app.worker.dequeue_graph_sync_job", side_effect=[None, _BreakLoop()]
        ), patch(
            "app.worker.dequeue_index_lifecycle_job", return_value=lifecycle_job
        ) as mock_dequeue_lifecycle, patch(
            "app.worker._process_index_lifecycle_job", return_value=("retry_locked", 1, "project_lock_busy")
        ), patch(
            "app.worker._requeue_index_lifecycle_job"
        ) as mock_requeue_lifecycle, patch(
            "app.worker.complete_index_lifecycle_job"
        ) as mock_complete_lifecycle, patch(
            "app.worker.dequeue_entity_merge_scan_job", return_value=None
        ), patch(
            "app.worker.dequeue_consistency_audit_job", return_value=None
        ):
            with self.assertRaises(_BreakLoop):
                worker_module.run()

        mock_dequeue_lifecycle.assert_called_once_with(9, worker_id="index-lifecycle-worker")
        mock_requeue_lifecycle.assert_called_once_with(lifecycle_job, 1, "project_lock_busy")
        mock_complete_lifecycle.assert_not_called()

    def test_run_completes_entity_merge_job_on_ok(self) -> None:
        settings.graph_sync_worker_block_seconds = 6
        settings.entity_merge_scan_worker_block_seconds = 13
        merge_job = {"attempt": 2, "project_id": 6}

        with patch("app.worker.init_db"), patch(
            "app.worker.dequeue_graph_sync_job", side_effect=[None, _BreakLoop()]
        ), patch("app.worker.dequeue_index_lifecycle_job", return_value=None), patch(
            "app.worker.dequeue_entity_merge_scan_job", return_value=merge_job
        ) as mock_dequeue_merge, patch(
            "app.worker._process_entity_merge_scan_job", return_value=("ok", 2, "done")
        ), patch(
            "app.worker._requeue_entity_merge_scan_job"
        ) as mock_requeue_merge, patch(
            "app.worker.complete_entity_merge_scan_job"
        ) as mock_complete_merge, patch(
            "app.worker.dequeue_consistency_audit_job", return_value=None
        ):
            with self.assertRaises(_BreakLoop):
                worker_module.run()

        mock_dequeue_merge.assert_called_once_with(13, worker_id="entity-merge-scan-worker")
        mock_requeue_merge.assert_not_called()
        mock_complete_merge.assert_called_once_with(merge_job, final_status="done", error="")

    def test_run_requeues_consistency_job_on_retry_error(self) -> None:
        settings.consistency_audit_worker_block_seconds = 7
        consistency_job = {"attempt": 4, "project_id": 7}

        with patch("app.worker.init_db"), patch(
            "app.worker.dequeue_graph_sync_job", side_effect=[None, _BreakLoop()]
        ), patch("app.worker.dequeue_index_lifecycle_job", return_value=None), patch(
            "app.worker.dequeue_entity_merge_scan_job", return_value=None
        ), patch(
            "app.worker.dequeue_consistency_audit_job", return_value=consistency_job
        ) as mock_dequeue_audit, patch(
            "app.worker._process_consistency_audit_job", return_value=("retry_error", 4, "audit_failed")
        ), patch(
            "app.worker._requeue_consistency_audit_job"
        ) as mock_requeue_audit, patch(
            "app.worker.complete_consistency_audit_job"
        ) as mock_complete_audit:
            with self.assertRaises(_BreakLoop):
                worker_module.run()

        mock_dequeue_audit.assert_called_once_with(7, worker_id="consistency-audit-worker")
        mock_requeue_audit.assert_called_once_with(consistency_job, 4, "audit_failed")
        mock_complete_audit.assert_not_called()

    def test_run_triggers_cleanup_and_scheduler_when_due(self) -> None:
        settings.job_cleanup_enabled = True
        settings.job_cleanup_interval_seconds = 5
        settings.job_cleanup_retention_seconds = 123
        settings.job_cleanup_batch_size = 11
        settings.consistency_audit_enabled = True
        settings.consistency_audit_auto_enqueue = True
        settings.consistency_audit_scheduler_interval_seconds = 10

        monotonic_ticks = [100.0, 100.0, 106.0, 106.0, 111.0, 111.0]
        perf_ticks = [1.0, 1.01]

        with patch("app.worker.init_db"), patch(
            "app.worker.time.monotonic", side_effect=monotonic_ticks
        ), patch(
            "app.worker.time.perf_counter", side_effect=perf_ticks
        ), patch(
            "app.worker.cleanup_async_jobs", return_value=4
        ) as mock_cleanup, patch(
            "app.worker._enqueue_due_consistency_audit_jobs", return_value=2
        ) as mock_enqueue_due, patch(
            "app.worker.dequeue_graph_sync_job", side_effect=_BreakLoop()
        ):
            with self.assertRaises(_BreakLoop):
                worker_module.run()

        mock_cleanup.assert_called_once_with(
            statuses=("done", "failed"),
            older_than_seconds=123,
            limit=11,
        )
        mock_enqueue_due.assert_called_once()


if __name__ == "__main__":
    unittest.main()
