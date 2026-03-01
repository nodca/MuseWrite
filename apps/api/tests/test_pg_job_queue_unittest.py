import unittest
from datetime import timedelta

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.core.config import settings
from app.models.chat import AsyncJob
from app.services import pg_job_queue as queue_module


class PgJobQueueTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

        self._snapshot = {
            "engine": queue_module.engine,
            "job_queue_poll_interval_seconds": settings.job_queue_poll_interval_seconds,
            "job_processing_timeout_seconds": settings.job_processing_timeout_seconds,
        }
        queue_module.engine = self.engine
        settings.job_queue_poll_interval_seconds = 0.05
        settings.job_processing_timeout_seconds = 15

    def tearDown(self) -> None:
        queue_module.engine = self._snapshot["engine"]
        settings.job_queue_poll_interval_seconds = self._snapshot["job_queue_poll_interval_seconds"]
        settings.job_processing_timeout_seconds = self._snapshot["job_processing_timeout_seconds"]
        SQLModel.metadata.drop_all(self.engine)
        self.engine.dispose()

    @staticmethod
    def _make_payload(value: str) -> dict:
        return {"value": value, "_job_private": "should-be-stripped"}

    def test_enqueue_and_dequeue_assigns_processing_lock_token(self) -> None:
        queued = queue_module.enqueue_async_job(
            queue_name="unit_jobs",
            payload=self._make_payload("a"),
            project_id=1,
            action_id=11,
            operator_id="tester",
            reason="unit",
            mutation_id="m1",
            expected_version=3,
            idempotency_key="idem-1",
            attempt=0,
            max_retries=2,
        )
        self.assertTrue(queued)

        job = queue_module.dequeue_async_job("unit_jobs", timeout_seconds=1, worker_id="worker-a")
        self.assertIsNotNone(job)
        if job is None:
            return
        self.assertEqual(job.get("project_id"), 1)
        self.assertEqual(job.get("action_id"), 11)
        self.assertEqual(job.get("operator_id"), "tester")
        self.assertEqual(job.get("idempotency_key"), "idem-1")
        self.assertEqual(job.get("value"), "a")
        self.assertNotIn("_job_private", job)
        self.assertTrue(str(job.get("_job_lock_token") or "").strip())

        with Session(self.engine) as db:
            row = db.get(AsyncJob, int(job.get("_job_id", 0)))
            self.assertIsNotNone(row)
            if row is not None:
                self.assertEqual(row.status, "processing")
                self.assertEqual(row.locked_by, "worker-a")

    def test_complete_job_rejects_lock_token_mismatch(self) -> None:
        queue_module.enqueue_async_job(
            queue_name="unit_jobs",
            payload=self._make_payload("complete"),
            project_id=1,
            action_id=12,
            operator_id="tester",
            reason="unit",
            mutation_id="m2",
            expected_version=4,
            idempotency_key="idem-2",
        )
        job = queue_module.dequeue_async_job("unit_jobs", timeout_seconds=1, worker_id="worker-a")
        self.assertIsNotNone(job)
        if job is None:
            return

        wrong = {**job, "_job_lock_token": "wrong-token"}
        self.assertFalse(queue_module.complete_async_job(wrong, final_status="done"))
        self.assertTrue(queue_module.complete_async_job(job, final_status="done"))

        with Session(self.engine) as db:
            row = db.get(AsyncJob, int(job.get("_job_id", 0)))
            self.assertIsNotNone(row)
            if row is not None:
                self.assertEqual(row.status, "done")
                self.assertEqual(row.lock_token, "")

    def test_reschedule_job_updates_attempt_and_available_at(self) -> None:
        queue_module.enqueue_async_job(
            queue_name="unit_jobs",
            payload=self._make_payload("retry"),
            project_id=1,
            action_id=13,
            operator_id="tester",
            reason="unit",
            mutation_id="m3",
            expected_version=5,
            idempotency_key="idem-3",
        )
        job = queue_module.dequeue_async_job("unit_jobs", timeout_seconds=1, worker_id="worker-a")
        self.assertIsNotNone(job)
        if job is None:
            return

        self.assertTrue(queue_module.reschedule_async_job(job, next_attempt=2, delay_seconds=2.0, error="retry-later"))
        with Session(self.engine) as db:
            row = db.get(AsyncJob, int(job.get("_job_id", 0)))
            self.assertIsNotNone(row)
            if row is not None:
                self.assertEqual(row.status, "queued")
                self.assertEqual(int(row.attempt), 2)
                self.assertIn("retry-later", row.last_error)
                self.assertEqual(row.lock_token, "")

    def test_recover_stale_processing_jobs_requeues_timeout_rows(self) -> None:
        stale_now = queue_module._utc_now()
        with Session(self.engine) as db:
            row = AsyncJob(
                queue_name="unit_jobs",
                project_id=1,
                action_id=14,
                operator_id="tester",
                reason="stale",
                mutation_id="m4",
                expected_version=1,
                idempotency_key="idem-stale",
                lifecycle_slot="default",
                payload={"value": "stale"},
                attempt=0,
                max_retries=3,
                status="processing",
                available_at=stale_now,
                locked_at=stale_now - timedelta(seconds=120),
                lock_token="lock-old",
                locked_by="worker-old",
                created_at=stale_now - timedelta(seconds=200),
                updated_at=stale_now - timedelta(seconds=200),
            )
            db.add(row)
            db.commit()

        job = queue_module.dequeue_async_job("unit_jobs", timeout_seconds=1, worker_id="worker-new")
        self.assertIsNotNone(job)
        if job is None:
            return
        self.assertEqual(job.get("idempotency_key"), "idem-stale")
        with Session(self.engine) as db:
            row = db.get(AsyncJob, int(job.get("_job_id", 0)))
            self.assertIsNotNone(row)
            if row is not None:
                self.assertEqual(row.status, "processing")
                self.assertEqual(row.locked_by, "worker-new")
                self.assertIn("recovered_from_stale_processing", row.last_error)

    def test_pop_queue_payloads_filters_by_project_and_cleanup_done_rows(self) -> None:
        queue_module.enqueue_async_job(
            queue_name="unit_jobs",
            payload=self._make_payload("p1"),
            project_id=1,
            action_id=21,
            operator_id="tester",
            reason="unit",
            mutation_id="m21",
            expected_version=1,
            idempotency_key="idem-p1",
        )
        queue_module.enqueue_async_job(
            queue_name="unit_jobs",
            payload=self._make_payload("p2"),
            project_id=2,
            action_id=22,
            operator_id="tester",
            reason="unit",
            mutation_id="m22",
            expected_version=1,
            idempotency_key="idem-p2",
        )
        popped = queue_module.pop_queue_payloads("unit_jobs", limit=5, project_id=1)
        self.assertEqual(len(popped), 1)
        self.assertEqual(popped[0].get("project_id"), 1)

        with Session(self.engine) as db:
            row = db.exec(select(AsyncJob).where(AsyncJob.idempotency_key == "idem-p2")).first()
            self.assertIsNotNone(row)
            if row is not None:
                row.status = "done"
                row.updated_at = queue_module._utc_now() - timedelta(days=10)
                db.add(row)
                db.commit()

        cleaned = queue_module.cleanup_async_jobs(older_than_seconds=7 * 86400, limit=20)
        self.assertEqual(cleaned, 1)


if __name__ == "__main__":
    unittest.main()
