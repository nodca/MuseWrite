from datetime import datetime, timezone

from sqlmodel import Session, select

from app.models.chat import PendingGraphMutation


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_pending_graph_mutation(
    db: Session,
    *,
    mutation_id: str,
) -> PendingGraphMutation | None:
    normalized = str(mutation_id or "").strip()
    if not normalized:
        return None
    stmt = select(PendingGraphMutation).where(PendingGraphMutation.mutation_id == normalized)
    return db.exec(stmt).first()


def upsert_pending_graph_mutation(
    db: Session,
    *,
    project_id: int,
    action_id: int,
    mutation_id: str,
    expected_version: int,
    status: str,
) -> PendingGraphMutation:
    normalized_mutation_id = str(mutation_id or "").strip()
    if not normalized_mutation_id:
        raise ValueError("mutation_id is required")

    existing = get_pending_graph_mutation(db, mutation_id=normalized_mutation_id)
    if existing is not None:
        existing.project_id = int(project_id)
        existing.action_id = int(action_id)
        existing.expected_version = int(expected_version)
        existing.status = str(status or "pending_queue").strip() or "pending_queue"
        existing.updated_at = _utc_now()
        db.add(existing)
        db.flush()
        return existing

    row = PendingGraphMutation(
        project_id=int(project_id),
        action_id=int(action_id),
        mutation_id=normalized_mutation_id,
        expected_version=int(expected_version),
        status=str(status or "pending_queue").strip() or "pending_queue",
        created_at=_utc_now(),
        updated_at=_utc_now(),
    )
    db.add(row)
    db.flush()
    return row


def mark_pending_graph_mutation_status(
    db: Session,
    *,
    mutation_id: str,
    status: str,
    cancel_reason: str = "",
    canceled_by_mutation_id: str = "",
) -> PendingGraphMutation | None:
    row = get_pending_graph_mutation(db, mutation_id=mutation_id)
    if row is None:
        return None
    row.status = str(status or row.status or "").strip() or "pending_queue"
    if cancel_reason:
        row.cancel_reason = str(cancel_reason).strip()[:255]
    if canceled_by_mutation_id:
        row.canceled_by_mutation_id = str(canceled_by_mutation_id).strip()[:128]
    row.updated_at = _utc_now()
    db.add(row)
    db.flush()
    return row


def mark_pending_graph_mutation_canceled(
    db: Session,
    *,
    mutation_id: str,
    cancel_reason: str,
    canceled_by_mutation_id: str,
) -> PendingGraphMutation | None:
    return mark_pending_graph_mutation_status(
        db,
        mutation_id=mutation_id,
        status="canceled",
        cancel_reason=cancel_reason,
        canceled_by_mutation_id=canceled_by_mutation_id,
    )
