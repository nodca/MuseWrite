from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import update
from sqlmodel import Session, select

from app.models.simulation import (
    SimulationDecisionTrace,
    SimulationEvent,
    SimulationSessionState,
    SimulationTurn,
)
from app.services.simulation.types import (
    PersistedTurnRecord,
    RefereeDecision,
    SessionStateRecord,
    TurnCandidate,
    TurnOutcome,
)


class RevisionConflictError(RuntimeError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_state_record(state: SimulationSessionState) -> SessionStateRecord:
    return SessionStateRecord(
        session_id=state.session_id,
        revision=state.revision,
        runtime_status=state.runtime_status,
        last_speaker_id=state.last_speaker_id,
        active_pressures=list(state.active_pressures or []),
        mind_states=dict(state.mind_states or {}),
    )


def load_or_create_session_state(
    db: Session,
    session_id: int,
) -> SessionStateRecord:
    state = db.exec(
        select(SimulationSessionState).where(
            SimulationSessionState.session_id == session_id
        )
    ).first()
    if state is None:
        state = SimulationSessionState(session_id=session_id)
        db.add(state)
        db.flush()
    return _to_state_record(state)


def save_session_state(
    db: Session,
    state: SessionStateRecord,
    *,
    expected_revision: int,
) -> SessionStateRecord:
    if state.revision != expected_revision:
        raise RevisionConflictError(
            f"state revision {state.revision} does not match expected {expected_revision}"
        )
    entity = db.exec(
        select(SimulationSessionState).where(
            SimulationSessionState.session_id == state.session_id
        )
    ).first()
    if entity is None:
        if expected_revision != 0:
            raise RevisionConflictError("session state does not exist")
        entity = SimulationSessionState(session_id=state.session_id)
        db.add(entity)
        db.flush()
    if entity.revision != expected_revision:
        raise RevisionConflictError(
            f"expected revision {expected_revision}, got {entity.revision}"
        )
    entity.runtime_status = state.runtime_status
    entity.last_speaker_id = state.last_speaker_id
    entity.active_pressures = list(state.active_pressures)
    entity.mind_states = dict(state.mind_states)
    entity.revision = expected_revision + 1
    entity.updated_at = _utc_now()
    db.add(entity)
    db.flush()
    return _to_state_record(entity)


def append_event(
    db: Session,
    session_id: int,
    *,
    event_type: str,
    source: str = "external",
    payload: dict | None = None,
    priority: int = 0,
) -> SimulationEvent:
    event = SimulationEvent(
        session_id=session_id,
        event_type=event_type,
        source=source,
        payload=dict(payload or {}),
        priority=priority,
    )
    db.add(event)
    db.flush()
    return event


def consume_pending_events(
    db: Session,
    session_id: int,
    *,
    limit: int | None = None,
) -> list[SimulationEvent]:
    stmt = (
        select(SimulationEvent)
        .where(
            SimulationEvent.session_id == session_id,
            SimulationEvent.consumed_at.is_(None),
        )
        .order_by(SimulationEvent.priority.desc(), SimulationEvent.id)
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    event_ids = list(db.exec(stmt.with_only_columns(SimulationEvent.id)).all())
    if not event_ids:
        return []
    now = _utc_now()
    claim_result = db.exec(
        update(SimulationEvent)
        .where(
            SimulationEvent.id.in_(event_ids),
            SimulationEvent.consumed_at.is_(None),
        )
        .values(consumed_at=now)
    )
    if claim_result.rowcount is not None and claim_result.rowcount != len(event_ids):
        raise RevisionConflictError("pending event claim conflict")
    db.flush()
    events = list(
        db.exec(
            select(SimulationEvent)
            .where(SimulationEvent.id.in_(event_ids))
            .order_by(SimulationEvent.priority.desc(), SimulationEvent.id)
        ).all()
    )
    return events


def record_turn_outcome(
    db: Session,
    *,
    session_id: int,
    turn_index: int,
    outcome: TurnOutcome,
    decision: RefereeDecision,
    candidates: list[TurnCandidate],
    turn: SimulationTurn | None = None,
    consumed_event_ids: list[int] | None = None,
) -> PersistedTurnRecord:
    persisted_turn: SimulationTurn | None = None
    if turn is not None:
        persisted_turn = turn
        db.add(persisted_turn)
        db.flush()
        outcome = outcome.model_copy(
            update={
                "turn_id": persisted_turn.id,
                "turn_index": persisted_turn.turn_index,
            }
        )

    trace = SimulationDecisionTrace(
        session_id=session_id,
        turn_index=turn_index,
        candidate_snapshot=[item.model_dump(mode="json") for item in candidates],
        decision_payload=decision.model_dump(mode="json"),
        failure_code=outcome.error_code,
    )
    db.add(trace)
    db.flush()
    return PersistedTurnRecord(
        outcome=outcome,
        turn_id=persisted_turn.id if persisted_turn is not None else None,
        trace_id=trace.id,
        consumed_event_ids=list(consumed_event_ids or []),
    )
