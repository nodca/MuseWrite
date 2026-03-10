from __future__ import annotations

import logging
from typing import Any

from sqlmodel import Session, select

from app.models.content import StoryCard
from app.models.simulation import SimulationSession, SimulationTurn
from app.services.simulation.actor_engine import ActorGenerationError, generate_turn
from app.services.simulation.bidding import produce_candidates
from app.services.simulation.mind_engine import advance_minds
from app.services.simulation.referee import pick_speaker
from app.services.simulation.repository import (
    RevisionConflictError,
    consume_pending_events,
    load_or_create_session_state,
    record_turn_outcome,
    save_session_state,
)
from app.services.simulation.tom_updater import (
    ToMUpdateError,
    update_second_order_beliefs_from_turns,
)
from app.services.simulation.types import RefereeDecision, TurnCandidate, TurnOutcome, WorldState

logger = logging.getLogger(__name__)


def _load_story_cards(
    db: Session,
    *,
    project_id: int,
    card_ids: list[int],
) -> dict[int, StoryCard]:
    if not card_ids:
        return {}
    stmt = select(StoryCard).where(
        StoryCard.project_id == project_id,
        StoryCard.id.in_(card_ids),
    )
    return {card.id: card for card in db.exec(stmt).all()}


def _turn_index(turns: list[SimulationTurn]) -> int:
    if not turns:
        return 1
    return int(turns[-1].turn_index) + 1


def _recent_turn_dicts(turns: list[SimulationTurn], *, limit: int = 12) -> list[dict[str, Any]]:
    recent = turns[-limit:] if len(turns) > limit else turns
    items: list[dict[str, Any]] = []
    for turn in recent:
        items.append(
            {
                "turn_index": turn.turn_index,
                "actor_id": turn.actor_card_id,
                "actor_name": turn.actor_name,
                "action_type": turn.action_type,
                "content": turn.content,
                "target_card_id": turn.target_card_id,
            }
        )
    return items


def _event_dicts(events: list[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for event in events:
        items.append(
            {
                "id": event.id,
                "event_type": event.event_type,
                "payload": event.payload,
                "priority": event.priority,
            }
        )
    return items


async def run_turn(
    db: Session,
    *,
    sim: SimulationSession,
    runtime_config: Any = None,
    current_chapter: int | None = None,
) -> tuple[TurnOutcome, SimulationTurn | None]:
    del current_chapter  # reserved for future context injection

    turns = list(
        db.exec(
            select(SimulationTurn)
            .where(SimulationTurn.session_id == sim.id)
            .order_by(SimulationTurn.turn_index)
        ).all()
    )
    next_index = _turn_index(turns)

    state = load_or_create_session_state(db, sim.id)
    events = consume_pending_events(db, sim.id, limit=8)
    consumed_event_ids = [int(e.id) for e in events if e.id is not None]

    world_state = WorldState(
        session_id=sim.id,
        scenario=sim.scenario,
        turn_index=next_index,
        last_speaker_id=state.last_speaker_id,
        active_pressures=list(state.active_pressures),
        pending_events=_event_dicts(events),
        recent_turns=_recent_turn_dicts(turns),
    )

    character_ids = [int(cid) for cid in (sim.character_card_ids or []) if int(cid) > 0]
    if not character_ids:
        raise ValueError("simulation has no characters")

    cards = _load_story_cards(db, project_id=sim.project_id, card_ids=character_ids)
    if len(cards) != len(character_ids):
        missing = [cid for cid in character_ids if cid not in cards]
        raise ValueError(f"simulation references missing StoryCard ids: {missing}")

    candidates: list[TurnCandidate] = []
    decision: RefereeDecision = RefereeDecision()
    updated_state = state
    try:
        updated_state = advance_minds(
            world_state,
            state,
            character_ids=character_ids,
        )
        updated_state = await update_second_order_beliefs_from_turns(
            world_state,
            updated_state,
            character_ids=character_ids,
            runtime_config=runtime_config,
        )

        candidates = produce_candidates(
            world_state,
            updated_state,
            character_ids=character_ids,
        )
        decision = pick_speaker(candidates, turn_index=next_index)

        if decision.winner_id is None:
            outcome = TurnOutcome(status="paused", turn_index=next_index)
            record_turn_outcome(
                db,
                session_id=sim.id,
                turn_index=next_index,
                outcome=outcome,
                decision=decision,
                candidates=candidates,
                turn=None,
                consumed_event_ids=consumed_event_ids,
            )
            next_state = updated_state.model_copy(update={"runtime_status": "paused"})
            save_session_state(db, next_state, expected_revision=state.revision)
            db.commit()
            return outcome, None

        actor_id = int(decision.winner_id)
        actor_card = cards.get(actor_id)
        if actor_card is None:
            raise ValueError(f"missing actor StoryCard id={actor_id}")

        turn = await generate_turn(
            db,
            sim=sim,
            actor_card=actor_card,
            turn_index=next_index,
            world_state=world_state,
            session_state=updated_state,
            runtime_config=runtime_config,
        )
        outcome = TurnOutcome(status="spoken", turn_index=next_index)
        record_turn_outcome(
            db,
            session_id=sim.id,
            turn_index=next_index,
            outcome=outcome,
            decision=decision,
            candidates=candidates,
            turn=turn,
            consumed_event_ids=consumed_event_ids,
        )
        next_state = updated_state.model_copy(
            update={
                "runtime_status": "idle",
                "last_speaker_id": actor_id,
            }
        )
        save_session_state(db, next_state, expected_revision=state.revision)
        db.commit()
        return outcome, turn
    except (ActorGenerationError, ToMUpdateError) as exc:
        db.rollback()
        outcome = TurnOutcome(
            status="failed",
            turn_index=next_index,
            error_code=exc.code,
            details=exc.details,
        )
        record_turn_outcome(
            db,
            session_id=sim.id,
            turn_index=next_index,
            outcome=outcome,
            decision=decision,
            candidates=candidates,
            turn=None,
            consumed_event_ids=consumed_event_ids,
        )
        next_state = state.model_copy(update={"runtime_status": "failed"})
        save_session_state(db, next_state, expected_revision=state.revision)
        db.commit()
        return outcome, None
    except RevisionConflictError:
        db.rollback()
        raise
