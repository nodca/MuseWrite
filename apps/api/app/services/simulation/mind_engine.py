from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.simulation.types import (
    Belief,
    CharacterMindState,
    SecondOrderBelief,
    SessionStateRecord,
    WorldState,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None
    return None


def _decay_confidence(value: float) -> float:
    return max(0.0, round(value - 0.05, 4))

def _decay_second_order_confidence(value: float) -> float:
    return max(0.0, round(value - 0.03, 4))


def _prune_beliefs(
    beliefs: list[Belief],
    *,
    now: datetime,
) -> list[Belief]:
    kept: list[Belief] = []
    for belief in beliefs:
        expires_at = _parse_datetime(belief.expires_at)
        if expires_at is not None and expires_at <= now:
            continue
        kept.append(
            belief.model_copy(
                update={"confidence": _decay_confidence(belief.confidence)}
            )
        )
    return kept


def _prune_second_order_beliefs(
    beliefs: list[SecondOrderBelief],
    *,
    cap: int = 80,
) -> list[SecondOrderBelief]:
    kept: list[SecondOrderBelief] = []
    for item in beliefs:
        next_confidence = _decay_second_order_confidence(item.confidence)
        if next_confidence < 0.1:
            continue
        kept.append(item.model_copy(update={"confidence": next_confidence}))
    kept.sort(key=lambda item: item.confidence, reverse=True)
    return kept[: max(int(cap), 1)]


def _upsert_second_order(
    beliefs: list[SecondOrderBelief],
    update: SecondOrderBelief,
) -> list[SecondOrderBelief]:
    for idx, item in enumerate(beliefs):
        if item.subject_id == update.subject_id and item.fact_key == update.fact_key:
            merged = item.model_copy(
                update={
                    "confidence": max(item.confidence, update.confidence),
                    "believed_knowledge_state": update.believed_knowledge_state,
                    "source_turn_id": update.source_turn_id or item.source_turn_id,
                }
            )
            next_list = list(beliefs)
            next_list[idx] = merged
            return next_list
    return [*beliefs, update]


def _load_mind_state(
    raw_states: dict[str, Any],
    character_id: int,
) -> CharacterMindState:
    raw = raw_states.get(str(character_id)) or {}
    return CharacterMindState(
        character_id=character_id,
        known_fact_keys=list(raw.get("known_fact_keys") or []),
        beliefs=[Belief.model_validate(item) for item in raw.get("beliefs") or []],
        beliefs_about_others=[
            SecondOrderBelief.model_validate(item)
            for item in raw.get("beliefs_about_others") or []
        ],
        goals=list(raw.get("goals") or []),
        agitation=float(raw.get("agitation") or 0.0),
        cooldown=int(raw.get("cooldown") or 0),
        focus_target=raw.get("focus_target"),
    )


def _apply_events(
    mind_state: CharacterMindState,
    *,
    events: list[dict[str, Any]],
) -> CharacterMindState:
    beliefs = list(mind_state.beliefs)
    second_order = list(mind_state.beliefs_about_others)
    agitation = mind_state.agitation
    focus_target = mind_state.focus_target
    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        target_id = payload.get("target_id")
        if target_id not in {None, mind_state.character_id}:
            continue
        intensity = float(payload.get("intensity") or 0.0)
        agitation += intensity
        fact_key = str(payload.get("fact_key") or "").strip()
        if fact_key:
            stance = str(payload.get("stance") or "known").strip() or "known"
            beliefs.append(
                Belief(
                    fact_key=fact_key,
                    stance=stance,
                    confidence=min(1.0, max(0.1, float(payload.get("confidence") or 0.7))),
                    source_turn_id=payload.get("source_turn_id"),
                    expires_at=payload.get("expires_at"),
                )
            )
        if payload.get("focus_target") is not None:
            focus_target = int(payload["focus_target"])
        if payload.get("subject_id") is not None and fact_key:
            second_order.append(
                SecondOrderBelief(
                    holder_id=mind_state.character_id,
                    subject_id=int(payload["subject_id"]),
                    fact_key=fact_key,
                    believed_knowledge_state=str(
                        payload.get("knowledge_state") or "suspects"
                    ),
                    confidence=min(
                        1.0, max(0.1, float(payload.get("confidence") or 0.65))
                    ),
                    source_turn_id=payload.get("source_turn_id"),
                )
            )
    return mind_state.model_copy(
        update={
            "beliefs": beliefs,
            "beliefs_about_others": second_order,
            "agitation": agitation,
            "focus_target": focus_target,
        }
    )


def _apply_recent_turns(
    mind_state: CharacterMindState,
    *,
    recent_turns: list[dict[str, Any]],
) -> CharacterMindState:
    agitation = mind_state.agitation
    focus_target = mind_state.focus_target
    for turn in recent_turns:
        actor_id = turn.get("actor_id")
        target_id = turn.get("target_card_id")
        if actor_id == mind_state.character_id:
            agitation = max(0.0, agitation - 0.1)
        elif target_id == mind_state.character_id:
            agitation += 0.35
            if isinstance(actor_id, int) and actor_id > 0:
                focus_target = actor_id
    return mind_state.model_copy(
        update={
            "agitation": agitation,
            "focus_target": focus_target,
        }
    )


def advance_minds(
    world_state: WorldState,
    session_state: SessionStateRecord,
    *,
    character_ids: list[int],
) -> SessionStateRecord:
    now = _utc_now()
    next_states: dict[str, Any] = {}
    for character_id in character_ids:
        current = _load_mind_state(session_state.mind_states, character_id)
        current = current.model_copy(
            update={
                "cooldown": max(0, current.cooldown - 1),
                "agitation": max(0.0, round(current.agitation * 0.6, 4)),
                "beliefs": _prune_beliefs(current.beliefs, now=now),
                "beliefs_about_others": _prune_second_order_beliefs(
                    current.beliefs_about_others
                ),
            }
        )
        current = _apply_events(
            current,
            events=list(world_state.pending_events or []),
        )
        current = _apply_recent_turns(
            current,
            recent_turns=list(world_state.recent_turns or []),
        )
        next_states[str(character_id)] = current.model_dump(mode="json")
    return session_state.model_copy(
        update={
            "active_pressures": list(world_state.active_pressures),
            "last_speaker_id": world_state.last_speaker_id,
            "mind_states": next_states,
        }
    )
