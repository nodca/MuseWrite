from __future__ import annotations

from typing import Any

from app.services.simulation.policy import DEFAULT_POLICY, SimulationPolicy
from app.services.simulation.types import (
    CharacterMindState,
    SessionStateRecord,
    TurnCandidate,
    WorldState,
)


def _load_mind_state(
    session_state: SessionStateRecord,
    character_id: int,
) -> CharacterMindState:
    raw = session_state.mind_states.get(str(character_id)) or {}
    return CharacterMindState(
        character_id=character_id,
        known_fact_keys=list(raw.get("known_fact_keys") or []),
        beliefs=list(raw.get("beliefs") or []),
        beliefs_about_others=list(raw.get("beliefs_about_others") or []),
        goals=list(raw.get("goals") or []),
        agitation=float(raw.get("agitation") or 0.0),
        cooldown=int(raw.get("cooldown") or 0),
        focus_target=raw.get("focus_target"),
    )


def _event_bonus(
    character_id: int,
    events: list[dict[str, Any]],
    policy: SimulationPolicy,
) -> tuple[float, str]:
    bonus = 0.0
    motive = "观望"
    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if payload.get("target_id") == character_id:
            bonus += policy.event_target_bonus
            motive = str(payload.get("motive") or "事件直接命中")
    return bonus, motive


def _recent_turn_bonus(
    mind_state: CharacterMindState,
    world_state: WorldState,
    policy: SimulationPolicy,
) -> tuple[float, str]:
    bonus = 0.0
    motive = "观望"
    for turn in world_state.recent_turns:
        actor_id = turn.get("actor_id")
        target_id = turn.get("target_card_id")
        if target_id == mind_state.character_id:
            bonus += policy.mention_target_bonus
            motive = "刚被针对"
        if (
            mind_state.focus_target is not None
            and actor_id == mind_state.focus_target
        ):
            bonus += policy.focus_target_bonus
            motive = "关注对象发言"
    return bonus, motive


def produce_candidates(
    world_state: WorldState,
    session_state: SessionStateRecord,
    *,
    character_ids: list[int],
    policy: SimulationPolicy = DEFAULT_POLICY,
) -> list[TurnCandidate]:
    candidates: list[TurnCandidate] = []
    for character_id in character_ids:
        mind_state = _load_mind_state(session_state, character_id)
        score = mind_state.agitation * policy.agitation_weight
        motive = "观望"

        event_bonus, event_motive = _event_bonus(
            character_id,
            list(world_state.pending_events),
            policy,
        )
        score += event_bonus
        if event_bonus > 0:
            motive = event_motive

        turn_bonus, turn_motive = _recent_turn_bonus(
            mind_state,
            world_state,
            policy,
        )
        score += turn_bonus
        if turn_bonus > 0:
            motive = turn_motive

        if session_state.last_speaker_id == character_id:
            score -= policy.last_speaker_penalty
        score -= float(mind_state.cooldown) * policy.cooldown_penalty
        confidence = max(0.0, min(1.0, 0.4 + score / 2.0))
        candidates.append(
            TurnCandidate(
                actor_id=character_id,
                agitation_score=round(max(0.0, score), 4),
                motive=motive,
                target_id=mind_state.focus_target,
                interrupt_kind="voluntary",
                confidence=round(confidence, 4),
            )
        )
    return sorted(
        candidates,
        key=lambda item: (item.agitation_score, item.confidence, -item.actor_id),
        reverse=True,
    )
