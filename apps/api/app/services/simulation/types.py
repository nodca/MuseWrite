from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Belief(BaseModel):
    fact_key: str = Field(min_length=1, max_length=255)
    stance: str = Field(default="known", max_length=32)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_turn_id: int | None = Field(default=None)
    expires_at: datetime | None = Field(default=None)


class SecondOrderBelief(BaseModel):
    holder_id: int
    subject_id: int
    fact_key: str = Field(min_length=1, max_length=255)
    believed_knowledge_state: str = Field(default="unknown", max_length=32)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_turn_id: int | None = Field(default=None)


class CharacterMindState(BaseModel):
    character_id: int
    known_fact_keys: list[str] = Field(default_factory=list)
    beliefs: list[Belief] = Field(default_factory=list)
    beliefs_about_others: list[SecondOrderBelief] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    agitation: float = Field(default=0.0, ge=0.0)
    cooldown: int = Field(default=0, ge=0)
    focus_target: int | None = Field(default=None)


class WorldState(BaseModel):
    session_id: int
    scenario: str = Field(default="")
    turn_index: int = Field(default=1, ge=1)
    last_speaker_id: int | None = Field(default=None)
    active_pressures: list[dict[str, Any]] = Field(default_factory=list)
    pending_events: list[dict[str, Any]] = Field(default_factory=list)
    recent_turns: list[dict[str, Any]] = Field(default_factory=list)


class TurnCandidate(BaseModel):
    actor_id: int
    agitation_score: float = Field(default=0.0, ge=0.0)
    motive: str = Field(default="", max_length=256)
    target_id: int | None = Field(default=None)
    interrupt_kind: str = Field(default="voluntary", max_length=32)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class RefereeDecision(BaseModel):
    winner_id: int | None = Field(default=None)
    reason: str = Field(default="", max_length=256)
    applied_rules: list[str] = Field(default_factory=list)
    pause_suggested: bool = Field(default=False)


class TurnOutcome(BaseModel):
    status: Literal["spoken", "paused", "failed"]
    turn_id: int | None = Field(default=None)
    turn_index: int | None = Field(default=None)
    error_code: str = Field(default="", max_length=64)
    details: dict[str, Any] = Field(default_factory=dict)


class SessionStateRecord(BaseModel):
    session_id: int
    revision: int = Field(default=0, ge=0)
    runtime_status: str = Field(default="idle", max_length=32)
    last_speaker_id: int | None = Field(default=None)
    active_pressures: list[dict[str, Any]] = Field(default_factory=list)
    mind_states: dict[str, Any] = Field(default_factory=dict)


class PersistedTurnRecord(BaseModel):
    outcome: TurnOutcome
    turn_id: int | None = Field(default=None)
    trace_id: int
    consumed_event_ids: list[int] = Field(default_factory=list)
