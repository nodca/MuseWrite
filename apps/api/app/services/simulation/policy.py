from __future__ import annotations

from pydantic import BaseModel, Field


class SimulationPolicy(BaseModel):
    agitation_weight: float = Field(default=1.0, ge=0.0)
    event_target_bonus: float = Field(default=0.7, ge=0.0)
    mention_target_bonus: float = Field(default=0.8, ge=0.0)
    cooldown_penalty: float = Field(default=0.6, ge=0.0)
    last_speaker_penalty: float = Field(default=0.5, ge=0.0)
    focus_target_bonus: float = Field(default=0.25, ge=0.0)
    min_candidate_score: float = Field(default=0.35, ge=0.0)
    min_candidate_confidence: float = Field(default=0.45, ge=0.0, le=1.0)


DEFAULT_POLICY = SimulationPolicy()
