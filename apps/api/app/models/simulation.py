from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Column, Index, Text, UniqueConstraint
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SimulationSession(SQLModel, table=True):
    __tablename__ = "simulationsession"

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    title: str = Field(default="未命名模拟", max_length=255)
    scenario: str = Field(default="", sa_column=Column(Text, nullable=False))
    character_card_ids: list[int] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    setting_keys: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    max_turns: int = Field(default=10, ge=1, le=50)
    status: str = Field(default="idle", max_length=32, index=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class SimulationSessionState(SQLModel, table=True):
    __tablename__ = "simulationsessionstate"
    __table_args__ = (
        UniqueConstraint(
            "session_id", name="uq_simulation_session_state_session"
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(
        foreign_key="simulationsession.id", index=True
    )
    revision: int = Field(default=0, nullable=False)
    runtime_status: str = Field(default="idle", max_length=32, index=True)
    last_speaker_id: Optional[int] = Field(default=None, index=True)
    active_pressures: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    mind_states: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class SimulationEvent(SQLModel, table=True):
    __tablename__ = "simulationevent"
    __table_args__ = (
        Index(
            "ix_simulation_event_session_consumed",
            "session_id",
            "consumed_at",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(
        foreign_key="simulationsession.id", index=True
    )
    event_type: str = Field(max_length=64, index=True)
    source: str = Field(default="external", max_length=32, index=True)
    payload: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    priority: int = Field(default=0, index=True)
    consumed_at: Optional[datetime] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class SimulationDecisionTrace(SQLModel, table=True):
    __tablename__ = "simulationdecisiontrace"
    __table_args__ = (
        Index(
            "ix_simulation_decision_trace_session_turn",
            "session_id",
            "turn_index",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(
        foreign_key="simulationsession.id", index=True
    )
    turn_index: int = Field(default=1, ge=1, index=True)
    candidate_snapshot: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    decision_payload: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    failure_code: str = Field(default="", max_length=64)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class SimulationTurn(SQLModel, table=True):
    __tablename__ = "simulationturn"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(
        foreign_key="simulationsession.id", index=True
    )
    turn_index: int = Field(default=1, ge=1)
    actor_card_id: int = Field(index=True)
    actor_name: str = Field(default="", max_length=128)
    action_type: str = Field(default="say", max_length=32)
    content: str = Field(default="", sa_column=Column(Text, nullable=False))
    target_card_id: Optional[int] = Field(default=None)
    emotion: Optional[str] = Field(default=None, max_length=64)
    is_injected_event: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
