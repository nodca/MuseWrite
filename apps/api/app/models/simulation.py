from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, Column, Text
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
        default_factory=list, sa_column=Column(JSON)
    )
    setting_keys: list[str] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    max_turns: int = Field(default=10, ge=1, le=50)
    status: str = Field(default="idle", max_length=32, index=True)
    pending_events: list[str] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


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
