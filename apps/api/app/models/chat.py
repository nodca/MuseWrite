from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Column, Index, UniqueConstraint
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChatSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    user_id: str = Field(index=True, max_length=128)
    title: str = Field(default="新对话", max_length=255)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class ChatMessage(SQLModel, table=True):
    __table_args__ = (Index("ix_chat_message_session_id_id", "session_id", "id"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="chatsession.id", index=True)
    role: str = Field(max_length=32, index=True)
    content: str = Field(default="")
    model: Optional[str] = Field(default=None, max_length=128)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class ChatAction(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("session_id", "idempotency_key", name="uq_chat_action_session_idem"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="chatsession.id", index=True)
    action_type: str = Field(max_length=64, index=True)
    status: str = Field(default="proposed", max_length=32, index=True)
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    apply_result: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    undo_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    idempotency_key: str = Field(index=True, max_length=128)
    operator_id: str = Field(max_length=128)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    applied_at: Optional[datetime] = Field(default=None)
    undone_at: Optional[datetime] = Field(default=None)


class ActionAuditLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    action_id: int = Field(foreign_key="chataction.id", index=True)
    event_type: str = Field(max_length=32, index=True)
    event_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    operator_id: str = Field(max_length=128)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class ProjectMutationVersion(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("project_id", name="uq_project_mutation_version_project"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    version: int = Field(default=0, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class AsyncJob(SQLModel, table=True):
    __table_args__ = (
        Index("ix_async_job_queue_status_available", "queue_name", "status", "available_at"),
        Index("ix_async_job_queue_project_status", "queue_name", "project_id", "status"),
        Index("ix_async_job_status_updated", "status", "updated_at"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    queue_name: str = Field(index=True, max_length=64)
    project_id: int = Field(index=True)
    action_id: int = Field(default=0, index=True)
    operator_id: str = Field(default="system", max_length=128)
    reason: str = Field(default="", max_length=255)
    mutation_id: str = Field(default="", max_length=128)
    expected_version: int = Field(default=0)
    idempotency_key: str = Field(default="", max_length=191, index=True)
    lifecycle_slot: str = Field(default="default", max_length=32)
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    attempt: int = Field(default=0)
    max_retries: int = Field(default=3)
    status: str = Field(default="queued", index=True, max_length=32)
    available_at: datetime = Field(default_factory=utc_now, nullable=False, index=True)
    locked_at: Optional[datetime] = Field(default=None, index=True)
    lock_token: str = Field(default="", max_length=64, index=True)
    locked_by: str = Field(default="", max_length=128)
    last_error: str = Field(default="", max_length=4000)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)
