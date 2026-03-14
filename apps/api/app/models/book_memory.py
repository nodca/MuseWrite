from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Column, Index, Text, UniqueConstraint
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WorldRule(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("project_id", "scope", "title", name="uq_world_rule_project_scope_title"),
        Index("ix_world_rule_project_status_priority", "project_id", "status", "priority"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    scope: str = Field(default="global", max_length=64, index=True)
    title: str = Field(max_length=255, index=True)
    statement: str = Field(default="", sa_column=Column(Text, nullable=False))
    priority: int = Field(default=100, index=True)
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    status: str = Field(default="active", max_length=32, index=True)
    source_refs: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class CharacterProfile(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("project_id", "canonical_name", name="uq_character_profile_project_name"),
        Index("ix_character_profile_project_status", "project_id", "status"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    canonical_name: str = Field(max_length=191, index=True)
    aliases: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    public_traits: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    private_traits: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    core_goals: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    fears: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    taboos: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    default_voice_notes: str = Field(default="", sa_column=Column(Text, nullable=False))
    status: str = Field(default="active", max_length=32, index=True)
    source_refs: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class StoryStateSnapshot(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("project_id", "chapter_id", "scene_beat_id", name="uq_story_state_scope"),
        Index("ix_story_state_project_updated", "project_id", "updated_at"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    volume_id: Optional[int] = Field(default=None, foreign_key="projectvolume.id", index=True)
    chapter_id: Optional[int] = Field(default=None, foreign_key="projectchapter.id", index=True)
    scene_beat_id: Optional[int] = Field(default=None, foreign_key="chapterscenebeat.id", index=True)
    chapter_goal: str = Field(default="", sa_column=Column(Text, nullable=False))
    active_characters: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    current_location: str = Field(default="", max_length=191)
    active_conflicts: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    open_questions: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    source_refs: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class StoryEpisode(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("project_id", "chapter_id", "scene_beat_id", "episode_index", name="uq_story_episode_scope_index"),
        Index("ix_story_episode_project_chapter", "project_id", "chapter_id", "episode_index"),
        Index("ix_story_episode_project_type", "project_id", "event_type"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    chapter_id: Optional[int] = Field(default=None, foreign_key="projectchapter.id", index=True)
    scene_beat_id: Optional[int] = Field(default=None, foreign_key="chapterscenebeat.id", index=True)
    episode_index: int = Field(default=1, ge=1, index=True)
    title: str = Field(default="未命名事件", max_length=255)
    summary: str = Field(default="", sa_column=Column(Text, nullable=False))
    event_type: str = Field(default="scene", max_length=64, index=True)
    participants: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    location: str = Field(default="", max_length=191)
    visibility: str = Field(default="public", max_length=32, index=True)
    importance: int = Field(default=50, index=True)
    source_text_ref: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class CharacterKnowledgeState(SQLModel, table=True):
    __table_args__ = (
        Index("ix_character_knowledge_project_character_key", "project_id", "character_profile_id", "knowledge_key"),
        Index("ix_character_knowledge_project_chapter", "project_id", "gained_at_chapter", "lost_at_chapter"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    character_profile_id: int = Field(foreign_key="characterprofile.id", index=True)
    knowledge_key: str = Field(max_length=191, index=True)
    knowledge_value: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    gained_at_chapter: Optional[int] = Field(default=None, ge=1, index=True)
    lost_at_chapter: Optional[int] = Field(default=None, ge=1, index=True)
    source_episode_id: Optional[int] = Field(default=None, foreign_key="storyepisode.id", index=True)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class MemoryMaterialization(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("project_id", "materialization_type", "scope_key", name="uq_memory_materialization_scope"),
        Index("ix_memory_materialization_project_type", "project_id", "materialization_type"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    materialization_type: str = Field(max_length=64, index=True)
    scope_key: str = Field(default="global", max_length=191, index=True)
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    source_versions: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)
