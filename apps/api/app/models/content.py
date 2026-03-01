from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Column, Index, Text, UniqueConstraint
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SettingEntry(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("project_id", "key", name="uq_setting_project_key"),
        Index("ix_setting_entry_project_id_id", "project_id", "id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    key: str = Field(index=True, max_length=191)
    value: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    aliases: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class StoryCard(SQLModel, table=True):
    __table_args__ = (Index("ix_story_card_project_id_id", "project_id", "id"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    title: str = Field(default="未命名卡片", max_length=255)
    content: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    aliases: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class ProjectChapter(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("project_id", "chapter_index", name="uq_project_chapter_index"),
        Index("ix_project_chapter_project_volume", "project_id", "volume_id", "chapter_index"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    volume_id: Optional[int] = Field(default=None, foreign_key="projectvolume.id", index=True)
    chapter_index: int = Field(default=1, ge=1, index=True)
    title: str = Field(default="第1章", max_length=255)
    content: str = Field(default="", sa_column=Column(Text, nullable=False))
    version: int = Field(default=1, nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class ProjectChapterRevision(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("chapter_id", "version", name="uq_project_chapter_revision"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    chapter_id: int = Field(foreign_key="projectchapter.id", index=True)
    project_id: int = Field(index=True)
    version: int = Field(default=1, nullable=False, index=True)
    title: str = Field(default="第1章", max_length=255)
    content: str = Field(default="", sa_column=Column(Text, nullable=False))
    operator_id: str = Field(default="system", max_length=128)
    source: str = Field(default="save", max_length=32)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class ProjectVolume(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("project_id", "volume_index", name="uq_project_volume_index"),
        Index("ix_project_volume_project_id_id", "project_id", "id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    volume_index: int = Field(default=1, ge=1, index=True)
    title: str = Field(default="第1卷", max_length=255)
    outline: str = Field(default="", sa_column=Column(Text, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class ChapterSceneBeat(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("chapter_id", "beat_index", name="uq_chapter_scene_beat_index"),
        Index("ix_chapter_scene_beat_project_chapter", "project_id", "chapter_id", "beat_index"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    chapter_id: int = Field(foreign_key="projectchapter.id", index=True)
    beat_index: int = Field(default=1, ge=1, index=True)
    content: str = Field(default="", sa_column=Column(Text, nullable=False))
    status: str = Field(default="pending", max_length=32, index=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class ForeshadowingCard(SQLModel, table=True):
    __table_args__ = (
        Index("ix_foreshadow_project_status", "project_id", "status"),
        Index("ix_foreshadow_project_planted", "project_id", "planted_in_chapter_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    title: str = Field(default="未命名伏笔", max_length=255)
    description: str = Field(default="", sa_column=Column(Text, nullable=False))
    status: str = Field(default="open", max_length=32, index=True)
    planted_in_chapter_id: Optional[int] = Field(default=None, foreign_key="projectchapter.id", index=True)
    resolved_in_chapter_id: Optional[int] = Field(default=None, foreign_key="projectchapter.id", index=True)
    source_action_id: Optional[int] = Field(default=None, foreign_key="chataction.id", index=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)
    resolved_at: Optional[datetime] = Field(default=None)


class PromptTemplate(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_prompt_template_project_name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    name: str = Field(default="默认模板", max_length=128)
    system_prompt: str = Field(default="", sa_column=Column(Text, nullable=False))
    user_prompt_prefix: str = Field(default="", sa_column=Column(Text, nullable=False))
    knowledge_setting_keys: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    knowledge_card_ids: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class PromptTemplateRevision(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("template_id", "version", name="uq_prompt_template_revision"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    template_id: int = Field(foreign_key="prompttemplate.id", index=True)
    project_id: int = Field(index=True)
    version: int = Field(default=1, nullable=False, index=True)
    name: str = Field(default="默认模板", max_length=128)
    system_prompt: str = Field(default="", sa_column=Column(Text, nullable=False))
    user_prompt_prefix: str = Field(default="", sa_column=Column(Text, nullable=False))
    knowledge_setting_keys: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    knowledge_card_ids: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    operator_id: str = Field(default="system", max_length=128)
    source: str = Field(default="save", max_length=32)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
