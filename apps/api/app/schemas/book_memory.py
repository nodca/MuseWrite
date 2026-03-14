from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class WorldRuleRead(BaseModel):
    id: int
    project_id: int
    scope: str
    title: str
    statement: str
    priority: int
    tags: list[str] = Field(default_factory=list)
    status: str
    source_refs: list[dict] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class CharacterProfileRead(BaseModel):
    id: int
    project_id: int
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    public_traits: list[str] = Field(default_factory=list)
    private_traits: list[str] = Field(default_factory=list)
    core_goals: list[str] = Field(default_factory=list)
    fears: list[str] = Field(default_factory=list)
    taboos: list[str] = Field(default_factory=list)
    default_voice_notes: str
    status: str
    source_refs: list[dict] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class StoryStateSnapshotRead(BaseModel):
    id: int
    project_id: int
    volume_id: Optional[int] = None
    chapter_id: Optional[int] = None
    scene_beat_id: Optional[int] = None
    chapter_goal: str
    active_characters: list[str] = Field(default_factory=list)
    current_location: str
    active_conflicts: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    source_refs: list[dict] = Field(default_factory=list)
    updated_at: datetime


class StoryEpisodeRead(BaseModel):
    id: int
    project_id: int
    chapter_id: Optional[int] = None
    scene_beat_id: Optional[int] = None
    episode_index: int
    title: str
    summary: str
    event_type: str
    participants: list[str] = Field(default_factory=list)
    location: str
    visibility: str
    importance: int
    source_text_ref: dict = Field(default_factory=dict)
    created_at: datetime


class CharacterKnowledgeStateRead(BaseModel):
    id: int
    project_id: int
    character_profile_id: int
    knowledge_key: str
    knowledge_value: dict = Field(default_factory=dict)
    gained_at_chapter: Optional[int] = None
    lost_at_chapter: Optional[int] = None
    source_episode_id: Optional[int] = None
    confidence: float
    created_at: datetime
    updated_at: datetime


class MemoryMaterializationRead(BaseModel):
    id: int
    project_id: int
    materialization_type: str
    scope_key: str
    payload: dict = Field(default_factory=dict)
    source_versions: dict = Field(default_factory=dict)
    updated_at: datetime
