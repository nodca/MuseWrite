"""Fiction-specific entity and edge types for Graphiti temporal graph.

These Pydantic models define custom entity types that guide Graphiti's
LLM-powered entity extraction from story episodes. Each model's docstring
is used by the extraction prompt as a label description.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FictionCharacter(BaseModel):
    """A character in the story — protagonist, antagonist, or supporting role."""

    role: str | None = Field(
        None,
        description="Role in narrative: protagonist, antagonist, supporting, minor",
    )
    status: str | None = Field(
        None,
        description="Current status: alive, dead, missing, unknown",
    )
    affiliation: str | None = Field(
        None,
        description="Primary faction or sect the character belongs to",
    )


class FictionLocation(BaseModel):
    """A named place in the story world — city, mountain, building, dimension."""

    location_type: str | None = Field(
        None,
        description="Type: city, village, mountain, forest, building, realm, dimension",
    )
    controlled_by: str | None = Field(
        None,
        description="Faction or character that controls this location",
    )


class FictionFaction(BaseModel):
    """An organization, sect, empire, guild, or political force in the story."""

    faction_type: str | None = Field(
        None,
        description="Type: sect, empire, guild, alliance, family, order",
    )
    alignment: str | None = Field(
        None,
        description="General alignment: righteous, evil, neutral, chaotic",
    )


class FictionItem(BaseModel):
    """A significant object — weapon, artifact, treasure, or resource."""

    item_type: str | None = Field(
        None,
        description="Type: weapon, artifact, medicine, treasure, tool",
    )
    rarity: str | None = Field(
        None,
        description="Rarity: common, rare, legendary, unique",
    )


class FictionWorldRule(BaseModel):
    """A rule, law, or constraint governing the story world — cultivation system, magic, taboo."""

    scope: str | None = Field(
        None,
        description="Scope: universal, regional, personal, conditional",
    )
    enforceable: bool | None = Field(
        None,
        description="Whether this rule is strictly enforced or merely convention",
    )


# ── Mapping passed to Graphiti.add_episode(entity_types=...) ──

FICTION_ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "Character": FictionCharacter,
    "Location": FictionLocation,
    "Faction": FictionFaction,
    "Item": FictionItem,
    "WorldRule": FictionWorldRule,
}

# ── Chinese-language extraction instructions for fiction domain ──

FICTION_EXTRACTION_INSTRUCTIONS: str = (
    "你正在分析一部中文长篇小说的章节片段。"
    "提取所有出现的角色(Character)、地点(Location)、势力(Faction)、重要物品(Item)、世界规则(WorldRule)。"
    "同时提取实体之间的关系，包括但不限于：信任、敌对、隶属、控制、知晓、位于、违反规则。"
    "关系应反映该章节片段中的事实状态，而非全书的全知视角。"
    "使用角色的中文名称（优先使用正式名而非别名）。"
    "如果无法确定，标记 confidence 为较低值。"
)
