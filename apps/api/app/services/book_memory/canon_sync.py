"""Canon Sync — 自然中文文本 → 结构化记忆。

作者用自然中文撰写世界设定和角色档案，系统自动解析入库。
格式与写作正文一致，无需学习特殊标记语法。

**世界规则格式** (一个文件，多条规则，用标题分隔)::

    # 世界规则

    ## 修炼体系
    修仙界分为九大境界。每个境界都需要消耗灵石修炼。

    ## 飞行限制
    凡人不可飞行，只有金丹期以上的修士才能御空飞行。

**角色档案格式** (每个角色一个文件)::

    # 林澈

    别名：小澈、林师弟
    身份：天剑宗弟子
    目标：为师父复仇、飞升
    性格：冷静、坚毅
    禁忌：不伤无辜
    语气：冷淡

    林澈是天剑宗外门弟子，因师父被害而踏上复仇之路。
    表面沉稳寡言，内心深处恐惧失去仅剩的同伴。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from app.models.book_memory import CharacterProfile, WorldRule

logger = logging.getLogger(__name__)


# ── Chinese label patterns ──────────────────────────────────────────

# Matches lines like "别名：小澈、林师弟" or "目标：复仇，飞升"
_LABEL_RE = re.compile(
    r"^(别名|又名|化名|身份|角色|职业|目标|追求|性格|特点|特征|"
    r"禁忌|底线|恐惧|弱点|语气|说话风格|声音|阵营|势力|门派|"
    r"外貌|年龄|境界|实力|武器|法宝|关系|备注)"
    r"[：:]\s*(.+)$"
)

# Goal-like labels.
_GOAL_LABELS = {"目标", "追求"}
_TRAIT_LABELS = {"性格", "特点", "特征"}
_ALIAS_LABELS = {"别名", "又名", "化名"}
_TABOO_LABELS = {"禁忌", "底线"}
_FEAR_LABELS = {"恐惧", "弱点"}
_VOICE_LABELS = {"语气", "说话风格", "声音"}
_AFFILIATION_LABELS = {"阵营", "势力", "门派"}
_ROLE_LABELS = {"身份", "角色", "职业"}


def _split_values(text: str) -> list[str]:
    """Split Chinese comma / 、 separated values."""
    return [v.strip() for v in re.split(r"[，,、；;]", text) if v.strip()]


# ── World Rules parsing ─────────────────────────────────────────────


def parse_world_rules_text(text: str) -> list[dict[str, Any]]:
    """Parse a world rules document into rule dicts.

    Rules are separated by ``## Title`` headings.
    Everything after the heading until the next heading is the statement.
    """
    rules: list[dict[str, Any]] = []
    current_title = ""
    current_lines: list[str] = []

    for line in text.strip().splitlines():
        stripped = line.strip()
        # Skip top-level heading (# 世界规则).
        if stripped.startswith("# ") and not stripped.startswith("## "):
            continue
        if stripped.startswith("## "):
            if current_title:
                statement = "\n".join(current_lines).strip()
                if statement:
                    rules.append({"title": current_title, "statement": statement})
            current_title = stripped.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last.
    if current_title:
        statement = "\n".join(current_lines).strip()
        if statement:
            rules.append({"title": current_title, "statement": statement})

    return rules


# ── Character parsing ───────────────────────────────────────────────


def parse_character_text(text: str) -> dict[str, Any]:
    """Parse a character document written in natural Chinese.

    Extracts structured fields from labeled lines (``别名：...``)
    and treats remaining prose as description.
    """
    lines = text.strip().splitlines()
    name = ""
    labels: dict[str, str] = {}
    prose_lines: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # First heading → character name.
        if i == 0 and stripped.startswith("#"):
            name = stripped.lstrip("#").strip()
            continue

        # Check for label line.
        match = _LABEL_RE.match(stripped)
        if match:
            labels[match.group(1)] = match.group(2)
        else:
            prose_lines.append(stripped)

    # Build structured fields from labels.
    aliases: list[str] = []
    goals: list[str] = []
    traits: list[str] = []
    private_traits: list[str] = []
    fears: list[str] = []
    taboos: list[str] = []
    voice = ""

    for label, value in labels.items():
        if label in _ALIAS_LABELS:
            aliases.extend(_split_values(value))
        elif label in _GOAL_LABELS:
            goals.extend(_split_values(value))
        elif label in _TRAIT_LABELS:
            traits.extend(_split_values(value))
        elif label in _TABOO_LABELS:
            taboos.extend(_split_values(value))
        elif label in _FEAR_LABELS:
            fears.extend(_split_values(value))
        elif label in _VOICE_LABELS:
            voice = value

    description = "\n".join(prose_lines).strip()

    # Try to extract traits from prose if not labeled.
    if not traits and description:
        # Look for patterns like "性格冷静" in prose.
        for hint in ("性格", "为人", "天性"):
            idx = description.find(hint)
            if idx >= 0:
                after = description[idx + len(hint):idx + len(hint) + 20]
                after = re.split(r"[，。；、]", after)[0].strip()
                if after:
                    traits.extend(_split_values(after))
                break

    return {
        "canonical_name": name,
        "aliases": aliases,
        "core_goals": goals,
        "public_traits": traits,
        "private_traits": private_traits,
        "fears": fears,
        "taboos": taboos,
        "default_voice_notes": voice,
        "description": description,
    }


# ── DB Sync ─────────────────────────────────────────────────────────


def sync_world_rules(
    db: Session,
    *,
    project_id: int,
    text: str,
) -> dict[str, Any]:
    """Parse rules text and upsert into WorldRule table."""
    parsed = parse_world_rules_text(text)
    created = updated = unchanged = 0

    for rule in parsed:
        title = rule["title"]
        stmt = select(WorldRule).where(
            WorldRule.project_id == project_id,
            WorldRule.title == title,
        )
        existing = db.exec(stmt).first()

        if existing is None:
            db.add(WorldRule(
                project_id=project_id,
                title=title,
                statement=rule["statement"],
                scope="universal",
                priority=100,
                status="active",
            ))
            created += 1
        elif existing.statement != rule["statement"]:
            existing.statement = rule["statement"]
            db.add(existing)
            updated += 1
        else:
            unchanged += 1

    db.commit()
    return {"created": created, "updated": updated, "unchanged": unchanged}


def sync_character(
    db: Session,
    *,
    project_id: int,
    text: str,
) -> dict[str, Any]:
    """Parse a character text and upsert into CharacterProfile table."""
    data = parse_character_text(text)
    name = data["canonical_name"]
    if not name:
        return {"error": "no_name"}

    stmt = select(CharacterProfile).where(
        CharacterProfile.project_id == project_id,
        CharacterProfile.canonical_name == name,
    )
    existing = db.exec(stmt).first()

    if existing is None:
        db.add(CharacterProfile(
            project_id=project_id,
            canonical_name=name,
            aliases=data["aliases"],
            core_goals=data["core_goals"],
            public_traits=data["public_traits"],
            private_traits=data["private_traits"],
            fears=data["fears"],
            taboos=data["taboos"],
            default_voice_notes=data["default_voice_notes"],
        ))
        db.commit()
        return {"status": "created", "name": name}

    existing.aliases = data["aliases"] or existing.aliases
    existing.core_goals = data["core_goals"] or existing.core_goals
    existing.public_traits = data["public_traits"] or existing.public_traits
    existing.private_traits = data["private_traits"] or existing.private_traits
    existing.fears = data["fears"] or existing.fears
    existing.taboos = data["taboos"] or existing.taboos
    if data["default_voice_notes"]:
        existing.default_voice_notes = data["default_voice_notes"]
    db.add(existing)
    db.commit()
    return {"status": "updated", "name": name}


def sync_canon_directory(
    db: Session,
    *,
    project_id: int,
    canon_dir: str | Path,
) -> dict[str, Any]:
    """Sync an entire canon directory to the database.

    Expected layout::

        canon_dir/
        ├── world/
        │   └── rules.md
        └── characters/
            ├── 林澈.md
            └── 苏青.md
    """
    root = Path(canon_dir)
    report: dict[str, Any] = {"rules": None, "characters": []}

    rules_file = root / "world" / "rules.md"
    if rules_file.exists():
        report["rules"] = sync_world_rules(
            db, project_id=project_id,
            text=rules_file.read_text(encoding="utf-8"),
        )

    chars_dir = root / "characters"
    if chars_dir.is_dir():
        for f in sorted(chars_dir.glob("*.md")):
            result = sync_character(
                db, project_id=project_id,
                text=f.read_text(encoding="utf-8"),
            )
            report["characters"].append(result)

    return report
