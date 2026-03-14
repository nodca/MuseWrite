"""Canon Sync — 自然中文文本 → 结构化记忆。

作者用自然中文撰写世界设定和角色档案，系统自动解析入库。

**解析策略** (LLM 优先 + 规则兜底)::

    Layer 1: 标签行直接解析（别名：小澈 → aliases）
    Layer 2: LLM 从散文中提取（有 Instructor 时）
    Layer 3: 正则模式匹配（LLM 不可用时兜底）

格式与写作正文一致，无需学习特殊标记语法。
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

_LABEL_RE = re.compile(
    r"^(别名|又名|化名|身份|角色|职业|目标|追求|性格|特点|特征|"
    r"禁忌|底线|恐惧|弱点|语气|说话风格|声音|阵营|势力|门派|"
    r"外貌|年龄|境界|实力|武器|法宝|关系|备注)"
    r"[：:]\s*(.+)$"
)

_GOAL_LABELS = {"目标", "追求"}
_TRAIT_LABELS = {"性格", "特点", "特征"}
_ALIAS_LABELS = {"别名", "又名", "化名"}
_TABOO_LABELS = {"禁忌", "底线"}
_FEAR_LABELS = {"恐惧", "弱点"}
_VOICE_LABELS = {"语气", "说话风格", "声音"}


def _split_values(text: str) -> list[str]:
    return [v.strip() for v in re.split(r"[，,、；;]", text) if v.strip()]


def _add_unique(target: list[str], items: list[str]) -> None:
    """Append items to target, skipping duplicates."""
    existing = {v.lower() for v in target}
    for item in items:
        if item.strip() and item.strip().lower() not in existing:
            target.append(item.strip())
            existing.add(item.strip().lower())


# ── World Rules parsing ─────────────────────────────────────────────


def parse_world_rules_text(text: str) -> list[dict[str, Any]]:
    """Parse a world rules document into rule dicts.

    Rules are separated by ``## Title`` headings.
    """
    rules: list[dict[str, Any]] = []
    current_title = ""
    current_lines: list[str] = []

    for line in text.strip().splitlines():
        stripped = line.strip()
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

    if current_title:
        statement = "\n".join(current_lines).strip()
        if statement:
            rules.append({"title": current_title, "statement": statement})

    return rules


# ── Character parsing ───────────────────────────────────────────────


def parse_character_text(text: str, *, use_llm: bool = True) -> dict[str, Any]:
    """Parse a character document written in natural Chinese.

    Three-layer extraction:
      1. Labeled lines (``别名：小澈``) → always
      2. LLM from prose → when *use_llm* is True and Instructor is available
      3. Regex from prose → fallback
    """
    lines = text.strip().splitlines()
    name = ""
    labels: dict[str, str] = {}
    prose_lines: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if i == 0 and stripped.startswith("#"):
            name = stripped.lstrip("#").strip()
            continue
        match = _LABEL_RE.match(stripped)
        if match:
            labels[match.group(1)] = match.group(2)
        else:
            prose_lines.append(stripped)

    # ── Layer 1: Labels ──
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

    if description:
        # ── Layer 2: LLM extraction from prose ──
        llm_ok = False
        if use_llm:
            llm_ok = _supplement_from_llm(
                name or "未知角色",
                description,
                aliases=aliases,
                goals=goals,
                traits=traits,
                private_traits=private_traits,
                fears=fears,
                taboos=taboos,
            )

        # ── Layer 3: Regex fallback ──
        if not llm_ok:
            _supplement_from_regex(
                description,
                goals=goals,
                traits=traits,
                private_traits=private_traits,
                fears=fears,
                taboos=taboos,
            )

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


# ── Layer 2: LLM extraction ────────────────────────────────────────


def _supplement_from_llm(
    character_name: str,
    prose: str,
    *,
    aliases: list[str],
    goals: list[str],
    traits: list[str],
    private_traits: list[str],
    fears: list[str],
    taboos: list[str],
) -> bool:
    """Use Instructor to extract structured fields from prose.

    Returns True if LLM extraction succeeded.
    """
    try:
        import instructor
        from openai import OpenAI
        from pydantic import BaseModel, Field

        from app.core.config import settings
    except ImportError:
        return False

    api_key = settings.lightrag_llm_api_key or settings.llm_api_key
    base_url = settings.lightrag_llm_base_url or settings.llm_base_url
    model = settings.lightrag_llm_model or settings.llm_model
    if not (api_key and base_url and model):
        return False

    class ProseExtraction(BaseModel):
        aliases: list[str] = Field(default_factory=list, description="角色的其他称呼/别名")
        goals: list[str] = Field(default_factory=list, description="角色的目标/追求")
        public_traits: list[str] = Field(default_factory=list, description="外在表现的性格特点")
        private_traits: list[str] = Field(default_factory=list, description="内心隐藏的特质")
        fears: list[str] = Field(default_factory=list, description="角色害怕/恐惧的事物")
        taboos: list[str] = Field(default_factory=list, description="角色绝不会做的事/底线")
        voice_style: str = Field(default="", description="说话语气/风格")

    try:
        client = instructor.from_openai(
            OpenAI(api_key=api_key, base_url=base_url, timeout=30),
            mode=instructor.Mode.JSON,
        )
        result = client.chat.completions.create(
            model=model,
            response_model=ProseExtraction,
            max_retries=1,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"从以下关于角色「{character_name}」的描述文本中提取结构化信息。"
                        "只提取文本明确表达或强烈暗示的内容，禁止编造。"
                        "如果某个字段没有相关信息，返回空数组或空字符串。"
                    ),
                },
                {"role": "user", "content": prose},
            ],
        )

        _add_unique(aliases, result.aliases)
        _add_unique(goals, result.goals)
        _add_unique(traits, result.public_traits)
        _add_unique(private_traits, result.private_traits)
        _add_unique(fears, result.fears)
        _add_unique(taboos, result.taboos)

        logger.debug("LLM prose extraction succeeded for %s", character_name)
        return True

    except Exception:
        logger.debug("LLM prose extraction failed for %s", character_name, exc_info=True)
        return False


# ── Layer 3: Regex fallback ─────────────────────────────────────────

# Common Chinese prose patterns for character attributes.
_PROSE_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # Goals
    ("goals", re.compile(r"(?:一心|立志|发誓|决心|渴望|梦想|毕生所求是?|为了)(.{2,20}?)(?:[，。；]|$)"), "goal"),
    ("goals", re.compile(r"(?:踏上了|走上了)(.{2,15}?)(?:之路|的道路)"), "goal"),
    # Traits (public)
    ("traits", re.compile(r"(?:性格|为人|天性|生性)(.{2,15}?)(?:[，。；]|$)"), "trait"),
    ("traits", re.compile(r"表面(?:上)?(.{2,10}?)(?:[，。；]|$)"), "trait"),
    # Private traits
    ("private_traits", re.compile(r"(?:实则|实际上?|内心深处?|暗地里)(.{2,15}?)(?:[，。；]|$)"), "private"),
    ("private_traits", re.compile(r"(?:看似.{2,6}?[，,])\s*(?:实则|其实|内心)(.{2,15}?)(?:[，。；]|$)"), "private"),
    # Fears
    ("fears", re.compile(r"(?:恐惧|害怕|最怕|不敢面对|畏惧)(.{2,15}?)(?:[，。；]|$)"), "fear"),
    # Taboos
    ("taboos", re.compile(r"(?:绝不|从不|永远不会?|底线是|原则是|不会主动)(.{2,15}?)(?:[，。；]|$)"), "taboo"),
    # Voice
    ("voice", re.compile(r"(?:说话|语气|总是.{0,4}?地说|口头禅)(.{2,15}?)(?:[，。；]|$)"), "voice"),
]


def _supplement_from_regex(
    prose: str,
    *,
    goals: list[str],
    traits: list[str],
    private_traits: list[str],
    fears: list[str],
    taboos: list[str],
) -> None:
    """Regex-based extraction from prose. Best-effort fallback."""
    target_map = {
        "goals": goals,
        "traits": traits,
        "private_traits": private_traits,
        "fears": fears,
        "taboos": taboos,
    }

    for field_name, pattern, _ in _PROSE_PATTERNS:
        if field_name == "voice":
            continue  # Voice is handled separately.
        target = target_map.get(field_name)
        if target is None:
            continue
        for m in pattern.finditer(prose):
            value = m.group(1).strip()
            if value:
                _add_unique(target, _split_values(value))


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
    use_llm: bool = True,
) -> dict[str, Any]:
    """Parse a character text and upsert into CharacterProfile table."""
    data = parse_character_text(text, use_llm=use_llm)
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
    use_llm: bool = True,
) -> dict[str, Any]:
    """Sync an entire canon directory to the database."""
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
                use_llm=use_llm,
            )
            report["characters"].append(result)

    return report
