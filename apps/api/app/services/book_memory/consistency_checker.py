"""Consistency checker powered by Book Memory OS.

Detects three categories of contradiction in draft text:

1. **World-rule violations** — draft contradicts established world rules
2. **Character knowledge leaks** — draft reveals information a character
   should not know at the current chapter (omniscience leak)
3. **Chronology violations** — draft implies events in the wrong order
   or references facts that are not yet valid

All checks are heuristic (keyword + temporal boundary matching).
LLM-based deep checks can be layered on top later.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from sqlmodel import Session

from app.services.book_memory.memory_pipeline import build_consistency_memory_context
from app.services.book_memory.temporal_query import CharacterKnowledgeBundle

logger = logging.getLogger(__name__)


# ── Result types ────────────────────────────────────────────────────


@dataclass
class ConsistencyIssue:
    """A single consistency violation or warning."""

    category: str  # "world_rule" | "knowledge_leak" | "chronology"
    severity: str  # "critical" | "warning" | "info"
    title: str = ""
    detail: str = ""
    evidence: str = ""
    suggestion: str = ""


@dataclass
class ConsistencyReport:
    """Result of a consistency check against draft text."""

    project_id: int = 0
    chapter_id: int = 0
    issues: list[ConsistencyIssue] = field(default_factory=list)
    checks_run: int = 0

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def clean(self) -> bool:
        return self.critical_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "chapter_id": self.chapter_id,
            "clean": self.clean,
            "critical": self.critical_count,
            "warnings": self.warning_count,
            "total_issues": len(self.issues),
            "checks_run": self.checks_run,
            "issues": [
                {
                    "category": i.category,
                    "severity": i.severity,
                    "title": i.title,
                    "detail": i.detail,
                    "evidence": i.evidence,
                    "suggestion": i.suggestion,
                }
                for i in self.issues
            ],
        }


# ── Public API ──────────────────────────────────────────────────────


def check_draft_consistency(
    db: Session,
    *,
    project_id: int,
    chapter_id: int,
    draft_text: str,
    chapter_index: int | None = None,
) -> ConsistencyReport:
    """Run all consistency checks against *draft_text*.

    Returns a ``ConsistencyReport`` with any detected issues.
    """
    report = ConsistencyReport(project_id=project_id, chapter_id=chapter_id)

    if not draft_text.strip():
        return report

    memory = build_consistency_memory_context(
        db,
        project_id=project_id,
        chapter_id=chapter_id,
        draft_text=draft_text,
        chapter_index=chapter_index,
    )

    # ── Check 1: World-rule violations ──
    report.checks_run += 1
    _check_world_rules(draft_text, memory.world_rules, report)

    # ── Check 2: Character knowledge leaks ──
    report.checks_run += 1
    _check_knowledge_leaks(draft_text, memory.character_knowledge, report)

    # ── Check 3: Chronology violations ──
    report.checks_run += 1
    _check_chronology(draft_text, memory.recent_episodes, report)

    return report


# ── Check implementations ───────────────────────────────────────────

# Negation words that often appear before a rule statement.
_NEGATION_PATTERN = re.compile(
    r"(不再|没有|无法|不能|不会|不可能|已经解除|已失效|突破了|打破了|无视)"
)


def _check_world_rules(
    draft: str,
    rules: list[dict[str, Any]],
    report: ConsistencyReport,
) -> None:
    """Detect potential world-rule violations in draft text."""
    draft_lower = draft.lower()

    for rule in rules:
        title = str(rule.get("title") or "").strip()
        statement = str(rule.get("statement") or "").strip()
        if not statement:
            continue

        # Extract key terms from the rule statement.
        keywords = _extract_keywords(statement)
        if not keywords:
            continue

        # Check if draft text contains the rule's key terms.
        matched_kw = [kw for kw in keywords if kw in draft_lower]
        if not matched_kw:
            continue

        # Check if the draft negates the rule.
        for kw in matched_kw:
            idx = draft_lower.find(kw)
            if idx < 0:
                continue
            # Look at the 20 chars before the keyword for negation.
            window_start = max(0, idx - 20)
            before_text = draft[window_start:idx]
            neg_match = _NEGATION_PATTERN.search(before_text)
            if neg_match:
                report.issues.append(
                    ConsistencyIssue(
                        category="world_rule",
                        severity="warning",
                        title=f"可能违反世界规则「{title}」",
                        detail=f"草稿中出现了对「{statement[:80]}」的否定表述",
                        evidence=draft[window_start : idx + len(kw) + 20].strip()[:200],
                        suggestion=f"请确认「{title}」是否已被剧情正式废除",
                    )
                )


def _check_knowledge_leaks(
    draft: str,
    character_knowledge: list[CharacterKnowledgeBundle],
    report: ConsistencyReport,
) -> None:
    """Detect when a character appears to know something they shouldn't."""
    for ck in character_knowledge:
        if not ck.withheld_facts:
            continue

        char_name = ck.character_name
        if char_name not in draft:
            continue

        for wf in ck.withheld_facts:
            withheld_text = wf.object.strip()
            if not withheld_text:
                continue

            # Extract key phrases from the withheld fact.
            keywords = _extract_keywords(withheld_text)
            for kw in keywords:
                if kw not in draft.lower():
                    continue

                # Check if the character name and the withheld fact
                # appear within 200 characters of each other.
                for m in re.finditer(re.escape(char_name), draft):
                    char_pos = m.start()
                    kw_pos = draft.lower().find(kw)
                    if kw_pos >= 0 and abs(char_pos - kw_pos) < 200:
                        report.issues.append(
                            ConsistencyIssue(
                                category="knowledge_leak",
                                severity="critical",
                                title=f"角色「{char_name}」可能全知泄漏",
                                detail=f"「{char_name}」此时不应知道「{withheld_text[:60]}」",
                                evidence=draft[
                                    max(0, min(char_pos, kw_pos) - 20) :
                                    max(char_pos, kw_pos) + 60
                                ].strip()[:200],
                                suggestion=f"考虑让「{char_name}」通过其他方式得知此信息，或移除相关描写",
                            )
                        )
                        break  # One issue per withheld fact per character.


def _check_chronology(
    draft: str,
    episodes: list[dict[str, Any]],
    report: ConsistencyReport,
) -> None:
    """Detect chronology problems using episode order."""
    if len(episodes) < 2:
        return

    # Build a simple event-order timeline.
    event_titles = [
        str(ep.get("title") or "").strip()
        for ep in episodes
        if str(ep.get("title") or "").strip()
    ]

    # Check if draft references later events before earlier ones.
    positions: list[tuple[int, str]] = []
    for title in event_titles:
        keywords = _extract_keywords(title)
        for kw in keywords:
            idx = draft.lower().find(kw)
            if idx >= 0:
                positions.append((idx, title))
                break  # First mention is enough.

    # If we found two or more events, check ordering.
    if len(positions) >= 2:
        # Sort by position in draft.
        positions.sort(key=lambda x: x[0])
        # Check if the order in draft matches the episode order.
        draft_order = [t for _, t in positions]
        episode_order = [t for t in event_titles if t in draft_order]

        if draft_order != episode_order:
            report.issues.append(
                ConsistencyIssue(
                    category="chronology",
                    severity="warning",
                    title="事件顺序可能不一致",
                    detail=(
                        f"草稿中的事件提及顺序「{'→'.join(draft_order[:4])}」"
                        f"与记录的事件顺序「{'→'.join(episode_order[:4])}」不同"
                    ),
                    evidence="",
                    suggestion="请确认这是否为刻意的叙事倒叙，还是时间线错误",
                )
            )


# ── Helpers ─────────────────────────────────────────────────────────


def _extract_keywords(text: str, min_len: int = 2, max_count: int = 5) -> list[str]:
    """Extract meaningful Chinese/English keywords from text.

    For Chinese text (no spaces), extracts overlapping 2-4 character
    windows since Chinese words don't have whitespace delimiters.
    """
    # Remove punctuation.
    cleaned = re.sub(r"[，。！？、；：\u201c\u201d\u2018\u2019（）\[\]{}【】\s]+", " ", text)
    tokens = cleaned.strip().split()
    keywords: list[str] = []

    for t in tokens:
        t = t.strip().lower()
        if not t:
            continue
        # If this is a long CJK string (no spaces), extract sub-phrases.
        if len(t) > 4 and any("\u4e00" <= c <= "\u9fff" for c in t):
            for size in (4, 3, 2):
                for i in range(len(t) - size + 1):
                    sub = t[i : i + size]
                    if sub not in keywords:
                        keywords.append(sub)
                    if len(keywords) >= max_count:
                        return keywords
        elif len(t) >= min_len and t not in keywords:
            keywords.append(t)
        if len(keywords) >= max_count:
            break
    return keywords
