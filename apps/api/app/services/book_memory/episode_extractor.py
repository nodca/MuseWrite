from __future__ import annotations

import re
from typing import Any, Sequence

from app.models.book_memory import CharacterProfile
from app.models.content import ChapterSceneBeat, ProjectChapter
from app.services.book_memory.extraction_service import (
    BookMemoryStructuredExtraction,
    extract_book_memory_structured,
)

_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
_LOCATION_PATTERNS = (
    re.compile(
        r"(?:在|于|来到|抵达|回到|潜入|进入|赶到)([^，。；：:、\s]{1,20}?(?:阁|殿|城|山|峰|谷|镇|村|宫|楼|台|寺|院|府|门|堂|海|河|湖|渊|岛|街|巷|林|原|关))"
    ),
    re.compile(r"([^，。；：:、\s]{1,20}(?:阁|殿|城|山|峰|谷|镇|村|宫|楼|台|寺|院|府|门|堂|海|河|湖|渊|岛|街|巷|林|原|关))"),
)
_CONFLICT_HINTS = ("对峙", "冲突", "交锋", "追杀", "追查", "怀疑", "背叛", "威胁", "阻止", "危机", "杀")
_REVELATION_HINTS = ("发现", "得知", "意识到", "真相", "秘密", "线索")
_DIALOGUE_HINTS = ("说道", "问道", "回答", "开口", "低声", "交谈", "商议")
_MOVEMENT_HINTS = ("来到", "抵达", "赶赴", "潜入", "回到", "离开", "前往")
_PRIVATE_HINTS = ("秘密", "真相", "私下", "暗中", "独自", "内心")


def normalize_memory_text(text: str | None) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip()


def match_character_names(text: str, character_profiles: Sequence[CharacterProfile] | None = None) -> list[str]:
    normalized = normalize_memory_text(text)
    if not normalized or not character_profiles:
        return []

    matched: list[str] = []
    for profile in character_profiles:
        canonical_name = str(profile.canonical_name or "").strip()
        if not canonical_name:
            continue
        aliases = [canonical_name, *[str(alias or "").strip() for alias in list(profile.aliases or [])]]
        if any(alias and alias in normalized for alias in aliases):
            matched.append(canonical_name)
    return matched


def extract_location(text: str | None) -> str:
    normalized = normalize_memory_text(text)
    if not normalized:
        return ""
    for pattern in _LOCATION_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        location = str(match.group(1) or "").strip(" ，。；：:、")
        if location:
            return location[:64]
    return ""


def _first_sentence(text: str) -> str:
    normalized = normalize_memory_text(text)
    if not normalized:
        return ""
    parts = [part.strip(" ，。；：:、") for part in _SENTENCE_SPLIT_RE.split(normalized)]
    for part in parts:
        if part:
            return part
    return normalized


def _episode_title(chapter: ProjectChapter, text: str, episode_index: int) -> str:
    sentence = _first_sentence(text)
    if 4 <= len(sentence) <= 18:
        return sentence
    chapter_title = str(chapter.title or "").strip() or f"第{int(chapter.chapter_index)}章"
    return f"{chapter_title}-事件{episode_index}"


def _episode_summary(text: str) -> str:
    normalized = normalize_memory_text(text)
    return normalized[:220]


def _infer_event_type(text: str) -> str:
    normalized = normalize_memory_text(text)
    if any(token in normalized for token in _REVELATION_HINTS):
        return "revelation"
    if any(token in normalized for token in _CONFLICT_HINTS):
        return "conflict"
    if any(token in normalized for token in _DIALOGUE_HINTS):
        return "dialogue"
    if any(token in normalized for token in _MOVEMENT_HINTS):
        return "movement"
    return "scene"


def _infer_visibility(text: str) -> str:
    normalized = normalize_memory_text(text)
    if any(token in normalized for token in _PRIVATE_HINTS):
        return "private"
    return "public"


def _infer_importance(text: str) -> int:
    normalized = normalize_memory_text(text)
    score = 50
    if any(token in normalized for token in _REVELATION_HINTS):
        score += 20
    if any(token in normalized for token in _CONFLICT_HINTS):
        score += 15
    if "？" in normalized or "?" in normalized:
        score += 10
    return max(10, min(score, 100))


def _profile_name_lookup(character_profiles: Sequence[CharacterProfile] | None = None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for profile in list(character_profiles or []):
        canonical_name = str(profile.canonical_name or "").strip()
        if not canonical_name:
            continue
        lookup[canonical_name] = canonical_name
        lookup[canonical_name.lower()] = canonical_name
        for alias in list(profile.aliases or []):
            normalized = str(alias or "").strip()
            if normalized:
                lookup[normalized] = canonical_name
                lookup[normalized.lower()] = canonical_name
    return lookup


def _normalize_participants(
    raw_participants: Sequence[str] | None,
    *,
    character_profiles: Sequence[CharacterProfile] | None,
    fallback_text: str,
) -> list[str]:
    profile_lookup = _profile_name_lookup(character_profiles)
    participants: list[str] = []
    for item in list(raw_participants or []):
        name = str(item or "").strip()
        if not name:
            continue
        canonical = profile_lookup.get(name) or profile_lookup.get(name.lower()) or name
        if canonical not in participants:
            participants.append(canonical)
    if participants:
        return participants
    return match_character_names(fallback_text, character_profiles)


def _coerce_structured_extraction(
    extraction_result: BookMemoryStructuredExtraction | dict[str, Any] | None,
) -> BookMemoryStructuredExtraction | None:
    if isinstance(extraction_result, BookMemoryStructuredExtraction):
        return extraction_result
    if isinstance(extraction_result, dict):
        try:
            return BookMemoryStructuredExtraction.model_validate(extraction_result)
        except Exception:
            return None
    return None


def extract_story_episode_candidates(
    *,
    project_id: int,
    chapter: ProjectChapter,
    scene_beats: Sequence[ChapterSceneBeat] | None = None,
    character_profiles: Sequence[CharacterProfile] | None = None,
    previous_snapshot: Any | None = None,
    extraction_result: BookMemoryStructuredExtraction | dict[str, Any] | None = None,
) -> list[dict]:
    structured = _coerce_structured_extraction(extraction_result)
    if structured is None:
        structured = extract_book_memory_structured(
            project_id=project_id,
            chapter=chapter,
            scene_beats=scene_beats,
            character_profiles=character_profiles,
            previous_snapshot=previous_snapshot,
        )

    beats = sorted(list(scene_beats or []), key=lambda item: (int(item.beat_index), int(item.id or 0)))
    beat_by_index = {int(item.beat_index): item for item in beats}
    beat_by_id = {int(item.id or 0): item for item in beats if int(item.id or 0) > 0}
    if structured is not None and structured.episodes:
        candidates: list[dict] = []
        for index, episode in enumerate(structured.episodes, start=1):
            matched_beat = None
            if episode.scene_beat_id and int(episode.scene_beat_id) in beat_by_id:
                matched_beat = beat_by_id[int(episode.scene_beat_id)]
            elif episode.beat_index and int(episode.beat_index) in beat_by_index:
                matched_beat = beat_by_index[int(episode.beat_index)]

            fallback_text = (
                str(episode.source_excerpt or "").strip()
                or str(getattr(matched_beat, "content", "") or "").strip()
                or str(chapter.content or "").strip()
            )
            text = normalize_memory_text(fallback_text)
            if not text:
                continue
            episode_index = len(candidates) + 1
            candidates.append(
                {
                    "project_id": int(project_id),
                    "chapter_id": int(chapter.id or 0) or None,
                    "scene_beat_id": int(getattr(matched_beat, "id", 0) or episode.scene_beat_id or 0) or None,
                    "episode_index": episode_index,
                    "title": str(episode.title or "").strip() or _episode_title(chapter, text, episode_index),
                    "summary": str(episode.summary or "").strip() or _episode_summary(text),
                    "event_type": str(episode.event_type or "").strip() or _infer_event_type(text),
                    "participants": _normalize_participants(
                        list(episode.participants or []),
                        character_profiles=character_profiles,
                        fallback_text=text,
                    ),
                    "location": str(episode.location or "").strip() or extract_location(text),
                    "visibility": str(episode.visibility or "").strip() or _infer_visibility(text),
                    "importance": int(episode.importance),
                    "source_text_ref": {
                        "chapter_id": int(chapter.id or 0) or None,
                        "chapter_index": int(chapter.chapter_index),
                        "scene_beat_id": int(getattr(matched_beat, "id", 0) or episode.scene_beat_id or 0) or None,
                        "beat_index": int(getattr(matched_beat, "beat_index", 0) or episode.beat_index or episode_index),
                        "excerpt": text[:160],
                    },
                }
            )
        if candidates:
            return candidates

    if not beats:
        beats = [
            ChapterSceneBeat(
                project_id=project_id,
                chapter_id=int(chapter.id or 0),
                beat_index=1,
                content=str(chapter.content or "").strip() or str(chapter.title or "").strip(),
                status="derived",
            )
        ]

    candidates: list[dict] = []
    for beat in beats:
        text = normalize_memory_text(beat.content)
        if not text:
            continue
        episode_index = len(candidates) + 1
        candidates.append(
            {
                "project_id": int(project_id),
                "chapter_id": int(chapter.id or 0) or None,
                "scene_beat_id": int(beat.id or 0) or None,
                "episode_index": episode_index,
                "title": _episode_title(chapter, text, episode_index),
                "summary": _episode_summary(text),
                "event_type": _infer_event_type(text),
                "participants": match_character_names(text, character_profiles),
                "location": extract_location(text),
                "visibility": _infer_visibility(text),
                "importance": _infer_importance(text),
                "source_text_ref": {
                    "chapter_id": int(chapter.id or 0) or None,
                    "chapter_index": int(chapter.chapter_index),
                    "scene_beat_id": int(beat.id or 0) or None,
                    "beat_index": int(beat.beat_index),
                    "excerpt": text[:160],
                },
            }
        )
    return candidates
