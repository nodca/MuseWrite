from __future__ import annotations

import re
from typing import Any, Sequence

from app.models.book_memory import CharacterProfile, StoryStateSnapshot
from app.models.content import ChapterSceneBeat, ProjectChapter
from app.services.book_memory.episode_extractor import (
    extract_story_episode_candidates,
    extract_location,
    match_character_names,
    normalize_memory_text,
)
from app.services.book_memory.extraction_service import (
    BookMemoryStructuredExtraction,
    extract_book_memory_structured,
)

_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+")
_GOAL_PATTERNS = (
    re.compile(r"(?:决定|必须|要|想要|试图|准备|开始|继续)([^。！？!?；;，,]{2,24})"),
    re.compile(r"(?:目标|任务)[：: ]?([^。！？!?；;，,]{2,24})"),
)
_QUESTION_PATTERNS = ("谁", "为何", "为什么", "真相", "是否", "能否", "到底", "幕后")
_CONFLICT_HINTS = ("对峙", "冲突", "交锋", "追杀", "追查", "怀疑", "背叛", "威胁", "阻止", "危机", "争执")
_KNOWLEDGE_PATTERNS = (
    "得知",
    "知道",
    "发现",
    "察觉",
    "意识到",
    "看见",
    "看到",
    "听说",
    "听闻",
    "明白",
    "确认",
    "获悉",
)
_NEGATIVE_KNOWLEDGE_PATTERNS = ("不知", "不知情", "并不知", "未曾得知", "尚未知", "还不知道")


def _sentence_list(text: str) -> list[str]:
    return [segment.strip(" ，。；：:、") for segment in _SENTENCE_RE.findall(text) if segment.strip(" ，。；：:、")]


def _pick_chapter_goal(text: str, chapter: ProjectChapter, previous_snapshot: StoryStateSnapshot | None) -> str:
    for pattern in _GOAL_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        goal = str(match.group(1) or "").strip(" ，。；：:、")
        if goal:
            return goal[:120]
    chapter_title = str(chapter.title or "").strip()
    if chapter_title:
        return chapter_title[:120]
    if previous_snapshot is not None:
        return str(previous_snapshot.chapter_goal or "").strip()[:120]
    return ""


def _collect_active_characters(
    *,
    text: str,
    episodes: Sequence[dict[str, Any]] | None,
    character_profiles: Sequence[CharacterProfile] | None,
    previous_snapshot: StoryStateSnapshot | None,
) -> list[str]:
    matched = match_character_names(text, character_profiles)
    for episode in list(episodes or []):
        for name in list(episode.get("participants") or []):
            normalized = str(name or "").strip()
            if normalized and normalized not in matched:
                matched.append(normalized)
    if matched:
        return matched
    if previous_snapshot is not None:
        return list(previous_snapshot.active_characters or [])
    return []


def _pick_location(
    *,
    text: str,
    episodes: Sequence[dict[str, Any]] | None,
    previous_snapshot: StoryStateSnapshot | None,
) -> str:
    direct = extract_location(text)
    if direct:
        return direct
    for episode in list(episodes or []):
        location = str(episode.get("location") or "").strip()
        if location:
            return location[:64]
    if previous_snapshot is not None:
        return str(previous_snapshot.current_location or "").strip()[:64]
    return ""


def _collect_active_conflicts(text: str) -> list[str]:
    conflicts: list[str] = []
    for sentence in _sentence_list(text):
        if any(token in sentence for token in _CONFLICT_HINTS):
            conflicts.append(sentence[:120])
    return conflicts[:5]


def _collect_open_questions(text: str) -> list[str]:
    questions: list[str] = []
    normalized = normalize_memory_text(text)
    for sentence in _sentence_list(normalized):
        if "？" in sentence or "?" in sentence or any(token in sentence for token in _QUESTION_PATTERNS):
            questions.append(sentence[:120])
    if not questions:
        question_segments = re.findall(r"([^。！？!?]{0,80}[？?])", normalized)
        for item in question_segments:
            value = str(item or "").strip(" ，。；：:、")
            if value:
                questions.append(value[:120])
    deduped: list[str] = []
    for item in questions:
        if item not in deduped:
            deduped.append(item)
    return deduped[:5]


def _knowledge_value_key(fragment: str) -> str:
    value = re.sub(r"\s+", "_", fragment.strip(" ，。；：:、"))
    return f"knows:{value[:48]}" if value else ""


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


def _profile_alias_lookup(character_profiles: Sequence[CharacterProfile] | None = None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for profile in list(character_profiles or []):
        canonical = str(profile.canonical_name or "").strip()
        if not canonical:
            continue
        lookup[canonical] = canonical
        lookup[canonical.lower()] = canonical
        for alias in list(profile.aliases or []):
            normalized = str(alias or "").strip()
            if normalized:
                lookup[normalized] = canonical
                lookup[normalized.lower()] = canonical
    return lookup


def _normalize_name_list(
    raw_names: Sequence[str] | None,
    *,
    character_profiles: Sequence[CharacterProfile] | None,
) -> list[str]:
    lookup = _profile_alias_lookup(character_profiles)
    names: list[str] = []
    for item in list(raw_names or []):
        token = str(item or "").strip()
        if not token:
            continue
        canonical = lookup.get(token) or lookup.get(token.lower()) or token
        if canonical not in names:
            names.append(canonical)
    return names


def _normalize_text_list(values: Sequence[str] | None, *, limit: int) -> list[str]:
    normalized: list[str] = []
    for item in list(values or []):
        token = str(item or "").strip(" ，。；：:、")
        if not token or token in normalized:
            continue
        normalized.append(token[:120])
        if len(normalized) >= limit:
            break
    return normalized


def compile_character_knowledge_updates(
    *,
    chapter: ProjectChapter,
    character_profiles: Sequence[CharacterProfile] | None,
    episodes: Sequence[dict[str, Any]] | None = None,
    scene_beats: Sequence[ChapterSceneBeat] | None = None,
    extraction_result: BookMemoryStructuredExtraction | dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not character_profiles:
        return []

    structured = _coerce_structured_extraction(extraction_result)
    if structured is None:
        structured = extract_book_memory_structured(
            project_id=int(chapter.project_id),
            chapter=chapter,
            scene_beats=scene_beats,
            character_profiles=character_profiles,
            previous_snapshot=None,
        )

    if structured is not None and structured.knowledge_claims:
        alias_lookup = _profile_alias_lookup(character_profiles)
        episode_scene_map: dict[int, int | None] = {}
        if structured.episodes:
            for idx, episode in enumerate(structured.episodes, start=1):
                episode_scene_map[idx] = int(episode.scene_beat_id or 0) or None
        knowledge_updates: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        for claim in structured.knowledge_claims:
            if not bool(claim.known):
                continue
            character_name = alias_lookup.get(str(claim.character_name or "").strip()) or alias_lookup.get(
                str(claim.character_name or "").strip().lower()
            )
            if not character_name:
                continue
            fact = str(claim.fact or "").strip(" ，。；：:、")
            if not fact:
                continue
            knowledge_key = _knowledge_value_key(fact)
            if not knowledge_key or (character_name, knowledge_key) in seen_keys:
                continue
            seen_keys.add((character_name, knowledge_key))
            episode_index = int(claim.source_episode_index or 0) or None
            knowledge_updates.append(
                {
                    "character_name": character_name,
                    "knowledge_key": knowledge_key,
                    "knowledge_value": {
                        "fact": fact[:120],
                        "marker": "llm_structured_extraction",
                        "evidence": fact[:160],
                    },
                    "gained_at_chapter": int(chapter.chapter_index),
                    "lost_at_chapter": None,
                    "source_episode_index": episode_index,
                    "scene_beat_id": episode_scene_map.get(episode_index or 0),
                    "confidence": max(0.0, min(float(claim.confidence or 0.8), 1.0)),
                }
            )
        if knowledge_updates:
            return knowledge_updates

    candidate_texts: list[tuple[str, dict[str, Any]]] = []
    for episode in list(episodes or []):
        summary = normalize_memory_text(episode.get("summary"))
        if summary:
            candidate_texts.append((summary, episode))

    if not candidate_texts:
        for beat in sorted(list(scene_beats or []), key=lambda item: (int(item.beat_index), int(item.id or 0))):
            content = normalize_memory_text(beat.content)
            if content:
                candidate_texts.append(
                    (
                        content,
                        {
                            "episode_index": int(beat.beat_index),
                            "scene_beat_id": int(beat.id or 0) or None,
                            "participants": [],
                        },
                    )
                )

    knowledge_updates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    chapter_scope = int(chapter.chapter_index)

    for text, source in candidate_texts:
        for profile in character_profiles:
            canonical_name = str(profile.canonical_name or "").strip()
            if not canonical_name:
                continue
            aliases = [canonical_name, *[str(alias or "").strip() for alias in list(profile.aliases or [])]]
            matched_alias = next((alias for alias in aliases if alias and alias in text), "")
            if not matched_alias:
                continue
            if any(
                re.search(rf"{re.escape(matched_alias)}[^。！？!?；;，,]{{0,8}}{re.escape(token)}", text)
                for token in _NEGATIVE_KNOWLEDGE_PATTERNS
            ):
                continue
            for marker in _KNOWLEDGE_PATTERNS:
                pattern = re.compile(
                    rf"{re.escape(matched_alias)}[^。！？!?；;，,]{{0,12}}{re.escape(marker)}([^。！？!?；;，,]{{2,40}})"
                )
                match = pattern.search(text)
                if not match:
                    continue
                fragment = str(match.group(1) or "").strip(" ，。；：:、")
                if not fragment:
                    continue
                knowledge_key = _knowledge_value_key(fragment)
                if not knowledge_key:
                    continue
                identity = (canonical_name, knowledge_key)
                if identity in seen_keys:
                    continue
                seen_keys.add(identity)
                knowledge_updates.append(
                    {
                        "character_name": canonical_name,
                        "knowledge_key": knowledge_key,
                        "knowledge_value": {
                            "fact": fragment[:120],
                            "marker": marker,
                            "evidence": text[:160],
                        },
                        "gained_at_chapter": chapter_scope,
                        "lost_at_chapter": None,
                        "source_episode_index": int(source.get("episode_index") or 0) or None,
                        "scene_beat_id": int(source.get("scene_beat_id") or 0) or None,
                        "confidence": 0.9,
                    }
                )
                break
    return knowledge_updates


def compile_story_state_payload(
    *,
    project_id: int,
    chapter: ProjectChapter,
    scene_beat: ChapterSceneBeat | None = None,
    scene_beats: Sequence[ChapterSceneBeat] | None = None,
    character_profiles: Sequence[CharacterProfile] | None = None,
    previous_snapshot: StoryStateSnapshot | None = None,
    episodes: Sequence[dict[str, Any]] | None = None,
    extraction_result: BookMemoryStructuredExtraction | dict[str, Any] | None = None,
) -> dict[str, Any]:
    structured = _coerce_structured_extraction(extraction_result)
    scoped_beats = scene_beats if scene_beat is None else [scene_beat]
    if structured is None:
        structured = extract_book_memory_structured(
            project_id=project_id,
            chapter=chapter,
            scene_beats=scoped_beats,
            character_profiles=character_profiles,
            previous_snapshot=previous_snapshot,
        )

    normalized_episodes = list(episodes or [])
    if not normalized_episodes and structured is not None:
        normalized_episodes = extract_story_episode_candidates(
            project_id=project_id,
            chapter=chapter,
            scene_beats=scoped_beats,
            character_profiles=character_profiles,
            previous_snapshot=previous_snapshot,
            extraction_result=structured,
        )

    if scene_beat is not None:
        texts = [normalize_memory_text(scene_beat.content)]
        source_scene_beat_id = int(scene_beat.id or 0) or None
    else:
        ordered_beats = sorted(list(scene_beats or []), key=lambda item: (int(item.beat_index), int(item.id or 0)))
        texts = [normalize_memory_text(beat.content) for beat in ordered_beats if normalize_memory_text(beat.content)]
        source_scene_beat_id = None

    if not texts:
        texts = [normalize_memory_text(chapter.content) or normalize_memory_text(chapter.title)]

    combined_text = " ".join(item for item in texts if item).strip()
    fallback_active_characters = _collect_active_characters(
        text=combined_text,
        episodes=normalized_episodes,
        character_profiles=character_profiles,
        previous_snapshot=previous_snapshot,
    )
    active_characters = (
        _normalize_name_list(getattr(structured, "active_characters", []), character_profiles=character_profiles)
        if structured is not None
        else []
    ) or fallback_active_characters
    knowledge_updates = compile_character_knowledge_updates(
        chapter=chapter,
        character_profiles=character_profiles,
        episodes=normalized_episodes,
        scene_beats=scoped_beats,
        extraction_result=structured,
    )
    fallback_location = _pick_location(
        text=combined_text,
        episodes=normalized_episodes,
        previous_snapshot=previous_snapshot,
    )
    fallback_active_conflicts = _collect_active_conflicts(combined_text)
    fallback_open_questions = _collect_open_questions(combined_text)

    return {
        "project_id": int(project_id),
        "volume_id": int(chapter.volume_id or 0) or None,
        "chapter_id": int(chapter.id or 0) or None,
        "scene_beat_id": source_scene_beat_id,
        "chapter_goal": (
            str(getattr(structured, "chapter_goal", "") or "").strip()
            if structured is not None
            else ""
        )
        or _pick_chapter_goal(combined_text, chapter, previous_snapshot),
        "active_characters": active_characters,
        "current_location": (str(getattr(structured, "current_location", "") or "").strip() if structured is not None else "")
        or fallback_location,
        "active_conflicts": (
            _normalize_text_list(getattr(structured, "active_conflicts", []), limit=5)
            if structured is not None
            else []
        )
        or fallback_active_conflicts,
        "open_questions": (
            _normalize_text_list(getattr(structured, "open_questions", []), limit=5)
            if structured is not None
            else []
        )
        or fallback_open_questions,
        "source_refs": [
            {
                "chapter_id": int(chapter.id or 0) or None,
                "chapter_index": int(chapter.chapter_index),
                "scene_beat_id": source_scene_beat_id,
                "episode_count": len(normalized_episodes),
            }
        ],
        "knowledge_updates": knowledge_updates,
    }
