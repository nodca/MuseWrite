from typing import Any, Callable

from app.services.context_compiler._utils import (
    _truncate_text,
    _safe_iso,
    _freshness_days,
    _serialize,
    _score_term_hits,
    _char_ngrams,
    _jaccard,
    _setting_value_text,
    _card_content_text,
    _setting_source_text,
    _card_source_text,
)


def _window_retrieval_rows(
    rows: list[Any],
    *,
    limit: int,
    terms: list[str],
    source_text_getter: Callable[[Any], str],
) -> list[Any]:
    max_items = max(int(limit), 0)
    if max_items <= 0:
        return []
    if len(rows) <= max_items:
        return rows

    selected_indexes: set[int] = set()
    if terms:
        ranked: list[tuple[int, int, int]] = []
        for idx, row in enumerate(rows):
            score = _score_term_hits(source_text_getter(row), terms)
            if score <= 0:
                continue
            row_id = int(getattr(row, "id", 0) or 0)
            ranked.append((score, row_id, idx))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, _, idx in ranked:
            selected_indexes.add(idx)
            if len(selected_indexes) >= max_items:
                break

    if len(selected_indexes) < max_items:
        for idx in range(len(rows) - 1, -1, -1):
            selected_indexes.add(idx)
            if len(selected_indexes) >= max_items:
                break

    ordered_indexes = sorted(selected_indexes)
    return [rows[idx] for idx in ordered_indexes]


def _apply_pov_filter(settings_rows: list[Any], cards_rows: list[Any], mode: str, anchor: str | None) -> tuple[list[Any], list[Any]]:
    if mode != "character" or not anchor:
        return settings_rows, cards_rows

    anchor_lower = anchor.lower()

    filtered_settings = []
    for row in settings_rows:
        text = _setting_source_text(row).lower()
        if anchor_lower in text:
            filtered_settings.append(row)

    filtered_cards = []
    for row in cards_rows:
        text = _card_source_text(row).lower()
        if anchor_lower in text:
            filtered_cards.append(row)

    if filtered_settings or filtered_cards:
        return filtered_settings or settings_rows, filtered_cards or cards_rows
    return settings_rows, cards_rows


def _build_dsl_hits(
    terms: list[str],
    settings_rows: list[Any],
    cards_rows: list[Any],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for row in settings_rows:
        source_text = _setting_source_text(row)
        score = _score_term_hits(source_text, terms)
        if score <= 0:
            continue
        hits.append(
            {
                "kind": "setting",
                "id": row.id,
                "project_id": getattr(row, "project_id", None),
                "title": row.key,
                "score": score,
                "snippet": _truncate_text(source_text, 180),
                "confidence": 0.95,
                "updated_at": _safe_iso(getattr(row, "updated_at", None)),
                "freshness_days": _freshness_days(getattr(row, "updated_at", None)),
            }
        )

    for row in cards_rows:
        source_text = _card_source_text(row)
        score = _score_term_hits(source_text, terms)
        if score <= 0:
            continue
        hits.append(
            {
                "kind": "card",
                "id": row.id,
                "project_id": getattr(row, "project_id", None),
                "title": row.title,
                "score": score,
                "snippet": _truncate_text(source_text, 180),
                "confidence": 0.9,
                "updated_at": _safe_iso(getattr(row, "updated_at", None)),
                "freshness_days": _freshness_days(getattr(row, "updated_at", None)),
            }
        )

    hits.sort(
        key=lambda item: (
            int(item.get("score", 0)),
            -(int(item.get("freshness_days")) if isinstance(item.get("freshness_days"), int) else 99999),
        ),
        reverse=True,
    )
    return hits[: max(int(limit), 1)]


def _build_graph_facts(
    cards_rows: list[Any],
    settings_rows: list[Any],
    anchor: str | None,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    relation_keys = {
        "relationship",
        "relationships",
        "relation",
        "faction",
        "affiliation",
        "motivation",
        "goal",
        "status",
        "secret",
    }

    facts: list[dict[str, Any]] = []

    for row in cards_rows:
        content = row.content if isinstance(row.content, dict) else {}
        for key in relation_keys:
            if key not in content:
                continue
            value_text = _serialize(content.get(key))
            if not value_text.strip():
                continue
            line = f"{row.title} / {key}: {value_text}"
            if anchor and anchor not in line:
                continue
            facts.append(
                {
                    "kind": "card_fact",
                    "id": row.id,
                    "project_id": getattr(row, "project_id", None),
                    "title": row.title,
                    "fact": _truncate_text(line, 180),
                    "confidence": 0.6,
                    "updated_at": _safe_iso(getattr(row, "updated_at", None)),
                    "freshness_days": _freshness_days(getattr(row, "updated_at", None)),
                }
            )
            if len(facts) >= max(int(limit), 1):
                return facts

    for row in settings_rows:
        key_text = (row.key or "").strip()
        if not key_text:
            continue
        if not any(marker in key_text for marker in ("关系", "阵营", "势力", "组织", "规则")):
            continue
        line = f"{key_text}: {_setting_value_text(row)}"
        if anchor and anchor not in line:
            continue
        facts.append(
            {
                "kind": "setting_fact",
                "id": row.id,
                "project_id": getattr(row, "project_id", None),
                "title": key_text,
                "fact": _truncate_text(line, 180),
                "confidence": 0.65,
                "updated_at": _safe_iso(getattr(row, "updated_at", None)),
                "freshness_days": _freshness_days(getattr(row, "updated_at", None)),
            }
        )
        if len(facts) >= max(int(limit), 1):
            break

    return facts[: max(int(limit), 1)]


def _build_semantic_hits(
    user_input: str,
    settings_rows: list[Any],
    cards_rows: list[Any],
    anchor: str | None,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    query_ngrams = _char_ngrams(user_input or "")
    if not query_ngrams:
        return []

    ranked: list[dict[str, Any]] = []
    for row in settings_rows:
        source = _setting_source_text(row)
        if anchor and anchor not in source:
            continue
        score = _jaccard(query_ngrams, _char_ngrams(source))
        if score <= 0:
            continue
        ranked.append(
            {
                "kind": "setting",
                "id": row.id,
                "project_id": getattr(row, "project_id", None),
                "title": row.key,
                "score": round(score, 4),
                "snippet": _truncate_text(source, 180),
                "confidence": round(score, 4),
                "updated_at": _safe_iso(getattr(row, "updated_at", None)),
                "freshness_days": _freshness_days(getattr(row, "updated_at", None)),
            }
        )

    for row in cards_rows:
        source = _card_source_text(row)
        if anchor and anchor not in source:
            continue
        score = _jaccard(query_ngrams, _char_ngrams(source))
        if score <= 0:
            continue
        ranked.append(
            {
                "kind": "card",
                "id": row.id,
                "project_id": getattr(row, "project_id", None),
                "title": row.title,
                "score": round(score, 4),
                "snippet": _truncate_text(source, 180),
                "confidence": round(score, 4),
                "updated_at": _safe_iso(getattr(row, "updated_at", None)),
                "freshness_days": _freshness_days(getattr(row, "updated_at", None)),
            }
        )

    ranked.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            -(int(item.get("freshness_days")) if isinstance(item.get("freshness_days"), int) else 99999),
        ),
        reverse=True,
    )
    return ranked[: max(int(limit), 1)]

