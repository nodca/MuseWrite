import re
import time
from typing import Any

from app.core.config import settings
from app.services.context_compiler._utils import _truncate_text, _extract_query_terms
from app.services.context_compiler._constants import (
    _NEGATIVE_CONSTRAINT_MARKERS,
    _NEGATIVE_CONSTRAINT_STRONG_MARKERS,
    _NEGATIVE_CONSTRAINT_RELATION_MARKERS,
    _COMPRESSION_SOURCE_TIER_ORDER,
    _COMPRESSION_SECTION_BY_SOURCE,
)
from app.services.context_compiler.normalization import _normalize_context_compression_mode
from app.services.context_compiler.reranker import _score_lines_with_reranker


def _hit_preview_text(hit: dict[str, Any]) -> str:
    if not isinstance(hit, dict):
        return ""
    return str(
        hit.get("snippet")
        or hit.get("fact")
        or hit.get("content")
        or hit.get("text")
        or ""
    ).strip()


def _contains_negative_constraint_marker(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    lower = value.lower()
    for marker in _NEGATIVE_CONSTRAINT_MARKERS:
        token = str(marker or "").strip()
        if not token:
            continue
        if token in value or token.lower() in lower:
            return True
    return False


def _normalize_negative_constraint_text(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"^[\s\-*•#]+", "", value)
    value = re.sub(r"^\d+[.)、:\s]+", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return _truncate_text(value, 160)


def _negative_constraint_identity(text: str) -> str:
    normalized = re.sub(r"[\W_]+", "", str(text or "").lower())
    return normalized[:120]


def _extract_negative_constraints_from_text(
    text: str,
    *,
    relation_hint: bool,
    limit: int,
) -> list[str]:
    normalized = str(text or "").replace("\r", "\n").strip()
    if not normalized:
        return []
    segments = re.split(r"[。！？!?；;\n]+", normalized)
    extracted: list[str] = []
    seen: set[str] = set()
    for raw_segment in segments:
        segment = _normalize_negative_constraint_text(raw_segment)
        if not segment:
            continue
        if not _contains_negative_constraint_marker(segment):
            continue
        key = _negative_constraint_identity(segment)
        if not key or key in seen:
            continue
        seen.add(key)
        extracted.append(segment)
        if len(extracted) >= max(limit, 1):
            return extracted
    if extracted:
        return extracted
    if relation_hint and _contains_negative_constraint_marker(normalized):
        segment = _normalize_negative_constraint_text(normalized)
        if segment:
            return [segment]
    return []


def _negative_constraint_source_weight(source: str) -> float:
    source_name = str(source or "").strip().upper()
    if source_name == "DSL":
        return 0.35
    if source_name == "GRAPH":
        return 0.25
    if source_name == "RAG":
        return 0.15
    return 0.0


def _build_negative_constraints(
    *,
    user_input: str,
    dsl_hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
    limit: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started_at = time.perf_counter()
    terms = _extract_query_terms(user_input)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for source, rows in (("DSL", dsl_hits), ("GRAPH", graph_facts), ("RAG", semantic_hits)):
        for hit in rows[: max(limit * 4, 12)]:
            if not isinstance(hit, dict):
                continue
            parts = [
                str(hit.get("title") or ""),
                str(hit.get("fact") or ""),
                str(hit.get("snippet") or ""),
            ]
            corpus = " ".join(part for part in parts if part).strip()
            if not corpus:
                continue
            relation_hint = False
            if source == "GRAPH":
                upper_corpus = corpus.upper()
                relation_hint = any(token in upper_corpus for token in _NEGATIVE_CONSTRAINT_RELATION_MARKERS)
            constraints = _extract_negative_constraints_from_text(
                corpus,
                relation_hint=relation_hint,
                limit=3,
            )
            if not constraints:
                continue

            score_raw = hit.get("score", hit.get("confidence"))
            try:
                base_score = max(min(float(score_raw), 1.0), 0.0)
            except Exception:
                base_score = 0.35

            for text in constraints:
                key = _negative_constraint_identity(text)
                if not key or key in seen:
                    continue
                seen.add(key)
                score = base_score + _negative_constraint_source_weight(source)
                lower_text = text.lower()
                for term in terms:
                    token = str(term or "").strip().lower()
                    if token and token in lower_text:
                        score += 0.08
                if any(marker in text or marker in lower_text for marker in _NEGATIVE_CONSTRAINT_STRONG_MARKERS):
                    score += 0.12
                candidates.append(
                    {
                        "text": text,
                        "source": source,
                        "title": _truncate_text(str(hit.get("title") or hit.get("kind") or source), 64),
                        "kind": str(hit.get("kind") or "").strip(),
                        "score": round(max(min(score, 2.0), 0.0), 4),
                        "confidence": round(base_score, 4),
                    }
                )

    candidates.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            float(item.get("confidence", 0.0)),
        ),
        reverse=True,
    )
    selected = candidates[: max(limit, 1)]
    source_counts = {"DSL": 0, "GRAPH": 0, "RAG": 0}
    for item in selected:
        source_name = str(item.get("source") or "").strip().upper()
        if source_name in source_counts:
            source_counts[source_name] += 1
    elapsed_ms = max(int((time.perf_counter() - started_at) * 1000), 0)
    metadata = {
        "enabled": True,
        "count": len(selected),
        "sources": source_counts,
        "elapsed_ms": elapsed_ms,
    }
    return selected, metadata


def _build_context_compression_corpus(
    *,
    dsl_hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    lines: list[str] = []
    for source, rows in (("DSL", dsl_hits), ("GRAPH", graph_facts), ("RAG", semantic_hits)):
        for item in rows:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("kind") or source).strip()
            preview = _truncate_text(_hit_preview_text(item), 220)
            score_raw = item.get("score", item.get("confidence"))
            try:
                score = float(score_raw)
            except Exception:
                score = 0.0
            line = f"[{source}] {title} :: {preview} (score={score:.3f})"
            lines.append(line)
    joined = "\n".join(lines)
    return joined, lines


def _line_task_relevance_score(
    line: str,
    *,
    terms: list[str],
    intent: str,
) -> float:
    score = 0.0
    lower_line = line.lower()
    for term in terms:
        if term and term.lower() in lower_line:
            score += 1.0
    if f"[{intent}]".lower() in lower_line:
        score += 0.8
    if "[dsl]" in lower_line:
        score += 0.45
    if "[graph]" in lower_line:
        score += 0.35
    if "[rag]" in lower_line:
        score += 0.25
    return score


def _heuristic_context_compress(
    *,
    user_input: str,
    intent: str,
    lines: list[str],
    max_chars: int,
) -> str:
    if not lines:
        return ""
    terms = _extract_query_terms(user_input)
    ranked: list[tuple[float, int, str]] = []
    for idx, line in enumerate(lines):
        ranked.append((_line_task_relevance_score(line, terms=terms, intent=intent), idx, line))
    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    selected_lines: list[str] = []
    used_indexes: set[int] = set()
    for _, idx, line in ranked:
        if idx in used_indexes:
            continue
        used_indexes.add(idx)
        selected_lines.append(line)
        if len(selected_lines) >= 24:
            break

    chunks = _apply_source_priority_tiering(selected_lines, max_chars=max_chars)
    return "\n".join(chunks)


def _compression_line_source(line: str) -> str:
    match = re.match(r"^\[([A-Za-z]+)\]", str(line or "").strip())
    if not match:
        return "OTHER"
    return str(match.group(1) or "").strip().upper() or "OTHER"


def _apply_source_priority_tiering(lines: list[str], *, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return []
    buckets: dict[str, list[str]] = {source: [] for source in _COMPRESSION_SOURCE_TIER_ORDER}
    for line in lines:
        piece = str(line or "").strip()
        if not piece:
            continue
        source = _compression_line_source(piece)
        key = source if source in buckets else "OTHER"
        buckets[key].append(piece)

    selected: list[str] = []
    total = 0
    for source in _COMPRESSION_SOURCE_TIER_ORDER:
        for piece in buckets[source]:
            next_total = total + len(piece) + (1 if selected else 0)
            if next_total > max_chars:
                if not selected:
                    selected.append(_truncate_text(piece, max_chars))
                return selected
            selected.append(piece)
            total = next_total
    return selected


def _parse_compression_line(line: str) -> tuple[str, str, str]:
    raw_line = str(line or "").strip()
    if not raw_line:
        return "OTHER", "", ""
    source = _compression_line_source(raw_line)
    body = re.sub(r"^\[[^\]]+\]\s*", "", raw_line)
    body = re.sub(r"\s+\(score=[^)]+\)\s*$", "", body).strip()
    if "::" in body:
        title, preview = body.split("::", 1)
    elif ":" in body:
        title, preview = body.split(":", 1)
    else:
        title, preview = source, body
    return source, str(title or "").strip(), str(preview or "").strip()


def _format_compression_section_item(*, source: str, title: str, preview: str) -> str:
    source_label = {
        "DSL": "设定",
        "GRAPH": "图谱",
        "RAG": "回忆",
        "OTHER": "证据",
    }.get(source, "证据")
    title_text = re.sub(r"\s+", " ", str(title or "").strip())
    preview_text = re.sub(r"\s+", " ", str(preview or "").strip())
    title_text = _truncate_text(title_text, 60)
    preview_text = _truncate_text(preview_text, 180)
    if title_text and preview_text:
        return f"[{source_label}] {title_text}: {preview_text}"
    if preview_text:
        return f"[{source_label}] {preview_text}"
    if title_text:
        return f"[{source_label}] {title_text}"
    return ""


def _build_compression_sections(
    *,
    summary: str,
    dsl_hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {
        "fact_entities": [],
        "fact_relations": [],
        "retrieved_events": [],
    }
    seen: dict[str, set[str]] = {key: set() for key in sections}

    def _append(section_key: str, text: str, *, limit: int = 6) -> None:
        item = str(text or "").strip()
        if not item:
            return
        if section_key not in sections:
            return
        if item in seen[section_key]:
            return
        if len(sections[section_key]) >= max(limit, 1):
            return
        sections[section_key].append(item)
        seen[section_key].add(item)

    for line in str(summary or "").splitlines():
        source, title, preview = _parse_compression_line(line)
        section_key = _COMPRESSION_SECTION_BY_SOURCE.get(source)
        if not section_key:
            continue
        _append(
            section_key,
            _format_compression_section_item(source=source, title=title, preview=preview),
            limit=6,
        )

    source_rows = (
        ("DSL", dsl_hits, "fact_entities"),
        ("GRAPH", graph_facts, "fact_relations"),
        ("RAG", semantic_hits, "retrieved_events"),
    )
    for source, rows, section_key in source_rows:
        for item in rows:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("kind") or source).strip()
            preview = _hit_preview_text(item)
            _append(
                section_key,
                _format_compression_section_item(source=source, title=title, preview=preview),
                limit=4,
            )
            if len(sections[section_key]) >= 4:
                break

    if not any(sections.values()) and str(summary or "").strip():
        _append(
            "retrieved_events",
            _format_compression_section_item(
                source="OTHER",
                title="压缩摘要",
                preview=_truncate_text(str(summary), 220),
            ),
            limit=2,
        )
    return sections


def _context_compression_rerank_bias(source: str, budget_mode: str = "") -> float:
    source_name = str(source or "").strip().upper()
    mode = str(budget_mode or "").strip().lower()

    _DYNAMIC_BIAS: dict[str, dict[str, float]] = {
        "action":        {"DSL": 0.15, "GRAPH": 0.04, "RAG": 0.02},
        "investigation": {"DSL": 0.08, "GRAPH": 0.12, "RAG": 0.06},
        "world":         {"DSL": 0.10, "GRAPH": 0.06, "RAG": 0.10},
        "dialogue":      {"DSL": 0.06, "GRAPH": 0.04, "RAG": 0.04},
    }

    if mode in _DYNAMIC_BIAS:
        return _DYNAMIC_BIAS[mode].get(source_name, 0.0)

    # fallback: 读静态配置（balanced 或未知 mode）
    if source_name == "DSL":
        return float(settings.context_compression_reranker_dsl_bias)
    if source_name == "GRAPH":
        return float(settings.context_compression_reranker_graph_bias)
    if source_name == "RAG":
        return float(settings.context_compression_reranker_rag_bias)
    return 0.0


def _call_context_compressor_reranker(
    *,
    user_input: str,
    intent: str,
    lines: list[str],
    max_chars: int,
    budget_mode: str = "",
    telemetry: dict[str, Any] | None = None,
) -> str | None:
    started_at = time.perf_counter()
    if telemetry is not None:
        telemetry.clear()
        telemetry.update(
            {
                "attempted": True,
                "line_count": len(lines),
                "max_chars": max(int(max_chars), 0),
            }
        )
    if not lines:
        if telemetry is not None:
            telemetry["reason"] = "empty_lines"
            telemetry["elapsed_ms"] = max(int((time.perf_counter() - started_at) * 1000), 0)
        return None
    query = (
        "你是写作检索压缩器，请优先保留实体、设定、动作约束。\n"
        + f"intent={_truncate_text(intent, 64)}\n"
        + f"user_input={_truncate_text(user_input, 600)}"
    )
    raw_scores = _score_lines_with_reranker(query=query, lines=lines)
    if raw_scores is None:
        if telemetry is not None:
            telemetry["reason"] = "reranker_unavailable"
            telemetry["runtime"] = "remote_api"
            telemetry["elapsed_ms"] = max(int((time.perf_counter() - started_at) * 1000), 0)
        return None

    min_score = max(min(float(settings.context_compression_reranker_min_score), 1.0), 0.0)
    top_k = max(int(settings.context_compression_reranker_top_k), 1)
    min_dsl_keep = max(int(settings.context_compression_reranker_min_dsl_keep), 0)
    if telemetry is not None:
        telemetry["min_score"] = round(min_score, 4)
        telemetry["top_k"] = top_k
        telemetry["min_dsl_keep"] = min_dsl_keep
        telemetry["runtime"] = "remote_api"

    ranked: list[tuple[float, float, int, str, str]] = []
    for idx, line in enumerate(lines):
        source = _compression_line_source(line)
        base_score = raw_scores[idx]
        dsl_hard_bias = 0.2 if source == "DSL" else 0.0
        adjusted = max(
            0.0,
            min(base_score + _context_compression_rerank_bias(source, budget_mode) + dsl_hard_bias, 1.5),
        )
        ranked.append((adjusted, base_score, idx, source, line))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    threshold_hit_count = sum(1 for item in ranked if item[0] >= min_score)

    forced: list[tuple[float, float, int, str, str]] = [
        item for item in ranked if item[3] == "DSL"
    ][:min_dsl_keep]
    selected_map: dict[int, tuple[float, float, int, str, str]] = {
        item[2]: item for item in forced
    }
    for item in ranked:
        if item[0] < min_score:
            continue
        selected_map[item[2]] = item
        if len(selected_map) >= top_k:
            break
    if not selected_map:
        for item in ranked[:top_k]:
            selected_map[item[2]] = item

    selected_ranked = sorted(selected_map.values(), key=lambda item: (item[0], item[1]), reverse=True)[:top_k]
    selected = sorted(selected_ranked, key=lambda item: item[2])

    chunks = _apply_source_priority_tiering([item[4] for item in selected_ranked], max_chars=max_chars)
    if not chunks and selected_ranked:
        chunks.append(_truncate_text(str(selected_ranked[0][4]), max_chars))
    result = "\n".join(chunks).strip()
    if telemetry is not None:
        selected_sources: dict[str, int] = {}
        for _, _, _, source, _ in selected_ranked:
            selected_sources[source] = selected_sources.get(source, 0) + 1
        kept_sources: dict[str, int] = {}
        for line in chunks:
            source = _compression_line_source(line)
            kept_sources[source] = kept_sources.get(source, 0) + 1
        kept_count = len(chunks)
        telemetry["threshold_hit_count"] = threshold_hit_count
        telemetry["selected_count"] = len(selected_ranked)
        telemetry["kept_count"] = kept_count
        telemetry["dropped_count"] = max(len(lines) - kept_count, 0)
        telemetry["selected_sources"] = selected_sources
        telemetry["kept_sources"] = kept_sources
        telemetry["output_chars"] = len(result)
        telemetry["reason"] = "ok" if result else "empty_result"
        telemetry["elapsed_ms"] = max(int((time.perf_counter() - started_at) * 1000), 0)
    return result or None


def _build_context_compression(
    *,
    user_input: str,
    intent: str,
    dsl_hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
    budget_mode: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not settings.context_compression_enabled:
        return None, {"enabled": False, "applied": False, "reason": "disabled"}

    mode = _normalize_context_compression_mode(settings.context_compression_mode)
    if mode == "off":
        return None, {"enabled": True, "applied": False, "reason": "mode_off", "mode": mode}

    min_chars = max(int(settings.context_compression_min_chars), 200)
    max_chars = max(int(settings.context_compression_max_chars), 200)
    corpus, lines = _build_context_compression_corpus(
        dsl_hits=dsl_hits,
        graph_facts=graph_facts,
        semantic_hits=semantic_hits,
    )
    chars_before = len(corpus)
    if chars_before < min_chars:
        return None, {
            "enabled": True,
            "applied": False,
            "reason": "below_threshold",
            "mode": mode,
            "chars_before": chars_before,
            "chars_after": 0,
            "min_chars": min_chars,
            "max_chars": max_chars,
        }

    summary = ""
    source = "rerank"
    reranker_telemetry: dict[str, Any] | None = {}
    rerank_summary = _call_context_compressor_reranker(
        user_input=user_input,
        intent=intent,
        lines=lines,
        max_chars=max_chars,
        budget_mode=budget_mode,
        telemetry=reranker_telemetry,
    )
    if rerank_summary:
        summary = rerank_summary

    if not summary:
        summary = _heuristic_context_compress(
            user_input=user_input,
            intent=intent,
            lines=lines,
            max_chars=max_chars,
        )
        source = "heuristic_after_rerank"

    summary = summary.strip()
    if not summary:
        metadata = {
            "enabled": True,
            "applied": False,
            "reason": "empty_summary",
            "mode": mode,
            "source": source,
            "chars_before": chars_before,
            "chars_after": 0,
            "min_chars": min_chars,
            "max_chars": max_chars,
        }
        if reranker_telemetry:
            metadata["reranker"] = reranker_telemetry
        return None, metadata

    summary_text = _truncate_text(summary, max_chars)
    sections = _build_compression_sections(
        summary=summary_text,
        dsl_hits=dsl_hits,
        graph_facts=graph_facts,
        semantic_hits=semantic_hits,
    )
    compressed = {
        "intent": intent,
        "summary": summary_text,
        "source": source,
        "max_chars": max_chars,
        "resolver_order": ["DSL", "GRAPH", "RAG"],
        "sections": sections,
    }
    metadata = {
        "enabled": True,
        "applied": True,
        "reason": "ok",
        "mode": mode,
        "source": source,
        "chars_before": chars_before,
        "chars_after": len(summary_text),
        "min_chars": min_chars,
        "max_chars": max_chars,
        "line_count": len(lines),
        "priority_tiering": "DSL>GRAPH>RAG",
        "sections": {key: len(value) for key, value in sections.items()},
    }
    if reranker_telemetry:
        metadata["reranker"] = reranker_telemetry
    return compressed, metadata

