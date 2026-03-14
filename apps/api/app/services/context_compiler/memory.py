import math
import re
from typing import Any

from app.core.config import settings
from app.services.context_compiler._utils import _safe_int


def _memory_decay_factor(freshness_days: Any) -> float:
    try:
        days = max(float(freshness_days), 0.0)
    except Exception:
        return 1.0
    half_life = max(float(settings.memory_decay_half_life_days), 1.0)
    floor = max(min(float(settings.memory_decay_floor), 1.0), 0.0)
    decay = math.exp(-math.log(2.0) * days / half_life)
    return max(floor, min(1.0, decay))


def _apply_memory_decay(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decayed: list[dict[str, Any]] = []
    for item in hits:
        if not isinstance(item, dict):
            continue
        next_item = dict(item)
        freshness = next_item.get("freshness_days")
        factor = _memory_decay_factor(freshness)
        next_item["memory_decay"] = round(factor, 4)
        score_raw = next_item.get("score")
        confidence_raw = next_item.get("confidence")
        if isinstance(score_raw, (int, float)):
            next_item["score"] = round(float(score_raw) * factor, 4)
        if isinstance(confidence_raw, (int, float)):
            next_item["confidence"] = round(float(confidence_raw) * factor, 4)
        decayed.append(next_item)
    decayed.sort(
        key=lambda row: (
            float(row.get("score", row.get("confidence", 0.0)) or 0.0),
            -_safe_int(row.get("freshness_days"), 99999),
        ),
        reverse=True,
    )
    return decayed


def _build_spatial_graph(settings_rows: list[Any], cards_rows: list[Any]) -> tuple[dict[str, set[str]], dict[str, str]]:
    relation_keys = {item.lower() for item in settings.spatial_relation_keys}
    adjacency: dict[str, set[str]] = {}
    canonical_display: dict[str, str] = {}

    def _norm(value: str) -> str:
        return re.sub(r"\s+", "", str(value or "").strip()).lower()

    def _ensure(name: str) -> str:
        display = str(name or "").strip()
        if not display:
            return ""
        key = _norm(display)
        if not key:
            return ""
        canonical_display.setdefault(key, display)
        adjacency.setdefault(key, set())
        return key

    def _link(left: str, right: str) -> None:
        a = _ensure(left)
        b = _ensure(right)
        if not a or not b or a == b:
            return
        adjacency[a].add(b)
        adjacency[b].add(a)

    for row in cards_rows:
        title = str(getattr(row, "title", "") or "").strip()
        content = getattr(row, "content", {}) if isinstance(getattr(row, "content", {}), dict) else {}
        if not title:
            continue
        for key, raw in content.items():
            if str(key or "").strip().lower() not in relation_keys:
                continue
            if isinstance(raw, list):
                for item in raw:
                    _link(title, str(item))
            elif isinstance(raw, dict):
                for target in raw.keys():
                    _link(title, str(target))
            else:
                _link(title, str(raw))

    for row in settings_rows:
        key_text = str(getattr(row, "key", "") or "").strip()
        value = getattr(row, "value", {})
        if not key_text or not isinstance(value, dict):
            continue
        if "地点" not in key_text and "地理" not in key_text and "区域" not in key_text:
            continue
        for source, raw in value.items():
            source_name = str(source or "").strip()
            if not source_name:
                continue
            if isinstance(raw, list):
                for item in raw:
                    _link(source_name, str(item))
            elif isinstance(raw, dict):
                for target in raw.keys():
                    _link(source_name, str(target))
            else:
                _link(source_name, str(raw))

    return adjacency, canonical_display


def _build_spatial_distance_map(
    adjacency: dict[str, set[str]],
    *,
    current_location: str,
    max_hops: int,
) -> dict[str, int]:
    start = re.sub(r"\s+", "", str(current_location or "").strip()).lower()
    if not start or start not in adjacency:
        return {}
    visited: dict[str, int] = {start: 0}
    queue: list[str] = [start]
    while queue:
        node = queue.pop(0)
        depth = visited.get(node, 0)
        if depth >= max_hops:
            continue
        for nxt in adjacency.get(node, set()):
            if nxt in visited:
                continue
            visited[nxt] = depth + 1
            queue.append(nxt)
    return visited


def _hit_spatial_distance(hit: dict[str, Any], distance_map: dict[str, int]) -> int | None:
    if not distance_map:
        return None
    corpus = " ".join(
        [
            str(hit.get("title", "") or ""),
            str(hit.get("snippet", "") or ""),
            str(hit.get("fact", "") or ""),
        ]
    )
    normalized = re.sub(r"\s+", "", corpus).lower()
    if not normalized:
        return None
    distances: list[int] = []
    for token, distance in distance_map.items():
        if token and token in normalized:
            distances.append(distance)
    if not distances:
        return None
    return min(distances)


def _apply_spatial_penalty(
    hits: list[dict[str, Any]],
    *,
    current_location: str | None,
    settings_rows: list[Any],
    cards_rows: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not settings.spatial_enabled:
        return hits, {"enabled": False, "reason": "disabled"}
    location = str(current_location or "").strip()
    if not location:
        return hits, {"enabled": False, "reason": "missing_location"}

    adjacency, canonical_display = _build_spatial_graph(settings_rows, cards_rows)
    if not adjacency:
        return hits, {"enabled": False, "reason": "no_spatial_graph"}
    distance_map = _build_spatial_distance_map(
        adjacency,
        current_location=location,
        max_hops=max(int(settings.spatial_max_hops), 1),
    )
    if not distance_map:
        return hits, {"enabled": False, "reason": "location_not_found", "current_location": location}

    lam = max(float(settings.spatial_penalty_lambda), 0.0)
    adjusted: list[dict[str, Any]] = []
    penalized_count = 0
    for item in hits:
        if not isinstance(item, dict):
            continue
        distance = _hit_spatial_distance(item, distance_map)
        if distance is None:
            adjusted.append(dict(item))
            continue
        factor = math.exp(-lam * float(distance))
        next_item = dict(item)
        next_item["spatial_distance"] = distance
        next_item["spatial_factor"] = round(factor, 4)
        score_raw = next_item.get("score")
        conf_raw = next_item.get("confidence")
        if isinstance(score_raw, (int, float)):
            next_item["score"] = round(float(score_raw) * factor, 4)
        if isinstance(conf_raw, (int, float)):
            next_item["confidence"] = round(float(conf_raw) * factor, 4)
        if distance > 0:
            penalized_count += 1
        adjusted.append(next_item)

    adjusted.sort(
        key=lambda row: (
            float(row.get("score", row.get("confidence", 0.0)) or 0.0),
            -_safe_int(row.get("freshness_days"), 99999),
        ),
        reverse=True,
    )
    return adjusted, {
        "enabled": True,
        "reason": "ok",
        "current_location": location,
        "graph_nodes": len(adjacency),
        "reachable_nodes": len(distance_map),
        "penalized_hits": penalized_count,
        "penalty_lambda": lam,
        "known_locations": [canonical_display.get(key, key) for key in list(distance_map.keys())[:16]],
    }


def _split_memory_layers(settings_rows: list[Any]) -> tuple[list[Any], list[Any]]:
    semantic_prefix = str(settings.memory_semantic_key_prefix or "memory.semantic.volume.").strip()
    if not semantic_prefix:
        return settings_rows, []
    working: list[Any] = []
    semantic: list[Any] = []
    for row in settings_rows:
        key_text = str(getattr(row, "key", "") or "")
        if key_text.startswith(semantic_prefix):
            semantic.append(row)
        else:
            working.append(row)
    return working, semantic

