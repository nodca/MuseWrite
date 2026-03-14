import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Callable


def _truncate_text(text: str, max_chars: int) -> str:
    content = (text or "").strip()
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip() + "..."


def _safe_iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    return None


def _freshness_days(value: Any) -> int | None:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    else:
        return None
    now = datetime.now(timezone.utc)
    delta = now - dt.astimezone(timezone.utc)
    return max(int(delta.total_seconds() // 86400), 0)


def _serialize(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def _extract_query_terms(user_input: str) -> list[str]:
    stop_words = {
        "请",
        "一下",
        "我们",
        "你们",
        "这个",
        "那个",
        "然后",
        "以及",
        "还有",
        "设定",
        "卡片",
        "角色",
        "剧情",
    }
    raw_tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", user_input or "")
    terms: list[str] = []
    for token in raw_tokens:
        if token in stop_words:
            continue
        if token not in terms:
            terms.append(token)
        # 对连续中文短语补充双字词，提升 DSL 命中稳定性（无分词器兜底）
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", token):
            for idx in range(0, len(token) - 1):
                gram = token[idx : idx + 2]
                if gram in stop_words or gram in terms:
                    continue
                terms.append(gram)
    return terms[:10]

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _score_term_hits(text: str, terms: list[str]) -> int:
    if not text or not terms:
        return 0
    return sum(1 for term in terms if term and term in text)


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    normalized = re.sub(r"\s+", "", text or "")
    if len(normalized) < n:
        return set()
    return {normalized[idx : idx + n] for idx in range(0, len(normalized) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _setting_value_text(row: Any) -> str:
    value_text = getattr(row, "value_text", None)
    if isinstance(value_text, str):
        return value_text
    return _serialize(getattr(row, "value", {}))


def _card_content_text(row: Any) -> str:
    content_text = getattr(row, "content_text", None)
    if isinstance(content_text, str):
        return content_text
    return _serialize(getattr(row, "content", {}))


def _setting_source_text(row: Any) -> str:
    return f"{getattr(row, 'key', '')}\n{_setting_value_text(row)}"


def _card_source_text(row: Any) -> str:
    return f"{getattr(row, 'title', '')}\n{_card_content_text(row)}"
