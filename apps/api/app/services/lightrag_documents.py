from __future__ import annotations

from typing import Any
from uuid import uuid4

import httpx

from app.core.config import settings


def _ensure_lightrag_enabled() -> None:
    if not settings.lightrag_enabled:
        raise RuntimeError("LIGHTRAG_ENABLED=false")
    if not settings.lightrag_base_url:
        raise RuntimeError("LIGHTRAG_BASE_URL is empty")


def _normalize_path(path: str, fallback: str) -> str:
    normalized = (path or "").strip() or fallback
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    return normalized


def _endpoint(path: str, fallback: str) -> str:
    normalized_path = _normalize_path(path, fallback)
    return settings.lightrag_base_url.rstrip("/") + normalized_path


def _auth_headers() -> tuple[dict[str, str], dict[str, str] | None]:
    headers: dict[str, str] = {}
    if settings.lightrag_api_key:
        headers["Authorization"] = f"Bearer {settings.lightrag_api_key}"
    params = {"api_key_header_value": settings.lightrag_api_key} if settings.lightrag_api_key else None
    return headers, params


def _request_json(
    method: str,
    *,
    path: str,
    fallback_path: str,
    json_body: dict[str, Any] | None = None,
    params_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ensure_lightrag_enabled()
    url = _endpoint(path, fallback_path)
    headers, params = _auth_headers()
    if params_override:
        params = {**(params or {}), **params_override}

    timeout = httpx.Timeout(float(settings.lightrag_timeout_seconds))
    with httpx.Client(timeout=timeout) as client:
        resp = client.request(method=method, url=url, json=json_body, headers=headers, params=params)
    if int(resp.status_code) >= 400:
        detail = resp.text.strip()
        raise RuntimeError(f"LightRAG request failed: {resp.status_code} {detail}")
    try:
        payload = resp.json()
    except Exception:
        payload = {"status": "ok", "raw_text": resp.text}
    return payload if isinstance(payload, dict) else {"status": "ok", "data": payload}


def _project_source_prefix(project_id: int) -> str:
    return f"np://project/{int(project_id)}/"


def _pick_doc_id(item: dict[str, Any]) -> str:
    for key in ("doc_id", "id", "document_id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)
    return ""


def _normalize_document(item: dict[str, Any]) -> dict[str, Any]:
    doc_id = _pick_doc_id(item)
    file_source = str(
        item.get("file_source")
        or item.get("file_path")
        or item.get("source")
        or item.get("path")
        or ""
    ).strip()
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        "doc_id": doc_id,
        "status": str(item.get("status") or item.get("pipeline_status") or ""),
        "track_id": str(item.get("track_id") or ""),
        "file_source": file_source,
        "metadata": metadata,
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "summary": str(item.get("summary") or item.get("content_summary") or ""),
    }


def _extract_documents(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = payload.get("data")
    if isinstance(data, dict):
        docs_raw = data.get("documents")
        pagination = data.get("pagination") if isinstance(data.get("pagination"), dict) else {}
        if isinstance(docs_raw, list):
            return [item for item in docs_raw if isinstance(item, dict)], pagination
    docs_fallback = payload.get("documents")
    if isinstance(docs_fallback, list):
        return [item for item in docs_fallback if isinstance(item, dict)], {}
    return [], {}


def insert_text_document(
    *,
    project_id: int,
    text: str,
    file_source: str | None = None,
) -> dict[str, Any]:
    content = (text or "").strip()
    if not content:
        raise ValueError("text is required")
    source = (file_source or "").strip()
    if not source:
        source = f"{_project_source_prefix(project_id)}manual-{uuid4().hex[:12]}.txt"

    payload = _request_json(
        "POST",
        path=settings.lightrag_documents_text_path,
        fallback_path="/documents/text",
        json_body={"text": content, "file_source": source},
    )
    return {
        "provider": "lightrag_native",
        "project_id": int(project_id),
        "file_source": source,
        "result": payload,
    }


def list_project_documents(
    *,
    project_id: int,
    page: int,
    page_size: int,
    status_filter: str | None = None,
    sort_field: str = "updated_at",
    sort_direction: str = "desc",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "page": max(int(page), 1),
        "page_size": max(int(page_size), 1),
        "sort_field": sort_field or "updated_at",
        "sort_direction": sort_direction or "desc",
    }
    if status_filter and status_filter.strip():
        body["status_filter"] = status_filter.strip()

    payload = _request_json(
        "POST",
        path=settings.lightrag_documents_paginated_path,
        fallback_path="/documents/paginated",
        json_body=body,
    )

    prefix = _project_source_prefix(project_id)
    documents_raw, pagination = _extract_documents(payload)
    normalized = [_normalize_document(item) for item in documents_raw]
    filtered: list[dict[str, Any]] = []
    for item in normalized:
        source = str(item.get("file_source") or "")
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        metadata_project = metadata.get("project_id")
        metadata_project_ok = str(metadata_project).strip() == str(project_id)
        if source.startswith(prefix) or metadata_project_ok:
            filtered.append(item)

    return {
        "provider": "lightrag_native",
        "project_id": int(project_id),
        "project_scope_prefix": prefix,
        "documents": filtered,
        "pagination": pagination,
        "page_scan_count": len(normalized),
        "project_hit_count": len(filtered),
        "raw_status": payload.get("status"),
        "raw_message": payload.get("message"),
    }


def delete_documents(
    *,
    doc_ids: list[str],
    delete_file: bool = False,
    delete_llm_cache: bool = False,
) -> dict[str, Any]:
    normalized = [str(item).strip() for item in doc_ids if str(item).strip()]
    if not normalized:
        raise ValueError("doc_ids is empty")

    body = {
        "doc_ids": normalized,
        "delete_file": bool(delete_file),
        "delete_llm_cache": bool(delete_llm_cache),
    }
    try:
        payload = _request_json(
            "DELETE",
            path=settings.lightrag_documents_delete_path,
            fallback_path="/documents/delete_document",
            json_body=body,
        )
    except RuntimeError as exc:
        # Compatibility fallback for older payload shapes.
        results: list[dict[str, Any]] = []
        for doc_id in normalized:
            single_payload = _request_json(
                "DELETE",
                path=settings.lightrag_documents_delete_path,
                fallback_path="/documents/delete_document",
                json_body={
                    "doc_id": doc_id,
                    "delete_file": bool(delete_file),
                    "delete_llm_cache": bool(delete_llm_cache),
                },
            )
            results.append({"doc_id": doc_id, "result": single_payload})
        return {
            "provider": "lightrag_native",
            "mode": "per_doc_fallback",
            "requested": len(normalized),
            "results": results,
            "fallback_reason": str(exc),
        }

    return {
        "provider": "lightrag_native",
        "mode": "batch",
        "requested": len(normalized),
        "result": payload,
    }


def get_pipeline_status() -> dict[str, Any]:
    try:
        return _request_json(
            "GET",
            path=settings.lightrag_documents_pipeline_status_path,
            fallback_path="/documents/pipeline_status",
        )
    except RuntimeError:
        return _request_json(
            "POST",
            path=settings.lightrag_documents_pipeline_status_path,
            fallback_path="/documents/pipeline_status",
        )
