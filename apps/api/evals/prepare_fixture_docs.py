#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError(f"fixture row#{line_no} must be object")
            rows.append(parsed)
    if not rows:
        raise ValueError("fixture docs file is empty")
    return rows


def _api_path(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def _post_json(client: httpx.Client, url: str, body: dict[str, Any]) -> dict[str, Any]:
    resp = client.post(url, json=body)
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, dict) else {"data": payload}


def _delete_json(client: httpx.Client, url: str, body: dict[str, Any]) -> dict[str, Any]:
    resp = client.request("DELETE", url, json=body)
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, dict) else {"data": payload}


def _list_docs(
    client: httpx.Client,
    *,
    base_url: str,
    project_id: int,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    url = _api_path(base_url, f"/api/chat/projects/{project_id}/documents/paginated")
    payload = _post_json(
        client,
        url,
        {"page": 1, "page_size": page_size, "sort_field": "updated_at", "sort_direction": "desc"},
    )
    docs = payload.get("documents")
    if not isinstance(docs, list):
        return []
    return [item for item in docs if isinstance(item, dict)]


def _build_file_source(prefix: str, row: dict[str, Any], index: int) -> str:
    explicit = str(row.get("file_source") or "").strip()
    if explicit:
        return explicit
    file_name = str(row.get("file_name") or "").strip() or f"fixture-{index:03d}.txt"
    return prefix.rstrip("/") + "/" + file_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare LightRAG fixture docs for offline eval.")
    parser.add_argument(
        "--fixture-docs",
        default=str(Path(__file__).parent / "data" / "novel_rag_docs.jsonl"),
        help="fixture docs JSONL file",
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base url")
    parser.add_argument("--project-id", type=int, default=1, help="project id")
    parser.add_argument("--timeout-seconds", type=int, default=180, help="max wait for processing")
    parser.add_argument("--poll-interval-seconds", type=int, default=2, help="poll interval")
    parser.add_argument("--reset-existing", action="store_true", help="delete existing fixture docs under prefix")
    parser.add_argument(
        "--fixture-prefix",
        default="",
        help="file_source prefix (default: np://project/{project_id}/eval)",
    )
    args = parser.parse_args()

    fixture_path = Path(args.fixture_docs).resolve()
    rows = _read_jsonl(fixture_path)
    prefix = (args.fixture_prefix or "").strip() or f"np://project/{int(args.project_id)}/eval"

    timeout = httpx.Timeout(30.0)
    with httpx.Client(timeout=timeout) as client:
        if args.reset_existing:
            existing = _list_docs(client, base_url=args.base_url, project_id=args.project_id, page_size=100)
            stale_doc_ids = [
                str(item.get("doc_id") or "").strip()
                for item in existing
                if str(item.get("file_source") or "").startswith(prefix)
                and str(item.get("doc_id") or "").strip()
            ]
            if stale_doc_ids:
                delete_url = _api_path(args.base_url, f"/api/chat/projects/{args.project_id}/documents")
                _delete_json(
                    client,
                    delete_url,
                    {"doc_ids": stale_doc_ids, "delete_file": False, "delete_llm_cache": False},
                )
                print(f"[fixture] reset_existing deleted={len(stale_doc_ids)}")

        inserted_sources: list[str] = []
        insert_url = _api_path(args.base_url, f"/api/chat/projects/{args.project_id}/documents/text")
        for idx, row in enumerate(rows, start=1):
            text = str(row.get("text") or "").strip()
            if not text:
                print(f"[fixture] skip row#{idx}: empty text")
                continue
            file_source = _build_file_source(prefix, row, idx)
            _post_json(client, insert_url, {"text": text, "file_source": file_source})
            inserted_sources.append(file_source)
            print(f"[fixture] inserted {file_source}")

        if not inserted_sources:
            print("[fixture] no docs inserted")
            return 2

        deadline = time.time() + max(int(args.timeout_seconds), 1)
        pending = set(inserted_sources)
        failed: dict[str, str] = {}
        while pending and time.time() < deadline:
            time.sleep(max(int(args.poll_interval_seconds), 1))
            docs = _list_docs(client, base_url=args.base_url, project_id=args.project_id, page_size=100)
            by_source = {str(item.get("file_source") or ""): item for item in docs}
            for source in list(pending):
                row = by_source.get(source)
                if not row:
                    continue
                status = str(row.get("status") or "").lower()
                if status == "processed":
                    pending.remove(source)
                    print(f"[fixture] processed {source}")
                elif status == "failed":
                    pending.remove(source)
                    failed[source] = status
                    print(f"[fixture] failed {source}")

        if pending:
            print(f"[fixture] timeout waiting docs: {len(pending)}")
            for source in sorted(pending):
                print(f"[fixture] pending {source}")
            return 3
        if failed:
            print(f"[fixture] failed docs: {len(failed)}")
            for source in sorted(failed):
                print(f"[fixture] failed {source}")
            return 4

        print(f"[fixture] ready docs={len(inserted_sources)} prefix={prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
