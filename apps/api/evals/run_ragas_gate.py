#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid jsonl at line {line_no}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"invalid jsonl row at line {line_no}: must be object")
            rows.append(parsed)
    if not rows:
        raise ValueError("dataset is empty")
    return rows


def _parse_sse_payload(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    raw = line[5:].strip()
    if not raw:
        return None
    if raw == "[DONE]":
        return {"type": "done"}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _collect_contexts(evidence_event: dict[str, Any] | None) -> list[str]:
    if not isinstance(evidence_event, dict):
        return []
    sources = evidence_event.get("sources")
    if not isinstance(sources, dict):
        return []

    contexts: list[str] = []
    for source_key in ("dsl", "graph", "rag"):
        rows = sources.get(source_key)
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, dict):
                continue
            text = (
                item.get("snippet")
                or item.get("fact")
                or item.get("content")
                or item.get("text")
                or ""
            )
            text_norm = str(text).strip()
            if text_norm:
                contexts.append(text_norm)
    # Stable dedupe to keep context ordering deterministic.
    return list(dict.fromkeys(contexts))


def _resolve_subset(row: dict[str, Any]) -> str:
    subset = str(row.get("subset") or "").strip().lower()
    if subset:
        return subset
    pov_mode = str(row.get("pov_mode") or "global").strip().lower()
    return "character" if pov_mode == "character" else "global"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_averages(rows: list[dict[str, Any]]) -> dict[str, float]:
    averages: dict[str, float] = {}
    for name in METRIC_NAMES:
        values: list[float] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            parsed = _safe_float(row.get(name))
            if parsed is not None:
                values.append(parsed)
        if values:
            averages[name] = round(statistics.fmean(values), 4)
    return averages


def _gate_score_from_averages(averages: dict[str, float]) -> float:
    if not averages:
        raise RuntimeError("ragas returned no metric columns; check dataset and model configuration")
    return round(statistics.fmean(averages.values()), 4)


def _build_subset_scores(
    *,
    samples: list[dict[str, Any]],
    ragas_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for idx, sample in enumerate(samples):
        subset = str(sample.get("subset") or "global").strip().lower() or "global"
        row = ragas_rows[idx] if idx < len(ragas_rows) else {}
        if not isinstance(row, dict):
            row = {}
        grouped_rows.setdefault(subset, []).append(row)

    result: dict[str, dict[str, Any]] = {}
    for subset, rows in grouped_rows.items():
        averages = _metric_averages(rows)
        result[subset] = {
            "sample_count": len(rows),
            "averages": averages,
            "gate_score": round(statistics.fmean(averages.values()), 4) if averages else None,
        }
    return result


def _metric_delta(current_value: float | None, baseline_value: float | None) -> float | None:
    if current_value is None or baseline_value is None:
        return None
    return round(current_value - baseline_value, 4)


def _load_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _build_trend_summary(
    *,
    gate_score: float,
    averages: dict[str, float],
    subset_scores: dict[str, dict[str, Any]],
    baseline_payload: dict[str, Any] | None,
    baseline_path: Path | None,
) -> dict[str, Any]:
    baseline_label = str(baseline_path) if baseline_path else ""
    if not baseline_payload:
        return {
            "available": False,
            "baseline_report": baseline_label,
            "reason": "baseline_report_missing_or_invalid",
        }

    baseline_gate_score = _safe_float(baseline_payload.get("gate_score"))
    baseline_averages_raw = baseline_payload.get("averages")
    baseline_averages = baseline_averages_raw if isinstance(baseline_averages_raw, dict) else {}

    averages_delta: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        averages_delta[name] = _metric_delta(averages.get(name), _safe_float(baseline_averages.get(name)))

    baseline_subsets_raw = baseline_payload.get("subset_scores")
    baseline_subsets = baseline_subsets_raw if isinstance(baseline_subsets_raw, dict) else {}
    subset_delta: dict[str, dict[str, Any]] = {}
    subset_names = sorted(set(subset_scores.keys()) | set(baseline_subsets.keys()))
    for subset in subset_names:
        current_subset = subset_scores.get(subset) if isinstance(subset_scores.get(subset), dict) else {}
        baseline_subset = baseline_subsets.get(subset) if isinstance(baseline_subsets.get(subset), dict) else {}

        current_subset_averages_raw = current_subset.get("averages")
        current_subset_averages = (
            current_subset_averages_raw if isinstance(current_subset_averages_raw, dict) else {}
        )
        baseline_subset_averages_raw = baseline_subset.get("averages")
        baseline_subset_averages = (
            baseline_subset_averages_raw if isinstance(baseline_subset_averages_raw, dict) else {}
        )

        metric_delta: dict[str, float | None] = {}
        for name in METRIC_NAMES:
            metric_delta[name] = _metric_delta(
                _safe_float(current_subset_averages.get(name)),
                _safe_float(baseline_subset_averages.get(name)),
            )

        subset_delta[subset] = {
            "gate_score_delta": _metric_delta(
                _safe_float(current_subset.get("gate_score")),
                _safe_float(baseline_subset.get("gate_score")),
            ),
            "averages_delta": metric_delta,
        }

    return {
        "available": True,
        "baseline_report": baseline_label,
        "baseline_generated_at": baseline_payload.get("generated_at"),
        "gate_score_delta": _metric_delta(gate_score, baseline_gate_score),
        "averages_delta": averages_delta,
        "subset_delta": subset_delta,
    }


def _run_chat_stream(
    *,
    base_url: str,
    request_payload: dict[str, Any],
    timeout_seconds: int,
) -> tuple[str, dict[str, Any] | None]:
    url = base_url.rstrip("/") + "/api/chat/stream"
    answer_chunks: list[str] = []
    evidence_event: dict[str, Any] | None = None

    with httpx.Client(timeout=httpx.Timeout(timeout_seconds)) as client:
        with client.stream("POST", url, json=request_payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line is None:
                    continue
                payload = _parse_sse_payload(str(line))
                if not payload:
                    continue
                event_type = str(payload.get("type") or "")
                if event_type == "delta":
                    answer_chunks.append(str(payload.get("text") or ""))
                elif event_type == "evidence":
                    evidence_event = payload
                elif event_type == "done":
                    break

    return "".join(answer_chunks).strip(), evidence_event


def _build_eval_dataset(samples: list[dict[str, Any]]) -> Any:
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "question": [str(item["question"]) for item in samples],
            "answer": [str(item["answer"]) for item in samples],
            "contexts": [item["contexts"] for item in samples],
            "ground_truth": [str(item["ground_truth"]) for item in samples],
        }
    )


def _evaluate_with_ragas(dataset: Any) -> tuple[dict[str, float], float, list[dict[str, Any]]]:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(dataset=dataset, metrics=metrics, raise_exceptions=False)
    frame = result.to_pandas()
    ragas_rows = frame.to_dict(orient="records")

    averages = _metric_averages(ragas_rows)
    gate_score = _gate_score_from_averages(averages)
    return averages, gate_score, ragas_rows


def _write_report(
    *,
    output_path: Path,
    threshold: float,
    gate_score: float,
    averages: dict[str, float],
    subset_scores: dict[str, dict[str, Any]],
    trend_summary: dict[str, Any],
    ragas_rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    rag_mode: str,
    deterministic_first: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "gate_score": gate_score,
        "averages": averages,
        "subset_scores": subset_scores,
        "trend": trend_summary,
        "rag_mode": rag_mode,
        "deterministic_first": deterministic_first,
        "sample_count": len(samples),
        "items": [
            {
                "id": str(item.get("id") or idx + 1),
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": item["answer"],
                "contexts_count": len(item["contexts"]),
                "subset": item.get("subset"),
                "pov_mode": item.get("pov_mode"),
                "pov_anchor": item.get("pov_anchor"),
            }
            for idx, item in enumerate(samples)
        ],
        "ragas_rows": ragas_rows,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline RAGAS regression gate against /api/chat/stream")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "data" / "novel_rag_eval.jsonl"),
        help="JSONL dataset path",
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base url")
    parser.add_argument("--threshold", type=float, default=0.55, help="Gate threshold for average score")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="HTTP timeout per sample")
    parser.add_argument("--rag-mode", default="mix", help="rag_mode sent to chat stream")
    parser.add_argument("--deterministic-first", action="store_true", help="enable deterministic_first")
    parser.add_argument("--model", default="", help="optional model override")
    parser.add_argument(
        "--baseline-report",
        default=str(Path(__file__).parent / "data" / "novel_rag_baseline.json"),
        help="optional baseline report path for trend diff",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "out" / "ragas-report.json"),
        help="output report path",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    output_path = Path(args.output).resolve()

    try:
        rows = _read_jsonl(dataset_path)
    except Exception as exc:
        print(f"[eval] dataset error: {exc}")
        return 2

    samples: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        project_id = int(row.get("project_id", 1))
        question = str(row.get("question") or "").strip()
        ground_truth = str(row.get("ground_truth") or "").strip()
        if not question or not ground_truth:
            print(f"[eval] skip row#{idx}: question/ground_truth required")
            continue
        pov_mode = str(row.get("pov_mode") or "global").strip().lower() or "global"
        pov_anchor = row.get("pov_anchor") or None
        subset = _resolve_subset(row)

        payload: dict[str, Any] = {
            "project_id": project_id,
            "content": question,
            "session_id": None,
            "model": args.model or None,
            "pov_mode": pov_mode,
            "pov_anchor": pov_anchor,
            "rag_mode": args.rag_mode,
            "deterministic_first": bool(args.deterministic_first),
        }

        print(f"[eval] ({idx}/{len(rows)}) project={project_id} q={question}")
        try:
            answer, evidence = _run_chat_stream(
                base_url=args.base_url,
                request_payload=payload,
                timeout_seconds=args.timeout_seconds,
            )
        except Exception as exc:
            print(f"[eval] request failed row#{idx}: {exc}")
            return 3

        contexts = _collect_contexts(evidence)
        samples.append(
            {
                "id": row.get("id"),
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "subset": subset,
                "pov_mode": pov_mode,
                "pov_anchor": pov_anchor,
            }
        )

    if not samples:
        print("[eval] no valid samples")
        return 2

    try:
        dataset = _build_eval_dataset(samples)
        averages, gate_score, ragas_rows = _evaluate_with_ragas(dataset)
    except Exception as exc:
        print(f"[eval] ragas failed: {exc}")
        print("[eval] hint: install eval deps and configure LLM/embedding keys for ragas")
        return 4

    subset_scores = _build_subset_scores(samples=samples, ragas_rows=ragas_rows)
    baseline_path = Path(args.baseline_report).resolve() if args.baseline_report else None
    baseline_payload = _load_report(baseline_path) if baseline_path else None
    trend_summary = _build_trend_summary(
        gate_score=gate_score,
        averages=averages,
        subset_scores=subset_scores,
        baseline_payload=baseline_payload,
        baseline_path=baseline_path,
    )

    ragas_rows_with_meta: list[dict[str, Any]] = []
    for idx, row in enumerate(ragas_rows):
        row_payload = dict(row) if isinstance(row, dict) else {}
        if idx < len(samples):
            sample = samples[idx]
            row_payload["id"] = sample.get("id")
            row_payload["subset"] = sample.get("subset")
            row_payload["pov_mode"] = sample.get("pov_mode")
            row_payload["pov_anchor"] = sample.get("pov_anchor")
        ragas_rows_with_meta.append(row_payload)

    _write_report(
        output_path=output_path,
        threshold=float(args.threshold),
        gate_score=gate_score,
        averages=averages,
        subset_scores=subset_scores,
        trend_summary=trend_summary,
        ragas_rows=ragas_rows_with_meta,
        samples=samples,
        rag_mode=args.rag_mode,
        deterministic_first=bool(args.deterministic_first),
    )

    print(f"[eval] averages={averages}")
    print(f"[eval] subset_scores={subset_scores}")
    print(f"[eval] trend_available={trend_summary.get('available')}")
    print(f"[eval] gate_score={gate_score} threshold={args.threshold}")
    print(f"[eval] report={output_path}")
    if gate_score < float(args.threshold):
        print("[eval] gate=FAIL")
        return 1
    print("[eval] gate=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
