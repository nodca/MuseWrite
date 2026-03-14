#!/usr/bin/env python3
"""Book Memory OS evaluation runner.

Loads evaluation fixtures from ``evals/data/book_memory_eval.jsonl``
and scores the memory pipeline's ability to answer each question
correctly.

Usage::

    python -m evals.run_book_memory_eval [--db-url DATABASE_URL] [--verbose]

The runner operates in **offline mode** (no live LLM calls).  It tests
whether the structured memory pipeline returns the correct information
by checking that expected keywords appear in the assembled context.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Ensure app is importable ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

EVAL_DATA_PATH = Path(__file__).resolve().parent / "data" / "book_memory_eval.jsonl"

# ── Types ───────────────────────────────────────────────────────────


@dataclass
class EvalCase:
    id: str
    capability: str
    question: str
    expected: str
    chapter_scope: int | None = None


@dataclass
class EvalResult:
    case_id: str
    capability: str
    passed: bool
    expected: str
    got: str
    reason: str = ""


@dataclass
class EvalReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    results: list[EvalResult] = field(default_factory=list)
    by_capability: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, result: EvalResult) -> None:
        self.total += 1
        if result.passed:
            self.passed += 1
        else:
            self.failed += 1
        self.results.append(result)

        cap = result.capability
        if cap not in self.by_capability:
            self.by_capability[cap] = {"total": 0, "passed": 0, "failed": 0}
        self.by_capability[cap]["total"] += 1
        if result.passed:
            self.by_capability[cap]["passed"] += 1
        else:
            self.by_capability[cap]["failed"] += 1

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Book Memory Eval Report",
            f"=======================",
            f"Total: {self.total}  Passed: {self.passed}  Failed: {self.failed}  Rate: {self.pass_rate:.1%}",
            "",
            "By Capability:",
        ]
        for cap, counts in sorted(self.by_capability.items()):
            rate = counts["passed"] / counts["total"] if counts["total"] > 0 else 0.0
            lines.append(f"  {cap:25s}  {counts['passed']}/{counts['total']}  ({rate:.0%})")

        if self.failed > 0:
            lines.append("")
            lines.append("Failed Cases:")
            for r in self.results:
                if not r.passed:
                    lines.append(f"  [{r.case_id}] {r.reason}")

        return "\n".join(lines)


# ── Loader ──────────────────────────────────────────────────────────


def load_eval_cases(path: Path | None = None) -> list[EvalCase]:
    p = path or EVAL_DATA_PATH
    if not p.exists():
        logger.error("eval data not found: %s", p)
        return []

    cases: list[EvalCase] = []
    for line in p.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            cases.append(
                EvalCase(
                    id=obj["id"],
                    capability=obj["capability"],
                    question=obj["question"],
                    expected=obj["expected"],
                    chapter_scope=obj.get("chapter_scope"),
                )
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("skipping invalid eval line: %s (%s)", line[:80], exc)

    return cases


# ── Evaluator ───────────────────────────────────────────────────────


def _keyword_match(expected: str, context: str) -> bool:
    """Check if key terms from expected answer appear in context."""
    import re

    # Extract 2+ char terms from expected.
    cleaned = re.sub(r"[，。！？、；：\s]+", " ", expected)
    tokens = [t.strip().lower() for t in cleaned.split() if len(t.strip()) >= 2]
    if not tokens:
        return True  # No meaningful tokens to check.

    context_lower = context.lower()
    matched = sum(1 for t in tokens if t in context_lower)
    return matched >= len(tokens) * 0.5  # At least 50% of tokens match.


def evaluate_case_offline(
    case: EvalCase,
    *,
    db_session: Any = None,
    project_id: int = 1,
) -> EvalResult:
    """Evaluate a single case by checking the memory pipeline output.

    This is an **offline** evaluation — no LLM calls.  It assembles
    memory context and checks if the expected answer's key terms
    appear in the assembled context.
    """
    if db_session is None:
        return EvalResult(
            case_id=case.id,
            capability=case.capability,
            passed=False,
            expected=case.expected,
            got="",
            reason="no_db_session",
        )

    try:
        from app.services.book_memory.memory_pipeline import build_planning_memory_context

        ctx = build_planning_memory_context(
            db_session,
            project_id=project_id,
            chapter_id=case.chapter_scope or 10,
            chapter_index=case.chapter_scope,
        )
        prompt_text = ctx.to_prompt_text()
    except Exception as exc:
        return EvalResult(
            case_id=case.id,
            capability=case.capability,
            passed=False,
            expected=case.expected,
            got="",
            reason=f"pipeline_error: {exc}",
        )

    passed = _keyword_match(case.expected, prompt_text)
    return EvalResult(
        case_id=case.id,
        capability=case.capability,
        passed=passed,
        expected=case.expected,
        got=prompt_text[:200] if not passed else "(matched)",
        reason="" if passed else "keyword_mismatch",
    )


# ── CLI ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Book Memory OS Evaluation")
    parser.add_argument("--db-url", type=str, default=None, help="Database URL")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--project-id", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cases = load_eval_cases()
    if not cases:
        print("No eval cases loaded.")
        sys.exit(1)

    print(f"Loaded {len(cases)} eval cases")
    print(f"Capabilities: {sorted(set(c.capability for c in cases))}")

    report = EvalReport()

    if args.db_url:
        from sqlmodel import Session, create_engine

        engine = create_engine(args.db_url)
        with Session(engine) as db:
            for case in cases:
                result = evaluate_case_offline(case, db_session=db, project_id=args.project_id)
                report.add(result)
                if args.verbose:
                    status = "PASS" if result.passed else "FAIL"
                    print(f"  [{status}] {case.id}: {case.question[:50]}")
    else:
        print("No --db-url provided. Running in dry-run mode (all cases marked as no_db).")
        for case in cases:
            result = EvalResult(
                case_id=case.id,
                capability=case.capability,
                passed=False,
                expected=case.expected,
                got="",
                reason="dry_run",
            )
            report.add(result)

    print()
    print(report.summary())

    sys.exit(0 if report.pass_rate >= 0.5 else 1)


if __name__ == "__main__":
    main()
