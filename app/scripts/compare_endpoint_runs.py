"""Score prototype and current endpoint runs against the same Oracle results."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.clients.mysql_client_manager import dw_mysql_client_manager
from app.evaluation.cases import EvalCase, load_eval_cases
from app.evaluation.endpoint_correctness import score_endpoint_result
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.scripts.run_eval import _run_oracle


async def compare_endpoint_runs(
    *,
    cases_path: Path,
    prototype_path: Path,
    current_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    cases = load_eval_cases(cases_path)
    prototype_report = _load_json(prototype_path)
    current_report = _load_json(current_path)
    prototype_by_id = _index_results(prototype_report)
    current_by_id = _index_results(current_report)
    rows: list[dict[str, Any]] = []

    dw_mysql_client_manager.init()
    try:
        async with dw_mysql_client_manager.session_factory() as session:
            repository = DWMySQLRepository(session)
            for case in cases:
                prototype = prototype_by_id.get(case.id)
                current = current_by_id.get(case.id)
                if prototype is None or current is None:
                    continue
                if case.expected_blocked_by:
                    oracle_rows, oracle_full_rows = [], None
                else:
                    oracle_rows, oracle_full_rows = await _run_oracle(
                        case, repository
                    )
                rows.append(
                    {
                        "case_id": case.id,
                        "query": case.query,
                        "difficulty": case.difficulty,
                        "prototype": _score_prototype(
                            case,
                            prototype,
                            oracle_rows,
                            oracle_full_rows,
                        ),
                        "current": _score_current(case, current),
                    }
                )
    finally:
        await dw_mysql_client_manager.close()

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "cases_path": str(cases_path),
        "prototype_report": str(prototype_path),
        "current_report": str(current_path),
        "summary": summarize_endpoint_comparison(rows),
        "results": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return payload


def summarize_endpoint_comparison(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(rows)
    prototype_correct = sum(
        item["prototype"]["correct"] is True for item in rows
    )
    current_correct = sum(item["current"]["correct"] is True for item in rows)
    prototype_accuracy = prototype_correct / total if total else 0.0
    current_accuracy = current_correct / total if total else 0.0
    return {
        "prototype": {
            "correct": prototype_correct,
            "total": total,
            "accuracy": round(prototype_accuracy, 4),
        },
        "current": {
            "correct": current_correct,
            "total": total,
            "accuracy": round(current_accuracy, 4),
        },
        "improvement_percentage_points": round(
            (current_accuracy - prototype_accuracy) * 100,
            2,
        ),
    }


def _score_prototype(
    case: EvalCase,
    result: dict[str, Any],
    oracle_rows: list[dict[str, Any]],
    oracle_full_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    score = score_endpoint_result(
        case,
        generated_sql=result.get("generated_sql"),
        actual_rows=result.get("rows"),
        oracle_rows=oracle_rows,
        oracle_full_rows=oracle_full_rows,
        blocked_by=result.get("blocked_by"),
    )
    return {
        "correct": score.correct,
        "reason": score.reason,
        "details": score.details,
        "generated_sql": result.get("generated_sql"),
        "rows": result.get("rows"),
        "error": result.get("error"),
        "latency_seconds": result.get("latency_seconds"),
    }


def _score_current(case: EvalCase, result: dict[str, Any]) -> dict[str, Any]:
    trace = result.get("trace") or {}
    if case.expected_blocked_by:
        score = score_endpoint_result(
            case,
            generated_sql=trace.get("generated_sql"),
            actual_rows=trace.get("final_answer"),
            oracle_rows=[],
            blocked_by=trace.get("blocked_by"),
        )
        correct = score.correct
        reason = score.reason
        details = score.details
    else:
        correct = result.get("endpoint_correct") is True
        reason = (
            result.get("endpoint_score_reason") or "missing_endpoint_score"
        )
        details = result.get("endpoint_score_details") or {}
    return {
        "correct": correct,
        "reason": reason,
        "details": details,
        "generated_sql": trace.get("generated_sql"),
        "rows": trace.get("final_answer"),
        "latency_seconds": result.get("latency_seconds"),
    }


def _index_results(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("case_id")): item
        for item in report.get("results", [])
        if item.get("case_id")
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"{value.__class__.__name__} is not JSON serializable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(
        compare_endpoint_runs(
            cases_path=args.cases,
            prototype_path=args.prototype,
            current_path=args.current,
            output_path=args.output,
        )
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
