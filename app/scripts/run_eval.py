"""Run diagnostic regression evaluations for the data agent."""

import argparse
import asyncio
import json
import subprocess
import time
import traceback
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlglot import expressions as exp
from sqlglot import parse

from app.agent.context import DataAgentContext
from app.agent.cost import CostRates, CostTracker
from app.agent.failure import build_failure
from app.agent.graph import graph
from app.agent.llm_usage import (
    reset_llm_cache_context_namespace,
    reset_llm_request_call_budget,
    set_llm_cache_context_namespace,
    set_llm_request_call_budget,
)
from app.agent.state import DataAgentState
from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import (
    dw_mysql_client_manager,
    meta_mysql_client_manager,
)
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.conf.app_config import app_config
from app.evaluation.cases import (
    EvalCase,
    evaluate_case,
    load_eval_cases,
    results_match,
)
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository
from app.repositories.qdrant.value_qdrant_repository import ValueQdrantRepository

ALL_CAPABILITIES = {
    "keyword_extraction",
    "rag_column_recall",
    "rag_metric_recall",
    "rag_value_hybrid_recall",
    "context_filter",
    "sql_generation",
    "sql_validation",
    "plan_consistency",
    "sql_correction_loop",
    "tool_execution",
    "safety",
}

ALL_SCENARIOS = {"smoke", "regression", "adversarial", "realistic"}


async def run_eval(
    cases_path: Path,
    output_path: Path | None = None,
    *,
    repeat: int = 1,
) -> int:
    if repeat < 1:
        raise ValueError("repeat must be a positive integer")
    started_at = _now_iso()
    started = time.perf_counter()
    run_id = started_at.replace(":", "-")

    qdrant_client_manager.init()
    embedding_client_manager.init()
    es_client_manager.init()
    meta_mysql_client_manager.init()
    dw_mysql_client_manager.init()

    try:
        cases = load_eval_cases(cases_path)
        results = []
        async with (
            meta_mysql_client_manager.session_factory() as meta_session,
            dw_mysql_client_manager.session_factory() as dw_session,
        ):
            repositories = {
                "column_qdrant_repository": ColumnQdrantRepository(
                    qdrant_client_manager.client
                ),
                "metric_qdrant_repository": MetricQdrantRepository(
                    qdrant_client_manager.client
                ),
                "value_es_repository": ValueESRepository(es_client_manager.client),
                "value_qdrant_repository": ValueQdrantRepository(
                    qdrant_client_manager.client
                ),
                "meta_mysql_repository": MetaMySQLRepository(meta_session),
                "dw_mysql_repository": DWMySQLRepository(dw_session),
            }

            for case in cases:
                for repeat_index in range(repeat):
                    case_payload = await _run_case(
                        case,
                        repositories,
                        repeat_index=repeat_index,
                    )
                    if (
                        case.oracle_sql
                        and not case.expected_blocked_by
                        and (case_payload.get("trace") or {}).get("final_answer")
                        is not None
                    ):
                        oracle_rows = await _run_oracle(
                            case,
                            repositories["dw_mysql_repository"],
                        )
                        _score_oracle_result(case_payload, case, oracle_rows)
                    case_payload["repeat_index"] = repeat_index
                    results.append(case_payload)

        finished_at = _now_iso()
        total_latency_seconds = round(time.perf_counter() - started, 3)
        passed = sum(1 for item in results if item["passed"])
        summary = {
            "passed": passed,
            "failed": len(results) - passed,
            "total": len(results),
            "pass_rate": round(passed / len(results), 4) if results else 0,
        }
        payload = {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "cases_path": str(cases_path),
            "git_commit": _git_commit(),
            "model": app_config.llm.model,
            "summary": summary,
            "repeat": repeat,
            "repeat_summary": summarize_repeat_results(results),
            "usage": _summarize_usage([item["usage"] for item in results]),
            "cost": _summarize_cost([item["usage"] for item in results]),
            "total_latency_seconds": total_latency_seconds,
            "capability_summary": _dimension_summary(
                results, "capabilities", ALL_CAPABILITIES
            ),
            "scenario_summary": _dimension_summary(results, "suite", ALL_SCENARIOS),
            "results": results,
        }

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    payload, ensure_ascii=False, indent=2, default=_json_default
                ),
                encoding="utf-8",
            )

        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
        return 0 if passed == len(results) else 1
    finally:
        await qdrant_client_manager.close()
        await es_client_manager.close()
        await meta_mysql_client_manager.close()
        await dw_mysql_client_manager.close()


async def _run_case(
    case: EvalCase,
    repositories: dict[str, Any],
    *,
    repeat_index: int = 0,
) -> dict[str, Any]:
    print(f"Running eval case: {case.id} - {case.query}")
    cost_tracker = _new_cost_tracker()
    metadata_build_version = await repositories[
        "meta_mysql_repository"
    ].get_active_build_version()
    metadata_cache_version = await repositories[
        "meta_mysql_repository"
    ].get_metadata_cache_version()
    context = DataAgentContext(
        column_qdrant_repository=repositories["column_qdrant_repository"],
        embedding_client=embedding_client_manager.client,
        metric_qdrant_repository=repositories["metric_qdrant_repository"],
        value_es_repository=repositories["value_es_repository"],
        value_qdrant_repository=repositories["value_qdrant_repository"],
        meta_mysql_repository=repositories["meta_mysql_repository"],
        dw_mysql_repository=repositories["dw_mysql_repository"],
        cost_tracker=cost_tracker,
        metadata_build_version=metadata_build_version,
        metadata_cache_version=metadata_cache_version,
        semantic_reference_date=date.today(),
        ablation_options=case.ablation_options or {},
    )
    state = DataAgentState(query=case.query)
    started = time.perf_counter()
    cache_namespace_token = set_llm_cache_context_namespace(
        f"eval:{case.id}:repeat:{repeat_index}:metadata:{metadata_cache_version}"
    )
    call_budget_token = set_llm_request_call_budget(
        app_config.llm.max_calls_per_request
    )
    try:
        final_state = await asyncio.wait_for(
            graph.ainvoke(input=state, context=context),
            timeout=case.timeout_seconds,
        )
        result = evaluate_case(case, final_state)
    except TimeoutError as exc:
        exception_stage = _infer_exception_stage(exc)
        result = evaluate_case(
            case,
            {
                "trace": {"keywords": []},
                "failure": build_failure(
                    category="system",
                    stage=exception_stage,
                    code="case_timeout",
                    message=f"节点或评测用例超时：{case.timeout_seconds} 秒",
                    disposition="failed",
                ),
                "sql": "",
                "sql_context": {"tables": [], "metrics": []},
            },
        )
    except Exception as exc:
        result = evaluate_case(
            case,
            {
                "trace": {"keywords": []},
                "failure": build_failure(
                    category="system",
                    stage=_infer_exception_stage(exc),
                    code=exc.__class__.__name__,
                    message=str(exc),
                    disposition="failed",
                ),
                "sql": "",
                "sql_context": {"tables": [], "metrics": []},
            },
        )
    finally:
        reset_llm_cache_context_namespace(cache_namespace_token)
        reset_llm_request_call_budget(call_budget_token)

    payload = result.to_dict()
    payload["case"] = case.to_dict()
    payload["usage"] = cost_tracker.summary()
    payload["latency_seconds"] = round(time.perf_counter() - started, 3)
    return payload


async def _run_oracle(case: EvalCase, dw_repository) -> list[dict[str, Any]]:
    sql = _validated_oracle_sql(case.oracle_sql or "")
    return await dw_repository.run(sql)


def _validated_oracle_sql(sql: str) -> str:
    statements = parse(sql, read="mysql")
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError("oracle_sql must be one read-only SELECT statement")
    return statements[0].sql(dialect="mysql")


def _score_oracle_result(
    payload: dict[str, Any],
    case: EvalCase,
    oracle_rows: list[dict[str, Any]],
) -> None:
    agent_rows = (payload.get("trace") or {}).get("final_answer")
    plan_has_order = bool(
        ((payload.get("trace") or {}).get("semantic_plan") or {}).get("order_by")
    )
    matched = results_match(
        agent_rows,
        oracle_rows,
        order_sensitive=case.order_sensitive or plan_has_order,
    )
    payload["oracle_result_match"] = matched
    payload["oracle_rows"] = oracle_rows
    if matched:
        return
    payload["passed"] = False
    payload["failure_stage"] = payload.get("failure_stage") or "answer_generation"
    payload.setdefault("failures", []).append(
        {
            "code": "oracle_result_mismatch",
            "message": "Agent SQL 结果与人工审核 Oracle SQL 结果不一致",
            "stage": "answer_generation",
            "fatal": True,
        }
    )


def summarize_repeat_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Fail a case if any repeated run violates plan or Oracle consistency."""

    by_case: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        by_case.setdefault(str(item.get("case_id") or ""), []).append(item)
    return {
        case_id: {
            "passed": all(
                item.get("passed") is True
                and (item.get("oracle_result_match") is not False)
                and (
                    ((item.get("trace") or {}).get("sql_plan_consistency") or {}).get(
                        "status"
                    )
                    != "failed"
                )
                for item in items
            ),
            "runs": len(items),
        }
        for case_id, items in by_case.items()
    }


def _new_cost_tracker() -> CostTracker:
    return CostTracker(
        CostRates(
            llm_input_per_1m_tokens=app_config.cost.llm_input_per_1m_tokens,
            llm_output_per_1m_tokens=app_config.cost.llm_output_per_1m_tokens,
            embedding_per_1m_tokens=app_config.cost.embedding_per_1m_tokens,
            currency=app_config.cost.currency,
        )
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def _infer_exception_stage(exc: Exception) -> str:
    text = "".join(traceback.format_exception(exc)).lower()
    if (
        "context_builder" in text
        or "retrieval_context" in text
        or "recall_value_context" in text
        or "recall_column_context" in text
        or "recall_metric_context" in text
    ):
        return "rag_recall"
    if "generate_sql" in text:
        return "sql_generation"
    if "sql_correction" in text or "correct_sql_candidate" in text:
        return "sql_validation"
    if "sql_guard" in text or "validate_sql" in text:
        return "sql_validation"
    if "sql_executor" in text:
        return "tool_execution"
    if "qdrant" in text or "elastic" in text or "connection attempts failed" in text:
        return "rag_recall"
    if "sql" in text or "mysql" in text:
        return "tool_execution"
    return "tool_execution"


def _summarize_usage(usages: list[dict]) -> dict[str, Any]:
    return {
        "llm_input_tokens": sum(item["llm_input_tokens"] for item in usages),
        "llm_output_tokens": sum(item["llm_output_tokens"] for item in usages),
        "llm_total_tokens": sum(item["llm_total_tokens"] for item in usages),
        "embedding_tokens": sum(item["embedding_tokens"] for item in usages),
        "currency": app_config.cost.currency,
        "embedding_estimated": any(item["embedding_estimated"] for item in usages),
    }


def _summarize_cost(usages: list[dict]) -> dict[str, Any]:
    return {
        "llm_cost": round(sum(item["llm_cost"] for item in usages), 8),
        "embedding_cost": round(sum(item["embedding_cost"] for item in usages), 8),
        "total_cost": round(sum(item["total_cost"] for item in usages), 8),
        "currency": app_config.cost.currency,
    }


def _dimension_summary(
    results: list[dict], field: str, required_values: set[str]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for value in sorted(required_values):
        summary[value] = {
            "passed": 0,
            "failed": 0,
            "total": 0,
            "pass_rate": 0,
            "failure_stages": {},
        }

    for result in results:
        values = result.get(field)
        if isinstance(values, str):
            values = [values]
        for value in values or []:
            item = summary.setdefault(
                value,
                {
                    "passed": 0,
                    "failed": 0,
                    "total": 0,
                    "pass_rate": 0,
                    "failure_stages": {},
                },
            )
            item["total"] += 1
            if result["passed"]:
                item["passed"] += 1
            else:
                item["failed"] += 1
                stage = result.get("failure_stage") or "unknown"
                item["failure_stages"][stage] = item["failure_stages"].get(stage, 0) + 1

    for item in summary.values():
        item["pass_rate"] = (
            round(item["passed"] / item["total"], 4) if item["total"] else 0
        )

    uncovered = sorted(value for value, item in summary.items() if item["total"] == 0)
    failure_rank = sorted(
        (
            {"name": value, "failed": item["failed"], "total": item["total"]}
            for value, item in summary.items()
            if item["failed"] > 0
        ),
        key=lambda item: item["failed"],
        reverse=True,
    )
    return {"items": summary, "uncovered": uncovered, "failure_rank": failure_rank}


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cases",
        default="examples/eval_cases.yaml",
        help="评测用例 YAML 文件路径",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="评测报告 JSON 输出路径，例如 eval/runs/latest.json",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="每条用例独立重复执行次数",
    )
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            run_eval(
                cases_path=Path(args.cases),
                output_path=Path(args.output) if args.output else None,
                repeat=args.repeat,
            )
        )
    )
