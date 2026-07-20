"""运行 110 条评测集的消融实验。

执行顺序固定为：先 seed 历史成功 SQL，再跑成本消融、校验消融，最后跑召回消融。
校验关闭组只做 dry-run 统计，不会执行危险 SQL。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

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
from app.agent.memory import build_sql_tool_memory
from app.agent.state import DataAgentState
from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import (
    dw_mysql_client_manager,
    meta_mysql_client_manager,
)
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.conf.app_config import app_config
from app.evaluation.cases import EvalCase, evaluate_case, load_eval_cases
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.repositories.mysql.meta.agent_memory_repository import AgentMemoryRepository
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.agent_memory_qdrant_repository import (
    AgentMemoryQdrantRepository,
)
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository
from app.repositories.qdrant.value_qdrant_repository import ValueQdrantRepository


@dataclass(frozen=True)
class AblationRunSpec:
    phase: str
    variant: str
    case_tag: str
    ablation_options: dict[str, Any]
    dry_run_validation_off: bool = False
    write_memory: bool = False


def select_ablation_cases(cases: list[EvalCase]) -> dict[str, list[EvalCase]]:
    """按评测标签切分 case，供 runner 固定顺序执行。"""

    return {
        "seed": [
            case
            for case in cases
            if not case.expected_blocked_by
            and {"ablation_cost", "ablation_guard"}.intersection(case.tags)
        ],
        "cost": [case for case in cases if "ablation_cost" in case.tags],
        "guard": [case for case in cases if "ablation_guard" in case.tags],
        "retrieval": [case for case in cases if "ablation_retrieval" in case.tags],
    }


def ablation_specs() -> list[AblationRunSpec]:
    return [
        AblationRunSpec(
            phase="seed",
            variant="current_config_seed",
            case_tag="seed",
            ablation_options={},
            write_memory=True,
        ),
        AblationRunSpec(
            phase="cost",
            variant="optimized",
            case_tag="cost",
            ablation_options={},
        ),
        AblationRunSpec(
            phase="cost",
            variant="unoptimized",
            case_tag="cost",
            ablation_options={
                "disable_non_sql_llm_cache": True,
                "disable_embedding_cache": True,
                # This disables pruning only: Meta authority checks, required/JOIN
                # enrichment, and runtime date/database context remain enabled.
                "disable_context_compaction": True,
            },
        ),
        AblationRunSpec(
            phase="guard",
            variant="validation_on",
            case_tag="guard",
            ablation_options={},
        ),
        AblationRunSpec(
            phase="guard",
            variant="dry_run_validation_off",
            case_tag="guard",
            ablation_options={},
            dry_run_validation_off=True,
        ),
        AblationRunSpec(
            phase="retrieval",
            variant="retrieval_basic",
            case_tag="retrieval",
            ablation_options={
                "disable_sql_memory": True,
                "disable_value_recall": True,
            },
        ),
        AblationRunSpec(
            phase="retrieval",
            variant="retrieval_full",
            case_tag="retrieval",
            ablation_options={},
        ),
    ]


def selected_ablation_specs(
    *, phase: str | None = None, variant: str | None = None
) -> list[AblationRunSpec]:
    specs = ablation_specs()
    if phase:
        specs = [spec for spec in specs if spec.phase == phase]
    if variant:
        specs = [spec for spec in specs if spec.variant == variant]
    if not specs:
        raise ValueError(f"no ablation specs matched phase={phase!r}, variant={variant!r}")
    return specs


async def run_ablation_eval(
    cases_path: Path,
    output_path: Path | None = None,
    *,
    smoke_limit: int | None = None,
    phase: str | None = None,
    variant: str | None = None,
    seed_user_id_override: str | None = None,
) -> int:
    started_at = _now_iso()
    run_id = started_at.replace(":", "-")
    seed_user_id = seed_user_id_override or f"ablation-seed:{run_id}"
    started = time.perf_counter()

    qdrant_client_manager.init()
    embedding_client_manager.init()
    es_client_manager.init()
    meta_mysql_client_manager.init()
    dw_mysql_client_manager.init()

    try:
        cases = load_eval_cases(cases_path)
        case_groups = select_ablation_cases(cases)
        results: list[dict[str, Any]] = []
        seed_memory_saved = 0

        async with (
            meta_mysql_client_manager.session_factory() as meta_session,
            dw_mysql_client_manager.session_factory() as dw_session,
        ):
            repositories = _build_repositories(meta_session, dw_session)
            metadata_build_version = await repositories[
                "meta_mysql_repository"
            ].get_active_build_version()
            metadata_cache_version = await repositories[
                "meta_mysql_repository"
            ].get_metadata_cache_version()

            for spec in selected_ablation_specs(phase=phase, variant=variant):
                spec_cases = _limit_cases(case_groups[spec.case_tag], smoke_limit)
                for case in spec_cases:
                    if spec.dry_run_validation_off:
                        payload = _dry_run_validation_off_case(case, spec)
                        results.append(payload)
                        continue

                    payload, final_state = await _run_graph_case(
                        case=case,
                        spec=spec,
                        repositories=repositories,
                        user_id=seed_user_id,
                        metadata_build_version=metadata_build_version,
                        metadata_cache_version=metadata_cache_version,
                    )
                    results.append(payload)
                    if spec.write_memory and payload["passed"]:
                        saved = await _save_seed_sql_memory(
                            case=case,
                            final_state=final_state,
                            repositories=repositories,
                            user_id=seed_user_id,
                            metadata_cache_version=metadata_cache_version,
                        )
                        seed_memory_saved += int(saved)

        payload = {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": _now_iso(),
            "cases_path": str(cases_path),
            "git_commit": _git_commit(),
            "seed_user_id": seed_user_id,
            "seed_memory_saved": seed_memory_saved,
            "total_latency_seconds": round(time.perf_counter() - started, 3),
            "summary": summarize_ablation_results(results),
            "results": results,
        }
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
        return 0
    finally:
        await qdrant_client_manager.close()
        await es_client_manager.close()
        await meta_mysql_client_manager.close()
        await dw_mysql_client_manager.close()


def _build_repositories(meta_session, dw_session) -> dict[str, Any]:
    agent_memory_store = AgentMemoryQdrantRepository(
        qdrant_client_manager.client,
        embedding_client_manager.client,
    )
    return {
        "column_qdrant_repository": ColumnQdrantRepository(qdrant_client_manager.client),
        "metric_qdrant_repository": MetricQdrantRepository(qdrant_client_manager.client),
        "value_es_repository": ValueESRepository(es_client_manager.client),
        "value_qdrant_repository": ValueQdrantRepository(qdrant_client_manager.client),
        "meta_mysql_repository": MetaMySQLRepository(meta_session),
        "agent_memory_repository": AgentMemoryRepository(meta_session, agent_memory_store),
        "dw_mysql_repository": DWMySQLRepository(dw_session),
    }


async def _run_graph_case(
    *,
    case: EvalCase,
    spec: AblationRunSpec,
    repositories: dict[str, Any],
    user_id: str,
    metadata_build_version: str | None,
    metadata_cache_version: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    print(f"Running {spec.phase}/{spec.variant}: {case.id} - {case.query}")
    cost_tracker = _new_cost_tracker()
    context = DataAgentContext(
        column_qdrant_repository=repositories["column_qdrant_repository"],
        embedding_client=embedding_client_manager.client,
        metric_qdrant_repository=repositories["metric_qdrant_repository"],
        value_es_repository=repositories["value_es_repository"],
        value_qdrant_repository=repositories["value_qdrant_repository"],
        meta_mysql_repository=repositories["meta_mysql_repository"],
        agent_memory_repository=repositories["agent_memory_repository"],
        dw_mysql_repository=repositories["dw_mysql_repository"],
        cost_tracker=cost_tracker,
        metadata_build_version=metadata_build_version,
        metadata_cache_version=metadata_cache_version,
        user_id=user_id,
        ablation_options=spec.ablation_options,
    )
    state = DataAgentState(query=case.query)
    started = time.perf_counter()
    cache_namespace_token = set_llm_cache_context_namespace(
        f"metadata:{metadata_cache_version}"
    )
    call_budget_token = set_llm_request_call_budget(app_config.llm.max_calls_per_request)
    try:
        final_state = await asyncio.wait_for(
            graph.ainvoke(input=state, context=context),
            timeout=case.timeout_seconds,
        )
        result = evaluate_case(case, final_state)
    except TimeoutError:
        final_state = _exception_state(case, "tool_execution", "评测用例执行超时")
        result = evaluate_case(case, final_state)
    except Exception as exc:
        final_state = _exception_state(case, "tool_execution", str(exc))
        result = evaluate_case(case, final_state)
    finally:
        reset_llm_cache_context_namespace(cache_namespace_token)
        reset_llm_request_call_budget(call_budget_token)

    payload = result.to_dict()
    payload["case"] = case.to_dict()
    payload["phase"] = spec.phase
    payload["variant"] = spec.variant
    payload["ablation_options"] = spec.ablation_options
    payload["usage"] = cost_tracker.summary()
    payload["latency_seconds"] = round(time.perf_counter() - started, 3)
    payload["sql_memory_hit"] = bool(final_state.get("sql_memory_examples") or [])
    return payload, dict(final_state)


async def _save_seed_sql_memory(
    *,
    case: EvalCase,
    final_state: dict[str, Any],
    repositories: dict[str, Any],
    user_id: str,
    metadata_cache_version: str,
) -> bool:
    tool_memory = build_sql_tool_memory(case.query, final_state)
    if not tool_memory:
        return False
    await repositories["agent_memory_repository"].save_tool_usage(
        question=tool_memory.question,
        tool_name=tool_memory.tool_name,
        args=tool_memory.args,
        user_id=user_id,
        metadata_cache_version=metadata_cache_version,
        success=tool_memory.success,
        metadata=tool_memory.metadata,
    )
    return True


def _dry_run_validation_off_case(
    case: EvalCase, spec: AblationRunSpec
) -> dict[str, Any]:
    would_continue = bool(case.expected_blocked_by)
    payload = {
        "case_id": case.id,
        "suite": case.suite,
        "difficulty": case.difficulty,
        "capabilities": case.capabilities,
        "tags": case.tags,
        "passed": True,
        "failure_stage": None,
        "failures": [],
        "trace": {
            "dry_run": True,
            "expected_blocked_by": case.expected_blocked_by,
            "would_continue_without_validation": would_continue,
            "generated_sql": "",
            "blocked_by": None,
            "final_answer": None,
        },
        "case": case.to_dict(),
        "phase": spec.phase,
        "variant": spec.variant,
        "ablation_options": spec.ablation_options,
        "usage": _empty_usage(),
        "latency_seconds": 0,
        "sql_memory_hit": False,
        "dry_run_validation_off": True,
        "would_continue_without_validation": would_continue,
    }
    return payload


def summarize_ablation_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[(item["phase"], item["variant"])].append(item)

    return {
        f"{phase}:{variant}": _summarize_group(items)
        for (phase, variant), items in sorted(grouped.items())
    }


def _summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    passed = sum(1 for item in items if item.get("passed"))
    failures = Counter(
        item.get("failure_stage") or "none"
        for item in items
        if not item.get("passed")
    )
    usage_items = [item["usage"] for item in items]
    sql_memory_hits = sum(1 for item in items if item.get("sql_memory_hit"))
    dry_run_risks = sum(
        1 for item in items if item.get("would_continue_without_validation")
    )
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "avg_latency_seconds": round(
            sum(float(item.get("latency_seconds") or 0) for item in items) / total,
            3,
        )
        if total
        else 0,
        "usage": _summarize_usage(usage_items),
        "cost": _summarize_cost(usage_items),
        "node_usage": _summarize_calls(usage_items, "node"),
        "llm_usage": _summarize_calls(usage_items, "llm"),
        "embedding_usage": _summarize_calls(usage_items, "embedding"),
        "failure_stages": dict(failures),
        "sql_memory_hits": sql_memory_hits,
        "would_continue_without_validation": dry_run_risks,
    }


def _summarize_calls(usages: list[dict[str, Any]], call_type: str) -> dict[str, Any]:
    by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for usage in usages:
        for call in usage.get("calls") or []:
            if call.get("type") == call_type:
                by_step[str(call.get("step") or "unknown")].append(call)

    summary = {}
    for step, calls in sorted(by_step.items()):
        latencies = [
            float(call["latency_ms"])
            for call in calls
            if call.get("latency_ms") is not None
        ]
        summary[step] = {
            "calls": len(calls),
            "cache_hits": sum(1 for call in calls if call.get("cache_hit")),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2)
            if latencies
            else 0,
            "tokens": sum(
                int(call.get("total_tokens") or call.get("tokens") or 0)
                for call in calls
            ),
        }
    return summary


def _summarize_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "llm_input_tokens": sum(item["llm_input_tokens"] for item in usages),
        "llm_output_tokens": sum(item["llm_output_tokens"] for item in usages),
        "llm_total_tokens": sum(item["llm_total_tokens"] for item in usages),
        "embedding_tokens": sum(item["embedding_tokens"] for item in usages),
        "embedding_estimated": any(item["embedding_estimated"] for item in usages),
    }


def _summarize_cost(usages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "llm_cost": round(sum(item["llm_cost"] for item in usages), 8),
        "embedding_cost": round(sum(item["embedding_cost"] for item in usages), 8),
        "total_cost": round(sum(item["total_cost"] for item in usages), 8),
        "currency": app_config.cost.currency,
    }


def _empty_usage() -> dict[str, Any]:
    return {
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "llm_total_tokens": 0,
        "embedding_tokens": 0,
        "llm_cost": 0,
        "embedding_cost": 0,
        "total_cost": 0,
        "currency": app_config.cost.currency,
        "embedding_estimated": False,
        "calls": [],
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


def _exception_state(case: EvalCase, stage: str, error: str) -> dict[str, Any]:
    return {
        "failure": build_failure(
            category="system",
            stage=stage,
            code="ablation_case_exception",
            message=error,
            disposition="failed",
        ),
        "sql": "",
        "sql_context": {"tables": [], "metrics": []},
        "retrieval_context": {"columns": [], "metrics": [], "values": []},
        "trace": {"keywords": [], "retrieved_columns": [], "retrieved_metrics": [], "retrieved_values": []},
    }


def _limit_cases(cases: list[EvalCase], smoke_limit: int | None) -> list[EvalCase]:
    if smoke_limit is None:
        return cases
    return cases[: max(0, smoke_limit)]


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


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cases",
        default="examples/eval_cases_110.yaml",
        help="评测用例 YAML 文件路径",
    )
    parser.add_argument(
        "--output",
        default="eval/runs/ablation-latest.json",
        help="消融评测报告 JSON 输出路径",
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=None,
        help="每个阶段/变体只跑前 N 条，用于快速检查报告结构",
    )
    parser.add_argument("--phase", default=None, help="只运行指定阶段，例如 cost/guard/retrieval/seed")
    parser.add_argument("--variant", default=None, help="只运行指定变体，例如 optimized/retrieval_full")
    parser.add_argument(
        "--seed-user-id",
        default=None,
        help="复用指定 seed 用户的 SQL memory，便于分阶段运行消融",
    )
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            run_ablation_eval(
                cases_path=Path(args.cases),
                output_path=Path(args.output) if args.output else None,
                smoke_limit=args.smoke_limit,
                phase=args.phase,
                variant=args.variant,
                seed_user_id_override=args.seed_user_id,
            )
        )
    )
