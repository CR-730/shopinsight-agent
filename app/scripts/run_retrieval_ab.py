"""Compare a jieba-only retrieval baseline with current hybrid retrieval."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.agent.cost import CostRates, CostTracker
from app.agent.llm_usage import (
    reset_llm_cache_context_namespace,
    reset_llm_request_call_budget,
    set_llm_cache_context_namespace,
    set_llm_request_call_budget,
)
from app.agent.retrieval_context import (
    _extend_keywords,
    build_route_retrieval_queries,
    extract_retrieval_keywords,
)
from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import meta_mysql_client_manager
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.conf.app_config import app_config
from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo
from app.entities.value_info import ValueInfo
from app.evaluation.cases import load_eval_cases
from app.evaluation.retrieval_ab import (
    RetrievalCandidates,
    score_retrieval_case,
    summarize_retrieval_scores,
)
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository
from app.repositories.qdrant.value_qdrant_repository import ValueQdrantRepository
from app.retrieval.fusion import (
    RankedList,
    fuse_candidate_rankings,
    fuse_value_rankings,
)

T = TypeVar("T")
CANDIDATE_BUDGET = app_config.agent.retrieval_candidate_limit
_COLUMN_FIELDS = {field.name for field in fields(ColumnInfo)}
_METRIC_FIELDS = {field.name for field in fields(MetricInfo)}


async def run_retrieval_ab(cases_path: Path, output_path: Path) -> int:
    qdrant_client_manager.init()
    embedding_client_manager.init()
    es_client_manager.init()
    meta_mysql_client_manager.init()
    try:
        cases = load_eval_cases(cases_path)
        async with meta_mysql_client_manager.session_factory() as meta_session:
            meta_repository = MetaMySQLRepository(meta_session)
            build_version = await meta_repository.get_active_build_version()
            metadata_cache_version = await meta_repository.get_metadata_cache_version()
            repositories = {
                "columns": ColumnQdrantRepository(qdrant_client_manager.client),
                "metrics": MetricQdrantRepository(qdrant_client_manager.client),
                "values": ValueQdrantRepository(qdrant_client_manager.client),
                "es": ValueESRepository(es_client_manager.client),
            }
            baseline_scores = []
            current_scores = []
            details = []
            for case in cases:
                print(f"Retrieval A/B: {case.id} - {case.query}")
                cache_namespace_token = set_llm_cache_context_namespace(
                    f"retrieval-ab:{case.id}:metadata:{metadata_cache_version}"
                )
                call_budget_token = set_llm_request_call_budget(
                    app_config.llm.max_calls_per_request
                )
                cost_tracker = _cost_tracker()
                try:
                    context = {
                        "cost_tracker": cost_tracker,
                        "metadata_build_version": build_version,
                        "metadata_cache_version": metadata_cache_version,
                        "ablation_options": {},
                    }
                    baseline_keywords = await _baseline_route_keywords(case.query)
                    current_keywords = await _current_route_keywords(
                        case.query,
                        context,
                    )
                    embeddings = await _embed_all(
                        {
                            route: list(
                                dict.fromkeys(
                                    [
                                        *baseline_keywords[route],
                                        *current_keywords[route],
                                    ]
                                )
                            )
                            for route in ("columns", "metrics", "values")
                        }
                    )
                    baseline, current = await asyncio.gather(
                        _baseline_candidates(
                            baseline_keywords,
                            embeddings,
                            build_version,
                        ),
                        _current_candidates(
                            current_keywords,
                            embeddings,
                            build_version,
                            repositories,
                        ),
                    )
                finally:
                    reset_llm_cache_context_namespace(cache_namespace_token)
                    reset_llm_request_call_budget(call_budget_token)
                baseline_score = score_retrieval_case(case, baseline)
                current_score = score_retrieval_case(case, current)
                baseline_scores.append(baseline_score)
                current_scores.append(current_score)
                details.append(
                    {
                        "case_id": case.id,
                        "query": case.query,
                        "queries": {
                            "baseline": baseline_keywords,
                            "current": current_keywords,
                        },
                        "gold": {
                            "columns": case.expected_retrieved_columns,
                            "metrics": case.expected_metrics,
                            "values": case.expected_values,
                        },
                        "baseline": {
                            "candidates": baseline.__dict__,
                            "candidate_counts": _candidate_counts(baseline),
                            "score": baseline_score.to_dict(),
                        },
                        "current": {
                            "candidates": current.__dict__,
                            "candidate_counts": _candidate_counts(current),
                            "score": current_score.to_dict(),
                        },
                        "usage": cost_tracker.summary(),
                    }
                )
        payload = {
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit(),
            "cases_path": str(cases_path),
            "candidate_budget": CANDIDATE_BUDGET,
            "metadata_build_version": build_version,
            "metadata_cache_version": metadata_cache_version,
            "embedding_model": app_config.embedding.model,
            "embedding_dimension": app_config.qdrant.embedding_size,
            "llm_model": app_config.llm.model,
            "protocol": {
                "baseline": (
                    "完整问题加 jieba 关键词；字段/指标的原始 Qdrant point "
                    "仅按 point ID 去重后全局竞争 Top-5，不按候选 ID 分组；"
                    "字段值仅用 ES；不使用 LLM 扩词"
                ),
                "current": (
                    "完整问题加三路领域化 LLM 扩词，不使用 jieba；"
                    f"字段/指标 Qdrant 候选分组后全局 RRF Top-{CANDIDATE_BUDGET}；"
                    f"字段值 ES+Qdrant 候选级加权 RRF Top-{CANDIDATE_BUDGET}"
                ),
                "primary_comparison": "baseline vs current",
                "disclosure": (
                    "A 是可复现的基础检索基线，不等同于开源原项目的原始性能；"
                    "A/B 使用同一评测集、索引、Embedding 模型和最终候选预算"
                ),
            },
            "baseline": summarize_retrieval_scores(baseline_scores),
            "current": summarize_retrieval_scores(current_scores),
            "details": details,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    key: payload[key]
                    for key in ("baseline", "current")
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        await qdrant_client_manager.close()
        await es_client_manager.close()
        await meta_mysql_client_manager.close()


async def _baseline_route_keywords(query: str) -> dict[str, list[str]]:
    keywords = (await extract_retrieval_keywords({"query": query}))["keywords"]
    return {route: list(keywords) for route in ("columns", "metrics", "values")}


async def _current_route_keywords(
    query: str,
    context: dict[str, Any],
) -> dict[str, list[str]]:
    prompts = {
        "columns": "extend_keywords_for_column_recall",
        "metrics": "extend_keywords_for_metric_recall",
        "values": "extend_keywords_for_value_recall",
    }
    result = {}
    for route, prompt_name in prompts.items():
        expanded = await _extend_keywords(
            prompt_name=prompt_name,
            query=query,
            step=f"A/B {route}",
            context=context,
        )
        result[route] = build_route_retrieval_queries(query, expanded)
    return result


async def _embed_all(
    route_keywords: dict[str, list[str]],
) -> dict[str, list[float]]:
    keywords = sorted({item for values in route_keywords.values() for item in values})
    vectors = await embedding_client_manager.client.aembed_documents(keywords)
    return dict(zip(keywords, vectors))


async def _baseline_candidates(
    route_keywords: dict[str, list[str]],
    embeddings: dict[str, list[float]],
    build_version: str | None,
) -> RetrievalCandidates:
    columns = await _baseline_qdrant_route(
        collection_name=ColumnQdrantRepository.collection_name,
        keywords=route_keywords["columns"],
        embeddings=embeddings,
        build_version=build_version,
        entity_factory=_column_from_payload,
        candidate_id_of=lambda item: item.id,
    )
    metrics = await _baseline_qdrant_route(
        collection_name=MetricQdrantRepository.collection_name,
        keywords=route_keywords["metrics"],
        embeddings=embeddings,
        build_version=build_version,
        entity_factory=_metric_from_payload,
        candidate_id_of=lambda item: item.id,
    )
    values = await _baseline_es_values(
        route_keywords["values"],
        build_version,
    )
    return RetrievalCandidates(
        columns=[item.id for item in columns],
        metrics=[item.id for item in metrics],
        values=[item.id for item in values],
    )


async def _baseline_qdrant_route(
    *,
    collection_name: str,
    keywords: list[str],
    embeddings: dict[str, list[float]],
    build_version: str | None,
    entity_factory: Callable[[dict[str, Any]], T],
    candidate_id_of: Callable[[T], str],
) -> list[T]:
    point_lists = []
    for keyword in keywords:
        result = await qdrant_client_manager.client.query_points(
            collection_name=collection_name,
            query=embeddings[keyword],
            query_filter=_build_filter(build_version),
            limit=CANDIDATE_BUDGET,
            score_threshold=0.6,
        )
        point_lists.append(list(result.points))

    # A 在 Top-5 之前只去除同一个 point 的重复命中，不按候选 ID 分组。
    # 因而字段名、别名、描述等多个 point 仍会分别占用候选预算。
    top_points = _top_raw_points_by_score(
        point_lists,
        limit=CANDIDATE_BUDGET,
    )
    ordered: dict[str, T] = {}
    for point in top_points:
        item = entity_factory(point.payload or {})
        ordered.setdefault(candidate_id_of(item), item)
    return list(ordered.values())


async def _baseline_es_values(
    keywords: list[str],
    build_version: str | None,
) -> list[ValueInfo]:
    hits_by_id: dict[str, dict[str, Any]] = {}
    for keyword in keywords:
        filters = [{"term": {"surface_type": "canonical"}}]
        if build_version:
            filters.append({"term": {"meta_build_version": build_version}})
        response = await es_client_manager.client.search(
            index=ValueESRepository.index_name,
            query={
                "bool": {
                    "must": [{"match": {"matched_text": keyword}}],
                    "filter": filters,
                }
            },
            size=CANDIDATE_BUDGET,
            min_score=0.6,
        )
        for hit in response["hits"]["hits"]:
            hit_id = str(hit["_id"])
            existing = hits_by_id.get(hit_id)
            if existing is None or float(hit["_score"]) > float(existing["_score"]):
                hits_by_id[hit_id] = hit

    top_hits = sorted(
        hits_by_id.values(),
        key=lambda hit: (float(hit["_score"]), str(hit["_id"])),
        reverse=True,
    )[:CANDIDATE_BUDGET]
    ordered: dict[str, ValueInfo] = {}
    for hit in top_hits:
        source = hit["_source"]
        item = ValueInfo(
            id=str(source["candidate_id"]),
            value=str(source["value"]),
            column_id=str(source["column_id"]),
            matched_texts=[str(source["matched_text"])],
        )
        ordered.setdefault(item.id, item)
    return list(ordered.values())


def _top_raw_points_by_score(point_lists, *, limit: int):
    """Select raw points globally without grouping different points by candidate."""

    points_by_id = {}
    for points in point_lists:
        for point in points:
            point_id = str(point.id)
            existing = points_by_id.get(point_id)
            if existing is None or float(point.score) > float(existing.score):
                points_by_id[point_id] = point
    return sorted(
        points_by_id.values(),
        key=lambda point: (float(point.score), str(point.id)),
        reverse=True,
    )[:limit]


async def _current_candidates(
    route_keywords: dict[str, list[str]],
    embeddings: dict[str, list[float]],
    build_version: str | None,
    repositories: dict[str, Any],
) -> RetrievalCandidates:
    column_lists = [
        RankedList(
            source=f"vector:{keyword}",
            items=await repositories["columns"].search(
                embeddings[keyword],
                meta_build_version=build_version,
            ),
        )
        for keyword in route_keywords["columns"]
    ]
    metric_lists = [
        RankedList(
            source=f"vector:{keyword}",
            items=await repositories["metrics"].search(
                embeddings[keyword],
                meta_build_version=build_version,
            ),
        )
        for keyword in route_keywords["metrics"]
    ]
    value_lists: list[RankedList[ValueInfo]] = []
    for keyword in route_keywords["values"]:
        es_items, vector_items = await asyncio.gather(
            repositories["es"].search(
                keyword,
                meta_build_version=build_version,
            ),
            repositories["values"].search(
                embeddings[keyword],
                score_threshold=app_config.agent.value_vector_score_threshold,
                meta_build_version=build_version,
            ),
        )
        if es_items:
            value_lists.append(
                RankedList(
                    source=f"es:{keyword}",
                    items=es_items,
                    weight=app_config.agent.value_hybrid_es_weight,
                )
            )
        if vector_items:
            value_lists.append(
                RankedList(
                    source=f"vector:{keyword}",
                    items=vector_items,
                    weight=app_config.agent.value_hybrid_vector_weight,
                )
            )
    columns = fuse_candidate_rankings(
        column_lists,
        candidate_id_of=lambda item: item.id,
        limit=CANDIDATE_BUDGET,
    )
    metrics = fuse_candidate_rankings(
        metric_lists,
        candidate_id_of=lambda item: item.id,
        limit=CANDIDATE_BUDGET,
    )
    values = fuse_value_rankings(value_lists, limit=CANDIDATE_BUDGET)
    return RetrievalCandidates(
        columns=[item.item.id for item in columns],
        metrics=[item.item.id for item in metrics],
        values=[item.id for item in values],
    )


def _column_from_payload(payload: dict[str, Any]) -> ColumnInfo:
    return ColumnInfo(
        **{key: value for key, value in payload.items() if key in _COLUMN_FIELDS}
    )


def _metric_from_payload(payload: dict[str, Any]) -> MetricInfo:
    return MetricInfo(
        **{key: value for key, value in payload.items() if key in _METRIC_FIELDS}
    )


def _build_filter(build_version: str | None) -> Filter | None:
    if not build_version:
        return None
    return Filter(
        must=[
            FieldCondition(
                key="meta_build_version",
                match=MatchValue(value=build_version),
            )
        ]
    )


def _cost_tracker() -> CostTracker:
    return CostTracker(
        CostRates(
            llm_input_per_1m_tokens=app_config.cost.llm_input_per_1m_tokens,
            llm_output_per_1m_tokens=app_config.cost.llm_output_per_1m_tokens,
            embedding_per_1m_tokens=app_config.cost.embedding_per_1m_tokens,
            currency=app_config.cost.currency,
        )
    )


def _candidate_counts(candidates: RetrievalCandidates) -> dict[str, int]:
    return {
        "columns": len(candidates.columns),
        "metrics": len(candidates.metrics),
        "values": len(candidates.values),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run_retrieval_ab(args.cases, args.output)))
