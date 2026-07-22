"""Single-call LLM interpreter that emits only an untrusted SemanticDraft."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from app.agent.llm import llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.semantic_planning.catalog import SemanticCandidateCatalog
from app.agent.semantic_planning.draft import (
    AmbiguityReport,
    MeasureMention,
    SemanticDraft,
)
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def interpret_semantics(
    query: str,
    runtime,
    *,
    conversation_history: str,
    catalog: SemanticCandidateCatalog,
) -> SemanticDraft:
    """Interpret user semantics once without producing canonical backend facts."""

    parser = JsonOutputParser(pydantic_object=SemanticDraft)
    prompt = PromptTemplate(
        template=load_prompt("semantic_planning"),
        input_variables=["query", "conversation_history", "catalog_json"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    try:
        raw_draft = await ainvoke_llm_with_usage(
            prompt,
            llm,
            parser,
            {
                "query": query,
                "conversation_history": conversation_history or "无",
                "catalog_json": _serialize_catalog_for_prompt(catalog),
            },
            "语义理解",
            runtime.context["cost_tracker"],
            app_config.llm.timeout_seconds,
            cacheable=not _ablation_options(runtime.context).get(
                "disable_non_sql_llm_cache"
            ),
        )
        draft = (
            raw_draft
            if isinstance(raw_draft, SemanticDraft)
            else SemanticDraft.model_validate(_sanitize_draft_payload(raw_draft))
        )
        return _drop_temporal_filter_dimensions(draft, query)
    except Exception as exc:
        logger.warning(f"语义理解结构化输出失败，保留显式证据并阻断执行: {exc}")
        fallback = fallback_semantic_draft(query=query, catalog=catalog)
        return fallback.model_copy(
            update={
                "ambiguity_reports": [
                    *fallback.ambiguity_reports,
                    AmbiguityReport(
                        raw_text=query,
                        candidate_ids=[],
                        reason="semantic_interpretation_failed",
                    ),
                ]
            }
        )


def fallback_semantic_draft(
    *,
    query: str,
    catalog: SemanticCandidateCatalog,
) -> SemanticDraft:
    """Keep exact evidence only; never infer role, operator, order, limit, or time."""

    metric_matches = _collect_exact_matches(
        query,
        catalog.metrics.values(),
        lambda candidate: (candidate.name, *candidate.aliases),
        lambda candidate: candidate.candidate_id,
    )
    value_matches = _collect_exact_matches(
        query,
        catalog.values.values(),
        lambda candidate: (candidate.canonical_value, *candidate.aliases),
        lambda candidate: candidate.candidate_id,
    )
    column_matches = _collect_exact_matches(
        query,
        catalog.columns.values(),
        lambda candidate: (candidate.name, *candidate.aliases),
        lambda candidate: candidate.candidate_id,
    )

    reports = [
        AmbiguityReport(
            raw_text=raw_text,
            candidate_ids=candidate_ids,
            reason="enum_operator_unresolved_after_fallback",
        )
        for raw_text, candidate_ids in value_matches
    ]
    reports.extend(
        AmbiguityReport(
            raw_text=raw_text,
            candidate_ids=candidate_ids,
            reason="dimension_role_unresolved_after_fallback",
        )
        for raw_text, candidate_ids in column_matches
    )
    return SemanticDraft(
        measure_mentions=[
            MeasureMention(raw_text=raw_text, candidate_ids=candidate_ids)
            for raw_text, candidate_ids in metric_matches
        ],
        ambiguity_reports=reports,
    )


def _serialize_catalog_for_prompt(catalog: SemanticCandidateCatalog) -> str:
    payload = {
        "metadata_version": catalog.metadata_version,
        "tables": [
            {
                "candidate_id": candidate.candidate_id,
                "name": candidate.name,
                "role": candidate.role,
                "description": candidate.description,
            }
            for candidate in catalog.tables.values()
        ],
        "metrics": [
            {
                "candidate_id": candidate.candidate_id,
                "name": candidate.name,
                "aliases": list(candidate.aliases),
                "description": candidate.description,
            }
            for candidate in catalog.metrics.values()
        ],
        "columns": [
            {
                "candidate_id": candidate.candidate_id,
                "table": candidate.table,
                "name": candidate.name,
                "aliases": list(candidate.aliases),
                "role": candidate.role,
                "data_type": candidate.data_type,
                "description": candidate.description,
                "projectable": candidate.projectable,
            }
            for candidate in catalog.columns.values()
        ],
        "values": [
            {
                "candidate_id": candidate.candidate_id,
                "value_text": candidate.canonical_value,
                "aliases": list(candidate.aliases),
                "column_id": candidate.column_id,
                "source": candidate.source,
            }
            for candidate in catalog.values.values()
        ],
        "relationships": [
            {
                "candidate_id": candidate.candidate_id,
                "left_table_id": candidate.left_table_id,
                "left_column_id": candidate.left_column_id,
                "right_table_id": candidate.right_table_id,
                "right_column_id": candidate.right_column_id,
            }
            for candidate in catalog.relationships.values()
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _sanitize_draft_payload(payload: Any) -> dict[str, Any]:
    """Allowlist LLM fields and ignore backend-owned temporal suggestions."""

    value = dict(payload or {})
    return {
        "measure_mentions": _allowlist_items(
            value.get("measure_mentions"), {"raw_text", "candidate_ids"}
        ),
        "dimension_mentions": _allowlist_items(
            value.get("dimension_mentions"),
            {"raw_text", "candidate_ids", "role"},
        ),
        "predicate_mentions": [
            _sanitize_predicate(item)
            for item in value.get("predicate_mentions") or []
            if isinstance(item, dict)
        ],
        "order_mentions": _allowlist_items(
            value.get("order_mentions"),
            {"raw_text", "target_candidate_ids", "direction"},
        ),
        "limit_mentions": _allowlist_items(value.get("limit_mentions"), {"raw_text"}),
        "join_mentions": _allowlist_items(
            value.get("join_mentions"),
            {
                "raw_text",
                "relationship_candidate_id",
                "join_type",
                "left_table_candidate_id",
            },
        ),
        "ambiguity_reports": _allowlist_items(
            value.get("ambiguity_reports"),
            {"raw_text", "candidate_ids", "reason"},
        ),
    }


def _sanitize_predicate(item: dict[str, Any]) -> dict[str, Any]:
    allowed_by_kind = {
        "enum": {
            "kind",
            "raw_text",
            "value_candidate_ids",
            "operator_intent",
        },
        "numeric": {
            "kind",
            "raw_text",
            "target_candidate_ids",
            "operator_intent",
            "value_texts",
        },
        "temporal": {"kind", "raw_text", "relation_intent"},
    }
    allowed = allowed_by_kind.get(str(item.get("kind") or ""), set())
    return {key: value for key, value in item.items() if key in allowed}


def _allowlist_items(items: Any, allowed: set[str]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in item.items() if key in allowed}
        for item in items or []
        if isinstance(item, dict)
    ]


def _drop_temporal_filter_dimensions(
    draft: SemanticDraft,
    query: str,
) -> SemanticDraft:
    temporal_spans = [
        item.raw_text for item in draft.predicate_mentions if item.kind == "temporal"
    ]
    dimensions = [
        item
        for item in draft.dimension_mentions
        if not (
            any(item.raw_text and item.raw_text in span for span in temporal_spans)
            and not _has_explicit_group_cue(query, item.raw_text)
        )
    ]
    return draft.model_copy(update={"dimension_mentions": dimensions})


def _has_explicit_group_cue(query: str, raw_text: str) -> bool:
    return any(f"{prefix}{raw_text}" in query for prefix in ("按", "各", "每"))


def _collect_exact_matches(
    query: str,
    candidates: Iterable[Any],
    terms: Callable[[Any], Iterable[str]],
    candidate_id: Callable[[Any], str],
) -> list[tuple[str, list[str]]]:
    matches: dict[str, set[str]] = {}
    for candidate in candidates:
        for term in terms(candidate):
            raw_text = str(term or "")
            if raw_text and raw_text in query:
                matches.setdefault(raw_text, set()).add(candidate_id(candidate))
    return [
        (raw_text, sorted(candidate_ids))
        for raw_text, candidate_ids in sorted(
            matches.items(), key=lambda item: (-len(item[0]), query.index(item[0]))
        )
    ]


def _ablation_options(context: dict[str, Any]) -> dict[str, Any]:
    return dict(context.get("ablation_options") or {})


__all__ = ["fallback_semantic_draft", "interpret_semantics"]
