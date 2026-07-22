from datetime import date
from types import MappingProxyType

from app.agent.semantic_planning.catalog import (
    ColumnCandidate,
    SemanticCandidateCatalog,
)
from app.agent.semantic_planning.draft import TemporalPredicateMention
from app.agent.semantic_planning.resolvers.temporal import (
    TemporalResolutionContext,
    resolve_temporal_predicate,
)
from app.agent.semantic_planning.time_adapter import parse_time_span


def _column(column_id: str, data_type: str = "bigint") -> ColumnCandidate:
    table, name = column_id.split(".", 1)
    return ColumnCandidate(
        candidate_id=column_id,
        table=table,
        name=name,
        aliases=("日期",),
        role="foreign_key",
        projectable=True,
        data_type=data_type,
    )


def _catalog() -> SemanticCandidateCatalog:
    return SemanticCandidateCatalog(
        metadata_version="meta-v2",
        tables=MappingProxyType({}),
        columns=MappingProxyType(
            {"fact_order.date_id": _column("fact_order.date_id")}
        ),
        relationships=MappingProxyType({}),
        metrics=MappingProxyType({}),
        values=MappingProxyType({}),
    )


def _resolve(raw_text="2025年第一季度", relation="during", **changes):
    values = {
        "catalog": _catalog(),
        "trusted_sources": (f"统计{raw_text}的销售额",),
        "reference_date": date(2026, 7, 19),
        "temporal_column_id": "fact_order.date_id",
    }
    values.update(changes)
    return resolve_temporal_predicate(
        TemporalPredicateMention(raw_text=raw_text, relation_intent=relation),
        TemporalResolutionContext(**values),
    )


def test_quarter_and_relative_month_use_explicit_reference_date():
    quarter = parse_time_span(
        "2025年第一季度", reference_date=date(2026, 7, 19)
    )
    last_month = parse_time_span(
        "上个月", reference_date=date(2026, 7, 19)
    )

    assert (quarter.start_date, quarter.end_date, quarter.grain) == (
        date(2025, 1, 1),
        date(2025, 3, 31),
        "quarter",
    )
    assert (last_month.start_date, last_month.end_date, last_month.grain) == (
        date(2026, 6, 1),
        date(2026, 6, 30),
        "month",
    )


def test_temporal_resolution_materializes_only_canonical_closed_dates():
    result = _resolve()

    assert result.status == "resolved"
    assert result.plan.column_id == "fact_order.date_id"
    assert result.plan.operator == "during"
    assert result.plan.start_date == "2025-01-01"
    assert result.plan.end_date == "2025-03-31"
    assert "start_date_id" not in type(result.plan).model_fields
    assert "end_date_id" not in type(result.plan).model_fields
    assert result.plan.grain == "quarter"


def test_relation_intent_is_preserved_after_backend_date_calculation():
    before = _resolve(raw_text="2025年1月", relation="before")
    after = _resolve(raw_text="2025年1月", relation="after")

    assert before.status == "resolved"
    assert before.plan.operator == "before"
    assert after.status == "resolved"
    assert after.plan.operator == "after"


def test_temporal_column_is_backend_controlled_and_must_exist():
    result = _resolve(temporal_column_id="llm.invented_date")

    assert result.status == "unresolved"
    assert result.issue.code == "temporal_column_invalid"


def test_vague_and_comparison_time_requests_are_blocked():
    vague = _resolve(raw_text="近期")
    comparison = _resolve(raw_text="去年同比")

    assert vague.status in {"unresolved", "ambiguous"}
    assert vague.issue.code == "temporal_ambiguous"
    assert comparison.status == "unresolved"
    assert comparison.issue.code == "temporal_comparison_unsupported"


def test_jionlp_failure_is_a_system_failure_without_regex_guess(monkeypatch):
    from app.agent.semantic_planning import time_adapter

    def fail(*args, **kwargs):
        raise RuntimeError("parser unavailable")

    monkeypatch.setattr(time_adapter, "_parse_with_jionlp", fail)
    result = _resolve()

    assert result.status == "failed"
    assert result.issue.phase == "system"
    assert result.issue.code == "temporal_parser_failed"
