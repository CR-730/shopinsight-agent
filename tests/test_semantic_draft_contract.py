import pytest
from pydantic import ValidationError

from app.agent.semantic_planning.draft import (
    DimensionMention,
    EnumPredicateMention,
    LimitMention,
    NumericPredicateMention,
    OrderMention,
    SemanticDraft,
    TemporalPredicateMention,
)


def test_draft_accepts_only_llm_owned_semantic_evidence():
    draft = SemanticDraft.model_validate(
        {
            "source_query": "按地区统计销售额大于10000元的前5名，限定2025年第一季度",
            "measure_mentions": [
                {"raw_text": "销售额", "candidate_ids": ["metric:sales"]}
            ],
            "dimension_mentions": [
                {
                    "raw_text": "地区",
                    "candidate_ids": ["column:region"],
                    "role": "group_by",
                }
            ],
            "predicate_mentions": [
                {
                    "kind": "numeric",
                    "raw_text": "大于10000元",
                    "target_candidate_ids": ["metric:sales"],
                    "operator_intent": "gt",
                    "value_texts": ["10000元"],
                },
                {
                    "kind": "temporal",
                    "raw_text": "2025年第一季度",
                    "relation_intent": "during",
                },
            ],
            "order_mentions": [
                {
                    "raw_text": "前5名",
                    "target_candidate_ids": ["metric:sales"],
                    "direction": "desc",
                }
            ],
            "limit_mentions": [{"raw_text": "前5名"}],
            "ambiguity_reports": [],
        }
    )

    assert draft.dimension_mentions[0].role == "group_by"
    assert isinstance(draft.predicate_mentions[0], NumericPredicateMention)
    assert isinstance(draft.predicate_mentions[1], TemporalPredicateMention)
    assert draft.order_mentions[0].direction == "desc"


@pytest.mark.parametrize("role", ["filter", "select", "dimension"])
def test_dimension_role_is_limited_to_group_by_or_projection(role: str):
    with pytest.raises(ValidationError):
        DimensionMention(raw_text="地区", candidate_ids=["column:region"], role=role)


@pytest.mark.parametrize("direction", ["ascending", "descending", "top"])
def test_order_direction_is_limited_to_asc_or_desc(direction: str):
    with pytest.raises(ValidationError):
        OrderMention(
            raw_text="最高",
            target_candidate_ids=["metric:sales"],
            direction=direction,
        )


def test_predicates_are_discriminated_by_kind():
    draft = SemanticDraft.model_validate(
        {
            "source_query": "华北销售额大于一万",
            "predicate_mentions": [
                {
                    "kind": "enum",
                    "raw_text": "华北",
                    "value_candidate_ids": ["value:north"],
                    "column_candidate_ids": ["column:region"],
                    "operator_intent": "eq",
                },
                {
                    "kind": "numeric",
                    "raw_text": "大于一万",
                    "target_candidate_ids": ["metric:sales"],
                    "operator_intent": "gt",
                    "value_texts": ["一万"],
                },
            ],
        }
    )

    assert isinstance(draft.predicate_mentions[0], EnumPredicateMention)
    assert isinstance(draft.predicate_mentions[1], NumericPredicateMention)


@pytest.mark.parametrize(
    ("payload", "forbidden_field"),
    [
        ({"source_query": "统计销售额", "canonical_metric": "GMV"}, "canonical"),
        ({"source_query": "统计销售额", "sql": "SELECT 1"}, "SQL"),
        ({"source_query": "统计销售额", "joins": ["a.id=b.id"]}, "JOIN"),
    ],
)
def test_draft_rejects_backend_owned_top_level_fields(
    payload: dict[str, object], forbidden_field: str
):
    del forbidden_field
    with pytest.raises(ValidationError):
        SemanticDraft.model_validate(payload)


def test_temporal_draft_rejects_computed_dates():
    with pytest.raises(ValidationError):
        TemporalPredicateMention.model_validate(
            {
                "kind": "temporal",
                "raw_text": "2025年第一季度",
                "relation_intent": "during",
                "start_date": "2025-01-01",
            }
        )


def test_enum_draft_rejects_canonical_value_and_free_column_name():
    with pytest.raises(ValidationError):
        EnumPredicateMention.model_validate(
            {
                "kind": "enum",
                "raw_text": "华北",
                "value_candidate_ids": ["value:north"],
                "column_candidate_ids": ["column:region"],
                "operator_intent": "eq",
                "canonical_value": "华北地区",
            }
        )

    with pytest.raises(ValidationError):
        EnumPredicateMention.model_validate(
            {
                "kind": "enum",
                "raw_text": "华北",
                "column_name": "region_name",
                "operator_intent": "eq",
            }
        )


def test_limit_keeps_only_the_raw_span():
    mention = LimitMention(raw_text="前五名")
    assert mention.model_dump() == {"raw_text": "前五名"}

    with pytest.raises(ValidationError):
        LimitMention(raw_text="前五名", value=5)
