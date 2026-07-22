import asyncio
from types import MappingProxyType, SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agent.semantic_planning import interpreter
from app.agent.semantic_planning.catalog import (
    ColumnCandidate,
    MetricCandidate,
    RelationshipCandidate,
    SemanticCandidateCatalog,
    TableCandidate,
    ValueCandidate,
)
from app.agent.semantic_planning.draft import JoinMention, SemanticDraft

RELATIONSHIP_ID = "rel:dim_region.region_id=fact_order.region_id"


def _catalog(*, ambiguous_values: bool = False) -> SemanticCandidateCatalog:
    values = {
        "value:region:south": ValueCandidate(
            candidate_id="value:region:south",
            canonical_value="华南地区",
            aliases=("华南",),
            column_id="dim_region.region_name",
            source="retrieval",
        )
    }
    if ambiguous_values:
        values["value:area:south"] = ValueCandidate(
            candidate_id="value:area:south",
            canonical_value="华南片区",
            aliases=("华南",),
            column_id="dim_sales_area.area_name",
            source="retrieval",
        )
    return SemanticCandidateCatalog(
        metadata_version="meta-v2",
        tables=MappingProxyType(
            {
                "dim_region": TableCandidate(
                    candidate_id="dim_region",
                    name="dim_region",
                    role="dim",
                    description="地区维表",
                ),
                "fact_order": TableCandidate(
                    candidate_id="fact_order",
                    name="fact_order",
                    role="fact",
                    description="订单事实表",
                ),
            }
        ),
        columns=MappingProxyType(
            {
                "dim_region.region_name": ColumnCandidate(
                    candidate_id="dim_region.region_name",
                    table="dim_region",
                    name="region_name",
                    aliases=("地区",),
                    role="dimension",
                    projectable=True,
                    data_type="varchar",
                    description="地区名称",
                ),
                "dim_region.region_id": ColumnCandidate(
                    candidate_id="dim_region.region_id",
                    table="dim_region",
                    name="region_id",
                    aliases=(),
                    role="primary_key",
                    projectable=True,
                ),
                "fact_order.region_id": ColumnCandidate(
                    candidate_id="fact_order.region_id",
                    table="fact_order",
                    name="region_id",
                    aliases=(),
                    role="foreign_key",
                    projectable=True,
                ),
            }
        ),
        relationships=MappingProxyType(
            {
                RELATIONSHIP_ID: RelationshipCandidate(
                    candidate_id=RELATIONSHIP_ID,
                    left_table_id="dim_region",
                    left_column_id="dim_region.region_id",
                    right_table_id="fact_order",
                    right_column_id="fact_order.region_id",
                )
            }
        ),
        metrics=MappingProxyType(
            {
                "GMV": MetricCandidate(
                    candidate_id="GMV",
                    name="GMV",
                    aliases=("销售额",),
                    relevant_columns=("fact_order.order_amount",),
                    aggregation="sum",
                    description="成交金额总和",
                )
            }
        ),
        values=MappingProxyType(values),
    )


def _runtime():
    return SimpleNamespace(context={"cost_tracker": object(), "ablation_options": {}})


def test_interpreter_uses_one_llm_call_and_returns_untrusted_draft(monkeypatch):
    calls = []

    async def fake_invoke(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "source_query": "model supplied value",
            "measure_mentions": [{"raw_text": "销售额", "candidate_ids": ["GMV"]}],
        }

    monkeypatch.setattr(interpreter, "ainvoke_llm_with_usage", fake_invoke)

    result = asyncio.run(
        interpreter.interpret_semantics(
            "统计销售额",
            _runtime(),
            conversation_history="",
            catalog=_catalog(),
        )
    )

    assert len(calls) == 1
    assert not hasattr(result, "source_query")
    assert result.measure_mentions[0].candidate_ids == ["GMV"]


def test_semantic_draft_accepts_only_controlled_join_types():
    mention = JoinMention.model_validate(
        {
            "raw_text": "包括没有订单的地区",
            "relationship_candidate_id": RELATIONSHIP_ID,
            "join_type": "left",
            "left_table_candidate_id": "dim_region",
        }
    )

    assert mention.join_type == "left"
    with pytest.raises(ValidationError):
        JoinMention.model_validate(
            {
                "raw_text": "包括所有地区",
                "relationship_candidate_id": RELATIONSHIP_ID,
                "join_type": "full",
            }
        )


def test_prompt_contains_controlled_ids_but_no_backend_owned_output_fields(
    monkeypatch,
):
    captured = {}

    async def fake_invoke(prompt, _llm, _parser, inputs, *_args, **_kwargs):
        captured["text"] = (await prompt.ainvoke(inputs)).to_string()
        captured["model"] = _parser.pydantic_object
        captured["schema"] = _parser.pydantic_object.model_json_schema()
        return {}

    monkeypatch.setattr(interpreter, "ainvoke_llm_with_usage", fake_invoke)
    asyncio.run(
        interpreter.interpret_semantics(
            "统计华南销售额",
            _runtime(),
            conversation_history="无",
            catalog=_catalog(),
        )
    )

    prompt_text = captured["text"]
    assert '"candidate_id": "GMV"' in prompt_text
    assert '"candidate_id": "value:region:south"' in prompt_text
    assert '"column_id": "dim_region.region_name"' in prompt_text
    assert f'"candidate_id": "{RELATIONSHIP_ID}"' in prompt_text
    assert '"left_column_id": "dim_region.region_id"' in prompt_text
    assert '"right_column_id": "fact_order.region_id"' in prompt_text
    assert "没有对应取值候选" in prompt_text
    assert "比较发生的语义层级" in prompt_text
    assert "聚合或派生结果" in prompt_text
    assert "单条记录的原始属性" in prompt_text
    for overfit_example in (
        "销售额、订单数、客单价",
        "单笔、每条记录",
        "示例 4",
        "销售额大于10000元",
    ):
        assert overfit_example not in prompt_text
    assert '"clause"' not in prompt_text
    assert '"source_query"' not in prompt_text
    assert captured["model"] is SemanticDraft
    assert "source_query" not in captured["schema"]["properties"]
    assert '"value_candidate_ids":[]' in prompt_text
    for forbidden in (
        "canonical_value",
        "start_date",
        "end_date",
        '"joins"',
        '"expression"',
    ):
        assert forbidden not in prompt_text


def test_fallback_keeps_exact_ids_but_does_not_infer_roles_operators_or_time(
    monkeypatch,
):
    async def fail(*_args, **_kwargs):
        raise ValueError("invalid structured output")

    monkeypatch.setattr(interpreter, "ainvoke_llm_with_usage", fail)
    result = asyncio.run(
        interpreter.interpret_semantics(
            "按地区统计2025年第一季度销售额最高的前5名",
            _runtime(),
            conversation_history="",
            catalog=_catalog(),
        )
    )

    assert result.measure_mentions[0].candidate_ids == ["GMV"]
    assert result.dimension_mentions == []
    assert result.predicate_mentions == []
    assert result.order_mentions == []
    assert result.limit_mentions == []
    assert result.join_mentions == []
    assert result.ambiguity_reports[0].candidate_ids == ["dim_region.region_name"]
    failure_report = next(
        report
        for report in result.ambiguity_reports
        if report.reason == "semantic_interpretation_failed"
    )
    assert failure_report.raw_text == "按地区统计2025年第一季度销售额最高的前5名"
    assert failure_report.candidate_ids == []


def test_fallback_never_selects_first_value_when_exact_term_has_multiple_ids(
    monkeypatch,
):
    async def fail(*_args, **_kwargs):
        raise ValueError("invalid structured output")

    monkeypatch.setattr(interpreter, "ainvoke_llm_with_usage", fail)
    result = asyncio.run(
        interpreter.interpret_semantics(
            "统计华南销售额",
            _runtime(),
            conversation_history="",
            catalog=_catalog(ambiguous_values=True),
        )
    )

    value_report = next(
        report for report in result.ambiguity_reports if report.raw_text == "华南"
    )
    assert value_report.candidate_ids == [
        "value:area:south",
        "value:region:south",
    ]
    assert result.predicate_mentions == []


def test_interpreter_ignores_backend_owned_temporal_target_ids(monkeypatch):
    async def fake_invoke(*args, **kwargs):
        return {
            "measure_mentions": [{"raw_text": "销售额", "candidate_ids": ["GMV"]}],
            "dimension_mentions": [],
            "predicate_mentions": [
                {
                    "kind": "temporal",
                    "raw_text": "2025年第一季度",
                    "relation_intent": "during",
                    "target_candidate_ids": ["dim_date.quarter"],
                }
            ],
        }

    monkeypatch.setattr(interpreter, "ainvoke_llm_with_usage", fake_invoke)

    result = asyncio.run(
        interpreter.interpret_semantics(
            "2025年第一季度销售额",
            _runtime(),
            conversation_history="",
            catalog=_catalog(),
        )
    )

    assert result.predicate_mentions[0].kind == "temporal"
    assert not hasattr(result.predicate_mentions[0], "target_candidate_ids")


def test_interpreter_ignores_redundant_enum_column_candidate_ids(monkeypatch):
    async def fake_invoke(*args, **kwargs):
        return {
            "predicate_mentions": [
                {
                    "kind": "enum",
                    "raw_text": "华北",
                    "value_candidate_ids": ["value:region:north"],
                    "column_candidate_ids": ["dim_region.region_name"],
                    "operator_intent": "eq",
                }
            ],
        }

    monkeypatch.setattr(interpreter, "ainvoke_llm_with_usage", fake_invoke)

    result = asyncio.run(
        interpreter.interpret_semantics(
            "统计华北销售额",
            _runtime(),
            conversation_history="",
            catalog=_catalog(),
        )
    )

    mention = result.predicate_mentions[0]
    assert mention.value_candidate_ids == ["value:region:north"]
    assert not hasattr(mention, "column_candidate_ids")


def test_interpreter_preserves_enum_mention_without_value_candidates(monkeypatch):
    async def fake_invoke(*args, **kwargs):
        return {
            "predicate_mentions": [
                {
                    "kind": "enum",
                    "raw_text": "火星区域",
                    "value_candidate_ids": [],
                    "operator_intent": "eq",
                }
            ],
        }

    monkeypatch.setattr(interpreter, "ainvoke_llm_with_usage", fake_invoke)

    result = asyncio.run(
        interpreter.interpret_semantics(
            "统计火星区域销售额",
            _runtime(),
            conversation_history="",
            catalog=_catalog(),
        )
    )

    mention = result.predicate_mentions[0]
    assert mention.raw_text == "火星区域"
    assert mention.value_candidate_ids == []


def test_temporal_filter_is_not_duplicated_as_group_dimension(monkeypatch):
    async def fake_invoke(*args, **kwargs):
        return {
            "measure_mentions": [{"raw_text": "销售额", "candidate_ids": ["GMV"]}],
            "dimension_mentions": [
                {
                    "raw_text": "第一季度",
                    "candidate_ids": ["dim_date.quarter"],
                    "role": "group_by",
                },
                {
                    "raw_text": "地区",
                    "candidate_ids": ["dim_region.region_name"],
                    "role": "group_by",
                },
            ],
            "predicate_mentions": [
                {
                    "kind": "temporal",
                    "raw_text": "2025年第一季度",
                    "relation_intent": "during",
                }
            ],
        }

    monkeypatch.setattr(interpreter, "ainvoke_llm_with_usage", fake_invoke)

    result = asyncio.run(
        interpreter.interpret_semantics(
            "2025年第一季度销售额最高的前5个商品",
            _runtime(),
            conversation_history="",
            catalog=_catalog(),
        )
    )

    assert [item.raw_text for item in result.dimension_mentions] == ["地区"]


def test_prompt_forbids_temporal_candidate_ids_and_implicit_time_grouping(
    monkeypatch,
):
    captured = {}

    async def fake_invoke(prompt, _llm, _parser, inputs, *_args, **_kwargs):
        captured["text"] = (await prompt.ainvoke(inputs)).to_string()
        return {}

    monkeypatch.setattr(interpreter, "ainvoke_llm_with_usage", fake_invoke)
    asyncio.run(
        interpreter.interpret_semantics(
            "2025年第一季度销售额",
            _runtime(),
            conversation_history="",
            catalog=_catalog(),
        )
    )

    assert "时间谓词禁止输出 target_candidate_ids" in captured["text"]
    assert "不能同时作为分组维度" in captured["text"]


def test_prompt_limits_join_choice_to_relationship_and_preserved_side(monkeypatch):
    captured = {}

    async def fake_invoke(prompt, _llm, _parser, inputs, *_args, **_kwargs):
        captured["text"] = (await prompt.ainvoke(inputs)).to_string()
        return {}

    monkeypatch.setattr(interpreter, "ainvoke_llm_with_usage", fake_invoke)
    asyncio.run(
        interpreter.interpret_semantics(
            "包括没有订单的地区",
            _runtime(),
            conversation_history="",
            catalog=_catalog(),
        )
    )

    assert "明确要求保留没有匹配记录的对象" in captured["text"]
    assert "relationship_candidate_id" in captured["text"]
    assert "left_table_candidate_id" in captured["text"]
    assert "不要编写连接条件" in captured["text"]


def test_interpreter_allowlists_join_mention_fields(monkeypatch):
    async def fake_invoke(*args, **kwargs):
        return {
            "join_mentions": [
                {
                    "raw_text": "包括没有订单的地区",
                    "relationship_candidate_id": RELATIONSHIP_ID,
                    "join_type": "left",
                    "left_table_candidate_id": "dim_region",
                    "on_expression": "1 = 1",
                }
            ],
        }

    monkeypatch.setattr(interpreter, "ainvoke_llm_with_usage", fake_invoke)

    result = asyncio.run(
        interpreter.interpret_semantics(
            "包括没有订单的地区",
            _runtime(),
            conversation_history="",
            catalog=_catalog(),
        )
    )

    assert result.join_mentions == [
        JoinMention(
            raw_text="包括没有订单的地区",
            relationship_candidate_id=RELATIONSHIP_ID,
            join_type="left",
            left_table_candidate_id="dim_region",
        )
    ]
    assert not hasattr(result.join_mentions[0], "on_expression")
