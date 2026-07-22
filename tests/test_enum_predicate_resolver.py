import asyncio
from types import MappingProxyType

from app.agent.semantic_planning.catalog import (
    ColumnCandidate,
    SemanticCandidateCatalog,
    ValueCandidate,
)
from app.agent.semantic_planning.draft import EnumPredicateMention
from app.agent.semantic_planning.resolvers.enum_predicate import (
    EnumResolutionContext,
    resolve_enum_predicate,
)


def _column(column_id: str):
    table, name = column_id.split(".", 1)
    return ColumnCandidate(
        candidate_id=column_id,
        table=table,
        name=name,
        aliases=(),
        role="dimension",
        projectable=True,
        data_type="varchar",
    )


def _value(candidate_id, canonical_value, column_id, source="retrieval"):
    return ValueCandidate(
        candidate_id=candidate_id,
        canonical_value=canonical_value,
        aliases=(),
        column_id=column_id,
        source=source,
    )


def _catalog():
    columns = {
        "dim_region.region_name": _column("dim_region.region_name"),
        "dim_sales_area.area_name": _column("dim_sales_area.area_name"),
    }
    values = {
        "v-north": _value(
            "v-north", "华北地区", "dim_region.region_name"
        ),
        "v-south-region": _value(
            "v-south-region", "华南地区", "dim_region.region_name"
        ),
        "v-south-area": _value(
            "v-south-area", "华南片区", "dim_sales_area.area_name"
        ),
    }
    return SemanticCandidateCatalog(
        metadata_version="meta-v2",
        tables=MappingProxyType({}),
        columns=MappingProxyType(columns),
        relationships=MappingProxyType({}),
        metrics=MappingProxyType({}),
        values=MappingProxyType(values),
    )


def _context():
    return EnumResolutionContext(
        catalog=_catalog(),
        trusted_sources=("统计华北和华南销售额",),
    )


def test_value_candidate_resolves_to_its_owned_column():
    mention = EnumPredicateMention(
        raw_text="华北",
        value_candidate_ids=["v-north"],
        operator_intent="eq",
    )

    result = asyncio.run(resolve_enum_predicate(mention, _context()))

    assert result.status == "resolved"
    assert result.plan.column_id == "dim_region.region_name"
    assert result.plan.canonical_values == ["华北地区"]


def test_values_from_different_columns_are_ambiguous():
    mention = EnumPredicateMention(
        raw_text="华南",
        value_candidate_ids=["v-south-region", "v-south-area"],
        operator_intent="eq",
    )

    result = asyncio.run(resolve_enum_predicate(mention, _context()))

    assert result.status == "ambiguous"
    assert result.issue.code == "filter_column_ambiguous"


def test_missing_value_candidate_blocks():
    mention = EnumPredicateMention(
        raw_text="华北",
        value_candidate_ids=[],
        operator_intent="eq",
    )

    result = asyncio.run(resolve_enum_predicate(mention, _context()))

    assert result.status == "unresolved"
    assert result.issue.code == "value_not_bound"


def test_in_operator_accepts_multiple_values_only_within_one_column():
    mention = EnumPredicateMention(
        raw_text="华北和华南",
        value_candidate_ids=["v-north", "v-south-region"],
        operator_intent="in",
    )

    result = asyncio.run(resolve_enum_predicate(mention, _context()))

    assert result.status == "resolved"
    assert result.plan.operator == "in"
    assert result.plan.canonical_values == ["华北地区", "华南地区"]
