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


class FakeDWRepository:
    def __init__(self, existing=(), error: Exception | None = None):
        self.existing = set(existing)
        self.error = error
        self.calls = []

    async def column_value_exists(self, table, column, value):
        self.calls.append((table, column, value))
        if self.error:
            raise self.error
        return (table, column, value) in self.existing


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
        "v-north-alias": _value(
            "v-north-alias",
            "华北地区",
            "dim_region.region_name",
            source="meta_alias",
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


def _context(repository=None):
    return EnumResolutionContext(
        catalog=_catalog(),
        dw_repository=repository or FakeDWRepository(),
        trusted_sources=("统计华北和华南销售额",),
    )


def test_value_candidate_resolves_to_its_owned_column():
    mention = EnumPredicateMention(
        raw_text="华北",
        value_candidate_ids=["v-north"],
        column_candidate_ids=["dim_region.region_name"],
        operator_intent="eq",
    )

    result = asyncio.run(resolve_enum_predicate(mention, _context()))

    assert result.status == "resolved"
    assert result.plan.column_id == "dim_region.region_name"
    assert result.plan.canonical_values == ["华北地区"]
    assert result.plan.allowed_sql_literals == ["华北地区"]


def test_values_from_different_columns_are_ambiguous_without_dw_query():
    repository = FakeDWRepository()
    mention = EnumPredicateMention(
        raw_text="华南",
        value_candidate_ids=["v-south-region", "v-south-area"],
        column_candidate_ids=[],
        operator_intent="eq",
    )

    result = asyncio.run(resolve_enum_predicate(mention, _context(repository)))

    assert result.status == "ambiguous"
    assert result.issue.code == "filter_column_ambiguous"
    assert repository.calls == []


def test_exact_dw_fallback_is_scoped_to_one_column():
    repository = FakeDWRepository(
        existing={("dim_region", "region_name", "华北")}
    )
    mention = EnumPredicateMention(
        raw_text="华北",
        value_candidate_ids=[],
        column_candidate_ids=["dim_region.region_name"],
        operator_intent="eq",
    )

    result = asyncio.run(resolve_enum_predicate(mention, _context(repository)))

    assert repository.calls == [("dim_region", "region_name", "华北")]
    assert result.status == "resolved"
    assert result.plan.canonical_values == ["华北"]


def test_exact_dw_fallback_does_not_fuzzy_match_longer_database_value():
    repository = FakeDWRepository(
        existing={("dim_region", "region_name", "华北地区")}
    )
    mention = EnumPredicateMention(
        raw_text="华北",
        value_candidate_ids=[],
        column_candidate_ids=["dim_region.region_name"],
        operator_intent="eq",
    )

    result = asyncio.run(resolve_enum_predicate(mention, _context(repository)))

    assert result.status == "unresolved"
    assert result.issue.code == "value_not_found"
    assert repository.calls == [("dim_region", "region_name", "华北")]


def test_multiple_column_hints_never_trigger_dw_fallback():
    repository = FakeDWRepository()
    mention = EnumPredicateMention(
        raw_text="华南",
        value_candidate_ids=[],
        column_candidate_ids=[
            "dim_region.region_name",
            "dim_sales_area.area_name",
        ],
        operator_intent="eq",
    )

    result = asyncio.run(resolve_enum_predicate(mention, _context(repository)))

    assert result.status == "ambiguous"
    assert result.issue.code == "filter_column_ambiguous"
    assert repository.calls == []


def test_repository_error_is_failed_not_value_not_found():
    repository = FakeDWRepository(error=RuntimeError("mysql unavailable"))
    mention = EnumPredicateMention(
        raw_text="华北",
        value_candidate_ids=[],
        column_candidate_ids=["dim_region.region_name"],
        operator_intent="eq",
    )

    result = asyncio.run(resolve_enum_predicate(mention, _context(repository)))

    assert result.status == "failed"
    assert result.issue.phase == "system"
    assert result.issue.code == "dw_value_lookup_failed"


def test_meta_alias_candidate_is_verified_by_exact_canonical_dw_value():
    repository = FakeDWRepository(
        existing={("dim_region", "region_name", "华北地区")}
    )
    mention = EnumPredicateMention(
        raw_text="华北",
        value_candidate_ids=["v-north-alias"],
        column_candidate_ids=["dim_region.region_name"],
        operator_intent="eq",
    )

    result = asyncio.run(resolve_enum_predicate(mention, _context(repository)))

    assert result.status == "resolved"
    assert repository.calls == [
        ("dim_region", "region_name", "华北地区")
    ]


def test_in_operator_accepts_multiple_values_only_within_one_column():
    mention = EnumPredicateMention(
        raw_text="华北和华南",
        value_candidate_ids=["v-north", "v-south-region"],
        column_candidate_ids=["dim_region.region_name"],
        operator_intent="in",
    )

    result = asyncio.run(resolve_enum_predicate(mention, _context()))

    assert result.status == "resolved"
    assert result.plan.operator == "in"
    assert result.plan.canonical_values == ["华北地区", "华南地区"]
