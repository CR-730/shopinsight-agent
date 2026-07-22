from types import MappingProxyType

from app.agent.semantic_planning.catalog import (
    ColumnCandidate,
    MetricCandidate,
    RelationshipCandidate,
    SemanticCandidateCatalog,
    TableCandidate,
    ValueCandidate,
)
from app.agent.semantic_planning.plan import (
    DimensionPlan,
    EnumPredicate,
    JoinPlan,
    MeasurePlan,
    NumericPredicate,
    OrderByPlan,
    SemanticQueryPlan,
)
from app.agent.semantic_planning.plan_validator import validate_plan
from app.agent.semantic_planning.resolvers.join import ResolvedJoinPreference


def _column(column_id, role, data_type="bigint"):
    table, name = column_id.split(".", 1)
    return ColumnCandidate(
        candidate_id=column_id,
        table=table,
        name=name,
        aliases=(),
        role=role,
        projectable=True,
        data_type=data_type,
    )


def _relationship(left_column, right_column):
    left_table = left_column.split(".", 1)[0]
    right_table = right_column.split(".", 1)[0]
    relationship_id = f"relationship:{left_column}:{right_column}"
    return RelationshipCandidate(
        candidate_id=relationship_id,
        left_table_id=left_table,
        left_column_id=left_column,
        right_table_id=right_table,
        right_column_id=right_column,
    )


def _catalog(*, extra_columns=None, extra_relationships=None):
    columns = {
        "fact_order.order_amount": _column(
            "fact_order.order_amount", "measure", "decimal"
        ),
        "fact_order.region_id": _column("fact_order.region_id", "foreign_key"),
        "dim_region.region_id": _column("dim_region.region_id", "primary_key"),
        "dim_region.region_name": _column(
            "dim_region.region_name", "dimension", "varchar"
        ),
    }
    columns.update(extra_columns or {})
    relationship = _relationship("fact_order.region_id", "dim_region.region_id")
    relationships = {relationship.candidate_id: relationship}
    relationships.update(extra_relationships or {})
    table_ids = {column.table for column in columns.values()}
    return SemanticCandidateCatalog(
        metadata_version="meta-v2",
        tables=MappingProxyType(
            {
                table_id: TableCandidate(
                    candidate_id=table_id,
                    name=table_id,
                    role="fact" if table_id == "fact_order" else "dim",
                    description="",
                )
                for table_id in table_ids
            }
        ),
        columns=MappingProxyType(columns),
        relationships=MappingProxyType(relationships),
        metrics=MappingProxyType(
            {
                "GMV": MetricCandidate(
                    candidate_id="GMV",
                    name="GMV",
                    aliases=("销售额",),
                    relevant_columns=("fact_order.order_amount",),
                    aggregation="sum",
                )
            }
        ),
        values=MappingProxyType(
            {
                "v-north": ValueCandidate(
                    candidate_id="v-north",
                    canonical_value="华北地区",
                    aliases=("华北",),
                    column_id="dim_region.region_name",
                    source="retrieval",
                )
            }
        ),
    )


def _plan(**changes):
    values = {
        "version": "1",
        "metadata_version": "meta-v2",
        "measures": [
            MeasurePlan(
                metric_id="GMV",
                name="GMV",
                aggregation="sum",
                source_column_ids=["fact_order.order_amount"],
                output_alias="销售额",
            )
        ],
        "dimensions": [
            DimensionPlan(
                column_id="dim_region.region_name",
                role="group_by",
                output_alias="地区",
            )
        ],
    }
    values.update(changes)
    return SemanticQueryPlan(**values)


def test_validator_materializes_unique_join_closure():
    result = validate_plan(_plan(), _catalog())

    assert result.status == "resolved"
    assert result.plan.required_table_ids == ["dim_region", "fact_order"]
    assert result.plan.required_column_ids == [
        "dim_region.region_id",
        "dim_region.region_name",
        "fact_order.order_amount",
        "fact_order.region_id",
    ]
    assert {
        item.column_id: item.data_type for item in result.plan.required_columns
    } == {
        "dim_region.region_id": "bigint",
        "dim_region.region_name": "varchar",
        "fact_order.order_amount": "decimal",
        "fact_order.region_id": "bigint",
    }
    assert result.plan.joins == [
        JoinPlan(
            left_column_id="dim_region.region_id",
            right_column_id="fact_order.region_id",
            join_type="inner",
        )
    ]


def test_validator_applies_left_join_preference_and_direction():
    catalog = _catalog()
    relationship_id = next(iter(catalog.relationships))

    result = validate_plan(
        _plan(),
        catalog,
        join_preferences=(
            ResolvedJoinPreference(
                relationship_candidate_id=relationship_id,
                join_type="left",
                left_table_candidate_id="dim_region",
            ),
        ),
    )

    assert result.status == "resolved"
    assert result.plan.joins == [
        JoinPlan(
            left_column_id="dim_region.region_id",
            right_column_id="fact_order.region_id",
            join_type="left",
        )
    ]


def test_validator_ignores_inner_join_preference_outside_required_closure():
    unused = _relationship("fact_order.product_id", "dim_product.product_id")
    catalog = _catalog(
        extra_columns={
            "fact_order.product_id": _column("fact_order.product_id", "foreign_key"),
            "dim_product.product_id": _column("dim_product.product_id", "primary_key"),
        },
        extra_relationships={unused.candidate_id: unused},
    )

    result = validate_plan(
        _plan(),
        catalog,
        join_preferences=(
            ResolvedJoinPreference(
                relationship_candidate_id=unused.candidate_id,
                join_type="inner",
            ),
        ),
    )

    assert result.status == "resolved"
    assert result.plan.required_table_ids == ["dim_region", "fact_order"]
    assert len(result.plan.joins) == 1


def test_validator_rejects_left_join_preference_outside_required_closure():
    unused = _relationship("fact_order.product_id", "dim_product.product_id")
    catalog = _catalog(
        extra_columns={
            "fact_order.product_id": _column("fact_order.product_id", "foreign_key"),
            "dim_product.product_id": _column("dim_product.product_id", "primary_key"),
        },
        extra_relationships={unused.candidate_id: unused},
    )

    result = validate_plan(
        _plan(),
        catalog,
        join_preferences=(
            ResolvedJoinPreference(
                relationship_candidate_id=unused.candidate_id,
                join_type="left",
                left_table_candidate_id="dim_product",
            ),
        ),
    )

    assert result.status == "unresolved"
    assert result.issues[0].code == "join_not_required"


def test_validator_rejects_metadata_version_and_prepopulated_closure():
    wrong_version = validate_plan(_plan(metadata_version="old-meta"), _catalog())
    prepopulated = validate_plan(_plan(required_table_ids=["fact_order"]), _catalog())

    assert wrong_version.status == "unresolved"
    assert wrong_version.issues[0].code == "metadata_version_mismatch"
    assert prepopulated.status == "unresolved"
    assert prepopulated.issues[0].code == "plan_closure_must_be_empty"


def test_validator_rejects_metric_definition_tampering():
    result = validate_plan(
        _plan(
            measures=[
                MeasurePlan(
                    metric_id="GMV",
                    name="GMV",
                    aggregation="avg",
                    source_column_ids=["fact_order.order_amount"],
                    output_alias="销售额",
                )
            ]
        ),
        _catalog(),
    )

    assert result.status == "unresolved"
    assert result.issues[0].code == "metric_definition_mismatch"


def test_validator_checks_numeric_scope_and_order_target():
    invalid_scope = validate_plan(
        _plan(
            predicates=[
                NumericPredicate(
                    target_type="measure",
                    target_id="GMV",
                    operator="gt",
                    values=["10000"],
                    clause="where",
                )
            ]
        ),
        _catalog(),
    )
    invalid_order = validate_plan(
        _plan(
            order_by=[
                OrderByPlan(
                    target_type="measure",
                    target_id="AOV",
                    direction="desc",
                )
            ]
        ),
        _catalog(),
    )

    assert invalid_scope.issues[0].code == "numeric_clause_mismatch"
    assert invalid_order.issues[0].code == "order_target_not_selected"


def test_validator_rejects_non_finite_numeric_literal():
    result = validate_plan(
        _plan(
            predicates=[
                NumericPredicate(
                    target_type="column",
                    target_id="fact_order.order_amount",
                    operator="gte",
                    values=["NaN"],
                    clause="where",
                )
            ]
        ),
        _catalog(),
    )

    assert result.status == "unresolved"
    assert result.issues[0].code == "numeric_value_invalid"


def test_validator_materializes_canonical_multi_value_enum_predicate():
    result = validate_plan(
        _plan(
            predicates=[
                EnumPredicate(
                    column_id="dim_region.region_name",
                    operator="in",
                    canonical_values=["华北地区"],
                ),
                EnumPredicate(
                    column_id="dim_region.region_name",
                    operator="in",
                    canonical_values=["华南地区"],
                ),
            ]
        ),
        _catalog(),
    )

    assert result.status == "resolved"
    assert result.plan.predicates == [
        EnumPredicate(
            column_id="dim_region.region_name",
            operator="in",
            canonical_values=["华北地区", "华南地区"],
        )
    ]


def test_validator_blocks_conflicting_enum_predicates_before_join_resolution():
    result = validate_plan(
        _plan(
            predicates=[
                EnumPredicate(
                    column_id="dim_region.region_name",
                    operator="eq",
                    canonical_values=["华北地区"],
                ),
                EnumPredicate(
                    column_id="dim_region.region_name",
                    operator="eq",
                    canonical_values=["华南地区"],
                ),
            ]
        ),
        _catalog(),
    )

    assert result.status == "unresolved"
    assert [issue.code for issue in result.issues] == ["predicate_conflict"]


def test_validator_requires_a_measure_or_projection_and_valid_limit():
    no_output = validate_plan(
        _plan(
            measures=[],
            dimensions=[
                DimensionPlan(
                    column_id="dim_region.region_name",
                    role="group_by",
                    output_alias="地区",
                )
            ],
        ),
        _catalog(),
    )
    bad_limit = validate_plan(_plan(limit=1001), _catalog())

    assert no_output.issues[0].code == "business_object_not_planned"
    assert bad_limit.issues[0].code == "limit_out_of_range"


def test_validator_reports_missing_and_ambiguous_join_paths():
    no_relationships = _catalog()
    no_relationships = SemanticCandidateCatalog(
        metadata_version=no_relationships.metadata_version,
        tables=no_relationships.tables,
        columns=no_relationships.columns,
        relationships=MappingProxyType({}),
        metrics=no_relationships.metrics,
        values=no_relationships.values,
    )
    missing = validate_plan(_plan(), no_relationships)

    extra_columns = {
        "fact_order.area_id": _column("fact_order.area_id", "foreign_key"),
        "dim_area.area_id": _column("dim_area.area_id", "primary_key"),
        "dim_area.region_id": _column("dim_area.region_id", "foreign_key"),
    }
    first = _relationship("fact_order.area_id", "dim_area.area_id")
    second = _relationship("dim_area.region_id", "dim_region.region_id")
    ambiguous_catalog = _catalog(
        extra_columns=extra_columns,
        extra_relationships={
            first.candidate_id: first,
            second.candidate_id: second,
        },
    )
    ambiguous = validate_plan(_plan(), ambiguous_catalog)

    assert missing.status == "unresolved"
    assert missing.issues[0].code == "join_path_not_found"
    assert ambiguous.status == "ambiguous"
    assert ambiguous.issues[0].code == "join_path_ambiguous"
