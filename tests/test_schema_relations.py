from app.agent.schema_relations import (
    build_schema_graph,
    find_unique_shortest_join_closure,
    is_valid_join_pair,
    stable_relationship_id,
    unique_join_edges,
)
from app.entities.column_info import ColumnInfo


def _column(table: str, name: str, role: str) -> ColumnInfo:
    return ColumnInfo(
        id=f"{table}.{name}",
        name=name,
        type="bigint",
        role=role,
        examples=[],
        description="",
        alias=[],
        table_id=table,
    )


def test_valid_join_pair_requires_same_name_fk_and_pk():
    foreign_key = _column("fact_order", "region_id", "foreign_key")
    primary_key = _column("dim_region", "region_id", "primary_key")

    assert is_valid_join_pair(foreign_key, primary_key) is True
    assert (
        is_valid_join_pair(
            foreign_key, _column("dim_region", "region_name", "primary_key")
        )
        is False
    )
    assert (
        is_valid_join_pair(
            _column("fact_order", "region_id", "dimension"), primary_key
        )
        is False
    )


def test_schema_graph_does_not_connect_ordinary_same_name_columns():
    graph = build_schema_graph(
        [
            _column("left_table", "region_name", "dimension"),
            _column("right_table", "region_name", "dimension"),
        ]
    )

    result = find_unique_shortest_join_closure(
        graph, {"left_table", "right_table"}
    )

    assert result.status == "unresolved"
    assert result.table_ids == frozenset()
    assert result.column_ids == frozenset()


def test_fk_with_multiple_global_same_name_primary_keys_is_ambiguous_not_edges():
    graph = build_schema_graph(
        [
            _column("fact_order", "region_id", "foreign_key"),
            _column("dim_region", "region_id", "primary_key"),
            _column("dim_area", "region_id", "primary_key"),
        ]
    )

    assert graph.adjacency["fact_order"] == ()
    assert graph.adjacency["dim_region"] == ()
    result = find_unique_shortest_join_closure(
        graph, {"fact_order", "dim_region"}
    )

    assert result.status == "ambiguous"


def test_fk_without_same_name_primary_key_does_not_create_relation():
    graph = build_schema_graph(
        [_column("fact_order", "region_id", "foreign_key")]
    )

    assert graph.adjacency["fact_order"] == ()
    assert (
        find_unique_shortest_join_closure(
            graph, {"fact_order", "dim_region"}
        ).status
        == "unresolved"
    )


def test_single_required_table_needs_no_join_path():
    result = find_unique_shortest_join_closure(
        build_schema_graph([]), {"fact_order"}
    )

    assert result.status == "success"
    assert result.table_ids == frozenset({"fact_order"})
    assert result.column_ids == frozenset()


def test_unique_direct_path_returns_both_join_keys():
    graph = build_schema_graph(
        [
            _column("fact_order", "region_id", "foreign_key"),
            _column("dim_region", "region_id", "primary_key"),
        ]
    )

    result = find_unique_shortest_join_closure(
        graph, {"fact_order", "dim_region"}
    )

    assert result.status == "success"
    assert result.table_ids == frozenset({"fact_order", "dim_region"})
    assert result.column_ids == frozenset(
        {"fact_order.region_id", "dim_region.region_id"}
    )
    assert len(result.edges) == 1
    assert result.edges[0].column_ids == frozenset(
        {"fact_order.region_id", "dim_region.region_id"}
    )


def test_unique_edges_and_relationship_ids_are_order_stable():
    forward = build_schema_graph(
        [
            _column("fact_order", "region_id", "foreign_key"),
            _column("dim_region", "region_id", "primary_key"),
        ]
    )
    reverse = build_schema_graph(
        [
            _column("dim_region", "region_id", "primary_key"),
            _column("fact_order", "region_id", "foreign_key"),
        ]
    )

    assert unique_join_edges(forward) == unique_join_edges(reverse)
    edge = unique_join_edges(forward)[0]
    assert stable_relationship_id(edge) == (
        "relationship:dim_region.region_id:fact_order.region_id"
    )


def test_unique_bridge_path_returns_bridge_table_and_both_sides_of_each_join():
    graph = build_schema_graph(
        [
            _column("fact_order", "shop_id", "foreign_key"),
            _column("bridge_shop_region", "shop_id", "primary_key"),
            _column("bridge_shop_region", "region_id", "foreign_key"),
            _column("dim_region", "region_id", "primary_key"),
        ]
    )

    result = find_unique_shortest_join_closure(
        graph, {"fact_order", "dim_region"}
    )

    assert result.status == "success"
    assert result.table_ids == frozenset(
        {"fact_order", "bridge_shop_region", "dim_region"}
    )
    assert result.column_ids == frozenset(
        {
            "fact_order.shop_id",
            "bridge_shop_region.shop_id",
            "bridge_shop_region.region_id",
            "dim_region.region_id",
        }
    )


def test_three_required_tables_are_connected_by_incremental_unique_bfs_paths():
    graph = build_schema_graph(
        [
            _column("fact_order", "shop_id", "foreign_key"),
            _column("bridge_shop", "shop_id", "primary_key"),
            _column("bridge_shop", "region_id", "foreign_key"),
            _column("dim_region", "region_id", "primary_key"),
            _column("bridge_shop", "channel_id", "foreign_key"),
            _column("dim_channel", "channel_id", "primary_key"),
        ]
    )

    result = find_unique_shortest_join_closure(
        graph, {"fact_order", "dim_region", "dim_channel"}
    )

    assert result.status == "success"
    assert result.table_ids == frozenset(
        {"fact_order", "bridge_shop", "dim_region", "dim_channel"}
    )
    assert result.column_ids == frozenset(
        {
            "fact_order.shop_id",
            "bridge_shop.shop_id",
            "bridge_shop.region_id",
            "dim_region.region_id",
            "bridge_shop.channel_id",
            "dim_channel.channel_id",
        }
    )


def test_graph_normalizes_table_and_column_identifiers():
    graph = build_schema_graph(
        [
            _column(" Fact_Order ", " Region_ID ", " FOREIGN_KEY "),
            _column("DIM_REGION", "REGION_ID", "PRIMARY_KEY"),
        ]
    )

    result = find_unique_shortest_join_closure(
        graph, {" FACT_ORDER ", "dim_region"}
    )

    assert result.status == "success"
    assert result.table_ids == frozenset({"fact_order", "dim_region"})
    assert result.column_ids == frozenset(
        {"fact_order.region_id", "dim_region.region_id"}
    )


def test_equal_length_shortest_paths_are_ambiguous():
    graph = build_schema_graph(
        [
            _column("fact_order", "shop_id", "foreign_key"),
            _column("bridge_shop_region", "shop_id", "primary_key"),
            _column("bridge_shop_region", "region_id", "foreign_key"),
            _column("dim_region", "region_id", "primary_key"),
            _column("fact_order", "area_id", "foreign_key"),
            _column("bridge_area_region", "area_id", "primary_key"),
            _column("bridge_area_region", "region_id", "foreign_key"),
            _column("dim_region", "region_id", "primary_key"),
        ]
    )

    result = find_unique_shortest_join_closure(
        graph, {"fact_order", "dim_region"}
    )

    assert result.status == "ambiguous"
    assert result.table_ids == frozenset()
    assert result.column_ids == frozenset()


def test_cycle_is_conservatively_ambiguous_even_with_unique_local_shortest_path():
    graph = build_schema_graph(
        [
            _column("table_a", "direct_id", "foreign_key"),
            _column("table_b", "direct_id", "primary_key"),
            _column("table_a", "bridge_id", "foreign_key"),
            _column("bridge_x", "bridge_id", "primary_key"),
            _column("bridge_x", "target_id", "foreign_key"),
            _column("table_b", "target_id", "primary_key"),
        ]
    )

    result = find_unique_shortest_join_closure(
        graph, {"table_a", "table_b"}
    )

    assert result.status == "ambiguous"


def test_local_equal_paths_stay_ambiguous_when_global_steiner_tree_is_unique():
    graph = build_schema_graph(
        [
            _column("table_a", "path_x_id", "foreign_key"),
            _column("bridge_x", "path_x_id", "primary_key"),
            _column("bridge_x", "x_to_b_id", "foreign_key"),
            _column("table_b", "x_to_b_id", "primary_key"),
            _column("table_a", "path_y_id", "foreign_key"),
            _column("bridge_y", "path_y_id", "primary_key"),
            _column("bridge_y", "y_to_b_id", "foreign_key"),
            _column("table_b", "y_to_b_id", "primary_key"),
            _column("bridge_x", "x_to_c_id", "foreign_key"),
            _column("table_c", "x_to_c_id", "primary_key"),
        ]
    )

    result = find_unique_shortest_join_closure(
        graph, {"table_a", "table_b", "table_c"}
    )

    # A global Steiner solver could uniquely prefer bridge_x because table_c
    # hangs from it. Version one intentionally refuses that global inference.
    assert result.status == "ambiguous"
