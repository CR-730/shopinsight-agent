from app.entities.column_info import ColumnInfo
from app.entities.value_info import ValueInfo
from app.retrieval.fusion import (
    RankedList,
    RankedValueInfo,
    fuse_candidate_rankings,
    fuse_ranked_value_infos,
    fuse_value_rankings,
)


def _column(column_id: str) -> ColumnInfo:
    table_id, name = column_id.split(".", 1)
    return ColumnInfo(
        id=column_id,
        name=name,
        type="varchar(32)",
        role="dimension",
        examples=[],
        description=name,
        alias=[],
        table_id=table_id,
    )


def test_fuse_ranked_value_infos_deduplicates_and_prefers_multi_source_hits():
    es_results = [
        RankedValueInfo(id="dim_region.region_name.华东", value="华东", column_id="dim_region.region_name"),
        RankedValueInfo(id="dim_region.region_name.华北", value="华北", column_id="dim_region.region_name"),
    ]
    vector_results = [
        RankedValueInfo(id="dim_region.region_name.华北", value="华北", column_id="dim_region.region_name"),
        RankedValueInfo(id="dim_region.region_name.华南", value="华南", column_id="dim_region.region_name"),
    ]

    fused = fuse_ranked_value_infos(
        {"es": es_results, "vector": vector_results},
        weights={"es": 1.0, "vector": 1.0},
        limit=3,
    )

    assert [item.id for item in fused] == [
        "dim_region.region_name.华北",
        "dim_region.region_name.华东",
        "dim_region.region_name.华南",
    ]


def test_fuse_ranked_value_infos_respects_source_weight():
    es_results = [
        RankedValueInfo(id="a", value="a", column_id="c"),
    ]
    vector_results = [
        RankedValueInfo(id="b", value="b", column_id="c"),
    ]

    fused = fuse_ranked_value_infos(
        {"es": es_results, "vector": vector_results},
        weights={"es": 1.0, "vector": 3.0},
    )

    assert [item.id for item in fused] == ["b", "a"]


def test_fuse_ranked_value_infos_merges_surface_evidence_by_candidate_id():
    fused = fuse_ranked_value_infos(
        {
            "es": [
                RankedValueInfo(
                    id="value:region:north",
                    value="华北",
                    column_id="dim_region.region_name",
                    matched_texts=["北方区域"],
                )
            ],
            "vector": [
                RankedValueInfo(
                    id="value:region:north",
                    value="华北",
                    column_id="dim_region.region_name",
                    matched_texts=["华北"],
                )
            ],
        }
    )

    assert len(fused) == 1
    assert fused[0].matched_texts == ["北方区域", "华北"]
    assert fused[0].sources == ["es", "vector"]


def test_global_rrf_fuses_same_entity_across_all_query_rankings():
    north = _column("dim_region.region_name")
    category = _column("dim_product.category")
    province = _column("dim_region.province")

    fused = fuse_candidate_rankings(
        [
            RankedList(source="vector", items=[category, north]),
            RankedList(source="vector", items=[province, north]),
        ],
        candidate_id_of=lambda item: item.id,
        limit=2,
    )

    assert [candidate.item.id for candidate in fused] == [
        "dim_region.region_name",
        "dim_region.province",
    ]


def test_global_value_rrf_merges_backends_queries_and_evidence_once():
    north = ValueInfo(
        id="value:north",
        value="华北",
        column_id="dim_region.region_name",
        matched_texts=["北方区域"],
    )
    canonical_north = ValueInfo(
        id="value:north",
        value="华北",
        column_id="dim_region.region_name",
        matched_texts=["华北"],
    )

    fused = fuse_value_rankings(
        [
            RankedList(source="es", weight=1.2, items=[north]),
            RankedList(source="vector", weight=1.0, items=[canonical_north]),
            RankedList(source="vector", weight=1.0, items=[north]),
        ],
        limit=20,
    )

    assert len(fused) == 1
    assert fused[0].id == "value:north"
    assert fused[0].matched_texts == ["北方区域", "华北"]
    assert fused[0].sources == ["es", "vector"]
    assert fused[0].score == (
        1.2 / 61
        + 1.0 / 61
        + 1.0 / 61
    )
