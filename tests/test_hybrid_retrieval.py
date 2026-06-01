from app.retrieval.fusion import RankedValueInfo, fuse_ranked_value_infos


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
