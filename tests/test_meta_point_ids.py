from app.services.meta_point_id import build_meta_point_id


def test_meta_point_id_is_stable_for_same_business_key():
    first = build_meta_point_id(
        object_type="column",
        object_id="dim_region.region_name",
        text_role="alias",
        text="大区",
    )
    second = build_meta_point_id(
        object_type="column",
        object_id="dim_region.region_name",
        text_role="alias",
        text="大区",
    )

    assert first == second


def test_meta_point_id_changes_when_indexed_text_changes():
    first = build_meta_point_id(
        object_type="column",
        object_id="dim_region.region_name",
        text_role="alias",
        text="大区",
    )
    second = build_meta_point_id(
        object_type="column",
        object_id="dim_region.region_name",
        text_role="alias",
        text="区域",
    )

    assert first != second
