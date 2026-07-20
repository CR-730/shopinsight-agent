from types import MappingProxyType

from app.agent.semantic_planning.catalog import (
    RelationshipCandidate,
    SemanticCandidateCatalog,
)
from app.agent.semantic_planning.draft import JoinMention
from app.agent.semantic_planning.resolvers.join import resolve_join_preferences

RELATIONSHIP_ID = "rel:dim_region.region_id=fact_order.region_id"


def _catalog():
    relationship = RelationshipCandidate(
        candidate_id=RELATIONSHIP_ID,
        left_table_id="dim_region",
        left_column_id="dim_region.region_id",
        right_table_id="fact_order",
        right_column_id="fact_order.region_id",
    )
    empty = MappingProxyType({})
    return SemanticCandidateCatalog(
        metadata_version="meta-v2",
        tables=empty,
        columns=empty,
        relationships=MappingProxyType({RELATIONSHIP_ID: relationship}),
        metrics=empty,
        values=empty,
    )


def test_resolves_controlled_left_join_preference():
    result = resolve_join_preferences(
        [
            JoinMention(
                raw_text="包括没有订单的地区",
                relationship_candidate_id=RELATIONSHIP_ID,
                join_type="left",
                left_table_candidate_id="dim_region",
            )
        ],
        catalog=_catalog(),
        trusted_sources=("包括没有订单的地区",),
    )

    assert result.status == "resolved"
    assert result.issues == ()
    assert result.preferences[0].join_type == "left"
    assert result.preferences[0].left_table_candidate_id == "dim_region"


def test_rejects_left_table_outside_relationship_endpoints():
    result = resolve_join_preferences(
        [
            JoinMention(
                raw_text="包括没有订单的地区",
                relationship_candidate_id=RELATIONSHIP_ID,
                join_type="left",
                left_table_candidate_id="dim_product",
            )
        ],
        catalog=_catalog(),
        trusted_sources=("包括没有订单的地区",),
    )

    assert result.status == "unresolved"
    assert result.preferences == ()
    assert result.issues[0].code == "join_left_table_invalid"


def test_conflicting_preferences_for_same_relationship_are_ambiguous():
    result = resolve_join_preferences(
        [
            JoinMention(
                raw_text="按地区统计",
                relationship_candidate_id=RELATIONSHIP_ID,
                join_type="inner",
            ),
            JoinMention(
                raw_text="包括没有订单的地区",
                relationship_candidate_id=RELATIONSHIP_ID,
                join_type="left",
                left_table_candidate_id="dim_region",
            ),
        ],
        catalog=_catalog(),
        trusted_sources=("按地区统计，包括没有订单的地区",),
    )

    assert result.status == "ambiguous"
    assert result.preferences == ()
    assert result.issues[0].code == "join_type_ambiguous"
