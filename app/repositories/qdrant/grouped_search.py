"""Small shared primitives for candidate-level Qdrant grouping."""

from weakref import WeakKeyDictionary

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
)

_validated_group_keys: WeakKeyDictionary[object, set[tuple[str, str]]] = (
    WeakKeyDictionary()
)


async def ensure_grouped_payload_indexes(
    client,
    *,
    collection_name: str,
    group_by: str,
) -> None:
    """Ensure grouping and metadata-version payload fields are indexed."""

    for field_name in (group_by, "meta_build_version"):
        await client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
        )


async def query_grouped_points(
    client,
    *,
    collection_name: str,
    embedding: list[float],
    group_by: str,
    group_size: int,
    limit: int,
    score_threshold: float,
    meta_build_version: str | None,
):
    """Run one grouped vector query with the shared version boundary."""

    await validate_grouped_payload_coverage(
        client,
        collection_name=collection_name,
        group_by=group_by,
    )
    return await client.query_points_groups(
        collection_name=collection_name,
        query=embedding,
        group_by=group_by,
        group_size=group_size,
        limit=limit,
        score_threshold=score_threshold,
        query_filter=metadata_build_filter(meta_build_version),
    )


async def validate_grouped_payload_coverage(
    client,
    *,
    collection_name: str,
    group_by: str,
) -> None:
    """Reject legacy collections whose existing points lack the grouping key."""

    cache_key = (collection_name, group_by)
    validated = _validated_group_keys.setdefault(client, set())
    if cache_key in validated:
        return

    collection = await client.get_collection(collection_name)
    points_count = int(collection.points_count or 0)
    index_info = (collection.payload_schema or {}).get(group_by)
    indexed_points = int(getattr(index_info, "points", 0) or 0)
    if points_count and indexed_points < points_count:
        raise RuntimeError(
            "qdrant_grouping_rebuild_required: "
            f"collection={collection_name}, group_by={group_by}, "
            f"indexed_points={indexed_points}, points_count={points_count}"
        )
    validated.add(cache_key)


def metadata_build_filter(meta_build_version: str | None) -> Filter | None:
    if not meta_build_version:
        return None
    return Filter(
        must=[
            FieldCondition(
                key="meta_build_version",
                match=MatchValue(value=meta_build_version),
            )
        ]
    )


__all__ = [
    "ensure_grouped_payload_indexes",
    "metadata_build_filter",
    "query_grouped_points",
    "validate_grouped_payload_coverage",
]
