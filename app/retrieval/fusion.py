"""Candidate-level rank fusion shared by metadata retrieval routes."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from app.entities.value_info import ValueInfo

T = TypeVar("T")


@dataclass(frozen=True)
class RankedList(Generic[T]):
    """One ordered candidate list produced by one query and one backend."""

    source: str
    items: Sequence[T]
    weight: float = 1.0


@dataclass
class FusedCandidate(Generic[T]):
    """A unique candidate with its accumulated RRF evidence."""

    candidate_id: str
    item: T
    score: float = 0.0
    sources: list[str] | None = None


@dataclass
class RankedValueInfo(ValueInfo):
    """A canonical value candidate with global fusion evidence."""

    score: float = 0.0
    sources: list[str] | None = None


def fuse_candidate_rankings(
    ranked_lists: Sequence[RankedList[T]],
    *,
    candidate_id_of: Callable[[T], str],
    merge_items: Callable[[T, T], T] | None = None,
    limit: int = 20,
    k: int = 60,
) -> list[FusedCandidate[T]]:
    """Fuse every query/backend ranking once into a global candidate Top-K."""

    fused: dict[str, FusedCandidate[T]] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list.items, start=1):
            candidate_id = str(candidate_id_of(item))
            existing = fused.get(candidate_id)
            if existing is None:
                existing = FusedCandidate(
                    candidate_id=candidate_id,
                    item=item,
                    score=0.0,
                    sources=[],
                )
                fused[candidate_id] = existing
            elif merge_items is not None:
                existing.item = merge_items(existing.item, item)

            existing.score += ranked_list.weight / (k + rank)
            if ranked_list.source not in (existing.sources or []):
                existing.sources.append(ranked_list.source)

    return sorted(
        fused.values(),
        key=lambda candidate: (candidate.score, candidate.candidate_id),
        reverse=True,
    )[:limit]


def fuse_value_rankings(
    ranked_lists: Sequence[RankedList[ValueInfo]],
    *,
    limit: int = 20,
    k: int = 60,
) -> list[RankedValueInfo]:
    """Fuse value rankings while retaining matched retrieval surfaces."""

    fused = fuse_candidate_rankings(
        ranked_lists,
        candidate_id_of=lambda item: item.id,
        merge_items=_merge_value_info,
        limit=limit,
        k=k,
    )
    return [
        RankedValueInfo(
            id=candidate.item.id,
            value=candidate.item.value,
            column_id=candidate.item.column_id,
            matched_texts=list(candidate.item.matched_texts),
            score=candidate.score,
            sources=list(candidate.sources or []),
        )
        for candidate in fused
    ]


def fuse_ranked_value_infos(
    ranked_results: Mapping[str, Sequence[ValueInfo]],
    weights: Mapping[str, float] | None = None,
    limit: int = 20,
    k: int = 60,
) -> list[RankedValueInfo]:
    """Compatibility wrapper for callers that provide one list per source."""

    weights = weights or {}
    return fuse_value_rankings(
        [
            RankedList(
                source=source,
                items=items,
                weight=weights.get(source, 1.0),
            )
            for source, items in ranked_results.items()
        ],
        limit=limit,
        k=k,
    )


def _merge_value_info(existing: ValueInfo, incoming: ValueInfo) -> ValueInfo:
    if (
        existing.value != incoming.value
        or existing.column_id != incoming.column_id
    ):
        raise ValueError(f"value_candidate_identity_conflict: {existing.id}")
    return ValueInfo(
        id=existing.id,
        value=existing.value,
        column_id=existing.column_id,
        matched_texts=list(
            dict.fromkeys([*existing.matched_texts, *incoming.matched_texts])
        ),
    )


__all__ = [
    "FusedCandidate",
    "RankedList",
    "RankedValueInfo",
    "fuse_candidate_rankings",
    "fuse_ranked_value_infos",
    "fuse_value_rankings",
]
