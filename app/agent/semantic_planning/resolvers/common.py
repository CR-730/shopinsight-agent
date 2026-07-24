"""Shared fail-closed candidate selection primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, Mapping, TypeVar

from app.agent.semantic_planning.issues import PlanningIssue

T = TypeVar("T")
SelectionStatus = Literal["resolved", "unresolved", "ambiguous"]


@dataclass(frozen=True)
class CandidateSelection(Generic[T]):
    status: SelectionStatus
    candidate_id: str | None = None
    candidate: T | None = None
    issue: PlanningIssue | None = None


def select_one_candidate(
    *,
    raw_text: str,
    candidate_ids: list[str],
    catalog: Mapping[str, T],
    issue_prefix: str = "value",
) -> CandidateSelection[T]:
    """Select a unique in-catalog ID.

    ``raw_text`` remains provenance for diagnostics. It is not a verbatim
    evidence gate because the semantic interpreter may paraphrase wording
    while preserving a controlled candidate ID.
    """
    unique_ids = list(dict.fromkeys(candidate_ids))
    invalid_ids = [
        candidate_id for candidate_id in unique_ids if candidate_id not in catalog
    ]
    if invalid_ids:
        return CandidateSelection(
            status="unresolved",
            issue=_issue(
                code="invalid_candidate_id",
                raw_text=raw_text,
                candidate_ids=invalid_ids,
            ),
        )
    if not unique_ids:
        return CandidateSelection(
            status="unresolved",
            issue=_issue(
                code=f"{issue_prefix}_not_bound",
                raw_text=raw_text,
                candidate_ids=[],
            ),
        )
    if len(unique_ids) > 1:
        return CandidateSelection(
            status="ambiguous",
            issue=_issue(
                code=f"{issue_prefix}_ambiguous",
                raw_text=raw_text,
                candidate_ids=unique_ids,
            ),
        )
    candidate_id = unique_ids[0]
    return CandidateSelection(
        status="resolved",
        candidate_id=candidate_id,
        candidate=catalog[candidate_id],
    )


def _issue(*, code: str, raw_text: str, candidate_ids: list[str]) -> PlanningIssue:
    return PlanningIssue(
        phase="resolution",
        code=code,
        source_span=raw_text,
        candidate_ids=candidate_ids,
        details={},
    )


__all__ = ["CandidateSelection", "SelectionStatus", "select_one_candidate"]
