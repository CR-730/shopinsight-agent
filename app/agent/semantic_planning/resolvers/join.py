"""Resolve controlled JOIN type choices without accepting SQL expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.agent.semantic_planning.catalog import SemanticCandidateCatalog
from app.agent.semantic_planning.draft import JoinMention
from app.agent.semantic_planning.issues import PlanningIssue

JoinPreferenceStatus = Literal["resolved", "unresolved", "ambiguous"]


@dataclass(frozen=True)
class ResolvedJoinPreference:
    relationship_candidate_id: str
    join_type: Literal["inner", "left"]
    left_table_candidate_id: str | None = None


@dataclass(frozen=True)
class JoinPreferenceResolution:
    status: JoinPreferenceStatus
    preferences: tuple[ResolvedJoinPreference, ...] = ()
    issues: tuple[PlanningIssue, ...] = ()


def resolve_join_preferences(
    mentions: list[JoinMention],
    *,
    catalog: SemanticCandidateCatalog,
    trusted_sources: tuple[str, ...],
) -> JoinPreferenceResolution:
    """Validate source spans, relationship IDs, direction, and conflicts."""

    preferences: dict[str, ResolvedJoinPreference] = {}
    issues: list[PlanningIssue] = []
    ambiguous = False
    for mention in mentions:
        if not mention.raw_text or not any(
            mention.raw_text in source for source in trusted_sources
        ):
            issues.append(_issue("untrusted_source_span", mention))
            continue
        relationship = catalog.relationships.get(
            mention.relationship_candidate_id
        )
        if relationship is None:
            issues.append(_issue("invalid_candidate_id", mention))
            continue
        endpoint_tables = {
            relationship.left_table_id,
            relationship.right_table_id,
        }
        if mention.join_type == "left" and (
            mention.left_table_candidate_id not in endpoint_tables
        ):
            issues.append(_issue("join_left_table_invalid", mention))
            continue
        preference = ResolvedJoinPreference(
            relationship_candidate_id=mention.relationship_candidate_id,
            join_type=mention.join_type,
            left_table_candidate_id=(
                mention.left_table_candidate_id
                if mention.join_type == "left"
                else None
            ),
        )
        existing = preferences.get(mention.relationship_candidate_id)
        if existing is not None and existing != preference:
            issues.append(_issue("join_type_ambiguous", mention))
            ambiguous = True
            continue
        preferences[mention.relationship_candidate_id] = preference

    if issues:
        return JoinPreferenceResolution(
            status="ambiguous" if ambiguous else "unresolved",
            issues=tuple(issues),
        )
    return JoinPreferenceResolution(
        status="resolved",
        preferences=tuple(preferences.values()),
    )


def _issue(code: str, mention: JoinMention) -> PlanningIssue:
    return PlanningIssue(
        phase="resolution",
        code=code,
        source_span=mention.raw_text,
        candidate_ids=[mention.relationship_candidate_id],
        details={},
    )


__all__ = [
    "JoinPreferenceResolution",
    "ResolvedJoinPreference",
    "resolve_join_preferences",
]
