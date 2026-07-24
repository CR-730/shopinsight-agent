"""Retrieval-only A/B scoring with explicit candidate-level coverage."""

from dataclasses import asdict, dataclass
from typing import Any

from app.evaluation.cases import EvalCase


@dataclass(frozen=True)
class RetrievalCandidates:
    columns: list[str]
    metrics: list[str]
    values: list[str]


def apply_candidate_budget(
    candidates: RetrievalCandidates,
    *,
    limit: int,
) -> RetrievalCandidates:
    return RetrievalCandidates(
        columns=candidates.columns[:limit],
        metrics=candidates.metrics[:limit],
        values=candidates.values[:limit],
    )


@dataclass(frozen=True)
class Coverage:
    hit_count: int
    gold_count: int
    recall: float


@dataclass(frozen=True)
class RetrievalCaseScore:
    case_id: str
    hit_count: int
    gold_count: int
    recall: float
    components: dict[str, Coverage]
    missing: dict[str, list[str]]
    gold: dict[str, list[str]]
    hits: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_retrieval_case(
    case: EvalCase,
    candidates: RetrievalCandidates,
) -> RetrievalCaseScore:
    expected = {
        "columns": set(case.expected_retrieved_columns),
        "metrics": set(case.expected_metrics),
        "values": set(case.expected_values),
    }
    actual = {
        "columns": set(candidates.columns),
        "metrics": set(candidates.metrics),
        "values": set(candidates.values),
    }
    components: dict[str, Coverage] = {}
    missing: dict[str, list[str]] = {}
    hit_ids: dict[str, list[str]] = {}
    for name in ("columns", "metrics", "values"):
        hits = expected[name] & actual[name]
        gold_count = len(expected[name])
        components[name] = Coverage(
            hit_count=len(hits),
            gold_count=gold_count,
            recall=(len(hits) / gold_count if gold_count else 1.0),
        )
        missing[name] = sorted(expected[name] - actual[name])
        hit_ids[name] = sorted(hits)

    hit_count = sum(item.hit_count for item in components.values())
    gold_count = sum(item.gold_count for item in components.values())
    return RetrievalCaseScore(
        case_id=case.id,
        hit_count=hit_count,
        gold_count=gold_count,
        recall=(hit_count / gold_count if gold_count else 1.0),
        components=components,
        missing=missing,
        gold={name: sorted(items) for name, items in expected.items()},
        hits=hit_ids,
    )


def summarize_retrieval_scores(
    scores: list[RetrievalCaseScore],
) -> dict[str, Any]:
    components: dict[str, dict[str, float | int]] = {}
    unique_components: dict[str, dict[str, float | int]] = {}
    for name in ("columns", "metrics", "values"):
        hit_count = sum(score.components[name].hit_count for score in scores)
        gold_count = sum(score.components[name].gold_count for score in scores)
        components[name] = {
            "hit_count": hit_count,
            "gold_count": gold_count,
            "recall": hit_count / gold_count if gold_count else 1.0,
        }
        unique_gold = {item for score in scores for item in score.gold[name]}
        unique_hits = {item for score in scores for item in score.hits[name]}
        unique_components[name] = {
            "hit_count": len(unique_hits),
            "gold_count": len(unique_gold),
            "recall": len(unique_hits) / len(unique_gold) if unique_gold else 1.0,
        }
    unique_hit_count = sum(
        int(component["hit_count"]) for component in unique_components.values()
    )
    unique_gold_count = sum(
        int(component["gold_count"]) for component in unique_components.values()
    )
    return {
        "case_count": len(scores),
        "average_recall": (
            sum(score.recall for score in scores) / len(scores) if scores else 0.0
        ),
        "micro_recall": (
            sum(score.hit_count for score in scores)
            / sum(score.gold_count for score in scores)
            if scores and sum(score.gold_count for score in scores)
            else 0.0
        ),
        "components": components,
        "unique_gold": {
            "hit_count": unique_hit_count,
            "gold_count": unique_gold_count,
            "recall": (
                unique_hit_count / unique_gold_count if unique_gold_count else 0.0
            ),
            "components": unique_components,
        },
    }


__all__ = [
    "Coverage",
    "RetrievalCandidates",
    "RetrievalCaseScore",
    "apply_candidate_budget",
    "score_retrieval_case",
    "summarize_retrieval_scores",
]
