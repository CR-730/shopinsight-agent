"""多路召回结果融合。"""

from dataclasses import dataclass

from app.entities.value_info import ValueInfo


@dataclass
class RankedValueInfo(ValueInfo):
    """带融合分数的字段值召回结果。"""

    score: float = 0.0
    sources: list[str] | None = None


def fuse_ranked_value_infos(
    ranked_results: dict[str, list[ValueInfo]],
    weights: dict[str, float] | None = None,
    limit: int = 20,
    k: int = 60,
) -> list[RankedValueInfo]:
    """用加权 RRF 融合 ES 和向量召回结果。"""

    weights = weights or {}
    fused: dict[str, RankedValueInfo] = {}
    for source, items in ranked_results.items():
        source_weight = weights.get(source, 1.0)
        for rank, item in enumerate(items, start=1):
            if item.id not in fused:
                fused[item.id] = RankedValueInfo(
                    id=item.id,
                    value=item.value,
                    column_id=item.column_id,
                    score=0.0,
                    sources=[],
                )
            fused[item.id].score += source_weight / (k + rank)
            fused[item.id].sources.append(source)

    return sorted(
        fused.values(),
        key=lambda item: (item.score, item.id),
        reverse=True,
    )[:limit]
