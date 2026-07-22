"""
字段取值业务实体

该对象用于承接字段值同步链路中的结构化结果，和表、字段、指标实体保持
一致的业务层表达方式
"""

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal


@dataclass
class ValueInfo:
    """字段具体取值及其所属字段的业务表达"""

    id: str
    value: str
    column_id: str
    matched_texts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValueSearchDocument:
    """One searchable surface form pointing to one canonical value candidate."""

    id: str
    candidate_id: str
    value: str
    column_id: str
    matched_text: str
    surface_type: Literal["canonical", "alias"]


def build_value_candidate_id(column_id: str, canonical_value: str) -> str:
    """Build the stable business ID shared by all surface-form documents."""

    digest = sha256(f"{column_id}\0{canonical_value}".encode()).hexdigest()
    return f"value:{column_id}:{digest}"


__all__ = [
    "ValueInfo",
    "ValueSearchDocument",
    "build_value_candidate_id",
]
