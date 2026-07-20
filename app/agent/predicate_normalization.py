"""Shared canonical predicate primitives for plans and SQL AST adapters."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Iterable


def canonical_set_values(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values}))


def canonical_number(value: object) -> str:
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("numeric_literal_must_be_finite")
    normalized = format(number.normalize(), "f")
    return "0" if Decimal(normalized) == 0 else normalized


def stable_fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "canonical_number",
    "canonical_set_values",
    "stable_fingerprint",
]
