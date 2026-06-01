"""Helpers for mapping JSON columns from ORM or raw SQL rows."""

import json
from typing import Any


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        return as_list(decoded)
    if isinstance(value, dict):
        return list(value.values())
    return [value]
