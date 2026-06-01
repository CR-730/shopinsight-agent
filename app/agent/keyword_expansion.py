"""Utilities for normalizing LLM keyword expansion output."""

from collections.abc import Iterable
from typing import Any


def normalize_keyword_list(value: Any) -> list[str]:
    """Convert common LLM JSON shapes into a flat list of non-empty strings."""

    normalized: list[str] = []

    def add(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            keyword = item.strip()
            if keyword:
                normalized.append(keyword)
            return
        if isinstance(item, dict):
            for key in ("keywords", "items", "result", "values", "fields"):
                if key in item:
                    add(item[key])
                    return
            for key in ("keyword", "name", "value", "text"):
                if key in item:
                    add(item[key])
                    return
            return
        if isinstance(item, Iterable):
            for child in item:
                add(child)

    add(value)
    return list(dict.fromkeys(normalized))
