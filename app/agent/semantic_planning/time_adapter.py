"""Narrow JioNLP adapter for deterministic, reference-time-bound parsing."""

from __future__ import annotations

import importlib
import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

TemporalGrain = Literal["day", "week", "month", "quarter", "year"]


class TimeParseAmbiguousError(ValueError):
    """The text cannot be converted to one accurate closed interval."""


class TimeParserFailure(RuntimeError):
    """The parser itself failed unexpectedly."""


@dataclass(frozen=True)
class ParsedTimeSpan:
    start_date: date
    end_date: date
    grain: TemporalGrain


def parse_time_span(
    raw_text: str,
    *,
    reference_date: date,
) -> ParsedTimeSpan:
    """Parse one LLM-selected time span; never scans or rewrites the query."""

    try:
        parsed = _parse_with_jionlp(raw_text, reference_date=reference_date)
    except ValueError as exc:
        raise TimeParseAmbiguousError("temporal_ambiguous") from exc
    except Exception as exc:
        raise TimeParserFailure("temporal_parser_failed") from exc

    if not isinstance(parsed, dict):
        raise TimeParseAmbiguousError("temporal_result_invalid")
    if parsed.get("definition") != "accurate":
        raise TimeParseAmbiguousError("temporal_not_accurate")
    if parsed.get("type") not in {"time_point", "time_span"}:
        raise TimeParseAmbiguousError("temporal_type_unsupported")
    boundaries = parsed.get("time")
    if not isinstance(boundaries, list) or len(boundaries) != 2:
        raise TimeParseAmbiguousError("temporal_boundary_invalid")
    try:
        start_date = datetime.fromisoformat(str(boundaries[0])).date()
        end_date = datetime.fromisoformat(str(boundaries[1])).date()
    except (TypeError, ValueError) as exc:
        raise TimeParseAmbiguousError("temporal_boundary_invalid") from exc
    if start_date > end_date:
        raise TimeParseAmbiguousError("temporal_boundary_reversed")
    return ParsedTimeSpan(
        start_date=start_date,
        end_date=end_date,
        grain=_infer_grain(raw_text, start_date, end_date),
    )


def _parse_with_jionlp(raw_text: str, *, reference_date: date) -> Any:
    # JioNLP prints a promotional line on first import. Keep library import
    # side effects out of API/service logs while preserving parser exceptions.
    with redirect_stdout(io.StringIO()):
        jio = importlib.import_module("jionlp")
    return jio.parse_time(
        raw_text,
        time_base={
            "year": reference_date.year,
            "month": reference_date.month,
            "day": reference_date.day,
        },
        ret_type="str",
        strict=False,
    )


def _infer_grain(
    raw_text: str,
    start_date: date,
    end_date: date,
) -> TemporalGrain:
    if "季度" in raw_text or "季" in raw_text:
        return "quarter"
    if "星期" in raw_text or "周" in raw_text:
        return "week"
    if "月" in raw_text:
        return "month"
    if "年" in raw_text:
        return "year"
    if start_date == end_date:
        return "day"
    if (end_date - start_date).days == 6:
        return "week"
    return "day"


__all__ = [
    "ParsedTimeSpan",
    "TimeParseAmbiguousError",
    "TimeParserFailure",
    "parse_time_span",
]
