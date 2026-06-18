"""Deterministic time expression resolver for business binding."""

from __future__ import annotations

import calendar
import re
from datetime import date

from app.agent.state import TimeBindingState


def _current_year() -> int:
    return date.today().year


def resolve_time_mentions(raw_texts: list[str]) -> TimeBindingState | None:
    best_binding: TimeBindingState | None = None
    for raw_text in raw_texts:
        binding = resolve_time_binding(raw_text)
        if binding and _specificity_score(binding) > _specificity_score(best_binding):
            best_binding = binding
    return best_binding


def _specificity_score(binding: TimeBindingState | None) -> int:
    if not binding:
        return 0
    score = 1
    for key in ("year", "start_date_id", "end_date_id", "month"):
        if binding.get(key) is not None:
            score += 1
    return score


def resolve_time_binding(text: str) -> TimeBindingState | None:
    quarter = _parse_quarter(text)
    if quarter:
        return quarter
    month = _parse_month(text)
    if month:
        return month
    day = _parse_day(text)
    if day:
        return day
    return None


def _parse_quarter(text: str) -> TimeBindingState | None:
    # 匹配 "2025年一季度" / "一季度" / "Q1" 
    pattern = re.compile(
        r"(?:(?P<year>\d{4})\s*年?\s*)?(?:第?\s*(?P<cn>[一二三四])|Q(?P<num>[1-4]))(?:季度)?",
        re.I,
    )
    match = pattern.search(text)
    if not match:
        return None
        
    quarter_num = int(match.group("num") or _cn_quarter_to_number(match.group("cn")))
    quarter = f"Q{quarter_num}"
    year_str = match.group("year")
    
    if not year_str:
        return {
            "raw_text": match.group(0).strip(),
            "grain": "quarter",
            "quarter": quarter,
            "strategy": "column_value",
            "required_columns": ["dim_date.quarter"],
        }
        
    year = int(year_str)
    start_month = (quarter_num - 1) * 3 + 1
    end_month = start_month + 2
    _, end_day = calendar.monthrange(year, end_month)
    return {
        "raw_text": match.group(0).strip(),
        "grain": "quarter",
        "year": year,
        "quarter": quarter,
        "start_date": f"{year:04d}-{start_month:02d}-01",
        "end_date": f"{year:04d}-{end_month:02d}-{end_day:02d}",
        "start_date_id": int(f"{year:04d}{start_month:02d}01"),
        "end_date_id": int(f"{year:04d}{end_month:02d}{end_day:02d}"),
        "strategy": "date_range",
        "required_columns": ["fact_order.date_id", "dim_date.year", "dim_date.quarter"],
    }


def _parse_month(text: str) -> TimeBindingState | None:
    # 优先匹配带年份，其次匹配 "x月" 无年份写法
    match = re.search(r"(?P<year>\d{4})\s*年\s*(?P<month>\d{1,2})\s*月", text)
    if not match:
        match = re.search(r"(?P<year>\d{4})-(?P<month>\d{1,2})(?!-\d)", text)
    if not match:
        m2 = re.search(r"(?<![\d-])(?P<month>\d{1,2})\s*月", text)
        if m2:
            return {
                "raw_text": m2.group(0).strip(),
                "grain": "month",
                "month": int(m2.group("month")),
                "strategy": "column_value",
                "required_columns": ["dim_date.month"],
            }
        return None
    year = int(match.group("year"))
    month = int(match.group("month"))
    if month < 1 or month > 12:
        return None
    _, end_day = calendar.monthrange(year, month)
    return {
        "raw_text": match.group(0).strip(),
        "grain": "month",
        "year": year,
        "month": month,
        "start_date": f"{year:04d}-{month:02d}-01",
        "end_date": f"{year:04d}-{month:02d}-{end_day:02d}",
        "start_date_id": int(f"{year:04d}{month:02d}01"),
        "end_date_id": int(f"{year:04d}{month:02d}{end_day:02d}"),
        "strategy": "date_range",
        "required_columns": ["fact_order.date_id"],
    }


def _parse_day(text: str) -> TimeBindingState | None:
    match = re.search(r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})", text)
    if not match:
        return None
    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    if month < 1 or month > 12:
        return None
    _, end_day = calendar.monthrange(year, month)
    if day < 1 or day > end_day:
        return None
    return {
        "raw_text": match.group(0),
        "grain": "day",
        "year": year,
        "start_date": f"{year:04d}-{month:02d}-{day:02d}",
        "end_date": f"{year:04d}-{month:02d}-{day:02d}",
        "start_date_id": int(f"{year:04d}{month:02d}{day:02d}"),
        "end_date_id": int(f"{year:04d}{month:02d}{day:02d}"),
        "strategy": "date_range",
        "required_columns": ["fact_order.date_id"],
    }


def _cn_quarter_to_number(value: str | None) -> int:
    return {"一": 1, "二": 2, "三": 3, "四": 4}[value or "一"]
