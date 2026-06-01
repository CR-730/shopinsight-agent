"""Post-RAG semantic guard for business concepts before SQL generation."""

import re
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.conf.policy_config import load_policy_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


class BusinessEnumValue(BaseModel):
    """One explicit enum filter extracted from a user question."""

    field: str = Field(description="Field semantics, such as 地区, 品牌, or 品类.")
    value: str = Field(description="The enum value exactly as expressed by the user.")


class BusinessIntentExtraction(BaseModel):
    """Structured business intent extracted by the LLM."""

    metrics: list[str] = Field(
        default_factory=list,
        description="Explicit business metrics requested by the user.",
    )
    enum_values: list[BusinessEnumValue] = Field(
        default_factory=list,
        description="Explicit enum filters requested by the user.",
    )


async def semantic_guard(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """Block unknown metrics and enum values after retrieval/context filtering."""

    writer = runtime.stream_writer
    step = "RAG后业务语义闸门"
    writer({"type": "progress", "step": step, "status": "running"})

    extracted_intent = await extract_business_intent(state, runtime)
    rule_error = await validate_business_semantics_from_extracted_intent(
        state,
        extracted_intent,
        runtime,
    )
    if rule_error:
        logger.warning(f"{step} blocked query: {rule_error}")
        writer(
            {"type": "progress", "step": step, "status": "blocked", "error": rule_error}
        )
        return {"safety_error": rule_error, "blocked_by": "semantic_guard"}

    writer({"type": "progress", "step": step, "status": "success"})
    return {"safety_error": None}


async def validate_business_semantics_from_extracted_intent(
    state: dict[str, Any],
    extracted_intent: dict[str, Any],
    runtime: Runtime[DataAgentContext] | None = None,
    columns: list[Any] | None = None,
) -> str | None:
    """Deterministically check extracted business intent against metadata."""

    known_metric_names = await _known_metric_names(state, runtime)
    has_bound_metric_candidate = _has_bound_metric_candidate(state)
    for metric in _extracted_metrics(extracted_intent):
        if not _is_known_or_allowed_metric(
            metric, known_metric_names, has_bound_metric_candidate
        ):
            return f"用户请求的指标未在元数据中确认：{metric}"

    known_values = _known_enum_values(state)
    columns = columns if columns is not None else await _catalog_columns(runtime)
    for enum_value in _extracted_enum_values(extracted_intent):
        value = enum_value.get("value")
        if value and not await _is_known_enum_value(
            value,
            known_values,
            enum_value.get("field") or "",
            columns,
            runtime,
        ):
            return f"用户请求的枚举值未在召回结果中确认：{value}"

    return None


async def extract_business_intent(
    state: dict[str, Any], runtime: Runtime[DataAgentContext]
) -> dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=BusinessIntentExtraction)
    prompt = PromptTemplate(
        template=load_prompt("extract_business_intent"),
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    try:
        result = await ainvoke_llm_with_usage(
            prompt,
            llm,
            parser,
            {
                "query": state.get("query") or "",
            },
            "RAG后业务意图抽取",
            runtime.context["cost_tracker"],
            app_config.llm.timeout_seconds,
        )
        extracted = result.model_dump()
        return _merge_rule_extracted_intent(state.get("query") or "", extracted)
    except Exception as exc:
        logger.warning(f"RAG 后业务意图抽取失败，降级为规则抽取: {exc}")
        return _merge_rule_extracted_intent(state.get("query") or "", {})


def _merge_rule_extracted_intent(
    query: str,
    extracted_intent: dict[str, Any],
) -> dict[str, Any]:
    metrics = list(_extracted_metrics(extracted_intent))
    enum_values = list(_extracted_enum_values(extracted_intent))

    for metric in _metric_candidates(query):
        if metric not in metrics:
            metrics.append(metric)

    region_candidate = _region_candidate(query)
    if region_candidate and not any(
        enum_value.get("value") == region_candidate for enum_value in enum_values
    ):
        enum_values.append({"field": "地区", "value": region_candidate})

    return {"metrics": metrics, "enum_values": enum_values}


def _metric_candidates(query: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(
        r"([\u4e00-\u9fffA-Za-z0-9]+?(指数|客单价|销售额|成交额|销售金额|销量))",
        query,
    ):
        candidate = match.group(1).split("的")[-1]
        candidate = re.sub(r"^(按|统计|查询|计算|看看|请问)+", "", candidate)
        if candidate:
            candidates.append(candidate)
    return candidates


def _known_enum_values(state: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for value_info in state.get("retrieved_value_infos") or []:
        value = getattr(value_info, "value", None)
        value_id = getattr(value_info, "id", None)
        if value:
            values.add(str(value))
        if value_id:
            values.add(str(value_id))
    return values


def _extracted_metrics(extracted_intent: dict[str, Any]) -> list[str]:
    metrics = extracted_intent.get("metrics") or []
    result = []
    for metric in metrics:
        normalized = _normalize_metric_text(str(metric))
        if normalized:
            result.append(normalized)
    return result


def _extracted_enum_values(extracted_intent: dict[str, Any]) -> list[dict[str, str]]:
    enum_values = extracted_intent.get("enum_values") or []
    normalized = []
    for enum_value in enum_values:
        if not isinstance(enum_value, dict):
            continue
        value = _normalize_enum_value(str(enum_value.get("value") or ""))
        if value and value not in _generic_enum_values():
            normalized.append(
                {
                    "field": str(enum_value.get("field") or "").strip(),
                    "value": value,
                }
            )
    return normalized


def _region_candidate(query: str) -> str | None:
    for match in re.finditer(r"([\u4e00-\u9fffA-Za-z0-9]+?)(地区|区域|大区)", query):
        candidate = _normalize_enum_value(match.group(1))
        if _looks_like_group_dimension_reference(candidate):
            continue
        if candidate and candidate not in _generic_enum_values():
            return candidate
    return None


def _normalize_metric_text(metric: str) -> str:
    metric = metric.strip()
    metric = metric.split("的")[-1]
    metric = re.sub(r"^(按|统计|查询|计算|看看|请问)+", "", metric)
    return metric.strip()


def _normalize_enum_value(value: str) -> str:
    value = value.strip()
    value = value.split("的")[-1]
    value = re.sub(r"^(按|统计|查询|计算|看看|请问)+", "", value)
    value = re.sub(r"(是多少|多少|如何|怎样)$", "", value)
    for suffix in load_policy_config().get("semantic", {}).get("enum_suffixes", []):
        if value.endswith(suffix) and len(value) > len(suffix):
            value = value[: -len(suffix)]
            break
    aliases = load_policy_config().get("semantic", {}).get("enum_aliases", {})
    return aliases.get(value, value)


def _is_known_or_allowed_metric(
    metric: str,
    known_metric_names: set[str],
    has_bound_metric_candidate: bool = False,
) -> bool:
    if has_bound_metric_candidate:
        return True

    derived_metrics = load_policy_config().get("semantic", {}).get(
        "derived_metric_columns", {}
    )
    if metric in derived_metrics:
        return True
    return any(
        metric == known_metric
        or metric in known_metric
        or known_metric in metric
        for known_metric in known_metric_names
    )


async def _is_known_enum_value(
    value: str,
    known_values: set[str],
    field_semantics: str,
    columns,
    runtime: Runtime[DataAgentContext] | None,
) -> bool:
    if any(
        value == known_value
        or value in known_value
        or known_value in value
        for known_value in known_values
    ):
        return True

    if _is_temporal_field(field_semantics):
        return True

    if _is_dimension_reference(value, columns):
        return True

    if runtime is None:
        return False

    candidate_columns = _candidate_enum_columns(columns, field_semantics)
    if not candidate_columns:
        return True

    for column in candidate_columns:
        exists = await runtime.context["dw_mysql_repository"].column_value_exists(
            column.table_id,
            column.name,
            value,
        )
        if exists:
            return True
    return False


async def _known_metric_names(
    state: dict[str, Any],
    runtime: Runtime[DataAgentContext] | None,
) -> set[str]:
    names = _metric_names_from_state(state)
    if runtime is not None:
        columns = await _catalog_columns(runtime)
        metrics = await runtime.context["meta_mysql_repository"].list_metric_infos()
        column_aliases = {
            column.id: set((column.alias or []) + [column.name]) for column in columns
        }
        for metric in metrics:
            names.add(metric.name)
            names.update(metric.alias or [])
            for column_id in metric.relevant_columns or []:
                names.update(column_aliases.get(column_id, set()))
    return names


def _has_bound_metric_candidate(state: dict[str, Any]) -> bool:
    for metric in state.get("metric_infos") or []:
        if isinstance(metric, dict):
            name = metric.get("name")
            relevant_columns = metric.get("relevant_columns") or []
        else:
            name = getattr(metric, "name", None)
            relevant_columns = getattr(metric, "relevant_columns", []) or []
        if name and relevant_columns:
            return True
    return False


def _metric_names_from_state(state: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for metric in state.get("metric_infos") or []:
        if isinstance(metric, dict):
            name = metric.get("name")
            aliases = metric.get("alias") or []
        else:
            name = getattr(metric, "name", None)
            aliases = getattr(metric, "alias", []) or []
        if name:
            names.add(str(name))
        names.update(str(alias) for alias in aliases if alias)
    return names


async def _catalog_columns(runtime: Runtime[DataAgentContext] | None):
    if runtime is None:
        return []
    return await runtime.context["meta_mysql_repository"].list_column_infos()


def _candidate_enum_columns(columns, field_semantics: str):
    field_semantics = field_semantics.strip()
    candidates = []
    for column in columns:
        if column.role != "dimension":
            continue
        names = {column.name, column.description, *(column.alias or [])}
        if not field_semantics or any(field_semantics in str(name) for name in names):
            candidates.append(column)
    return candidates


def _generic_enum_values() -> set[str]:
    return set(load_policy_config().get("semantic", {}).get("generic_enum_values", []))


def _is_temporal_field(field_semantics: str) -> bool:
    temporal_fields = load_policy_config().get("semantic", {}).get(
        "temporal_fields", []
    )
    return any(field and field in field_semantics for field in temporal_fields)


def _looks_like_group_dimension_reference(value: str) -> bool:
    value = value.strip()
    if value in {"各", "每个", "不同"}:
        return True
    return bool(re.search(r"(年|季度|月份?|日期|时间).*(各|每个|不同)$", value))


def _is_dimension_reference(value: str, columns) -> bool:
    for column in columns:
        if getattr(column, "role", None) != "dimension":
            continue
        names = {
            str(getattr(column, "name", "") or ""),
            str(getattr(column, "description", "") or ""),
            *(str(alias) for alias in (getattr(column, "alias", []) or [])),
        }
        if any(value and value in name for name in names):
            return True
    return False
