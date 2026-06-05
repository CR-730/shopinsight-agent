"""
电商问数 Agent 状态定义

State 是 LangGraph 各节点之间传递和更新的共享数据
本章在用户原始问题之外，新增关键词列表和三路召回结果
并把召回到的实体整理成后续提示词更容易消费的表信息和指标信息
SQL 生成闭环会继续写入候选 SQL 以及校验错误信息，用于控制校正或执行分支
"""

from typing import NotRequired, TypedDict

from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo
from app.entities.value_info import ValueInfo


class MetricInfoState(TypedDict):
    """面向 SQL 生成提示词的指标信息"""

    name: str
    description: str
    # 指标依赖的字段 id，用来提示模型不要脱离业务口径随意计算
    relevant_columns: list[str]
    alias: list[str]


class ColumnInfoState(TypedDict):
    """表上下文中的字段信息"""

    name: str
    type: str
    role: str
    # 字段真实样例值，尤其用于辅助 where 条件里的枚举值选择
    examples: list
    description: str
    alias: list[str]


class TableInfoState(TypedDict):
    """SQL 生成阶段真正传给模型的表结构上下文"""

    name: str
    role: str
    description: str
    columns: list[ColumnInfoState]


class DateInfoState(TypedDict):
    """SQL 生成阶段使用的当前日期上下文"""

    date: str
    weekday: str
    quarter: str


class DBInfoState(TypedDict):
    """SQL 生成阶段使用的数据库环境信息"""

    dialect: str
    version: str


class MetricBindingState(TypedDict):
    """Canonical metric resolved from user language and metric catalog."""

    raw_mention: str
    canonical_metric: str
    matched_by: str
    evidence: str
    relevant_columns: list[str]
    confidence: str


class ResolvedFilterState(TypedDict):
    """Canonical enum filter resolved from user language and value catalog."""

    raw_value: str
    canonical_value: str
    column: str
    field_alias: str
    matched_by: str
    allowed_sql_literals: list[str]


class GroupByBindingState(TypedDict):
    """Canonical grouping dimension resolved from user language."""

    raw_mention: str
    column: str
    field_alias: str
    matched_by: str
    confidence: str


class TimeBindingState(TypedDict, total=False):
    """Structured time constraint resolved from user language."""

    raw_text: str
    grain: str
    year: int
    quarter: str
    month: int
    start_date: str
    end_date: str
    start_date_id: int
    end_date_id: int
    strategy: str
    required_columns: list[str]


class BindingIssueState(TypedDict):
    """Unresolved or ambiguous business object found during binding."""

    type: str
    raw_text: str
    reason: str
    candidate_column: str


class BusinessBindingState(TypedDict):
    """Single business binding object produced by business_binding."""

    metrics: list[MetricBindingState]
    filters: list[ResolvedFilterState]
    groups: list[GroupByBindingState]
    time: TimeBindingState | None
    unresolved: list[BindingIssueState]
    ambiguous: list[BindingIssueState]


class DataAgentState(TypedDict):
    """一次问数链路中的核心状态"""

    query: str  # 用户输入的查询
    conversation_history: NotRequired[str]
    sql_memory_context: NotRequired[str]
    binding_candidates: NotRequired[dict]
    keywords: list[str]  # 抽取的关键词
    retrieved_column_infos: list[ColumnInfo]  # 检索到的字段信息
    retrieved_metric_infos: list[MetricInfo]  # 检索到的指标信息
    retrieved_value_infos: list[ValueInfo]  # 检索到的取值信息

    table_infos: list[TableInfoState]  # 合并和补齐后的表结构上下文
    metric_infos: list[MetricInfoState]  # 合并后的指标上下文
    date_info: DateInfoState  # 当前日期 星期和季度信息
    db_info: DBInfoState  # 数据库方言和版本信息

    business_binding: BusinessBindingState
    metric_bindings: list[MetricBindingState]
    resolved_filters: list[ResolvedFilterState]
    groupby_bindings: list[GroupByBindingState]
    time_binding: TimeBindingState | None
    validated_enum_values: list[str]
    unresolved_bindings: list[BindingIssueState]
    ambiguous_bindings: list[BindingIssueState]

    sql: str  # 生成或校正后的SQL
    sql_explanation: NotRequired[str]
    final_answer: list[dict]  # SQL 执行结果，用于评测 trace 和最终返回

    error: str  # 校验SQL时出现的错误信息
    safety_error: str  # SQL 执行前安全/语义闸门错误信息
    user_facing_message: NotRequired[str]
    blocked_by: str  # 拦截请求的闸门节点名称
    correction_attempts: int  # SQL 已修正次数
    max_correction_attempts: int  # 单次查询允许的 SQL 最大修正次数
