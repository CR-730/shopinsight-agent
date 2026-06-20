"""电商问数 Agent 的 LangGraph 状态定义。

设计原则：
- State 只保存跨节点必须共享的原始结构化数据。
- 能从其他字段推导出的子产物，不再重复写入 state。
- 面向 prompt 的文本格式化在节点内按需完成，不放进 state。
"""

from typing import Literal, NotRequired, Required, TypedDict

from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo
from app.entities.value_info import ValueInfo


class MetricInfoState(TypedDict, total=False):
    """SQL 生成可使用的指标上下文。"""

    name: str
    description: str
    relevant_columns: list[str]
    alias: list[str]


class ColumnInfoState(TypedDict, total=False):
    """SQL 生成可使用的字段上下文。"""

    name: str
    type: str
    role: str
    description: str
    alias: list[str]
    examples: list


class TableInfoState(TypedDict, total=False):
    """SQL 生成可使用的表结构上下文。"""

    name: str
    role: str
    description: str
    columns: list[ColumnInfoState]


class DateInfoState(TypedDict):
    """当前日期上下文。"""

    date: str
    weekday: str
    quarter: str


class DBInfoState(TypedDict):
    """数据库方言和版本上下文。"""

    dialect: str
    version: str


class MessageState(TypedDict, total=False):
    """会话历史中的原始消息。"""

    role: str
    content: str


class SqlMemoryExampleState(TypedDict, total=False):
    """历史成功 SQL 样例的原始结构。"""

    rank: int
    question: str
    sql: str
    similarity: float


class MetricBindingState(TypedDict, total=False):
    """用户指标表达解析出的 canonical 指标。"""

    raw_mention: str
    canonical_metric: str
    matched_by: str
    evidence: str
    relevant_columns: list[str]
    confidence: str


class ResolvedFilterState(TypedDict, total=False):
    """用户枚举值表达解析出的 canonical 筛选条件。"""

    raw_value: str
    canonical_value: str
    column: str
    field_alias: str
    matched_by: str
    allowed_sql_literals: list[str]


class GroupByBindingState(TypedDict, total=False):
    """用户分组表达解析出的 canonical 维度。"""

    raw_mention: str
    column: str
    field_alias: str
    matched_by: str
    confidence: str


class TimeBindingState(TypedDict, total=False):
    """用户时间表达解析出的结构化时间约束。"""

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


class BindingIssueState(TypedDict, total=False):
    """业务绑定中未解析或存在歧义的对象。"""

    type: str
    raw_text: str
    reason: str
    candidate_column: str


class BusinessBindingState(TypedDict, total=False):
    """业务绑定层产出的唯一 canonical 业务语义对象。

    下游如需指标、筛选、分组、时间、未解析项，都从这个对象中读取，
    不再在顶层 state 里保存 metric_bindings/resolved_filters 等展开字段。
    """

    metrics: list[MetricBindingState]
    filters: list[ResolvedFilterState]
    groups: list[GroupByBindingState]
    time: TimeBindingState | None
    unresolved: list[BindingIssueState]
    ambiguous: list[BindingIssueState]


class RetrievalContextState(TypedDict, total=False):
    """生产链路可消费的召回原始对象。"""

    columns: list[ColumnInfo]
    metrics: list[MetricInfo]
    values: list[ValueInfo]


class SqlContextState(TypedDict, total=False):
    """SQL 生成和修正阶段使用的上下文。"""

    tables: list[TableInfoState]
    metrics: list[MetricInfoState]
    date: DateInfoState
    db: DBInfoState


class TraceState(TypedDict, total=False):
    """仅用于调试和评测的链路轨迹。"""

    keywords: list[str]
    retrieved_columns: list[str]
    retrieved_metrics: list[str]
    retrieved_values: list[str]
    node_timings: list[dict]
    sql_explanation: str
    sql_correction_attempts: int


class OutputState(TypedDict, total=False):
    """最终返回给调用方的查询输出。"""

    rows: list[dict]
    analysis: str
    meta: dict


class FailureState(TypedDict, total=False):
    """跨节点共享的最终失败状态。

    SQL 纠错过程中的临时校验错误不写入这里；只有需要阻断链路或向调用方
    报告的最终失败才进入 Graph State。
    """

    category: Required[
        Literal[
            "input_guard",
            "business_binding",
            "sql_validation",
            "sql_execution",
            "system",
        ]
    ]
    stage: Required[str]
    code: Required[str]
    message: Required[str]
    user_message: NotRequired[str]
    disposition: Required[Literal["blocked", "failed"]]


class DataAgentState(TypedDict, total=False):
    """一次问数链路中的共享状态。"""

    # 输入与会话上下文
    query: Required[str]
    conversation_messages: NotRequired[list[MessageState]]
    sql_memory_examples: NotRequired[list[SqlMemoryExampleState]]

    # 召回上下文与调试轨迹分开：生产节点读 retrieval_context，评测读 trace。
    retrieval_context: RetrievalContextState
    trace: TraceState

    # SQL 生成上下文
    sql_context: SqlContextState
    business_binding: BusinessBindingState

    # SQL 生成结果。SQL 执行结果统一放在 output。
    sql: str
    output: NotRequired[OutputState]

    # 最终失败统一收口；节点内部可修复的临时错误不写入 Graph State。
    failure: NotRequired[FailureState | None]
