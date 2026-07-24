"""
应用主配置

定义 conf/app_config.yaml 在程序中的结构化配置对象
项目启动后会在这里一次性完成配置文件加载和类型化转换，其他模块只需要导入 app_config
就可以按属性方式读取日志 MySQL Qdrant Embedding Elasticsearch 和 LLM 配置
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf


@dataclass
class File:
    """文件日志配置"""

    enable: bool
    level: str
    path: str
    rotation: str
    retention: str


@dataclass
class Console:
    """控制台日志配置"""

    enable: bool
    level: str


@dataclass
class LoggingConfig:
    """日志总配置"""

    file: File
    console: Console


@dataclass
class DBConfig:
    """MySQL 连接配置"""

    host: str
    port: int
    user: str
    password: str
    database: str


@dataclass
class QdrantConfig:
    """Qdrant 连接与向量维度配置"""

    host: str
    port: int
    embedding_size: int


@dataclass
class EmbeddingConfig:
    """Embedding 服务配置"""

    base_url: str
    api_key: str
    model: str


@dataclass
class ESConfig:
    """Elasticsearch 配置"""

    host: str
    port: int
    index_name: str


@dataclass
class LLMConfig:
    """大模型调用配置"""

    provider: str
    model: str
    fast_model: str
    api_key: str
    base_url: str
    timeout_seconds: int
    structured_enable_thinking: bool
    generate_sql_enable_thinking: bool
    correct_sql_enable_thinking: bool
    input_per_1m_tokens: float
    output_per_1m_tokens: float
    max_retries: int
    retry_backoff_seconds: float
    concurrency_limit: int
    quota_circuit_breaker_seconds: int
    rate_limit_breaker_threshold: int
    error_window_seconds: int
    error_window_min_calls: int
    error_rate_threshold: float
    max_calls_per_request: int
    fast_max_retries: int
    fast_concurrency_limit: int
    fast_quota_circuit_breaker_seconds: int
    sql_max_retries: int
    sql_concurrency_limit: int
    sql_quota_circuit_breaker_seconds: int


@dataclass
class AgentConfig:
    """Agent 运行配置"""

    max_sql_correction_attempts: int = 2
    embedding_timeout_seconds: int = 30
    retrieval_timeout_seconds: int = 30
    retrieval_candidate_limit: int = 5
    sql_execution_timeout_seconds: int = 60
    value_hybrid_es_weight: float = 1.2
    value_hybrid_vector_weight: float = 1.0
    value_vector_score_threshold: float = 0.65


@dataclass
class MetadataBuildConfig:
    """元数据知识库后台构建配置"""

    enabled: bool = False
    config_path: str = "conf/meta_config.yaml"
    poll_interval_seconds: int = 300
    build_on_start: bool = False


@dataclass
class CostConfig:
    """Token 成本配置，价格单位为每 100 万 token。"""

    llm_input_per_1m_tokens: float = 0.0
    llm_output_per_1m_tokens: float = 0.0
    embedding_per_1m_tokens: float = 0.0
    currency: str = "CNY"


@dataclass
class AppConfig:
    """项目级总配置入口"""

    logging: LoggingConfig
    db_meta: DBConfig
    db_dw: DBConfig
    qdrant: QdrantConfig
    embedding: EmbeddingConfig
    es: ESConfig
    llm: LLMConfig
    agent: AgentConfig
    metadata_build: MetadataBuildConfig
    cost: CostConfig


@dataclass
class FileAppConfig:
    """只从 conf/app_config.yaml 读取的非模型配置。"""

    logging: LoggingConfig
    db_meta: DBConfig
    db_dw: DBConfig
    qdrant: QdrantConfig
    es: ESConfig
    agent: AgentConfig = field(default_factory=AgentConfig)
    metadata_build: MetadataBuildConfig = field(default_factory=MetadataBuildConfig)
    cost: CostConfig = field(default_factory=CostConfig)


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_optional_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


# 从当前文件位置回到项目根目录，再定位到 conf/app_config.yaml
def _get_bool_env(name: str) -> bool:
    value = _get_env(name).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean environment variable: {name}={value}")


def _get_optional_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_optional_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


project_root = Path(__file__).parents[2]
config_file = project_root / "conf" / "app_config.yaml"

# 先读取本地 .env，让 YAML 中的 ${oc.env:...} 可以解析到敏感配置
load_dotenv(project_root / ".env")

# 读取 YAML 配置内容
context = OmegaConf.load(config_file)

# 根据 AppConfig 生成结构化配置 schema
schema = OmegaConf.structured(FileAppConfig)

# 把配置结构和配置值合并，再转换成可以直接按属性访问的对象
file_config: FileAppConfig = OmegaConf.to_object(OmegaConf.merge(schema, context))

app_config = AppConfig(
    logging=file_config.logging,
    db_meta=file_config.db_meta,
    db_dw=file_config.db_dw,
    qdrant=file_config.qdrant,
    es=file_config.es,
    embedding=EmbeddingConfig(
        base_url=_get_env("LLM_BASE_URL"),
        api_key=_get_env("LLM_API_KEY"),
        model=_get_env("EMBEDDING_MODEL"),
    ),
    llm=LLMConfig(
        provider=_get_env("LLM_PROVIDER"),
        model=_get_env("LLM_MODEL"),
        fast_model=_get_optional_env(
            "LLM_FAST_MODEL",
            _get_optional_env("FAST_MODEL", _get_env("LLM_MODEL")),
        ),
        api_key=_get_env("LLM_API_KEY"),
        base_url=_get_env("LLM_BASE_URL"),
        timeout_seconds=int(_get_env("LLM_TIMEOUT_SECONDS")),
        structured_enable_thinking=_get_bool_env("LLM_STRUCTURED_ENABLE_THINKING"),
        generate_sql_enable_thinking=_get_bool_env("LLM_GENERATE_SQL_ENABLE_THINKING"),
        correct_sql_enable_thinking=_get_bool_env("LLM_CORRECT_SQL_ENABLE_THINKING"),
        input_per_1m_tokens=float(_get_env("LLM_INPUT_PER_1M_TOKENS")),
        output_per_1m_tokens=float(_get_env("LLM_OUTPUT_PER_1M_TOKENS")),
        max_retries=_get_optional_int_env("LLM_MAX_RETRIES", 2),
        retry_backoff_seconds=_get_optional_float_env("LLM_RETRY_BACKOFF_SECONDS", 0.2),
        concurrency_limit=_get_optional_int_env("LLM_CONCURRENCY_LIMIT", 4),
        quota_circuit_breaker_seconds=_get_optional_int_env(
            "LLM_QUOTA_CIRCUIT_BREAKER_SECONDS", 300
        ),
        rate_limit_breaker_threshold=_get_optional_int_env(
            "LLM_RATE_LIMIT_BREAKER_THRESHOLD", 3
        ),
        error_window_seconds=_get_optional_int_env("LLM_ERROR_WINDOW_SECONDS", 60),
        error_window_min_calls=_get_optional_int_env("LLM_ERROR_WINDOW_MIN_CALLS", 20),
        error_rate_threshold=_get_optional_float_env("LLM_ERROR_RATE_THRESHOLD", 0.5),
        max_calls_per_request=_get_optional_int_env("LLM_MAX_CALLS_PER_REQUEST", 40),
        fast_max_retries=_get_optional_int_env("LLM_FAST_MAX_RETRIES", 2),
        fast_concurrency_limit=_get_optional_int_env("LLM_FAST_CONCURRENCY_LIMIT", 4),
        fast_quota_circuit_breaker_seconds=_get_optional_int_env(
            "LLM_FAST_QUOTA_CIRCUIT_BREAKER_SECONDS", 60
        ),
        sql_max_retries=_get_optional_int_env("LLM_SQL_MAX_RETRIES", 1),
        sql_concurrency_limit=_get_optional_int_env("LLM_SQL_CONCURRENCY_LIMIT", 1),
        sql_quota_circuit_breaker_seconds=_get_optional_int_env(
            "LLM_SQL_QUOTA_CIRCUIT_BREAKER_SECONDS", 300
        ),
    ),
    agent=file_config.agent,
    metadata_build=file_config.metadata_build,
    cost=CostConfig(
        llm_input_per_1m_tokens=float(_get_env("LLM_INPUT_PER_1M_TOKENS")),
        llm_output_per_1m_tokens=float(_get_env("LLM_OUTPUT_PER_1M_TOKENS")),
        embedding_per_1m_tokens=file_config.cost.embedding_per_1m_tokens,
        currency=file_config.cost.currency,
    ),
)

if __name__ == "__main__":
    # 简单测试：验证配置是否能正常读取
    print(app_config.es.host)
