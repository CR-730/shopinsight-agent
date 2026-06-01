"""Agent 使用的大模型实例。"""

from langchain.chat_models import init_chat_model

from app.conf.app_config import app_config


def build_llm_kwargs(enable_thinking: bool, model: str | None = None) -> dict:
    """构造 OpenAI-compatible Chat Model 配置，保留 thinking 开关用于测试和配置化。"""

    return {
        "model": model or app_config.llm.model,
        "model_provider": app_config.llm.provider,
        "base_url": app_config.llm.base_url,
        "api_key": app_config.llm.api_key,
        "temperature": 0,
        "extra_body": {"enable_thinking": enable_thinking},
    }


# 结构化、召回扩展、过滤类节点默认关闭 thinking，降低延迟和 token 成本。
llm = init_chat_model(
    **build_llm_kwargs(
        app_config.llm.structured_enable_thinking,
        model=app_config.llm.fast_model,
    )
)

# SQL 生成和 SQL 错误修正可以开启 thinking，换取更强的规划/修正能力。
sql_llm = init_chat_model(
    **build_llm_kwargs(app_config.llm.sql_enable_thinking)
)


if __name__ == "__main__":
    print(llm.invoke("你好").content)
