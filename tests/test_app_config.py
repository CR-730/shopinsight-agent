import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def reload_app_config(monkeypatch):
    env_values = {
        "LLM_API_KEY": "llm-key",
        "LLM_BASE_URL": "https://llm.example/v1",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "Pro/zai-org/GLM-5.1",
        "LLM_FAST_MODEL": "Pro/zai-org/GLM-5.1-Air",
        "LLM_TIMEOUT_SECONDS": "60",
        "LLM_STRUCTURED_ENABLE_THINKING": "false",
        "LLM_SQL_ENABLE_THINKING": "true",
        "LLM_INPUT_PER_1M_TOKENS": "0.8",
        "LLM_OUTPUT_PER_1M_TOKENS": "4.8",
        "EMBEDDING_BASE_URL": "http://10.0.24.102:3000/v1",
        "EMBEDDING_API_KEY": "embedding-key",
        "EMBEDDING_MODEL": "Qwen3-Embedding-8B",
    }
    for key, value in env_values.items():
        monkeypatch.setenv(key, value)

    module_name = "app.conf.app_config"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def test_llm_config_is_loaded_from_environment(monkeypatch):
    app_config_module = reload_app_config(monkeypatch)

    llm_config = app_config_module.app_config.llm

    assert llm_config.provider == "openai"
    assert llm_config.model == "Pro/zai-org/GLM-5.1"
    assert llm_config.fast_model == "Pro/zai-org/GLM-5.1-Air"
    assert llm_config.api_key == "llm-key"
    assert llm_config.base_url == "https://llm.example/v1"
    assert llm_config.timeout_seconds == 60
    assert llm_config.structured_enable_thinking is False
    assert llm_config.sql_enable_thinking is True
    assert llm_config.input_per_1m_tokens == 0.8
    assert llm_config.output_per_1m_tokens == 4.8


def test_embedding_config_is_loaded_from_environment(monkeypatch):
    app_config_module = reload_app_config(monkeypatch)

    assert app_config_module.app_config.qdrant.embedding_size == 4096
    embedding_config = app_config_module.app_config.embedding
    assert embedding_config.base_url == "http://10.0.24.102:3000/v1"
    assert embedding_config.api_key == "embedding-key"
    assert embedding_config.model == "Qwen3-Embedding-8B"
