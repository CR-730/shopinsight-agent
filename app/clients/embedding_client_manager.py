"""
Embedding 客户端管理器。

通过 OpenAI-compatible /embeddings 接口统一接入远程向量模型。
"""

import asyncio
from typing import Optional

from langchain_openai import OpenAIEmbeddings

from app.agent.cached_clients import CachedEmbeddingClient
from app.conf.app_config import EmbeddingConfig, app_config


class EmbeddingClientManager:
    """管理 Embedding 客户端的初始化与复用。"""

    def __init__(self, config: EmbeddingConfig):
        self.client: Optional[OpenAIEmbeddings] = None
        self.config = config

    def init(self):
        """显式初始化客户端，避免模块导入时立即建立外部连接。"""
        self.client = CachedEmbeddingClient(
            OpenAIEmbeddings(
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
            )
        )


embedding_client_manager = EmbeddingClientManager(app_config.embedding)


if __name__ == "__main__":
    embedding_client_manager.init()
    client = embedding_client_manager.client

    async def test():
        """执行一次最小化向量化调用，验证服务是否可用。"""
        text = "What is deep learning?"
        query_result = await client.aembed_query(text)
        print(query_result[:3])

    asyncio.run(test())
