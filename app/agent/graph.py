"""Agent graph for the e-commerce NL2SQL flow.

The visible graph is intentionally compact:
intent recognition -> context builder -> business binding -> context compaction
-> SQL generation -> SQL executor.

Detailed retrieval, context pruning, SQL validation, correction, and execution
live inside helper modules so graph.py shows product-level stages only.
"""

import asyncio

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agent.context import DataAgentContext
from app.agent.cost import CostRates, CostTracker
from app.agent.node_observer import traced_node
from app.agent.nodes.business_binding import business_binding
from app.agent.nodes.context_builder import context_builder
from app.agent.nodes.context_compaction import context_compaction
from app.agent.nodes.generate_sql import generate_sql
from app.agent.nodes.intent_recognition import intent_recognition
from app.agent.nodes.sql_executor import sql_executor
from app.agent.sql_loop import (
    route_after_safety_guard,
)
from app.agent.state import DataAgentState
from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import (
    dw_mysql_client_manager,
    meta_mysql_client_manager,
)
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository
from app.repositories.qdrant.value_qdrant_repository import ValueQdrantRepository

# StateGraph 声明整张图使用的状态结构和运行时上下文结构
graph_builder = StateGraph(state_schema=DataAgentState, context_schema=DataAgentContext)

# 注册节点：每个节点负责问数链路中的一个清晰步骤
graph_builder.add_node(
    "intent_recognition", traced_node("intent_recognition", intent_recognition)
)
graph_builder.add_node("context_builder", traced_node("context_builder", context_builder))
graph_builder.add_node("business_binding", traced_node("business_binding", business_binding))
graph_builder.add_node(
    "context_compaction", traced_node("context_compaction", context_compaction)
)
graph_builder.add_node("generate_sql", traced_node("generate_sql", generate_sql))
graph_builder.add_node("sql_executor", traced_node("sql_executor", sql_executor))

# Start with intent recognition, then build retrieval context before binding.
graph_builder.add_edge(START, "intent_recognition")
graph_builder.add_conditional_edges(
    source="intent_recognition",
    path=route_after_safety_guard,
    path_map={
        "continue": "context_builder",
        "blocked": END,
    },
)
graph_builder.add_edge("context_builder", "business_binding")

# Business binding is the only business-level blocking decision.
graph_builder.add_conditional_edges(
    source="business_binding",
    path=route_after_safety_guard,
    path_map={
        "continue": "context_compaction",
        "blocked": END,
    },
)

# Compact table and metric context, then add runtime SQL context.
graph_builder.add_edge("context_compaction", "generate_sql")
graph_builder.add_edge("generate_sql", "sql_executor")

# SQL validation, correction, and execution are hidden behind one graph node.
graph_builder.add_edge("sql_executor", END)

# 编译后的 graph 是对外使用的 Agent 执行入口
graph = graph_builder.compile()

if __name__ == "__main__":

    async def test():
        """本地调试关键词抽取和字段 指标 取值三路召回链路"""

        # 多路召回和上下文补全会访问 Qdrant、Embedding、ES、Meta MySQL 和 DW MySQL
        qdrant_client_manager.init()
        embedding_client_manager.init()
        es_client_manager.init()
        meta_mysql_client_manager.init()
        dw_mysql_client_manager.init()

        # Meta MySQL 用来补齐元数据，DW MySQL 用来读取数据库方言和版本
        async with (
            meta_mysql_client_manager.session_factory() as meta_session,
            dw_mysql_client_manager.session_factory() as dw_session,
        ):
            meta_mysql_repository = MetaMySQLRepository(meta_session)
            dw_mysql_repository = DWMySQLRepository(dw_session)

            # 字段和指标分别使用不同 Qdrant collection，取值检索使用 ES index
            column_qdrant_repository = ColumnQdrantRepository(
                qdrant_client_manager.client
            )
            metric_qdrant_repository = MetricQdrantRepository(
                qdrant_client_manager.client
            )
            value_es_repository = ValueESRepository(es_client_manager.client)
            value_qdrant_repository = ValueQdrantRepository(qdrant_client_manager.client)

            # 当前只需要传入原始问题，后续节点会逐步写回召回、过滤和额外上下文结果
            state = DataAgentState(query="统计华北地区的销售总额")
            context = DataAgentContext(
                column_qdrant_repository=column_qdrant_repository,
                embedding_client=embedding_client_manager.client,
                metric_qdrant_repository=metric_qdrant_repository,
                value_es_repository=value_es_repository,
                value_qdrant_repository=value_qdrant_repository,
                meta_mysql_repository=meta_mysql_repository,
                dw_mysql_repository=dw_mysql_repository,
                cost_tracker=CostTracker(CostRates()),
            )

            # stream_mode="custom" 会接收各节点通过 runtime.stream_writer 写出的进度信息
            async for chunk in graph.astream(
                input=state, context=context, stream_mode="custom"
            ):
                print(chunk)

        # 关闭显式创建的异步客户端，避免本地调试时连接资源悬挂
        await qdrant_client_manager.close()
        await es_client_manager.close()
        await meta_mysql_client_manager.close()
        await dw_mysql_client_manager.close()

    asyncio.run(test())

