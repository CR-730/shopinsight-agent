from app.agent.graph import graph


def test_sql_memory_recall_runs_only_after_intent_recognition_continue():
    edges = {
        (edge.source, edge.target, edge.data, edge.conditional)
        for edge in graph.get_graph().edges
    }

    assert ("intent_recognition", "context_builder", "continue", True) in edges
    assert ("intent_recognition", "__end__", "blocked", True) in edges

    assert ("intent_recognition", "extract_keywords", "continue", True) not in edges
    assert not any(edge[0] == "pre_rag_guard" for edge in edges)
    assert not any(edge[0] == "recall_sql_memory" for edge in edges)
    assert not any(edge[0] == "extract_keywords" for edge in edges)
    assert not any(edge[0] == "merge_retrieved_info" for edge in edges)
    assert not any(edge[0] == "semantic_guard" for edge in edges)
    assert not any(edge[0] == "filter_table" for edge in edges)
    assert not any(edge[0] == "filter_metric" for edge in edges)
    assert not any(edge[0] == "add_extra_context" for edge in edges)


def test_graph_exposes_single_context_compaction_node():
    edges = {
        (edge.source, edge.target, edge.data, edge.conditional)
        for edge in graph.get_graph().edges
    }

    assert ("business_binding", "context_compaction", "continue", True) in edges
    assert ("business_binding", "__end__", "blocked", True) in edges
    assert ("context_compaction", "generate_sql", None, False) in edges


def test_graph_exposes_single_sql_executor_node():
    edges = {
        (edge.source, edge.target, edge.data, edge.conditional)
        for edge in graph.get_graph().edges
    }

    assert ("generate_sql", "sql_executor", None, False) in edges
    assert ("sql_executor", "__end__", None, False) in edges

    assert not any(edge[0] == "pre_sql_execution_validation" for edge in edges)
    assert not any(edge[0] == "correct_sql" for edge in edges)
    assert not any(edge[0] == "run_sql" for edge in edges)
