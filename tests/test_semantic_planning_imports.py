def test_semantic_planning_is_canonical():
    from app.agent.nodes.semantic_planning import semantic_planning

    assert callable(semantic_planning)
