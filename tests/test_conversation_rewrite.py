import asyncio

from app.agent.rewrite import ConversationRewriteResult, rewrite_query


def test_rewrite_query_uses_structured_llm_result(monkeypatch):
    async def fake_ainvoke_llm_with_usage(
        prompt,
        llm,
        output_parser,
        inputs,
        step,
        cost_tracker,
        timeout_seconds,
        cacheable=True,
    ):
        assert step == "追问改写"
        assert cacheable is True
        assert inputs["query"] == "那华东呢"
        assert "last_metric_bindings" in inputs["snapshot"]
        return ConversationRewriteResult(
            mode="rewritten",
            standalone_query="统计华东地区 GMV",
            reason="继承上一轮指标，覆盖地区",
            inherited_slots={"metric": ["GMV"]},
            overridden_slots={"filters": ["华东"]},
        )

    monkeypatch.setattr("app.agent.rewrite.ainvoke_llm_with_usage", fake_ainvoke_llm_with_usage)

    result = asyncio.run(
        rewrite_query(
            "那华东呢",
            {"last_metric_bindings": [{"canonical_metric": "GMV"}]},
            cost_tracker=object(),
        )
    )

    assert result.mode == "rewritten"
    assert result.standalone_query == "统计华东地区 GMV"
    assert result.inherited_slots == {"metric": ["GMV"]}
    assert result.overridden_slots == {"filters": ["华东"]}


def test_rewrite_query_falls_back_to_needs_context_on_llm_failure(monkeypatch):
    async def fail_ainvoke_llm_with_usage(*args, **kwargs):
        raise RuntimeError("rewrite llm unavailable")

    monkeypatch.setattr("app.agent.rewrite.ainvoke_llm_with_usage", fail_ainvoke_llm_with_usage)

    result = asyncio.run(
        rewrite_query(
            "那华东呢",
            None,
            cost_tracker=object(),
        )
    )

    assert result.mode == "needs_context"
    assert result.standalone_query == "那华东呢"


def test_rewrite_query_blocks_when_llm_fails_even_with_snapshot(monkeypatch):
    async def fail_ainvoke_llm_with_usage(*args, **kwargs):
        raise RuntimeError("rewrite llm unavailable")

    monkeypatch.setattr("app.agent.rewrite.ainvoke_llm_with_usage", fail_ainvoke_llm_with_usage)

    result = asyncio.run(
        rewrite_query(
            "那华东呢",
            {"last_metric_bindings": [{"canonical_metric": "GMV"}]},
            cost_tracker=object(),
        )
    )

    assert result.mode == "needs_context"
    assert result.standalone_query == "那华东呢"
    assert result.inherited_slots == {}
    assert result.overridden_slots == {}


def test_rewrite_query_enforces_unchanged_query_invariant(monkeypatch):
    async def fake_ainvoke_llm_with_usage(*args, **kwargs):
        return ConversationRewriteResult(
            mode="unchanged",
            standalone_query="统计华北地区 GMV",
            reason="完整新问题",
            inherited_slots={"metric": ["GMV"]},
            overridden_slots={"filter": ["华北"]},
        )

    monkeypatch.setattr("app.agent.rewrite.ainvoke_llm_with_usage", fake_ainvoke_llm_with_usage)

    result = asyncio.run(
        rewrite_query("统计各品类销量", {"last_metric_bindings": []}, object())
    )

    assert result.mode == "unchanged"
    assert result.standalone_query == "统计各品类销量"
    assert result.inherited_slots == {}
    assert result.overridden_slots == {}


def test_rewrite_query_keeps_original_query_for_needs_context(monkeypatch):
    async def fake_ainvoke_llm_with_usage(*args, **kwargs):
        return ConversationRewriteResult(
            mode="needs_context",
            standalone_query="",
            reason="缺少上下文",
            inherited_slots={"metric": ["GMV"]},
            overridden_slots={"filter": ["华东"]},
        )

    monkeypatch.setattr("app.agent.rewrite.ainvoke_llm_with_usage", fake_ainvoke_llm_with_usage)

    result = asyncio.run(rewrite_query("那华东呢", None, object()))

    assert result.mode == "needs_context"
    assert result.standalone_query == "那华东呢"
    assert result.inherited_slots == {}
    assert result.overridden_slots == {}
