import asyncio

from app.evaluation.conversation_memory_eval import run_minimal_conversation_memory_eval


def test_minimal_conversation_memory_eval_passes():
    report = asyncio.run(run_minimal_conversation_memory_eval())

    assert report["passed"] == report["total"]
