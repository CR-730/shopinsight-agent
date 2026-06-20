"""Compatibility imports for the renamed intent recognition node."""

from app.agent.nodes.intent_recognition import (  # noqa: F401
    IntentRecognitionDecision,
    _should_block_classifier_result,
    classify_query_intent,
    intent_recognition,
)

pre_rag_guard = intent_recognition
PreRagGuardDecision = IntentRecognitionDecision
