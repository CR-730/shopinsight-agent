"""Helpers for LLM-emitted stop signals that must not reach users."""

STOP_SIGNAL = "find_error"


def split_stop_signal(text: str) -> tuple[str, bool]:
    """Return user-visible text and whether the LLM asked to stop the turn."""

    content = str(text or "")
    if STOP_SIGNAL not in content:
        return content, False
    visible, _, _ = content.partition(STOP_SIGNAL)
    return visible.rstrip(), True
