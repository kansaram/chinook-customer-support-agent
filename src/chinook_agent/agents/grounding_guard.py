import re
from langchain_core.messages import AIMessage

# Heuristic: a response "looks like a data claim" if it contains a number, or a
# capitalized multi-word phrase (a plausible album/artist/track name), and the
# customer's message wasn't just a greeting/thanks. This is intentionally loose —
# it only exists to catch the case where the LLM skipped tools entirely.
_NUMBER_RE = re.compile(r"\$?\d[\d,]*\.?\d*")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\b")


def response_looks_like_data_claim(text: str) -> bool:
    return bool(_NUMBER_RE.search(text) or _PROPER_NOUN_RE.search(text))


def turn_had_tool_call(messages: list) -> bool:
    """Check whether any tool was actually invoked earlier in this turn's message run."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            return True
        # Stop scanning once we pass the start of this turn (a prior human message)
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role in ("human", "user"):
            break
    return False


def enforce_tool_grounding(response: AIMessage, recent_messages: list) -> tuple[bool, str]:
    """Returns (needs_retry, reason). If the response makes a data-shaped claim but
    called no tool this turn and requested none now, flag it for a retry with a
    stricter instruction rather than letting it reach the customer unverified."""
    if response.tool_calls:
        return False, ""  # it's calling a tool now — fine, let it proceed

    content = response.content or ""
    if not response_looks_like_data_claim(content):
        return False, ""  # plain conversational text, nothing to ground

    if turn_had_tool_call(recent_messages):
        return False, ""  # a tool already ran earlier this turn — the claim may be grounded in that

    return True, (
        "Your previous response stated specific details without calling a tool first. "
        "Call the appropriate tool now to verify this information before answering, "
        "or tell the customer you don't have that information if no tool applies."
    )