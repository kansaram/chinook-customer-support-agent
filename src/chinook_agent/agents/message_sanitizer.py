from langchain_core.messages import AIMessage, ToolMessage
from ..config.logging import get_logger

logger = get_logger(__name__)


def sanitize_message_history(messages: list) -> list:
    """Ensure every tool_call_id in the message list has a matching ToolMessage.

    OpenAI's API rejects a request if any AIMessage with tool_calls isn't
    immediately followed by ToolMessages responding to EVERY one of those
    tool_call_ids. If something in the graph (a handoff navigating away, a tool
    erroring before returning, etc.) leaves one orphaned, this patches it with a
    placeholder response instead of letting the next LLM call crash with a 400.

    Safe to call on every node's message list before invoke() — it's a no-op
    (returns the same list) when nothing is actually missing.

    NOTE: this returns a LOCAL patched copy only. It does not persist into graph
    state, so the same orphaned entry will be re-detected and re-patched by every
    node that reads state["messages"] for the rest of the conversation. Use
    get_sanitized_state_update() instead when you want the fix to persist once.
    """
    patched, _ = _sanitize(messages)
    return patched


def get_sanitized_state_update(messages: list) -> dict | None:
    """Like sanitize_message_history, but returns a state update dict (suitable
    for merging into a node's `updates` return value) ONLY if a fix was actually
    needed — so the healed messages get written back into the persisted,
    checkpointed state via the `add_messages` reducer, and the orphaned entry
    never needs to be re-detected again on later turns. Returns None when nothing
    needed patching, so callers can skip adding anything to their updates dict.
    """
    _, inserted = _sanitize(messages)
    if not inserted:
        return None
    # Only the NEWLY inserted placeholder ToolMessages need to be added — the
    # add_messages reducer appends them onto the existing persisted list.
    return {"messages": inserted}


def _sanitize(messages: list) -> tuple[list, list]:
    """Shared implementation. Returns (patched_full_list, newly_inserted_messages)."""
    responded_ids = {
        msg.tool_call_id for msg in messages if isinstance(msg, ToolMessage)
    }

    patched: list = []
    inserted: list = []

    for msg in messages:
        patched.append(msg)
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
                if call_id and call_id not in responded_ids:
                    logger.warning("patching orphaned tool_call_id", extra={"tool_call_id": call_id})
                    placeholder = ToolMessage(
                        content="(This tool call did not complete and has no result.)",
                        tool_call_id=call_id,
                    )
                    patched.append(placeholder)
                    inserted.append(placeholder)
                    responded_ids.add(call_id)

    return patched, inserted