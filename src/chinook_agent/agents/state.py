from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages


def add_preferences(existing: list[str], new: list[str]) -> list[str]:
    return existing + [p for p in new if p not in existing]

def replace_or_append_preferences(existing: list[str], new: list[str] | None) -> list[str]:
    """If new is explicitly an empty list, treat it as a reset. Otherwise append."""
    if new == []:
        return []
    return existing + [p for p in (new or []) if p not in existing]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    customer_id: Optional[int]
    customer_email: Optional[str]
    customer_phone: Optional[str]
    authenticated: bool
    preferences: Annotated[list[str], add_preferences]
    pending_preferences: Annotated[list[str], replace_or_append_preferences]
    preferences_loaded: bool
    next_agent: Optional[str]
    response: Optional[str]