from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages

from ..tools.preference_tools import parse_preference_statement


def _append_unique(existing: list[str], new: list[str] | None) -> list[str]:
    existing_lower = {item.lower() for item in existing}
    additions: list[str] = []
    for item in new or []:
        item_lower = item.lower()
        if item_lower not in existing_lower:
            additions.append(item)
            existing_lower.add(item_lower)
    return existing + additions


def add_preferences(existing: list[str], new: list[str]) -> list[str]:
    merged = list(existing)
    for item in new or []:
        subject, _ = parse_preference_statement(item)
        subject_lower = subject.casefold()
        merged = [
            value
            for value in merged
            if parse_preference_statement(value)[0].casefold() != subject_lower
        ]
        if item.lower() not in {value.lower() for value in merged}:
            merged.append(item)
    return merged


def replace_or_append_preferences(existing: list[str], new: list[str] | None) -> list[str]:
    """If new is explicitly an empty list, treat it as a reset. Otherwise append."""
    if new == []:
        return []
    return _append_unique(existing, new)


def set_response(existing: Optional[str], new: Optional[str]) -> Optional[str]:
    """Reducer for `response`: required because memory_agent can now set a response
    even when it's running in the background (not the supervisor's chosen primary
    agent) — e.g. when the user explicitly asks to see their preferences. Without a
    reducer here, two nodes writing to the same plain field in one parallel graph
    step raises a LangGraph 'multiple values per step' error. `None` means "this
    node has nothing to say", so it never overwrites an existing answer.
    """
    return new if new is not None else existing


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    customer_id: Optional[int]
    customer_email: Optional[str]
    customer_phone: Optional[str]
    authenticated: bool
    preferences: Annotated[list[str], add_preferences]
    pending_preferences: Annotated[list[str], replace_or_append_preferences]
    preferences_loaded: bool
    preferences_scanned_count: int
    next_agent: Optional[str]
    response: Annotated[Optional[str], set_response]