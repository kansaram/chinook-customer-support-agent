import re
from typing import Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

from ..database.memory_repository import save_preferences_list, load_preferences_list


def resolve_identifier(
    customer_id: Optional[int] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
) -> Optional[str]:
    """Always produce the SAME identifier string for the same person, regardless of
    which fields happen to be populated in state this session."""
    if customer_id is not None:
        return f"cid:{customer_id}"
    if email:
        return f"email:{email.strip().lower()}"
    if phone:
        digits = re.sub(r"\D", "", phone)
        return f"phone:{digits[-10:] if len(digits) >= 10 else digits}"
    return None


@tool("save_preference", description="Remember a preference the customer explicitly stated during this conversation.")
def save_preference(
    input: Annotated[dict, "input"],
    tool_call_id: Annotated[str, "tool_call_id"],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Append a stated preference to state AND persist the full list to customer_memory.db."""
    preference_text = input.get("preference")
    if not preference_text:
        message = "No preference text provided."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    identifier = resolve_identifier(
        customer_id=state.get("customer_id"),
        email=state.get("customer_email"),
        phone=state.get("customer_phone"),
    )
    if identifier is None:
        message = "I need your email, phone, or customer ID before I can save preferences."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    existing = state.get("preferences", [])
    updated_list = existing + [preference_text] if preference_text not in existing else existing

    save_preferences_list(identifier, updated_list)

    return Command(
        update={
            "preferences": [preference_text],  # the add_preferences reducer appends this
            "messages": [ToolMessage(content=f"Noted: {preference_text}", tool_call_id=tool_call_id)],
        }
    )


@tool("get_preferences", description="Retrieve the customer's previously saved preferences.")
def get_preferences(
    input: Annotated[dict, "input"],
    tool_call_id: Annotated[str, "tool_call_id"],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Load stored preferences for this customer from customer_memory.db."""
    identifier = resolve_identifier(
        customer_id=state.get("customer_id"),
        email=state.get("customer_email"),
        phone=state.get("customer_phone"),
    )
    if identifier is None:
        message = "I don't have enough information yet to look up saved preferences."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    preferences = load_preferences_list(identifier)
    if not preferences:
        message = "No saved preferences found for this customer."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    summary = "Saved preferences:\n" + "\n".join(f"- {p}" for p in preferences)
    return Command(
        update={
            "preferences": preferences,
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )