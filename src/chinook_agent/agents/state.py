from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages

def add_preferences(existing: list[str], new: list[str]) -> list[str]:
    """Reducer: append new preferences, avoid exact duplicates."""
    combined = existing + [p for p in new if p not in existing]
    return combined

    
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    customer_id: Optional[int]
    customer_email: Optional[str]
    customer_phone: Optional[str]
    authenticated: bool
    preferences: Annotated[list[str], add_preferences]