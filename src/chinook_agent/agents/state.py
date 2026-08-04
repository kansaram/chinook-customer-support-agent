from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    customer_id: Optional[int]