from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from ..config.logging import get_logger

logger = get_logger(__name__)


def make_handoff_tool(*, agent_name: str, description: str):
    """Build a tool that lets an agent hand this turn off to another agent node.

    Returns a Command(goto=agent_name, ...) — LangGraph routes execution straight
    to that node next, regardless of the graph's normal static edges. This is what
    makes the handoff a genuine agent decision rather than a supervisor rule: any
    agent can call this itself, mid-reasoning, the moment it realizes the request
    isn't its job.
    """

    @tool(f"transfer_to_{agent_name}", description=description)
    def handoff(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        logger.info("agent handoff", extra={"target": agent_name})
        tool_message = ToolMessage(
            content=f"Handing this off to {agent_name}.",
            name=f"transfer_to_{agent_name}",
            tool_call_id=tool_call_id,
        )
        return Command(
            goto=agent_name,
            update={
                "next_agent": agent_name,  # keep response-surfacing logic consistent
                "messages": [tool_message],
            },
        )

    return handoff


transfer_to_invoice_agent = make_handoff_tool(
    agent_name="invoice_agent",
    description="Hand off to the invoice specialist when the customer's request is about "
    "orders, invoices, billing, or their account — not something you can answer yourself.",
)

transfer_to_catalog_agent = make_handoff_tool(
    agent_name="catalog_agent",
    description="Hand off to the catalog specialist when the customer's request is about "
    "finding artists, albums, or tracks, or wants a music recommendation — not something "
    "you can answer yourself.",
)

transfer_to_memory_agent = make_handoff_tool(
    agent_name="memory_agent",
    description="Hand off to the preferences specialist when the customer wants to save a "
    "preference or asks what you remember about them — not something you can answer yourself.",
)