from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from ..config.logging import get_logger

logger = get_logger(__name__)


MAX_HANDOFFS_PER_TURN = 3


def make_handoff_tool(*, agent_name: str, description: str):
    """Build a tool that lets an agent hand this turn off to another agent node.

    Returns a Command(goto=agent_name, ...) — LangGraph routes execution straight
    to that node next, regardless of the graph's normal static edges. This is what
    makes the handoff a genuine agent decision rather than a supervisor rule: any
    agent can call this itself, mid-reasoning, the moment it realizes the request
    isn't its job.

    Caps total handoffs per turn at MAX_HANDOFFS_PER_TURN to prevent two or more
    agents ping-ponging a request neither wants — once the cap is hit, the tool
    refuses the handoff and tells the calling agent to answer directly instead.
    """

    @tool(f"transfer_to_{agent_name}", description=description)
    def handoff(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        handoff_count = state.get("handoff_count", 0)

        if handoff_count >= MAX_HANDOFFS_PER_TURN:
            logger.warning("handoff cap reached, refusing further handoff", extra={"target": agent_name, "count": handoff_count})
            tool_message = ToolMessage(
                content=(
                    "Handoff limit reached for this turn — you must answer the customer "
                    "directly now, even if imperfectly, rather than handing off again."
                ),
                name=f"transfer_to_{agent_name}",
                tool_call_id=tool_call_id,
            )
            return Command(update={"messages": [tool_message]})

        logger.info("agent handoff", extra={"target": agent_name, "count": handoff_count + 1})
        tool_message = ToolMessage(
            content=f"Handing this off to {agent_name}.",
            name=f"transfer_to_{agent_name}",
            tool_call_id=tool_call_id,
        )
        return Command(
            goto=agent_name,
            update={
                "next_agent": agent_name,
                "handoff_count": handoff_count + 1,
                "messages": [tool_message],
            },
        )

    return handoff


transfer_to_invoice_agent = make_handoff_tool(
    agent_name="invoice_agent",
    description="Hand off to the invoice specialist when the customer wants to LOOK UP "
    "existing orders, invoices, billing history, or account details. The invoice "
    "specialist can only look up existing information — it CANNOT process refunds, "
    "cancellations, returns, or any changes to an order or account. Do not hand off a "
    "refund/change/cancellation request implying it will be resolved; say plainly that "
    "this system can only look up information, not take that action, whether or not you "
    "hand off.",
)

transfer_to_catalog_agent = make_handoff_tool(
    agent_name="catalog_agent",
    description="Hand off to the catalog specialist when the customer wants to find "
    "artists, albums, or tracks, or wants a music recommendation. The catalog specialist "
    "can only search and describe the existing catalog — it CANNOT process a purchase, "
    "add to a cart, play, or download anything. Do not hand off a purchase/playback "
    "request implying it will be resolved; say plainly that this system can only help "
    "find and learn about music, not take that action, whether or not you hand off.",
)

transfer_to_memory_agent = make_handoff_tool(
    agent_name="memory_agent",
    description="Hand off to the preferences specialist ONLY for MUSIC preferences — genre, "
    "artist, style, or preferred contact method — or when the customer asks what music "
    "preferences you remember about them. Do NOT use this for anything unrelated to music "
    "(astrology signs, birthdays, favorite foods, or any other personal attribute) — the "
    "preferences specialist has no data or capability for those; say plainly that the system "
    "doesn't track that instead of transferring.",
)