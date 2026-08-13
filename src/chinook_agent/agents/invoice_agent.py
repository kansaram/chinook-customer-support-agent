from langchain_openai import ChatOpenAI
from ..config.settings import settings
from ..agents.state import AgentState
from ..agents.prompts import INVOICE_AGENT_PROMPT
from ..tools.invoice_tools import (
    customer_lookup,
    get_invoice_history,
    get_tracks_for_invoices_for_customer,
    get_tracks_for_invoice_for_customer,
    get_support_rep_for_customer_by_invoiceId,
)
from ..tools.handoff_tools import transfer_to_catalog_agent, transfer_to_memory_agent
from ..config.logging import get_logger
from ..agents.grounding_guard import enforce_tool_grounding
from ..agents.grounding_guard import enforce_tool_grounding
logger = get_logger(__name__)


INVOICE_TOOLS = [
    customer_lookup,
    get_invoice_history,
    get_tracks_for_invoices_for_customer,
    get_tracks_for_invoice_for_customer,
    get_support_rep_for_customer_by_invoiceId,
    transfer_to_catalog_agent,
    transfer_to_memory_agent,
]

llm = ChatOpenAI(model=settings.DEFAULT_MODEL, temperature=0).bind_tools(INVOICE_TOOLS)




def invoice_llm_node(state: AgentState) -> dict:
    messages = [{"role": "system", "content": INVOICE_AGENT_PROMPT}] + state["messages"]
    logger.info("invoice_agent invoked", extra={"customer_id": state.get("customer_id")})

    response = llm.invoke(messages)

    needs_retry, reason = enforce_tool_grounding(response, state["messages"])
    if needs_retry:
        logger.warning("grounding guard triggered a retry", extra={"agent": "invoice_agent"})
        response = llm.invoke(messages + [response, {"role": "system", "content": reason}])

    updates = {"messages": [response]}
    if state.get("next_agent") == "invoice_agent" and not response.tool_calls:
        updates["response"] = response.content
    return updates