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
from ..agents.message_sanitizer import sanitize_message_history
from langchain_core.messages import AIMessage
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
    updates: dict = {}
    messages = [{"role": "system", "content": INVOICE_AGENT_PROMPT}] + sanitize_message_history(state["messages"])
    logger.info("invoice_agent invoked", extra={"customer_id": state.get("customer_id")})

    try:
        response = llm.invoke(messages)
    except Exception:
        logger.error("invoice_agent LLM call failed even after sanitization", exc_info=True)
        fallback = AIMessage(content="Sorry, something went wrong on my end — could you try asking that again?")
        updates["messages"] = [fallback]
        if state.get("next_agent") == "invoice_agent":
            updates["response_parts"] = [fallback.content]
            updates["response"] = fallback.content
        return updates

    needs_retry, reason = enforce_tool_grounding(response, state["messages"])
    if needs_retry:
        logger.warning("grounding guard triggered a retry", extra={"agent": "invoice_agent"})
        try:
            response = llm.invoke(messages + [response, {"role": "system", "content": reason}])
        except Exception:
            logger.error("invoice_agent grounding-retry LLM call failed", exc_info=True)
            # Fall back to the ORIGINAL response rather than crashing — it may be
            # imperfectly grounded, but it's better than no response at all.
            pass

    updates["messages"] = [response]
    if response.content:
        updates["response_parts"] = [response.content]
    if not response.tool_calls:
        updates["response"] = response.content  # kept for any code still reading this directly
    return updates