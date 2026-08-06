# agents/supervisor.py

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from ..config.settings import settings
from ..agents.state import AgentState
from ..agents.prompts import SUPERVISOR_PROMPT
from ..config.logging import get_logger

logger = get_logger(__name__)

VALID_AGENTS = ["invoice_agent", "catalog_agent", "memory_agent"]

# No tools bound here — the supervisor only makes a routing decision, it never calls tools itself
router_llm = ChatOpenAI(model=settings.DEFAULT_MODEL, temperature=0)


INVOICE_KEYWORDS = {
    "invoice",
    "invoices",
    "billing",
    "bill",
    "payment",
    "payments",
    "purchase",
    "purchases",
    "order",
    "orders",
    "receipt",
    "receipts",
}

CATALOG_KEYWORDS = {
    "song",
    "songs",
    "track",
    "tracks",
    "artist",
    "artists",
    "album",
    "albums",
    "genre",
    "genres",
    "music",
    "catalog",
    "recommend",
    "recommendation",
    "recommendations",
}


def _latest_user_text(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        role = getattr(msg, "type", None)
        if role == "human":
            content = getattr(msg, "content", "") or ""
            return content if isinstance(content, str) else ""
        if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else ""
    return ""


def _is_mixed_intent(user_text: str) -> bool:
    lowered = user_text.lower()
    has_invoice = any(k in lowered for k in INVOICE_KEYWORDS)
    has_catalog = any(k in lowered for k in CATALOG_KEYWORDS)
    return has_invoice and has_catalog


def supervisor_node(state: AgentState) -> dict:
    """Decide which specialist agent should be primary for this turn."""
    latest_user_text = _latest_user_text(state)

    # Deterministic rule for mixed intent: prioritize invoice flow first so account
    # authentication/details are handled before any catalog follow-up.
    if latest_user_text and _is_mixed_intent(latest_user_text):
        logger.info("supervisor routed mixed-intent request", extra={"next_agent": "invoice_agent"})
        return {"next_agent": "invoice_agent"}

    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    response = router_llm.invoke(messages)
    raw = response.content.strip().lower()

    agent_name = next((a for a in VALID_AGENTS if a in raw), None)

    if agent_name is None:
        logger.warning("supervisor produced unparseable routing decision", extra={"raw_response": raw})
        agent_name = "catalog_agent"  # safe default

    logger.info("supervisor routed request", extra={"next_agent": agent_name})

    return {"next_agent": agent_name}