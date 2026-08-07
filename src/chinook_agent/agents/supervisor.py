# agents/supervisor.py

import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from ..config.settings import settings
from ..agents.state import AgentState
from ..agents.prompts import SUPERVISOR_PROMPT
from ..config.logging import get_logger

logger = get_logger(__name__)

VALID_AGENTS = ["invoice_agent", "catalog_agent", "memory_agent"]

router_llm = ChatOpenAI(model=settings.DEFAULT_MODEL, temperature=0)


INVOICE_KEYWORDS = {
    "invoice", "invoices", "billing", "bill", "payment", "payments",
    "purchase", "purchases", "order", "orders", "receipt", "receipts",
}

ACCOUNT_KEYWORDS = {
    "account", "accounts", "profile", "my details", "my info", "my information",
}

CATALOG_KEYWORDS = {
    "song", "songs", "track", "tracks", "artist", "artists", "album", "albums",
    "genre", "genres", "music", "catalog", "recommend", "recommendation", "recommendations",
}

_FOLLOWUP_PROMISE_RE = re.compile(
    r"\b(?:suggest|recommend)\w*\s+(?:you\s+)?(?:some\s+)?(?:songs?|tracks?|albums?|music)\b",
    re.IGNORECASE,
)

# invoice_agent's own identification prompt is the only place "customer ID" is
# ever mentioned in an assistant message — memory_agent's identifier ask only says
# "email or phone number". This distinctive phrase lets us tell the two apart and
# route a bare identifier reply back to whichever agent actually asked for it,
# instead of always defaulting bare-identifier replies to memory_agent (Rule 3).
_INVOICE_IDENTIFICATION_REQUEST_RE = re.compile(r"\bcustomer\s*id\b", re.IGNORECASE)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")


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


def _last_assistant_text(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        role = getattr(msg, "type", None)
        content = getattr(msg, "content", None)
        if role == "ai":
            return str(content or "")
        if isinstance(msg, dict) and str(msg.get("role", "")).lower() in ("assistant", "ai"):
            return str(msg.get("content", "") or "")
    return ""


def _is_mixed_intent(user_text: str) -> bool:
    lowered = user_text.lower()
    has_invoice = any(k in lowered for k in INVOICE_KEYWORDS)
    has_catalog = any(k in lowered for k in CATALOG_KEYWORDS)
    return has_invoice and has_catalog


def _assistant_promised_catalog_followup(state: AgentState) -> bool:
    return bool(_FOLLOWUP_PROMISE_RE.search(_last_assistant_text(state)))


def _invoice_agent_awaiting_identifier(state: AgentState) -> bool:
    """True if the assistant's last turn was invoice_agent asking for an email,
    phone, or customer ID to continue an invoice/account lookup it already started."""
    return bool(_INVOICE_IDENTIFICATION_REQUEST_RE.search(_last_assistant_text(state)))


def _is_bare_identifier(user_text: str) -> bool:
    has_email = bool(EMAIL_RE.search(user_text))
    has_phone = bool(PHONE_RE.search(user_text))
    if not (has_email or has_phone):
        return False
    lowered = user_text.lower()
    has_other_intent = any(k in lowered for k in INVOICE_KEYWORDS) or any(k in lowered for k in CATALOG_KEYWORDS) \
        or any(k in lowered for k in ACCOUNT_KEYWORDS)
    return not has_other_intent


def _is_invoice_or_account_intent(user_text: str) -> bool:
    lowered = user_text.lower()
    has_invoice_or_account = any(k in lowered for k in INVOICE_KEYWORDS) or any(k in lowered for k in ACCOUNT_KEYWORDS)
    has_catalog = any(k in lowered for k in CATALOG_KEYWORDS)
    return has_invoice_or_account and not has_catalog


def supervisor_node(state: AgentState) -> dict:
    """Decide which specialist agent should be primary for this turn."""
    latest_user_text = _latest_user_text(state)

    # Rule 0: invoice_agent already asked for an identifier this conversation to
    # continue an invoice/account lookup — send the reply back to it, even if the
    # reply itself is just a bare email/phone that Rule 3 would otherwise claim.
    # Checked first because it's the highest-confidence signal available: it means
    # a specific agent is mid-flow and waiting on exactly this piece of information.
    if _invoice_agent_awaiting_identifier(state):
        logger.info("supervisor routed continuation of invoice identification", extra={"next_agent": "invoice_agent"})
        return {"next_agent": "invoice_agent"}

    # Rule 1: honor a promised catalog follow-up.
    if _assistant_promised_catalog_followup(state):
        logger.info("supervisor routed via followup-promise override", extra={"next_agent": "catalog_agent"})
        return {"next_agent": "catalog_agent"}

    # Rule 2: mixed intent (invoice + catalog keywords both present) -> invoice first.
    if latest_user_text and _is_mixed_intent(latest_user_text):
        logger.info("supervisor routed mixed-intent request", extra={"next_agent": "invoice_agent"})
        return {"next_agent": "invoice_agent"}

    # Rule 3: bare identifier (just an email/phone, no other intent) -> memory_agent.
    if latest_user_text and _is_bare_identifier(latest_user_text):
        logger.info("supervisor routed bare identifier", extra={"next_agent": "memory_agent"})
        return {"next_agent": "memory_agent"}

    # Rule 4: standalone invoice/billing/order OR account/profile intent.
    if latest_user_text and _is_invoice_or_account_intent(latest_user_text):
        logger.info("supervisor routed invoice/account intent", extra={"next_agent": "invoice_agent"})
        return {"next_agent": "invoice_agent"}

    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    response = router_llm.invoke(messages)
    raw = response.content.strip().lower()

    agent_name = next((a for a in VALID_AGENTS if a in raw), None)

    if agent_name is None:
        logger.warning("supervisor produced unparseable routing decision", extra={"raw_response": raw})
        agent_name = "catalog_agent"

    logger.info("supervisor routed request", extra={"next_agent": agent_name})

    return {"next_agent": agent_name}