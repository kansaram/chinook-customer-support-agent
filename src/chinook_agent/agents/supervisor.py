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

CATALOG_KEYWORDS = {
    "song", "songs", "track", "tracks", "artist", "artists", "album", "albums",
    "genre", "genres", "music", "catalog", "recommend", "recommendation", "recommendations",
}

_FOLLOWUP_PROMISE_RE = re.compile(
    r"\b(?:suggest|recommend)\w*\s+(?:you\s+)?(?:some\s+)?(?:songs?|tracks?|albums?|music)\b",
    re.IGNORECASE,
)

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


def _is_bare_identifier(user_text: str) -> bool:
    """True if the message is essentially just an email or phone number, with no
    other invoice/catalog keywords — e.g. someone replying to 'what's your email?'
    with nothing but the address. This should go to memory_agent, since it's the
    only agent that both captures identity AND will acknowledge it in a response;
    catalog_agent/invoice_agent have no tool for a bare identifier and tend to
    reply with a generic refusal."""
    has_email = bool(EMAIL_RE.search(user_text))
    has_phone = bool(PHONE_RE.search(user_text))
    if not (has_email or has_phone):
        return False

    lowered = user_text.lower()
    has_other_intent = any(k in lowered for k in INVOICE_KEYWORDS) or any(k in lowered for k in CATALOG_KEYWORDS)
    return not has_other_intent


def supervisor_node(state: AgentState) -> dict:
    """Decide which specialist agent should be primary for this turn."""
    latest_user_text = _latest_user_text(state)

    # Rule 1: honor a promised catalog follow-up (e.g. assistant just asked
    # "what genre?" in order to suggest songs).
    if _assistant_promised_catalog_followup(state):
        logger.info("supervisor routed via followup-promise override", extra={"next_agent": "catalog_agent"})
        return {"next_agent": "catalog_agent"}

    # Rule 2: mixed intent (invoice + catalog keywords both present) prioritizes
    # invoice flow first so account authentication/details are handled before any
    # catalog follow-up.
    if latest_user_text and _is_mixed_intent(latest_user_text):
        logger.info("supervisor routed mixed-intent request", extra={"next_agent": "invoice_agent"})
        return {"next_agent": "invoice_agent"}

    # Rule 3: a bare email/phone with no other intent goes to memory_agent, so it
    # can acknowledge the identity and surface any preferences already on file —
    # instead of landing on catalog_agent or invoice_agent, which have nothing to
    # do with a plain identifier and tend to reply with an unhelpful refusal.
    if latest_user_text and _is_bare_identifier(latest_user_text):
        logger.info("supervisor routed bare identifier", extra={"next_agent": "memory_agent"})
        return {"next_agent": "memory_agent"}

    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    response = router_llm.invoke(messages)
    raw = response.content.strip().lower()

    agent_name = next((a for a in VALID_AGENTS if a in raw), None)

    if agent_name is None:
        logger.warning("supervisor produced unparseable routing decision", extra={"raw_response": raw})
        agent_name = "catalog_agent"

    logger.info("supervisor routed request", extra={"next_agent": agent_name})

    return {"next_agent": agent_name}