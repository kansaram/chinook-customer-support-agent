# agents/supervisor.py

import re
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from ..config.settings import settings
from ..agents.state import AgentState
from ..agents.prompts import SUPERVISOR_PROMPT
from ..config.logging import get_logger

logger = get_logger(__name__)


class RouteDecision(BaseModel):
    agent: Literal["invoice_agent", "catalog_agent", "memory_agent"] = Field(
        description="Which specialist should handle this turn"
    )
    reason: str = Field(description="One short phrase explaining the choice, for logging")


router_llm = ChatOpenAI(model=settings.DEFAULT_MODEL, temperature=0).with_structured_output(RouteDecision)

_INVOICE_IDENTIFICATION_REQUEST_RE = re.compile(r"\bcustomer\s*id\b", re.IGNORECASE)

# A message asking to see saved preferences must ALWAYS go to memory_agent, even if
# it also contains a phone/email/other keyword that would otherwise nudge the router
# toward invoice_agent (e.g. "give me my preferences and my phone number 555-1234").
# Only memory_agent can correctly answer this, and it already has the logic to
# extract any identifier mentioned in the same message. Mirrors PREFERENCE_LOOKUP_PATTERNS
# in memory_agent.py — kept as a separate copy intentionally (same reasoning as the
# other duplicated pattern lists: avoids coupling supervisor routing to memory_agent's
# internals, at the cost of needing to update both if wording changes).
_PREFERENCE_LOOKUP_RE = re.compile(
    r"\b(?:what|which)\s+(?:are\s+)?(?:my|the)\s+(?:saved\s+)?preferences\b"
    r"|\bwhat\s+do\s+you\s+remember\b"
    r"|\b(?:show|tell|give|send)\s+me\s+(?:my\s+)?preferences\b",
    re.IGNORECASE,
)


def _last_assistant_text(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        role = getattr(msg, "type", None)
        content = getattr(msg, "content", None)
        if role == "ai":
            return str(content or "")
        if isinstance(msg, dict) and str(msg.get("role", "")).lower() in ("assistant", "ai"):
            return str(msg.get("content", "") or "")
    return ""


def _latest_user_text(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        role = getattr(msg, "type", None)
        if role == "human":
            return str(getattr(msg, "content", "") or "")
        if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
            return str(msg.get("content", "") or "")
    return ""


def _invoice_agent_awaiting_identifier(state: AgentState) -> bool:
    return bool(_INVOICE_IDENTIFICATION_REQUEST_RE.search(_last_assistant_text(state)))


def _wants_saved_preferences(user_text: str) -> bool:
    return bool(_PREFERENCE_LOOKUP_RE.search(user_text))


def supervisor_node(state: AgentState) -> dict:
    """Decide which specialist agent should be primary for this turn."""

    latest_user_text = _latest_user_text(state)

    # Highest priority: an explicit "show my preferences" request always goes to
    # memory_agent, even if the same message also mentions a phone/email that would
    # otherwise pull the router toward invoice_agent.
    if latest_user_text and _wants_saved_preferences(latest_user_text):
        logger.info("supervisor routed preference-lookup override", extra={"next_agent": "memory_agent"})
        return {"next_agent": "memory_agent", "handoff_count": 0}

    # Fast path: invoice_agent already asked for an identifier — keep it with
    # invoice_agent regardless of what the reply looks like. No LLM call needed.
    if _invoice_agent_awaiting_identifier(state):
        logger.info("supervisor routed continuation of invoice identification", extra={"next_agent": "invoice_agent"})
        return {"next_agent": "invoice_agent", "handoff_count": 0}

    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]

    try:
        decision = router_llm.invoke(messages)
        agent_name = decision.agent
        logger.info("supervisor routed request", extra={"next_agent": agent_name, "reason": decision.reason})
    except Exception:
        logger.warning("structured routing failed, falling back to catalog_agent", exc_info=True)
        agent_name = "catalog_agent"

    return {"next_agent": agent_name, "handoff_count": 0}