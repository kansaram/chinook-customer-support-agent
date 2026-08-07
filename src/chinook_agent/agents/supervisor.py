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
    """Structured routing output — replaces free-text parsing of the LLM's answer.
    A schema is far more reliable than asking for a bare word and regex-matching
    it, which is what caused the inconsistent routing you were seeing."""
    agent: Literal["invoice_agent", "catalog_agent", "memory_agent"] = Field(
        description="Which specialist should handle this turn"
    )
    reason: str = Field(description="One short phrase explaining the choice, for logging")


router_llm = ChatOpenAI(model=settings.DEFAULT_MODEL, temperature=0).with_structured_output(RouteDecision)

# Kept as the one deterministic rule: invoice_agent's identification prompt is the
# only place "customer ID" appears in an assistant message (memory_agent's ask only
# says "email or phone"). This is a near-zero-ambiguity signal that doesn't need an
# LLM call to resolve, so it stays as a fast-path check rather than being folded
# into the prompt's few-shot examples.
_INVOICE_IDENTIFICATION_REQUEST_RE = re.compile(r"\bcustomer\s*id\b", re.IGNORECASE)


def _last_assistant_text(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        role = getattr(msg, "type", None)
        content = getattr(msg, "content", None)
        if role == "ai":
            return str(content or "")
        if isinstance(msg, dict) and str(msg.get("role", "")).lower() in ("assistant", "ai"):
            return str(msg.get("content", "") or "")
    return ""


def _invoice_agent_awaiting_identifier(state: AgentState) -> bool:
    return bool(_INVOICE_IDENTIFICATION_REQUEST_RE.search(_last_assistant_text(state)))


def supervisor_node(state: AgentState) -> dict:
    """Decide which specialist agent should be primary for this turn."""

    # Fast path: invoice_agent already asked for an identifier — keep it with
    # invoice_agent regardless of what the reply looks like. No LLM call needed.
    if _invoice_agent_awaiting_identifier(state):
        logger.info("supervisor routed continuation of invoice identification", extra={"next_agent": "invoice_agent"})
        return {"next_agent": "invoice_agent"}

    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]

    try:
        decision = router_llm.invoke(messages)
        agent_name = decision.agent
        logger.info("supervisor routed request", extra={"next_agent": agent_name, "reason": decision.reason})
    except Exception:
        logger.warning("structured routing failed, falling back to catalog_agent", exc_info=True)
        agent_name = "catalog_agent"

    return {"next_agent": agent_name}