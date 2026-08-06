# agents/memory_agent.py

import re
from langchain_openai import ChatOpenAI
from ..config.settings import settings
from ..agents.state import AgentState
from ..agents.prompts import MEMORY_AGENT_PROMPT
from ..tools.preference_tools import save_preference, get_preferences, resolve_identifier
from ..database.memory_repository import load_preferences_list
from ..config.logging import get_logger
from ..database.memory_repository import save_preferences_list
logger = get_logger(__name__)

MEMORY_TOOLS = [save_preference, get_preferences]
llm = ChatOpenAI(model=settings.DEFAULT_MODEL, temperature=0).bind_tools(MEMORY_TOOLS)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")


def _extract_identifier_from_messages(messages: list) -> dict:
    """Scan recent messages for an email or phone, so preferences work even for
    people who never match a Chinook customer record."""
    for msg in reversed(messages):
        content = getattr(msg, "content", "") or ""
        if not isinstance(content, str):
            continue
        email_match = EMAIL_RE.search(content)
        phone_match = PHONE_RE.search(content)
        if email_match or phone_match:
            return {
                "email": email_match.group(0) if email_match else None,
                "phone": phone_match.group(0) if phone_match else None,
            }
    return {"email": None, "phone": None}


def memory_llm_node(state: AgentState) -> dict:
    updates: dict = {}
    is_primary = state.get("next_agent") == "memory_agent"

    email = state.get("customer_email")
    phone = state.get("customer_phone")

    if not email and not phone:
        extracted = _extract_identifier_from_messages(state["messages"])
        email = extracted["email"]
        phone = extracted["phone"]
        if email:
            updates["customer_email"] = email
        if phone:
            updates["customer_phone"] = phone

    identifier = resolve_identifier(
        customer_id=state.get("customer_id"),
        email=email,
        phone=phone,
    )

    if identifier and not state.get("preferences_loaded"):
        loaded = load_preferences_list(identifier)
        updates["preferences"] = loaded
        updates["preferences_loaded"] = True
        logger.info("preferences loaded", extra={"identifier": identifier, "count": len(loaded)})

    # Flush anything that was queued while no identifier was known
    pending = state.get("pending_preferences", [])
    if identifier and pending:
        current = updates.get("preferences", state.get("preferences", []))
        merged = current + [p for p in pending if p not in current]
        save_preferences_list(identifier, merged)
        updates["preferences"] = pending  # reducer appends these onto current
        updates["pending_preferences"] = []  # NOTE: see caveat below — plain assign won't clear an additive reducer
        logger.info("flushed pending preferences", extra={"identifier": identifier, "count": len(pending)})

    messages = [{"role": "system", "content": MEMORY_AGENT_PROMPT}] + state["messages"]
    response = llm.invoke(messages)
    updates["messages"] = [response]

    if not response.tool_calls and is_primary:
        updates["response"] = response.content

    return updates