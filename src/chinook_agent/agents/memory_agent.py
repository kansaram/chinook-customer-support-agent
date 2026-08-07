# agents/memory_agent.py

import re
from langchain_openai import ChatOpenAI
from ..config.settings import settings
from ..agents.state import AgentState
from ..agents.prompts import MEMORY_AGENT_PROMPT
from ..tools.preference_tools import (
    save_preference,
    get_preferences,
    resolve_identifier,
    apply_preference_update,
)
from ..database.memory_repository import load_preferences_list
from ..config.logging import get_logger
from ..database.memory_repository import save_preferences_list
logger = get_logger(__name__)

MEMORY_TOOLS = [save_preference, get_preferences]
llm = ChatOpenAI(model=settings.DEFAULT_MODEL, temperature=0).bind_tools(MEMORY_TOOLS)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
PREFERENCE_PATTERNS = [
    re.compile(r"\bi\s+(?:really\s+)?(?:like|love|prefer)\s+(.+)$", re.IGNORECASE),
    re.compile(r"\b(.+?)\s+is\s+my\s+favorite\b", re.IGNORECASE),
    re.compile(r"\bi\s+am\s+interested\s+in\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bi'?m\s+interested\s+in\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bi\s+am\s+interested\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bi'?m\s+interested\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bi\s+(?:do\s+not|don't)\s+like\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bi\s+(?:do\s+not|don't)\s+prefer\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bi\s+(?:dislike|hate)\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bi\s+(?:am\s+not|'?m\s+not)\s+interested\s+in\s+(.+)$", re.IGNORECASE),
]


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


def _get_message_role(message) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "")

    msg_type = getattr(message, "type", None)
    if msg_type == "human":
        return "user"
    if msg_type == "ai":
        return "assistant"

    role = getattr(message, "role", None)
    return str(role or "")


def _get_message_content(message) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _extract_explicit_music_preferences(messages: list) -> list[str]:
    """Extract explicit preference statements from user messages only."""
    extracted: list[str] = []
    seen_lower: set[str] = set()

    for msg in messages:
        if _get_message_role(msg).lower() not in {"user", "human"}:
            continue

        content = _get_message_content(msg).strip()
        if not content:
            continue

        for chunk in re.split(r"[\n.!?]+", content):
            sentence = chunk.strip(" \t\r\n,;:")
            if not sentence:
                continue

            for pattern in PREFERENCE_PATTERNS:
                if pattern.search(sentence):
                    key = sentence.lower()
                    if key not in seen_lower:
                        extracted.append(sentence)
                        seen_lower.add(key)
                    break

    return extracted


def _append_unique_preferences(existing: list[str], new_items: list[str]) -> list[str]:
    existing_lower = {item.lower() for item in existing}
    merged = list(existing)
    for item in new_items:
        item_lower = item.lower()
        if item_lower not in existing_lower:
            merged.append(item)
            existing_lower.add(item_lower)
    return merged


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
        before = list(current)
        for pending_preference in pending:
            current, normalized_preference, _, _ = apply_preference_update(current, pending_preference)

        save_preferences_list(identifier, current)
        before_lower = {p.lower() for p in before}
        updates["preferences"] = [p for p in current if p.lower() not in before_lower]
        updates["pending_preferences"] = []  # NOTE: see caveat below — plain assign won't clear an additive reducer
        logger.info("flushed pending preferences", extra={"identifier": identifier, "count": len(pending)})

    # Deterministic preference capture each turn for explicit user statements.
    detected = _extract_explicit_music_preferences(state["messages"])
    if detected:
        current = updates.get("preferences", state.get("preferences", []))
        pending_now = updates.get("pending_preferences", state.get("pending_preferences", []))
        queued_updates: list[str] = []

        for preference_text in detected:
            if preference_text.lower() in {p.lower() for p in current}:
                continue
            if preference_text.lower() in {p.lower() for p in pending_now}:
                continue

            if identifier:
                current, normalized_preference, _, _ = apply_preference_update(current, preference_text)
                queued_updates.append(normalized_preference)
            else:
                queued_updates.append(preference_text)

        if queued_updates:
            if identifier:
                save_preferences_list(identifier, current)
                updates["preferences"] = queued_updates
                logger.info(
                    "captured explicit preferences",
                    extra={"identifier": identifier, "count": len(queued_updates)},
                )
            else:
                updates["pending_preferences"] = queued_updates
                logger.info("queued explicit preferences", extra={"count": len(queued_updates)})

    # In background mode, only sync preference state and avoid producing LLM/tool messages.
    # This prevents cross-agent tool-call ordering issues during parallel fan-out.
    if not is_primary:
        return updates

    messages = [{"role": "system", "content": MEMORY_AGENT_PROMPT}] + state["messages"]
    response = llm.invoke(messages)
    updates["messages"] = [response]

    if not response.tool_calls and is_primary:
        updates["response"] = response.content

    return updates