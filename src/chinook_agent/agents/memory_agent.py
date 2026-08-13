# agents/memory_agent.py

import re
from urllib import response
from langchain_openai import ChatOpenAI
from ..config.settings import settings
from ..agents.state import AgentState
from ..agents.prompts import MEMORY_AGENT_PROMPT
from ..tools.preference_tools import (
    save_preference,
    get_preferences,
    resolve_identifier,
    merge_preference_statements,
    preference_subject_key,
    llm_parse_preference,
)
from ..tools.handoff_tools import transfer_to_invoice_agent, transfer_to_catalog_agent
from ..agents.grounding_guard import enforce_tool_grounding
from ..database.memory_repository import load_preferences_list, save_preferences_list
from ..config.logging import get_logger
from langchain_core.messages import AIMessage
from ..agents.message_sanitizer import sanitize_message_history
logger = get_logger(__name__)

# save_preference is intentionally NOT bound here. The deterministic scan below
# (_extract_explicit_music_preferences + _persist_preferences) already guarantees
# every explicit preference statement gets saved, in both primary and background
# mode, before the LLM ever runs. Having the LLM also call save_preference itself
# was a redundant round-trip that never actually changed the outcome (the merge
# in _persist_preferences is idempotent) — removing it simplifies the tool surface
# without losing any capability. get_preferences stays bound as a fallback for
# preference-lookup phrasings the deterministic PREFERENCE_LOOKUP_PATTERNS list
# doesn't recognize.
MEMORY_TOOLS = [get_preferences, transfer_to_invoice_agent, transfer_to_catalog_agent]
llm = ChatOpenAI(model=settings.DEFAULT_MODEL, temperature=0).bind_tools(MEMORY_TOOLS)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")

PREFERENCE_LOOKUP_PATTERNS = [
    re.compile(r"\b(?:what|which)\s+(?:are\s+)?(?:my|the)\s+(?:saved\s+)?preferences\b", re.IGNORECASE),
    re.compile(r"\b(?:what\s+do\s+you\s+remember|what\s+preferences\s+do\s+you\s+remember)\b", re.IGNORECASE),
    re.compile(r"\b(?:show|tell)\s+me\s+(?:my\s+)?preferences\b", re.IGNORECASE),
    re.compile(r"\b(?:give|send)\s+me\s+(?:my\s+)?preferences\b", re.IGNORECASE),
]

# NOTE: intentionally narrower/differently-anchored than preference_tools' patterns
# (uses \b instead of ^, and only covers a subset of phrasings). Kept as documented
# debt — unifying the two lists risks changing which sentences trigger detection.
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


# ============================================================
# Message helpers
# ============================================================

def _get_message_role(message) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "")
    msg_type = getattr(message, "type", None)
    if msg_type == "human":
        return "user"
    if msg_type == "ai":
        return "assistant"
    return str(getattr(message, "role", None) or "")


def _get_message_content(message) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _extract_identifier_from_messages(messages: list) -> dict:
    """Scan recent messages for an email or phone, so preferences work even for
    people who never match a Chinook customer record."""
    for msg in reversed(messages):
        content = _get_message_content(msg)
        email_match = EMAIL_RE.search(content)
        phone_match = PHONE_RE.search(content)
        if email_match or phone_match:
            return {
                "email": email_match.group(0) if email_match else None,
                "phone": phone_match.group(0) if phone_match else None,
            }
    return {"email": None, "phone": None}


def _extract_identifier_from_latest_message(messages: list) -> dict:
    """Check ONLY the most recent user message for an explicit email/phone. Used to
    detect when the customer is intentionally providing a NEW/different identifier
    mid-conversation (e.g. "check with my phone instead") — this must take priority
    over whatever was already resolved earlier, so the system actually re-queries
    under the new identifier instead of silently reusing stale results."""
    for msg in reversed(messages):
        if _get_message_role(msg).lower() not in {"user", "human"}:
            continue
        content = _get_message_content(msg)
        email_match = EMAIL_RE.search(content)
        phone_match = PHONE_RE.search(content)
        return {
            "email": email_match.group(0) if email_match else None,
            "phone": phone_match.group(0) if phone_match else None,
        }
    return {"email": None, "phone": None}


def _sentence_matches_regex(sentence: str) -> bool:
    return any(pattern.search(sentence) for pattern in PREFERENCE_PATTERNS)


def _wants_saved_preferences(messages: list) -> bool:
    """True if the most recent user message explicitly asks to see saved preferences
    (e.g. "what are my preferences", "what do you remember about me")."""
    for msg in reversed(messages):
        if _get_message_role(msg).lower() not in {"user", "human"}:
            continue
        content = _get_message_content(msg).strip()
        if not content:
            continue
        for chunk in re.split(r"[\n.!?]+", content):
            sentence = chunk.strip(" \t\r\n,;:")
            if sentence and any(pattern.search(sentence) for pattern in PREFERENCE_LOOKUP_PATTERNS):
                return True
        return False  # only check the single most recent user message
    return False


def _format_preferences_response(identifier: str, preferences: list[str]) -> str:
    if not preferences:
        return "I couldn't find any saved preferences for your account yet — let me know what you like and I'll remember it."

    lines: list[str] = []
    for preference in preferences:
        lines.extend(f"- {part.strip()}" for part in re.split(r"\s*;\s*|\n+", preference) if part.strip())

    return "Here are your saved preferences:\n\n" + "\n".join(lines)


def _extract_explicit_music_preferences(messages: list) -> list[str]:
    """Extract explicit preference statements from user messages only.

    `messages` is expected to be ONLY the slice of messages not yet scanned (see
    memory_llm_node) — this function itself doesn't know or care about scan
    position, it just processes whatever it's given.

    Order: regex patterns first (cheap, deterministic, unchanged from before).
    Any sentence regex doesn't recognize is then checked with the LLM fallback
    (second option) — this catches phrasings like "I'm really into jazz" that
    the fixed regex list doesn't cover, without changing what regex already caught.
    """
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

            key = sentence.lower()
            if key in seen_lower:
                continue

            if _sentence_matches_regex(sentence):
                extracted.append(sentence)
                seen_lower.add(key)
                continue

            # Second option: ask the LLM only when regex found nothing for this sentence.
            llm_result = llm_parse_preference(sentence)
            if llm_result is not None:
                extracted.append(sentence)
                seen_lower.add(key)

    return extracted


# ============================================================
# Persistence helper — shared by the "flush pending" and "capture new" steps
# ============================================================

def _persist_preferences(identifier: str, base_preferences: list[str], new_items: list[str]) -> list[str]:
    """Merge new_items into the stored preference list for `identifier` and save it."""
    current = load_preferences_list(identifier)
    current = merge_preference_statements(current, base_preferences)
    current = merge_preference_statements(current, new_items)
    save_preferences_list(identifier, current)
    logger.info("persisted preferences to database", extra={"identifier": identifier, "preferences": current})
    return current


# ============================================================
# Node
# ============================================================




def memory_llm_node(state: AgentState) -> dict:
    updates: dict = {}
    is_primary = state.get("next_agent") == "memory_agent"
    all_messages = state["messages"]

    email = state.get("customer_email")
    phone = state.get("customer_phone")

    # If the customer's latest message explicitly mentions an identifier, prefer it
    # over whatever was already resolved earlier — this is what makes "check with
    # my phone instead" actually trigger a fresh lookup instead of silently reusing
    # results loaded under a previously-given email.
    latest_mentioned = _extract_identifier_from_latest_message(all_messages)
    new_email = latest_mentioned["email"]
    new_phone = latest_mentioned["phone"]

    identifier_changed = False
    if new_email and new_email.lower() != (email or "").lower():
        email, phone = new_email, None
        identifier_changed = True
    elif new_phone and new_phone != phone:
        phone, email = new_phone, None
        identifier_changed = True

    if not email and not phone:
        extracted = _extract_identifier_from_messages(all_messages)
        email = extracted["email"]
        phone = extracted["phone"]

    if email:
        updates["customer_email"] = email
    if phone:
        updates["customer_phone"] = phone
    if identifier_changed:
        # Force a real reload under the new identifier rather than trusting
        # whatever was cached in state["preferences"] from the old one.
        updates["preferences_loaded"] = False

    identifier = resolve_identifier(
        customer_id=state.get("customer_id"),
        email=email,
        phone=phone,
    )

    preferences_already_loaded = updates.get("preferences_loaded", state.get("preferences_loaded"))
    if identifier and not preferences_already_loaded:
        loaded = load_preferences_list(identifier)
        sanitized_loaded = merge_preference_statements([], loaded)
        if sanitized_loaded != loaded:
            save_preferences_list(identifier, sanitized_loaded)
            logger.info(
                "sanitized stale/duplicate preferences on load",
                extra={"identifier": identifier, "before": loaded, "after": sanitized_loaded},
            )
        updates["preferences"] = sanitized_loaded
        updates["preferences_loaded"] = True
        logger.info("preferences loaded", extra={"identifier": identifier, "count": len(sanitized_loaded)})

    # Explicit "show my preferences" request — answer directly, even in background
    # mode. This is what makes "here's my email, what are my preferences" work in
    # a single turn: the supervisor may have routed this turn to a different
    # primary agent, but only memory_agent can actually answer this part, so it
    # sets `response` itself. The `set_response` reducer on state["response"]
    # makes this safe even if the primary agent also sets a response this turn.
    if _wants_saved_preferences(all_messages):
        if identifier:
            preferences = updates.get("preferences", state.get("preferences", []))
            sanitized = merge_preference_statements([], preferences)
            if sanitized != preferences:
                save_preferences_list(identifier, sanitized)
                logger.info(
                    "sanitized stale/duplicate preferences before display",
                    extra={"identifier": identifier, "before": preferences, "after": sanitized},
                )
                preferences = sanitized
                updates["preferences"] = sanitized
            answer = _format_preferences_response(identifier, preferences)
        else:
            answer = "I'd love to look that up, but I need your email or phone number first."
        updates["response_parts"] = [answer]
        updates["response"] = answer  # kept for any code still reading this directly
        return updates

    # Flush anything that was queued while no identifier was known
    pending = state.get("pending_preferences", [])
    if identifier and pending:
        current = _persist_preferences(identifier, state.get("preferences", []), pending)
        updates["preferences"] = current
        updates["pending_preferences"] = []
        logger.info("flushed pending preferences", extra={"identifier": identifier, "count": len(pending)})

    # Deterministic preference capture — only scan messages not yet scanned, instead
    # of reprocessing the entire conversation every turn. Dedup within a turn still
    # applies via seen_lower/pending_subjects, but this avoids the ever-growing cost
    # (including repeated LLM-fallback calls for the same already-failed sentences)
    # that came from rescanning the full history on every single node invocation.
    scanned_count = state.get("preferences_scanned_count", 0)
    new_messages = all_messages[scanned_count:]
    updates["preferences_scanned_count"] = len(all_messages)

    detected = _extract_explicit_music_preferences(new_messages)
    if detected:
        pending_now = updates.get("pending_preferences", state.get("pending_preferences", []))
        pending_subjects = {preference_subject_key(item) for item in pending_now}
        new_for_identifier: list[str] = []
        changed = False

        for preference_text in detected:
            subject_key = preference_subject_key(preference_text)
            if subject_key in pending_subjects:
                continue
            if identifier:
                new_for_identifier.append(preference_text)
                changed = True
            else:
                pending_now.append(preference_text)
                pending_subjects.add(subject_key)
                changed = True

        if changed:
            if identifier and new_for_identifier:
                current = _persist_preferences(identifier, state.get("preferences", []), new_for_identifier)
                updates["preferences"] = current
                updates["pending_preferences"] = []
                logger.info(
                    "captured explicit preferences",
                    extra={"identifier": identifier, "count": len(detected)},
                )
            elif not identifier:
                updates["pending_preferences"] = pending_now
                logger.info("queued explicit preferences", extra={"count": len(pending_now)})

    # In background mode, only sync preference state and avoid producing LLM/tool messages.
    # This prevents cross-agent tool-call ordering issues during parallel fan-out.
    if not is_primary:
        return updates

    messages = [{"role": "system", "content": MEMORY_AGENT_PROMPT}]

    if identifier:
        # Tell the LLM directly that identity is already confirmed, instead of
        # relying on it to notice an email/phone somewhere earlier in the message
        # history. This is a stronger guarantee than the prompt instruction alone —
        # the model doesn't have to infer anything, it's stated as a fact for THIS
        # specific call.
        messages.append({
            "role": "system",
            "content": (
                f"The customer's identity is ALREADY confirmed for this conversation "
                f"(identifier on file: {identifier}). Do NOT ask them for an email, "
                f"phone number, or customer ID — you already have what you need."
            ),
        })

    messages += sanitize_message_history(all_messages)

    try:
        response = llm.invoke(messages)
    except Exception:
        logger.error("memory_agent LLM call failed even after sanitization", exc_info=True)
        fallback = AIMessage(content="Sorry, something went wrong on my end — could you try asking that again?")
        updates["messages"] = [fallback]
        updates["response_parts"] = [fallback.content]
        updates["response"] = fallback.content
        return updates

    needs_retry, reason = enforce_tool_grounding(response, all_messages)
    if needs_retry:
        logger.warning("grounding guard triggered a retry", extra={"agent": "memory_agent"})
        try:
            response = llm.invoke(messages + [response, {"role": "system", "content": reason}])
        except Exception:
            logger.error("memory_agent grounding-retry LLM call failed", exc_info=True)
            # Fall back to the ORIGINAL response rather than crashing — it may be
            # imperfectly grounded, but it's better than no response at all.
            pass

    updates["messages"] = [response]

    if not response.tool_calls:
        updates["response_parts"] = [response.content]
        updates["response"] = response.content

    return updates