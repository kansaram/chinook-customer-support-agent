import re
from typing import Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

from ..database.memory_repository import save_preferences_list, load_preferences_list

POSITIVE_PREFERENCE_PATTERNS = [
    re.compile(r"^i\s+(?:really\s+)?(?:like|love|prefer)\s+(.+)$", re.IGNORECASE),
    re.compile(r"^you\s+(?:really\s+)?(?:like|love|prefer)\s+(.+)$", re.IGNORECASE),
    re.compile(r"^i\s+am\s+interested\s+in\s+(.+)$", re.IGNORECASE),
    re.compile(r"^i'?m\s+interested\s+in\s+(.+)$", re.IGNORECASE),
    re.compile(r"^i\s+am\s+interested\s+(.+)$", re.IGNORECASE),
    re.compile(r"^i'?m\s+interested\s+(.+)$", re.IGNORECASE),
]

NEGATIVE_PREFERENCE_PATTERNS = [
    re.compile(r"^i\s+(?:do\s+not|don't)\s+like\s+(.+)$", re.IGNORECASE),
    re.compile(r"^you\s+(?:do\s+not|don't)\s+like\s+(.+)$", re.IGNORECASE),
    re.compile(r"^i\s+(?:do\s+not|don't)\s+prefer\s+(.+)$", re.IGNORECASE),
    re.compile(r"^you\s+(?:do\s+not|don't)\s+prefer\s+(.+)$", re.IGNORECASE),
    re.compile(r"^i\s+(?:dislike|hate)\s+(.+)$", re.IGNORECASE),
    re.compile(r"^you\s+(?:dislike|hate)\s+(.+)$", re.IGNORECASE),
    re.compile(r"^i\s+(?:am\s+not|'?m\s+not)\s+interested\s+in\s+(.+)$", re.IGNORECASE),
    re.compile(r"^you\s+are\s+not\s+interested\s+in\s+(.+)$", re.IGNORECASE),
]

_TRAILING_MODIFIERS_RE = re.compile(
    r"(?:\s*,?\s*(?:now|currently|at\s+the\s+moment|these\s+days|unfortunately))+$",
    re.IGNORECASE,
)


def _collapse_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .,!?")


def _normalize_subject(text: str) -> str:
    collapsed = _collapse_text(text)
    collapsed = _TRAILING_MODIFIERS_RE.sub("", collapsed)
    return _collapse_text(collapsed)


def parse_preference_statement(preference_text: str) -> tuple[str, bool]:
    """Return a normalized preference subject and whether it is negative."""
    cleaned = _collapse_text(preference_text)
    for pattern in NEGATIVE_PREFERENCE_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            return _normalize_subject(match.group(1)), True

    for pattern in POSITIVE_PREFERENCE_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            return _normalize_subject(match.group(1)), False

    return _normalize_subject(cleaned), False


def normalize_preference_statement(preference_text: str) -> str:
    """Normalize preference statements into a canonical stored form."""
    return _collapse_text(preference_text)


def apply_preference_update(existing: list[str], preference_text: str) -> tuple[list[str], str, bool, str | None]:
    """Apply a preference statement to an existing list.

    Returns the updated list, the normalized statement, whether it is negative,
    and any conflicting preference that was replaced.
    """
    normalized_preference = normalize_preference_statement(preference_text)
    subject, is_negative = parse_preference_statement(preference_text)

    if is_negative:
        updated_list = [p for p in existing if subject.lower() not in p.lower()]
        if normalized_preference.lower() not in {p.lower() for p in updated_list}:
            updated_list.append(normalized_preference)
        conflicting = next(
            (item for item in existing if subject.lower() in item.lower() and item.lower() != normalized_preference.lower()),
            None,
        )
        return updated_list, normalized_preference, True, conflicting

    updated_list = existing + [normalized_preference] if normalized_preference.lower() not in {p.lower() for p in existing} else existing
    return updated_list, normalized_preference, False, None


def resolve_identifier(
    customer_id: Optional[int] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
) -> Optional[str]:
    """Always produce the SAME identifier string for the same person, regardless of
    which fields happen to be populated in state this session."""
    if email:
        return f"email:{email.strip().lower()}"
    if phone:
        digits = re.sub(r"\D", "", phone)
        return f"phone:{digits[-10:] if len(digits) >= 10 else digits}"
    if customer_id is not None:
        return f"cid:{customer_id}"
    return None


@tool("save_preference", description="Remember a preference the customer explicitly stated during this conversation.")
def save_preference(
    input: Annotated[dict, "input"],
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    preference_text = input.get("preference")
    if not preference_text:
        message = "No preference text provided."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    normalized_preference = normalize_preference_statement(preference_text)

    identifier = resolve_identifier(
        customer_id=state.get("customer_id"),
        email=state.get("customer_email"),
        phone=state.get("customer_phone"),
    )

    if identifier is None:
        # Queue it instead of dropping it — will be flushed once an identifier is known.
        message = (
            "I can save that once I have your email, phone, or customer ID. "
            "I've noted the preference for now."
        )
        return Command(
            update={
                "pending_preferences": [normalized_preference],
                "messages": [ToolMessage(content=message, tool_call_id=tool_call_id)],
            }
        )

    existing = state.get("preferences", [])
    updated_list, normalized_preference, is_negative, conflicting = apply_preference_update(existing, preference_text)
    save_preferences_list(identifier, updated_list)

    conflict_message = None
    if is_negative and conflicting:
        conflict_message = f"Updated your preference: I replaced {conflicting} with {normalized_preference}."

    return Command(
        update={
            "preferences": [normalized_preference],
            "messages": [
                ToolMessage(
                    content=conflict_message or f"Noted: {normalized_preference}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool("get_preferences", description="Retrieve the customer's previously saved preferences.")
def get_preferences(
    input: Annotated[dict, "input"],
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Load stored preferences for this customer from customer_memory.db."""
    identifier = resolve_identifier(
        customer_id=state.get("customer_id"),
        email=state.get("customer_email"),
        phone=state.get("customer_phone"),
    )
    if identifier is None:
        message = "I don't have enough information yet to look up saved preferences."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    preferences = load_preferences_list(identifier)
    if not preferences:
        message = "No saved preferences found for this customer."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    summary = "Saved preferences:\n" + "\n".join(f"- {p}" for p in preferences)
    return Command(
        update={
            "preferences": preferences,
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )