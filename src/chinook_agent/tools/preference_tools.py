import json
import re
from typing import Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from rapidfuzz import process, fuzz

from ..database.memory_repository import save_preferences_list, load_preferences_list
from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)


# ============================================================
# Regex patterns for parsing natural-language preference statements
# ============================================================

POSITIVE_PREFERENCE_PATTERNS = [
    re.compile(r"^i\s+(?:really\s+)?(?:like|love|prefer)\s+(.+)$", re.IGNORECASE),
    re.compile(r"^you\s+(?:really\s+)?(?:like|love|prefer)\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(?:like|love|prefer)\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(.+?)\s+is\s+my\s+favorite\b", re.IGNORECASE),
    re.compile(r"^(.+?)\s+is\s+my\s+favourite\b", re.IGNORECASE),
    re.compile(r"^i\s+am\s+interested\s+in\s+(.+)$", re.IGNORECASE),
    re.compile(r"^i'?m\s+interested\s+in\s+(.+)$", re.IGNORECASE),
    re.compile(r"^i\s+am\s+interested\s+(.+)$", re.IGNORECASE),
    re.compile(r"^i'?m\s+interested\s+(.+)$", re.IGNORECASE),
]

NEGATIVE_PREFERENCE_PATTERNS = [
    re.compile(r"^i\s+(?:do\s+not|don't)\s+like\s+(.+)$", re.IGNORECASE),
    re.compile(r"^you\s+(?:do\s+not|don't)\s+like\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(?:do\s+not|don't)\s+like\s+(.+)$", re.IGNORECASE),
    re.compile(r"^i\s+(?:do\s+not|don't)\s+prefer\s+(.+)$", re.IGNORECASE),
    re.compile(r"^you\s+(?:do\s+not|don't)\s+prefer\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(?:do\s+not|don't)\s+prefer\s+(.+)$", re.IGNORECASE),
    re.compile(r"^i\s+(?:dislike|hate)\s+(.+)$", re.IGNORECASE),
    re.compile(r"^you\s+(?:dislike|hate)\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(?:dislike|hate)\s+(.+)$", re.IGNORECASE),
    re.compile(r"^i\s+(?:am\s+not|'?m\s+not)\s+interested\s+in\s+(.+)$", re.IGNORECASE),
    re.compile(r"^you\s+are\s+not\s+interested\s+in\s+(.+)$", re.IGNORECASE),
]

_TRAILING_MODIFIERS_RE = re.compile(
    r"(?:\s*,?\s*(?:now|currently|at\s+the\s+moment|these\s+days|unfortunately))+$",
    re.IGNORECASE,
)
_LEADING_FILLERS_RE = re.compile(
    r"^(?:(?:now|currently|sorry|but|well|actually|so|then)\s+)+",
    re.IGNORECASE,
)
_LEADING_THINK_RE = re.compile(r"^i\s+think\s+(?:that\s+)?", re.IGNORECASE)
_NESTED_SUBJECT_RE = re.compile(
    r"^(?:(?:i|you)\s+)?(?:do\s+not\s+like|don't\s+like|dislike|hate|like|love|prefer)\s+",
    re.IGNORECASE,
)

# Strips generic gerund lead-ins so "listening to Classical music" and "Classical
# music" normalize to the SAME subject, instead of being stored as two entries.
_LEADING_GERUND_RE = re.compile(
    r"^(?:listening\s+to|listening|playing|watching)\s+",
    re.IGNORECASE,
)


# ============================================================
# Text normalization helpers
# ============================================================

def _collapse_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .,!?")


def _normalize_statement_text(text: str) -> str:
    cleaned = _collapse_text(text)
    cleaned = _LEADING_FILLERS_RE.sub("", cleaned)
    cleaned = _LEADING_THINK_RE.sub("", cleaned)
    return _collapse_text(cleaned)


def _normalize_subject(text: str) -> str:
    collapsed = _normalize_statement_text(text)
    collapsed = _NESTED_SUBJECT_RE.sub("", collapsed)
    collapsed = _normalize_statement_text(collapsed)
    collapsed = _LEADING_GERUND_RE.sub("", collapsed)
    collapsed = _TRAILING_MODIFIERS_RE.sub("", collapsed)
    return _collapse_text(collapsed)


def normalize_preference_statement(preference_text: str) -> str:
    """Normalize preference statements into a canonical stored form."""
    return _collapse_text(preference_text)


# ============================================================
# LLM fallback — only used when NO regex pattern matches
# ============================================================

_preference_llm = ChatOpenAI(model=settings.DEFAULT_MODEL, temperature=0)

_LLM_PARSE_PROMPT = """Does this sentence express a personal preference (something the \
speaker likes, loves, prefers, dislikes, or hates)?

Sentence: "{text}"

If NO, respond with exactly: NONE
If YES, respond in exactly this format and nothing else:
SUBJECT: <short noun phrase for what they like/dislike>
SENTIMENT: <POSITIVE or NEGATIVE>
"""


def llm_parse_preference(text: str) -> Optional[tuple[str, bool]]:
    """Second-option parser: only called when regex patterns find no match.
    Returns (subject, is_negative), or None if the LLM finds no preference either
    or the call fails for any reason."""
    try:
        response = _preference_llm.invoke(_LLM_PARSE_PROMPT.format(text=text))
        content = (response.content or "").strip()
    except Exception:
        logger.warning("LLM preference fallback parsing failed", exc_info=True)
        return None

    if content.upper().startswith("NONE"):
        return None

    subject_match = re.search(r"SUBJECT:\s*(.+)", content, re.IGNORECASE)
    sentiment_match = re.search(r"SENTIMENT:\s*(POSITIVE|NEGATIVE)", content, re.IGNORECASE)
    if not subject_match:
        return None

    subject = _collapse_text(subject_match.group(1))
    is_negative = bool(sentiment_match and sentiment_match.group(1).upper() == "NEGATIVE")
    return subject, is_negative


# ============================================================
# Preference parsing / canonicalization
# ============================================================

def parse_preference_statement(preference_text: str) -> tuple[str, bool]:
    """Return a normalized preference subject and whether it is negative.

    Order: regex patterns first (fast, deterministic). If nothing matches, fall
    back to the LLM to catch phrasings the regex list doesn't cover (e.g. "I'm
    really into rock" or "I dig jazz"). If the LLM also finds nothing, the whole
    cleaned sentence is used as the subject, treated as positive — same default
    as before this fallback existed.
    """
    cleaned = _normalize_statement_text(preference_text)

    for pattern in NEGATIVE_PREFERENCE_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            return _normalize_subject(match.group(1)), True

    for pattern in POSITIVE_PREFERENCE_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            return _normalize_subject(match.group(1)), False

    llm_result = llm_parse_preference(cleaned)
    if llm_result is not None:
        subject, is_negative = llm_result
        return _normalize_subject(subject), is_negative

    return _normalize_subject(cleaned), False


def canonicalize_preference_statement(preference_text: str) -> str:
    subject, is_negative = parse_preference_statement(preference_text)
    if not subject:
        return _collapse_text(preference_text)
    return f"You {'dislike' if is_negative else 'like'} {subject}".strip()


def preference_subject_key(preference_text: str) -> str:
    subject, _ = parse_preference_statement(preference_text)
    return (subject or _collapse_text(preference_text)).casefold()


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


SUBJECT_FUZZY_MATCH_THRESHOLD = 82


def _find_matching_subject_key(key: str, existing_keys: list[str]) -> Optional[str]:
    """Find an existing subject key that's close enough to `key` to be the same
    subject (e.g. 'classical music' vs 'listening to classical music' vs a typo
    like 'classcial music'). Returns None if no existing key is a close enough match."""
    if not existing_keys:
        return None
    match = process.extractOne(key, existing_keys, scorer=fuzz.WRatio)
    if match is None:
        return None
    matched_key, score, _ = match
    return matched_key if score >= SUBJECT_FUZZY_MATCH_THRESHOLD else None


def merge_preference_statements(existing: list[str], preference_texts: list[str]) -> list[str]:
    """Collapse preference statements into one canonical summary entry.

    The stored database shape is a single JSON array item per customer, where that
    item is a semicolon-delimited summary of the customer's current preferences.
    Later statements replace earlier ones for the same subject.
    """
    subject_order: list[str] = []
    subject_map: dict[str, str] = {}

    def _ingest(statement: str) -> None:
        parts = [part.strip() for part in re.split(r"\s*;\s*|\n+", statement) if part.strip()]
        for part in parts:
            key = preference_subject_key(part)
            canonical = canonicalize_preference_statement(part)

            target_key = key if key in subject_map else _find_matching_subject_key(key, list(subject_map.keys()))

            if target_key is None:
                subject_order.append(key)
                subject_map[key] = canonical
                continue

            existing_canonical = subject_map[target_key]
            _, existing_negative = parse_preference_statement(existing_canonical)
            _, new_negative = parse_preference_statement(canonical)
            if existing_negative != new_negative:
                subject_map[target_key] = canonical

    for item in existing:
        _ingest(item)
    for item in preference_texts:
        _ingest(item)

    if not subject_order:
        return []

    return ["; ".join(subject_map[key] for key in subject_order)]


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


# ============================================================
# Tool helpers
# ============================================================

def _tool_message(payload: dict, tool_call_id: str, **extra_updates) -> Command:
    """Build a Command whose ToolMessage.content is a JSON string, plus any extra state updates."""
    return Command(update={"messages": [ToolMessage(content=json.dumps(payload), tool_call_id=tool_call_id)], **extra_updates})


# ============================================================
# Tools
# ============================================================

@tool("save_preference", description="Remember a preference the customer explicitly stated during this conversation.")
def save_preference(
    input: Annotated[dict, "input"],
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    preference_text = input.get("preference")
    if not preference_text:
        return _tool_message({"status": "error", "message": "No preference text provided."}, tool_call_id)

    identifier = resolve_identifier(
        customer_id=state.get("customer_id"),
        email=state.get("customer_email"),
        phone=state.get("customer_phone"),
    )

    if identifier is None:
        # Queue it instead of dropping it — will be flushed once an identifier is known.
        normalized_preference = normalize_preference_statement(preference_text)
        payload = {
            "status": "pending",
            "message": "I can save that once I have your email, phone, or customer ID. I've noted the preference for now.",
        }
        return _tool_message(payload, tool_call_id, pending_preferences=[normalized_preference])

    existing = load_preferences_list(identifier)
    merged = merge_preference_statements(existing, state.get("preferences", []))
    merged = merge_preference_statements(merged, state.get("pending_preferences", []))
    merged = merge_preference_statements(merged, [preference_text])
    save_preferences_list(identifier, merged)
    logger.info("saved preferences to database", extra={"identifier": identifier, "preferences": merged})

    current_text = merged[0] if merged else canonicalize_preference_statement(preference_text)
    is_update = "dislike" in current_text.lower() and len(merged) == 1
    payload = {
        "status": "ok",
        "updated": is_update,
        "current_preference": current_text,
        "message": f"Updated your preference: {current_text}." if is_update else f"Noted: {current_text}",
    }

    return _tool_message(payload, tool_call_id, preferences=merged, pending_preferences=[])


MAX_GET_PREFERENCES_CALLS_PER_TURN = 2


@tool("get_preferences", description="Retrieve the customer's previously saved preferences.")
def get_preferences(
    input: Annotated[dict, "input"],
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Load stored preferences for this customer from customer_memory.db."""
    call_count = state.get("get_preferences_call_count", 0)

    # Hard stop: if this tool has already been called this many times in the
    # SAME turn, refuse and force the LLM to answer from what it already has —
    # guarantees the loop terminates regardless of why the LLM keeps re-calling
    # it (ambiguous phrasing, model quirk, etc.), rather than relying on a
    # prompt instruction alone, which proved unreliable in testing.
    if call_count >= MAX_GET_PREFERENCES_CALLS_PER_TURN:
        payload = {
            "status": "already_fetched",
            "message": "You already called get_preferences this turn — use that result to answer now instead of calling it again.",
        }
        return _tool_message(payload, tool_call_id, get_preferences_call_count=call_count + 1)

    identifier = resolve_identifier(
        customer_id=state.get("customer_id"),
        email=state.get("customer_email"),
        phone=state.get("customer_phone"),
    )
    if identifier is None:
        payload = {"status": "error", "message": "I don't have enough information yet to look up saved preferences."}
        return _tool_message(payload, tool_call_id, get_preferences_call_count=call_count + 1)

    preferences = load_preferences_list(identifier)
    sanitized = merge_preference_statements([], preferences)
    if sanitized != preferences:
        save_preferences_list(identifier, sanitized)
        preferences = sanitized

    if not preferences:
        payload = {"status": "not_found", "message": "No saved preferences found for this customer.", "preferences": []}
        return _tool_message(payload, tool_call_id, get_preferences_call_count=call_count + 1)

    bullet_items: list[str] = []
    for preference in preferences:
        bullet_items.extend(part.strip() for part in re.split(r"\s*;\s*|\n+", preference) if part.strip())

    payload = {"status": "ok", "preferences": bullet_items}
    return _tool_message(
        payload, tool_call_id,
        preferences=preferences,
        get_preferences_call_count=call_count + 1,
    )