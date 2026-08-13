from langchain_openai import ChatOpenAI

from ..agents.prompts import CATALOG_AGENT_PROMPT
from ..agents.state import AgentState
from ..config.logging import get_logger
from ..config.settings import settings
from ..tools.catalog_tools import (
    browse_songs_by_genre_tool,
    get_albums_for_artist_tool,
    get_track_details_by_id_tool,
    search_tracks_by_composer_tool,
    search_song_by_title_fuzzy_tool,
    suggest_catalog_from_preferences_tool,
    search_tracks_by_artist_tool,
)
from ..tools.preference_tools import get_preferences, save_preference
from ..tools.handoff_tools import transfer_to_invoice_agent, transfer_to_memory_agent
from ..agents.grounding_guard import enforce_tool_grounding
from ..agents.message_sanitizer import sanitize_message_history
from langchain_core.messages import AIMessage

logger = get_logger(__name__)


CATALOG_TOOLS = [
    get_albums_for_artist_tool,
    search_tracks_by_artist_tool,
    browse_songs_by_genre_tool,
    search_song_by_title_fuzzy_tool,
    search_tracks_by_composer_tool,
    get_track_details_by_id_tool,
    suggest_catalog_from_preferences_tool,
    get_preferences,
    save_preference,
    transfer_to_invoice_agent,
    transfer_to_memory_agent,
]

llm = ChatOpenAI(model=settings.DEFAULT_MODEL, temperature=0).bind_tools(CATALOG_TOOLS)


def catalog_llm_node(state: AgentState) -> dict:
    """The catalog agent's reasoning step — decides whether to call a tool, hand off
    to another agent, or respond directly."""
    updates: dict = {}
    preferences = state.get("preferences", [])
    preference_context = (
        "Known customer preferences in state:\n"
        + "\n".join(f"- {p}" for p in preferences)
        if preferences
        else "Known customer preferences in state: none loaded yet."
    )
    system_prompt = CATALOG_AGENT_PROMPT + "\n\n" + preference_context

    messages = [{"role": "system", "content": system_prompt}] + sanitize_message_history(state["messages"])
    logger.info("catalog_agent invoked", extra={"preference_count": len(preferences)})

    try:
        response = llm.invoke(messages)
    except Exception:
        logger.error("catalog_agent LLM call failed even after sanitization", exc_info=True)
        fallback = AIMessage(content="Sorry, something went wrong on my end — could you try asking that again?")
        updates["messages"] = [fallback]
        if state.get("next_agent") == "catalog_agent":
            updates["response_parts"] = [fallback.content]
            updates["response"] = fallback.content
        return updates

    # Grounding check: if the response makes a specific claim without a tool call
    # this turn, retry once with an explicit corrective instruction.
    needs_retry, reason = enforce_tool_grounding(response, state["messages"])
    if needs_retry:
        logger.warning("grounding guard triggered a retry", extra={"agent": "catalog_agent"})
        corrective_messages = messages + [response, {"role": "system", "content": reason}]
        try:
            response = llm.invoke(corrective_messages)
        except Exception:
            logger.error("catalog_agent grounding-retry LLM call failed", exc_info=True)
            # Fall back to the ORIGINAL response rather than crashing — it may be
            # imperfectly grounded, but it's better than no response at all.
            pass

    updates["messages"] = [response]

    # Append to response_parts whenever real text is present — even if the LLM
    # ALSO attached a tool call (e.g. a handoff) to the same message, as OpenAI
    # allows content + tool_calls together and this model uses that combination.
    # Previously this only captured content when tool_calls was empty, which
    # silently dropped legitimate answers bundled with a handoff request.
    if response.content:
        updates["response_parts"] = [response.content]
    if not response.tool_calls:
        updates["response"] = response.content  # kept for any code still reading this directly

    return updates