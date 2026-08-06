from langchain_openai import ChatOpenAI

from ..agents.prompts import CATALOG_AGENT_PROMPT
from ..agents.state import AgentState
from ..config.logging import get_logger
from ..config.settings import settings
from ..tools.catalog_tools import (
	browse_songs_by_genre_tool,
	get_albums_for_artist_tool,
	get_track_details_by_id_tool,
	search_song_by_title_fuzzy_tool,
	suggest_catalog_from_preferences_tool,
	search_tracks_by_artist_tool,
)
from ..tools.preference_tools import get_preferences, save_preference

logger = get_logger(__name__)


CATALOG_TOOLS = [
	get_albums_for_artist_tool,
	search_tracks_by_artist_tool,
	browse_songs_by_genre_tool,
	search_song_by_title_fuzzy_tool,
	get_track_details_by_id_tool,
	suggest_catalog_from_preferences_tool,
	get_preferences,
	save_preference,
]

llm = ChatOpenAI(model=settings.DEFAULT_MODEL, temperature=0).bind_tools(CATALOG_TOOLS)


def catalog_llm_node(state: AgentState) -> dict:
	"""The catalog agent's reasoning step — decides whether to call a tool or respond directly."""
	preferences = state.get("preferences", [])
	preference_context = (
		"Known customer preferences in state:\n"
		+ "\n".join(f"- {p}" for p in preferences)
		if preferences
		else "Known customer preferences in state: none loaded yet."
	)
	system_prompt = CATALOG_AGENT_PROMPT + "\n\n" + preference_context

	messages = [{"role": "system", "content": system_prompt}] + state["messages"]
	logger.info("catalog_agent invoked", extra={"preference_count": len(preferences)})

	response = llm.invoke(messages)
	return {"messages": [response]}
