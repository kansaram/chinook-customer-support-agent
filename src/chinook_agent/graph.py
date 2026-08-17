from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from .agents.state import AgentState
from .agents.supervisor import supervisor_node

from .agents.invoice_agent import invoice_llm_node
from .tools.invoice_tools import (
    customer_lookup,
    get_invoice_history,
    get_tracks_for_invoices_for_customer,
    get_tracks_for_invoice_for_customer,
    get_support_rep_for_customer_by_invoiceId,
)

from .agents.catalog_agent import catalog_llm_node
from .tools.catalog_tools import (
    get_albums_for_artist_tool,
    search_tracks_by_artist_tool,
    browse_songs_by_genre_tool,
    search_song_by_title_fuzzy_tool,
    search_tracks_by_composer_tool,
    get_track_details_by_id_tool,
    suggest_catalog_from_preferences_tool,
)

from .agents.memory_agent import memory_llm_node
from .tools.preference_tools import save_preference, get_preferences

from .tools.handoff_tools import (
    transfer_to_invoice_agent,
    transfer_to_catalog_agent,
    transfer_to_memory_agent,
)
from langchain_core.messages import ToolMessage

def _route_after_tools(agent_name: str):
    """Build a conditional-edge function for the given agent's tools node.

    If the last tool message was a SUCCESSFUL handoff (transfer_to_X whose
    Command already redirected execution via goto), don't also fall back to
    the static edge — that would re-invoke this agent unnecessarily on top
    of the handoff that already happened. Only take the static edge back to
    the calling agent for normal (non-handoff, or refused-handoff) tool calls.
    """
    def route(state: AgentState) -> str:
        last_message = state["messages"][-1]
        is_handoff_tool = (
            isinstance(last_message, ToolMessage)
            and last_message.name
            and last_message.name.startswith("transfer_to_")
        )
        handoff_succeeded = is_handoff_tool and last_message.content.startswith("Handing this off to")
        if handoff_succeeded:
            return END  # Command(goto=...) already routed elsewhere — don't also loop back
        return agent_name

    return route

def route_after_supervisor(state: AgentState) -> list[str]:
    """Fan out to the primary agent AND memory_agent in parallel, unless memory_agent
    IS primary, or the supervisor already declined the request directly (off_topic)."""
    primary = state["next_agent"]
    if primary == "off_topic":
        return [END]
    branches = [primary]
    if primary != "memory_agent":
        branches.append("memory_agent")
    return branches


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.set_entry_point("supervisor")

    # Each ToolNode now also includes the OTHER two agents' handoff tools, so a
    # Command(goto=...) returned by a handoff call is recognized and executed.
    graph.add_node("invoice_agent", invoice_llm_node)
    graph.add_node(
        "invoice_tools",
        ToolNode([
            customer_lookup,
            get_invoice_history,
            get_tracks_for_invoices_for_customer,
            get_tracks_for_invoice_for_customer,
            get_support_rep_for_customer_by_invoiceId,
            transfer_to_catalog_agent,
            transfer_to_memory_agent,
        ]),
        destinations=("catalog_agent", "memory_agent"),
    )
    graph.add_conditional_edges(
    "invoice_tools",
    _route_after_tools("invoice_agent"),
    {"invoice_agent": "invoice_agent", END: END},
    )

    graph.add_conditional_edges(
        "catalog_tools",
        _route_after_tools("catalog_agent"),
        {"catalog_agent": "catalog_agent", END: END},
    )

    graph.add_conditional_edges(
        "memory_tools",
        _route_after_tools("memory_agent"),
        {"memory_agent": "memory_agent", END: END},
    )
    graph.add_conditional_edges("invoice_agent", tools_condition, {"tools": "invoice_tools", END: END})
    

    graph.add_node("catalog_agent", catalog_llm_node)
    graph.add_node(
        "catalog_tools",
        ToolNode([
            get_albums_for_artist_tool,
            search_tracks_by_artist_tool,
            browse_songs_by_genre_tool,
            search_song_by_title_fuzzy_tool,
            search_tracks_by_composer_tool,
            get_track_details_by_id_tool,
            suggest_catalog_from_preferences_tool,
            save_preference,
            get_preferences,
            transfer_to_invoice_agent,
            transfer_to_memory_agent,
        ]),
        destinations=("invoice_agent", "memory_agent"),
    )
    graph.add_conditional_edges("catalog_agent", tools_condition, {"tools": "catalog_tools", END: END})
    

    graph.add_node("memory_agent", memory_llm_node)
    graph.add_node(
        "memory_tools",
        ToolNode([
            save_preference,
            get_preferences,
            transfer_to_invoice_agent,
            transfer_to_catalog_agent,
        ]),
        destinations=("invoice_agent", "catalog_agent"),
    )
    graph.add_conditional_edges("memory_agent", tools_condition, {"tools": "memory_tools", END: END})
    

    graph.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
        "invoice_agent": "invoice_agent",
        "catalog_agent": "catalog_agent",
        "memory_agent": "memory_agent",
        END: END,  # off_topic routes here directly
    },
)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)