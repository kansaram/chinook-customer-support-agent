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


def route_after_supervisor(state: AgentState) -> list[str]:
    primary = state["next_agent"]
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
    )
    graph.add_conditional_edges("invoice_agent", tools_condition, {"tools": "invoice_tools", END: END})
    graph.add_edge("invoice_tools", "invoice_agent")

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
    )
    graph.add_conditional_edges("catalog_agent", tools_condition, {"tools": "catalog_tools", END: END})
    graph.add_edge("catalog_tools", "catalog_agent")

    graph.add_node("memory_agent", memory_llm_node)
    graph.add_node(
        "memory_tools",
        ToolNode([
            save_preference,
            get_preferences,
            transfer_to_invoice_agent,
            transfer_to_catalog_agent,
        ]),
    )
    graph.add_conditional_edges("memory_agent", tools_condition, {"tools": "memory_tools", END: END})
    graph.add_edge("memory_tools", "memory_agent")

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"invoice_agent": "invoice_agent", "catalog_agent": "catalog_agent", "memory_agent": "memory_agent"},
    )

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)