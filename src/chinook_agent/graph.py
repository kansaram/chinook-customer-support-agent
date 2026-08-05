from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from chinook_agent.agents.state import AgentState

from chinook_agent.agents.supervisor import supervisor_node

from chinook_agent.agents.invoice_agent import invoice_llm_node

from chinook_agent.tools.invoice_tools import (
    customer_lookup,
    get_invoice_history,
    get_tracks_for_invoices_for_customer,
    get_tracks_for_invoice_for_customer,
    get_support_rep_for_customer_by_invoiceId,
)

from chinook_agent.agents.catalog_agent import catalog_llm_node
from chinook_agent.tools.catalog_tools import search_artists_fuzzy, search_tracks_fuzzy

from chinook_agent.agents.memory_agent import memory_llm_node
from chinook_agent.tools.preference_tools import get_preferences, save_preference


def load_preferences_node(state: AgentState) -> dict:
    """Non-LLM node: runs once per conversation, loads preferences if we know the customer's email."""
    email = state.get("customer_email")
    loaded = get_preferences(email) if email else {}
    return {"preferences_loaded": True, "known_preferences": loaded}


def route_after_supervisor(state: AgentState) -> str:
    """Force one preferences-load pass before the first agent handoff each conversation."""
    if not state.get("preferences_loaded"):
        return "load_preferences"
    return state["next_agent"]


def route_after_load_preferences(state: AgentState) -> str:
    """After loading preferences, continue to whichever agent the supervisor originally picked."""
    return state["next_agent"]


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # --- Supervisor: routing decision only ---
    graph.add_node("supervisor", supervisor_node)
    graph.set_entry_point("supervisor")

    # --- One-time preferences load (no LLM call) ---
    graph.add_node("load_preferences", load_preferences_node)

    # --- Invoice agent ---
    graph.add_node("invoice_agent", invoice_llm_node)
    graph.add_node("invoice_tools", ToolNode([
        customer_lookup,
        get_invoice_history,
        get_tracks_for_invoices_for_customer,
        get_tracks_for_invoice_for_customer,
        get_support_rep_for_customer_by_invoiceId,
    ]))
    graph.add_conditional_edges("invoice_agent", tools_condition, {"tools": "invoice_tools", END: END})
    graph.add_edge("invoice_tools", "invoice_agent")

    # --- Catalog agent ---
    graph.add_node("catalog_agent", catalog_llm_node)
    graph.add_node("catalog_tools", ToolNode([search_artists_fuzzy, search_tracks_fuzzy]))
    graph.add_conditional_edges("catalog_agent", tools_condition, {"tools": "catalog_tools", END: END})
    graph.add_edge("catalog_tools", "catalog_agent")

    # --- Memory agent (explicit save/query requests) ---
    graph.add_node("memory_agent", memory_llm_node)
    graph.add_node("memory_tools", ToolNode([get_preferences, save_preference]))
    graph.add_conditional_edges("memory_agent", tools_condition, {"tools": "memory_tools", END: END})
    graph.add_edge("memory_tools", "memory_agent")

    # --- Supervisor -> (load_preferences once, then chosen agent) OR (straight to chosen agent) ---
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "load_preferences": "load_preferences",
            "invoice_agent": "invoice_agent",
            "catalog_agent": "catalog_agent",
            "memory_agent": "memory_agent",
        },
    )

    # --- After loading preferences, continue to the originally chosen agent ---
    graph.add_conditional_edges(
        "load_preferences",
        route_after_load_preferences,
        {
            "invoice_agent": "invoice_agent",
            "catalog_agent": "catalog_agent",
            "memory_agent": "memory_agent",
        },
    )

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)