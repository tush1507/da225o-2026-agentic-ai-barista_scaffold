"""
Block 3 — LangGraph: Graph Wiring
------------------------------------
This is where orchestration becomes visible.
The graph makes routing logic explicit — no hidden loops.

Key concepts demonstrated:
  - StateGraph with typed state
  - Conditional edges (route based on agent output)
  - Linear edges (always go to next node)
  - compile() → runnable app
"""

from langgraph.graph import StateGraph, END
from state import BaristaState
from agents import order_agent, inventory_agent, billing_agent


# ── Routing functions (conditional edge logic) ───────────────────────────────

def route_after_order(state: BaristaState) -> str:
    """
    After OrderAgent runs, decide where to go next.
    This is the conditional edge — explicit, inspectable, testable.
    """
    if state.get("order_valid"):
        return "inventory"  # → InventoryAgent
    else:
        return "end_invalid"  # → END with error message


def route_after_inventory(state: BaristaState) -> str:
    """
    After InventoryAgent runs, decide whether to bill or abort.
    """
    if state.get("in_stock"):
        return "billing"  # → BillingAgent
    else:
        return "end_unavailable"  # → END with out-of-stock message


# ── Terminal nodes (for graceful exits) ──────────────────────────────────────

def handle_invalid_order(state: BaristaState) -> dict:
    return {
        "response": f"Sorry, I couldn't process that order: {state.get('order_error', 'unknown error')}. "
                    f"Please try again!"
    }


def handle_unavailable(state: BaristaState) -> dict:
    return {
        "response": f"Sorry! {state.get('stock_error', 'That item is unavailable.')} "
                    f"Can I suggest something else from the menu?"
    }


# ── Build the graph ───────────────────────────────────────────────────────────

def build_barista_graph() -> StateGraph:
    graph = StateGraph(BaristaState)

    # Add nodes — each wraps one specialist agent
    graph.add_node("order", order_agent)
    graph.add_node("inventory", inventory_agent)
    graph.add_node("billing", billing_agent)
    graph.add_node("end_invalid", handle_invalid_order)
    graph.add_node("end_unavailable", handle_unavailable)

    # Entry point
    graph.set_entry_point("order")

    # Conditional edge after OrderAgent
    graph.add_conditional_edges(
        "order",
        route_after_order,
        {
            "inventory": "inventory",
            "end_invalid": "end_invalid",
        },
    )

    # Conditional edge after InventoryAgent
    graph.add_conditional_edges(
        "inventory",
        route_after_inventory,
        {
            "billing": "billing",
            "end_unavailable": "end_unavailable",
        },
    )

    # Terminal edges — both billing and error nodes go to END
    graph.add_edge("billing", END)
    graph.add_edge("end_invalid", END)
    graph.add_edge("end_unavailable", END)

    return graph.compile()


# ── Visual: print the graph structure ────────────────────────────────────────

def print_graph_structure():
    print("\nGraph structure:")
    print("  [START]")
    print("     ↓")
    print("  [OrderAgent]")
    print("     ├── valid=True   → [InventoryAgent]")
    print("     │                       ├── in_stock=True  → [BillingAgent] → [END]")
    print("     │                       └── in_stock=False → [end_unavailable] → [END]")
    print("     └── valid=False  → [end_invalid] → [END]")
    print()
