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
#
# In Block 2 (agent_loop.py) this routing was buried inside run_barista_agent():
#
#   if response.stop_reason == "end_turn":
#       ...
#   if response.stop_reason == "tool_use":
#       ...
#
# That worked for one agent, but with multiple agents the logic sprawls:
# "did OrderAgent succeed? if so, call InventoryAgent, then check its result..."
# It becomes hard to read, test, or change without breaking something else.
#
# LangGraph extracts that routing into standalone functions with clear names.
# Each function answers exactly one question about the state.
# You can unit-test route_after_order() by passing a dict — no LLM needed.

def route_after_order(state: BaristaState) -> str:
    # In Block 2 this check was an if/else inside the while loop.
    # Here it's a named function — you can see the full routing logic at a glance.
    if state.get("order_valid"):
        return "inventory"  # → InventoryAgent
    else:
        return "end_invalid"  # → END with error message


def route_after_inventory(state: BaristaState) -> str:
    # Another routing decision that would have been tangled in the Block 2 loop.
    # Extracting it here makes the "what happens after inventory check?" question
    # answerable by reading one small function instead of scanning the whole loop.
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
    # In Block 2, the "graph" was implicit — just a sequence of function calls
    # inside run_barista_agent(). Adding a new agent meant editing that function
    # and carefully wiring it in by hand.
    #
    # Here the graph is a first-class object. Adding an agent is:
    #   graph.add_node("new_agent", my_agent_fn)
    #   graph.add_edge("previous_node", "new_agent")
    #   graph.add_edge("new_agent", "next_node")
    # Nothing else changes. Existing agents don't need to know about the new one.
    graph = StateGraph(BaristaState)

    # Each node is one specialist agent function from agents.py.
    # Block 2 had one agent doing everything; here each agent has one job.
    graph.add_node("order", order_agent)
    graph.add_node("inventory", inventory_agent)
    graph.add_node("billing", billing_agent)
    graph.add_node("end_invalid", handle_invalid_order)
    graph.add_node("end_unavailable", handle_unavailable)

    graph.set_entry_point("order")

    # Conditional edges call the routing functions defined above.
    # The dict maps each possible return value to a node name.
    # LangGraph validates this at compile time — a typo in a node name
    # raises an error immediately, not silently at runtime.
    graph.add_conditional_edges(
        "order",
        route_after_order,
        {
            "inventory": "inventory",
            "end_invalid": "end_invalid",
        },
    )

    graph.add_conditional_edges(
        "inventory",
        route_after_inventory,
        {
            "billing": "billing",
            "end_unavailable": "end_unavailable",
        },
    )

    # Linear edges — no decision needed, always go to END.
    graph.add_edge("billing", END)
    graph.add_edge("end_invalid", END)
    graph.add_edge("end_unavailable", END)

    # compile() freezes the graph and returns a runnable object.
    # In Block 2 the "runnable" was just the run_barista_agent() function.
    # compile() also validates that every node is reachable and every
    # conditional edge target exists — catching wiring errors early.
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
