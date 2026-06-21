"""
Block 3 — LangGraph: Entry Point
-----------------------------------
Run this to see the full multi-agent barista in action.

Block 2 comparison:
  In Block 2, the entry point was run_barista_agent() — one function that
  contained the entire agent loop, tool dispatch, and state management.
  Invoking it meant starting that loop from scratch every time.

  Here the entry point calls app.invoke(initial_state) on the compiled graph.
  The graph handles execution order, state passing, and routing internally.
  This file only needs to know the shape of the initial state — it has no
  knowledge of which agents exist or how they are connected.

Run:
    python block3_langgraph/run.py
"""

from graph import build_barista_graph, print_graph_structure
from state import BaristaState

app = build_barista_graph()


def run(user_request: str):
    print(f"\n{'='*60}")
    print(f"User: {user_request}")
    print(f"{'='*60}")

    # In Block 2, state was implicit — just the messages list inside the loop.
    # Here we construct the full state explicitly before handing it to the graph.
    # Every field starts as None; each agent fills in only the fields it owns.
    initial_state: BaristaState = {
        "user_request": user_request,
        "drink_name": None,
        "size": None,
        "milk": None,
        "order_valid": None,
        "order_error": None,
        "in_stock": None,
        "stock_error": None,
        "discount_pct": None,
        "price": None,
        "discount": None,
        "final_price": None,
        "order_id": None,
        "response": None,
        "messages": [],
    }

    final_state = app.invoke(initial_state)

    print(f"\n{'─'*60}")
    print(f"Final response: {final_state.get('response', '(no response)')}")
    print(f"{'─'*60}")
    return final_state


if __name__ == "__main__":
    print_graph_structure()

    print("Barista Agent (LangGraph) — type your order, or 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        run(user_input)
        print()

    # ── EXERCISE ──────────────────────────────────────────────────────────
    # 1. Add a LoyaltyAgent between InventoryAgent and BillingAgent.
    #    It reads state["drink_name"] and decides the discount percentage.
    # 2. Modify route_after_inventory to route to "loyalty" instead of "billing".
    # 3. Add an edge: loyalty → billing.
    # Observe: the graph structure changes, nothing else does.
