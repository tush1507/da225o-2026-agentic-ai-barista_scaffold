"""
Block 3 — LangGraph: Entry Point
-----------------------------------
Run this to see the full multi-agent barista in action.

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

    initial_state: BaristaState = {
        "user_request": user_request,
        "drink_name": None,
        "size": None,
        "milk": None,
        "order_valid": None,
        "order_error": None,
        "in_stock": None,
        "stock_error": None,
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
