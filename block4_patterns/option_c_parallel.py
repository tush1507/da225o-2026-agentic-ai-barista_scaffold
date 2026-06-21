"""
Block 4 — Pattern C: Parallel Fan-out
---------------------------------------
Pattern: Fork execution into independent branches, then merge results.

Block 3 comparison:
  Block 3's pipeline was strictly sequential — OrderAgent finished, then
  InventoryAgent started, then BillingAgent. But InventoryAgent and
  NutritionAgent are completely independent: neither needs the other's
  output, and neither modifies shared fields the other reads.
  Running them sequentially wastes time equal to one full agent round-trip.

  This pattern adds a fan_out node that forks to both agents simultaneously.
  LangGraph detects that two edges leave fan_out and two edges arrive at
  merge_and_bill — it runs the middle nodes in parallel and automatically
  waits for both before allowing merge_and_bill to proceed.

  No threading code, no asyncio, no explicit synchronisation — just two
  add_edge calls and LangGraph handles the rest.

Run:
    python block4_patterns/option_c_parallel.py
"""

import json
import time
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"


# ── State ─────────────────────────────────────────────────────────────────────
# Block 3 comparison: Block 3's BaristaState only needed one agent's output
# at a time — each node wrote and the next read sequentially.
# Here both parallel agents write to the SAME state object simultaneously.
# LangGraph merges their partial updates before handing state to merge_and_bill,
# so that node sees both in_stock (from InventoryAgent) AND nutrition_summary
# (from NutritionAgent) already populated.

class BaristaStateParallel(TypedDict):
    user_request: str
    drink_name: str
    size: str
    milk: str

    # Set by InventoryAgent
    in_stock: Optional[bool]
    stock_message: Optional[str]

    # Set by NutritionAgent
    calories: Optional[int]
    caffeine_mg: Optional[int]
    nutrition_summary: Optional[str]

    # Set by AllergenAgent (exercise)
    allergen_warning: Optional[str]

    # Set by BillingAgent (after merge)
    final_price: Optional[float]
    order_id: Optional[str]
    response: Optional[str]


# ── Nutrition data (simulated — replace with real API) ───────────────────────

NUTRITION_DB = {
    "Latte":        {"calories": 190, "caffeine_mg": 75},
    "Cappuccino":   {"calories": 120, "caffeine_mg": 75},
    "Espresso":     {"calories": 5,   "caffeine_mg": 63},
    "Cold Brew":    {"calories": 15,  "caffeine_mg": 200},
    "Iced Latte":   {"calories": 130, "caffeine_mg": 75},
    "Frappuccino":  {"calories": 400, "caffeine_mg": 95},
    "Flat White":   {"calories": 150, "caffeine_mg": 130},
    "Americano":    {"calories": 15,  "caffeine_mg": 150},
    "Iced Matcha":  {"calories": 200, "caffeine_mg": 70},
}

INVENTORY = {
    "Espresso": True, "Cappuccino": True, "Latte": True,
    "Flat White": False, "Americano": True,
    "Cold Brew": True, "Iced Latte": True,
    "Frappuccino": True, "Iced Matcha": False,
}

PRICES = {
    "Espresso": 2.5, "Cappuccino": 3.5, "Latte": 4.0, "Flat White": 3.8,
    "Americano": 3.0, "Cold Brew": 4.5, "Iced Latte": 4.2,
    "Frappuccino": 5.0, "Iced Matcha": 4.8,
}
SIZE_MULTIPLIER = {"small": 0.85, "medium": 1.0, "large": 1.25}


def _canonical(name: str) -> str:
    """Case-insensitive drink name lookup."""
    lower = name.lower()
    for key in INVENTORY:
        if key.lower() == lower:
            return key
    return name


# ── InventoryAgent ────────────────────────────────────────────────────────────

def inventory_agent(state: BaristaStateParallel) -> dict:
    """
    Check stock. Runs concurrently with NutritionAgent — no changes needed
    to this function to make it parallel. LangGraph handles the scheduling.

    Block 3 comparison: identical responsibility to Block 3's inventory_agent.
    The only difference is that it now runs at the same time as NutritionAgent
    instead of after OrderAgent completes. The function itself is unchanged.
    """
    drink = _canonical(state["drink_name"])
    print(f"[InventoryAgent] Checking stock for {drink}...")
    time.sleep(0.3)  # simulate async I/O

    available = INVENTORY.get(drink, False)
    msg = f"{drink} is {'available' if available else 'OUT OF STOCK'}."
    print(f"[InventoryAgent] {msg}")
    return {"in_stock": available, "stock_message": msg}


# ── NutritionAgent ────────────────────────────────────────────────────────────

NUTRITION_TOOLS = [
    {
        "name": "lookup_nutrition",
        "description": "Look up nutritional information for a drink.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drink_name": {
                    "type": "string",
                    "description": "The base drink name only — no size or milk modifier.",
                    "enum": list(NUTRITION_DB.keys()),
                }
            },
            "required": ["drink_name"],
        },
    }
]


def nutrition_agent(state: BaristaStateParallel) -> dict:
    """
    Look up nutritional info and generate a human-friendly summary.
    Runs in parallel with InventoryAgent.
    """
    print(f"[NutritionAgent] Looking up nutrition for {state['drink_name']}...")
    time.sleep(0.3)  # simulate async I/O

    messages = [
        {"role": "user", "content": f"Get the nutrition facts for a {state['size']} {state['drink_name']} with {state['milk']} milk."}
    ]

    while True:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=256,
            system="You are a nutrition information agent. Look up facts and give a one-line friendly summary.",
            tools=NUTRITION_TOOLS,
            messages=messages,
        )

        if resp.stop_reason == "end_turn":
            summary = next((b.text for b in resp.content if hasattr(b, "text")), "Nutrition info unavailable.")
            return {
                "calories": None,
                "caffeine_mg": None,
                "nutrition_summary": summary,
            }

        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            nutrition = None

            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if block.name == "lookup_nutrition":
                    nutrition = NUTRITION_DB.get(_canonical(block.input["drink_name"]), {"calories": 100, "caffeine_mg": 50})
                    result = json.dumps(nutrition)
                else:
                    result = json.dumps({"error": "unknown tool"})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

            messages.append({"role": "user", "content": results})

            if nutrition:
                return {
                    "calories": nutrition["calories"],
                    "caffeine_mg": nutrition["caffeine_mg"],
                    "nutrition_summary": f"{nutrition['calories']} kcal, {nutrition['caffeine_mg']}mg caffeine",
                }


# ── Exercise: AllergenAgent (third parallel branch) ──────────────────────────
# Runs concurrently with InventoryAgent and NutritionAgent.
# Zero changes to those two agents — just two add_edge calls in the graph.

ALLERGENS = {
    "whole":  ["dairy"],
    "oat":    [],
    "almond": ["tree nuts"],
    "soy":    ["soy"],
    "none":   [],
}


def allergen_agent(state: BaristaStateParallel) -> dict:
    milk = state.get("milk", "whole")
    warnings = ALLERGENS.get(milk, [])
    if warnings:
        msg = f"Contains: {', '.join(warnings)}."
    else:
        msg = "No common allergens detected."
    print(f"[AllergenAgent] Milk '{milk}' — {msg}")
    return {"allergen_warning": msg}


# ── Merge + Bill node (runs after both parallel agents complete) ──────────────

def merge_and_bill(state: BaristaStateParallel) -> dict:
    """
    Runs only after BOTH parallel agents complete — LangGraph guarantees this.
    By the time this node executes, state already contains in_stock (from
    InventoryAgent) and nutrition_summary (from NutritionAgent).

    Block 3 comparison: Block 3 had no merge node — each agent handed off to
    the next in a chain. The merge node is a new concept that only appears when
    you have parallel branches. Its job is to combine the partial results and
    make a single decision (bill or reject) based on all of them.
    """
    print(f"\n[MergeAndBill] Merging parallel results...")
    print(f"  Stock: {state['stock_message']}")
    print(f"  Nutrition: {state['nutrition_summary']}")
    print(f"  Allergens: {state['allergen_warning']}")

    if not state["in_stock"]:
        return {
            "response": f"Oh no — {state['stock_message']} Can I get you something else?",
            "final_price": None,
            "order_id": None,
        }

    base = PRICES.get(state["drink_name"], 4.0) * SIZE_MULTIPLIER.get(state["size"], 1.0)
    final = round(base, 2)
    order_id = f"ORD-PAR-{hash(state['drink_name']) % 9000 + 1000}"

    response = (
        f"Perfect! One {state['size']} {state['drink_name']} with {state['milk']} milk for ${final:.2f}.\n"
        f"Just so you know: {state['nutrition_summary']}.\n"
        f"Allergen info: {state['allergen_warning']}"
    )

    return {"final_price": final, "order_id": order_id, "response": response}


# ── Build graph with parallel branches ───────────────────────────────────────
#
# LangGraph parallel execution structure:
#
#   [START]
#      ↓
#   [fan_out]  ← dummy node to trigger parallel execution
#    ↙       ↘
# [inventory] [nutrition]   ← run in parallel (both receive same state)
#    ↘       ↙
#   [merge_and_bill]        ← LangGraph waits for both before running this
#      ↓
#    [END]

def fan_out(state: BaristaStateParallel) -> dict:
    """Entry node — just passes state through to trigger parallel branches."""
    print(f"\n[FanOut] Dispatching parallel agents for: {state['drink_name']}")
    return {}


def build_parallel_graph():
    graph = StateGraph(BaristaStateParallel)

    graph.add_node("fan_out", fan_out)
    graph.add_node("inventory", inventory_agent)
    graph.add_node("nutrition", nutrition_agent)
    graph.add_node("allergen", allergen_agent)   # exercise: third parallel branch
    graph.add_node("merge_and_bill", merge_and_bill)

    graph.set_entry_point("fan_out")

    # Fork: three edges from fan_out — all three run simultaneously.
    graph.add_edge("fan_out", "inventory")
    graph.add_edge("fan_out", "nutrition")
    graph.add_edge("fan_out", "allergen")        # exercise: one new line

    # Join: all three must complete before merge_and_bill runs.
    graph.add_edge("inventory", "merge_and_bill")
    graph.add_edge("nutrition", "merge_and_bill")
    graph.add_edge("allergen", "merge_and_bill") # exercise: one new line

    graph.add_edge("merge_and_bill", END)

    return graph.compile()


# ── Run ───────────────────────────────────────────────────────────────────────

MENU_DRINKS = [
    "Espresso", "Cappuccino", "Latte", "Flat White", "Americano",
    "Cold Brew", "Iced Latte", "Frappuccino", "Iced Matcha",
]


def _parse_order(user_request: str) -> tuple[str, str, str]:
    """Extract drink, size, and milk from a natural language request."""
    req = user_request.lower()
    drink = next((d for d in MENU_DRINKS if d.lower() in req), "Latte")
    size = next((s for s in ["small", "medium", "large"] if s in req), "medium")
    milk = next((m for m in ["oat", "almond", "soy", "whole"] if m in req), "whole")
    return drink, size, milk


if __name__ == "__main__":
    app = build_parallel_graph()

    def run(user_request: str):
        drink, size, milk = _parse_order(user_request)
        print(f"\n{'='*60}")
        print(f"Order: {size} {drink} with {milk} milk")
        print(f"{'='*60}")

        t0 = time.time()
        state = app.invoke({
            "user_request": user_request,
            "drink_name": drink, "size": size, "milk": milk,
            "in_stock": None, "stock_message": None,
            "calories": None, "caffeine_mg": None, "nutrition_summary": None,
            "allergen_warning": None,
            "final_price": None, "order_id": None, "response": None,
        })
        elapsed = time.time() - t0
        print(f"\nFinal ({elapsed:.1f}s): {state['response']}")

    print("Barista Agent (Parallel) — type your order, or 'quit' to exit.\n")
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
    # 1. Add a third parallel branch: AllergenAgent
    #    It checks if the chosen milk contains common allergens.
    # 2. Add it as another fork from fan_out: graph.add_edge("fan_out", "allergen")
    # 3. Add it to the join: graph.add_edge("allergen", "merge_and_bill")
    # 4. MergeAndBill should print the allergen warning.
    # Observe: adding a parallel agent requires zero changes to existing agents.