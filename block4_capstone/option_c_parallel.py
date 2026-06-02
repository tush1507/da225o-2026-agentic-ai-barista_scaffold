"""
Block 4 — Capstone Option C: Parallel Fan-out
-----------------------------------------------
Pattern: Fire InventoryAgent and NutritionAgent simultaneously.
Orchestrator waits for both, merges results, then bills.

LangGraph supports parallel execution natively via Send() or
by adding multiple edges from one node to many.

Concept demonstrated: parallel agent execution + result merging.

Run:
    python block4_capstone/option_c_parallel.py
"""

import json
import asyncio
import time
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"


# ── State ─────────────────────────────────────────────────────────────────────

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


# ── InventoryAgent ────────────────────────────────────────────────────────────

def inventory_agent(state: BaristaStateParallel) -> dict:
    """Check stock. In parallel execution, this runs concurrently with NutritionAgent."""
    print(f"[InventoryAgent] Checking stock for {state['drink_name']}...")
    time.sleep(0.3)  # simulate async I/O

    available = INVENTORY.get(state["drink_name"], False)
    msg = f"{state['drink_name']} is {'available' if available else 'OUT OF STOCK'}."
    print(f"[InventoryAgent] {msg}")
    return {"in_stock": available, "stock_message": msg}


# ── NutritionAgent ────────────────────────────────────────────────────────────

NUTRITION_TOOLS = [
    {
        "name": "lookup_nutrition",
        "description": "Look up nutritional information for a drink.",
        "input_schema": {
            "type": "object",
            "properties": {"drink_name": {"type": "string"}},
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
                    nutrition = NUTRITION_DB.get(block.input["drink_name"], {"calories": 100, "caffeine_mg": 50})
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


# ── Merge + Bill node (runs after both parallel agents complete) ──────────────

def merge_and_bill(state: BaristaStateParallel) -> dict:
    """
    This node runs only after BOTH InventoryAgent and NutritionAgent complete.
    LangGraph ensures both upstream nodes have written to state before this runs.
    """
    print(f"\n[MergeAndBill] Merging parallel results...")
    print(f"  Stock: {state['stock_message']}")
    print(f"  Nutrition: {state['nutrition_summary']}")

    if not state["in_stock"]:
        return {
            "response": f"Sorry! {state['stock_message']} Please choose another drink.",
            "final_price": None,
            "order_id": None,
        }

    base = PRICES.get(state["drink_name"], 4.0) * SIZE_MULTIPLIER.get(state["size"], 1.0)
    final = round(base, 2)
    order_id = f"ORD-PAR-{hash(state['drink_name']) % 9000 + 1000}"

    response = (
        f"Order confirmed: {state['size']} {state['drink_name']} with {state['milk']} milk.\n"
        f"Price: ${final:.2f}\n"
        f"Nutrition: {state['nutrition_summary']}"
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
    graph.add_node("merge_and_bill", merge_and_bill)

    graph.set_entry_point("fan_out")

    # Fork: fan_out → both inventory and nutrition simultaneously
    graph.add_edge("fan_out", "inventory")
    graph.add_edge("fan_out", "nutrition")

    # Join: both must complete before merge_and_bill
    graph.add_edge("inventory", "merge_and_bill")
    graph.add_edge("nutrition", "merge_and_bill")

    graph.add_edge("merge_and_bill", END)

    return graph.compile()


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_parallel_graph()

    def run(drink: str, size: str = "medium", milk: str = "oat"):
        print(f"\n{'='*60}")
        print(f"Order: {size} {drink} with {milk} milk")
        print(f"{'='*60}")

        t0 = time.time()
        state = app.invoke({
            "user_request": f"I'd like a {size} {drink} with {milk} milk",
            "drink_name": drink, "size": size, "milk": milk,
            "in_stock": None, "stock_message": None,
            "calories": None, "caffeine_mg": None, "nutrition_summary": None,
            "final_price": None, "order_id": None, "response": None,
        })
        elapsed = time.time() - t0

        print(f"\nFinal ({elapsed:.1f}s): {state['response']}")

    run("Cold Brew", "large", "oat")
    run("Flat White", "medium", "whole")  # out of stock

    # ── EXERCISE ──────────────────────────────────────────────────────────
    # 1. Add a third parallel branch: AllergenAgent
    #    It checks if the chosen milk contains common allergens.
    # 2. Add it as another fork from fan_out: graph.add_edge("fan_out", "allergen")
    # 3. Add it to the join: graph.add_edge("allergen", "merge_and_bill")
    # 4. MergeAndBill should print the allergen warning.
    # Observe: adding a parallel agent requires zero changes to existing agents.
