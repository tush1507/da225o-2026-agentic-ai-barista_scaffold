"""
Block 4 — Capstone Option B: Dynamic Tool + Shared State
----------------------------------------------------------
Pattern: SurgeAgent reads a live tool (queue length),
writes a surge multiplier to shared state,
and BillingAgent reads it to compute the final price.

Concept demonstrated: agents sharing state — one writes, another reads.

Run:
    python block4_capstone/option_b_surge.py
"""

import json
import random
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-20250514"


# ── Extended state with surge field ──────────────────────────────────────────

class BaristaStateWithSurge(TypedDict):
    user_request: str
    drink_name: str
    size: str
    milk: str
    base_price: Optional[float]
    queue_length: Optional[int]
    surge_multiplier: Optional[float]
    surge_reason: Optional[str]
    final_price: Optional[float]
    order_id: Optional[str]
    response: Optional[str]


# ── Simulated live data (replace with real API in production) ─────────────────

def get_current_queue_length() -> int:
    """Simulate a live queue — randomised so you see different surge levels."""
    return random.choice([2, 5, 8, 12, 18, 25])  # 0-10: normal, 11-20: busy, 20+: surge


PRICES = {
    "Espresso": 2.5, "Cappuccino": 3.5, "Latte": 4.0, "Flat White": 3.8,
    "Americano": 3.0, "Cold Brew": 4.5, "Iced Latte": 4.2,
    "Frappuccino": 5.0, "Iced Matcha": 4.8,
}
SIZE_MULTIPLIER = {"small": 0.85, "medium": 1.0, "large": 1.25}


# ── SurgeAgent ────────────────────────────────────────────────────────────────

SURGE_TOOLS = [
    {
        "name": "get_queue_length",
        "description": "Get the current number of pending orders in the queue.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_surge_pricing",
        "description": "Set the surge pricing multiplier based on demand.",
        "input_schema": {
            "type": "object",
            "properties": {
                "multiplier": {
                    "type": "number",
                    "description": "Price multiplier. 1.0 = normal, 1.2 = 20% surge.",
                },
                "reason": {"type": "string"},
            },
            "required": ["multiplier", "reason"],
        },
    },
]


def surge_agent(state: BaristaStateWithSurge) -> dict:
    """
    Reads live queue length → decides surge multiplier → writes to state.
    BillingAgent will read surge_multiplier from state.
    """
    print(f"\n[SurgeAgent] Checking queue demand...")

    messages = [{"role": "user", "content": "Check the current queue and set appropriate surge pricing."}]
    surge_multiplier = 1.0
    surge_reason = "Normal pricing"

    while True:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=(
                "You are a surge pricing agent. Check the queue length and apply pricing rules:\n"
                "- Queue 0-10: multiplier 1.0 (normal)\n"
                "- Queue 11-20: multiplier 1.15 (busy)\n"
                "- Queue 20+: multiplier 1.30 (peak demand)\n"
                "Call set_surge_pricing with your decision and a brief reason."
            ),
            tools=SURGE_TOOLS,
            messages=messages,
        )

        if resp.stop_reason == "end_turn":
            break

        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            done = False

            for block in resp.content:
                if block.type != "tool_use":
                    continue

                if block.name == "get_queue_length":
                    q = get_current_queue_length()
                    print(f"[SurgeAgent] Queue length: {q}")
                    result = json.dumps({"queue_length": q, "unit": "orders"})
                    state = {**state, "queue_length": q}  # update local copy

                elif block.name == "set_surge_pricing":
                    surge_multiplier = block.input["multiplier"]
                    surge_reason = block.input["reason"]
                    print(f"[SurgeAgent] Surge: {surge_multiplier}x — {surge_reason}")
                    result = json.dumps({"applied": True})
                    done = True

                else:
                    result = json.dumps({"error": "unknown tool"})

                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

            messages.append({"role": "user", "content": results})
            if done:
                break

    return {
        "queue_length": state.get("queue_length", 0),
        "surge_multiplier": surge_multiplier,
        "surge_reason": surge_reason,
    }


# ── BillingAgent (surge-aware) ────────────────────────────────────────────────

def billing_agent_with_surge(state: BaristaStateWithSurge) -> dict:
    """
    Reads surge_multiplier from state — written by SurgeAgent.
    This is the key teaching point: agents communicate via state, not direct calls.
    """
    print(f"\n[BillingAgent] Computing price with surge: {state['surge_multiplier']}x")

    base = PRICES.get(state["drink_name"], 4.0) * SIZE_MULTIPLIER.get(state["size"], 1.0)
    surge_multiplier = state.get("surge_multiplier", 1.0)
    final = round(base * surge_multiplier, 2)

    surge_note = ""
    if surge_multiplier > 1.0:
        surge_note = f" (includes {int((surge_multiplier - 1) * 100)}% surge: {state['surge_reason']})"

    response = (
        f"Order confirmed: {state['size']} {state['drink_name']} with {state['milk']} milk.\n"
        f"Price: ${final:.2f}{surge_note}"
    )

    return {
        "base_price": round(base, 2),
        "final_price": final,
        "response": response,
        "order_id": f"ORD-SURGE-{random.randint(1000, 9999)}",
    }


# ── Build graph ───────────────────────────────────────────────────────────────

def build_surge_graph():
    graph = StateGraph(BaristaStateWithSurge)
    graph.add_node("surge", surge_agent)
    graph.add_node("billing", billing_agent_with_surge)
    graph.set_entry_point("surge")
    graph.add_edge("surge", "billing")
    graph.add_edge("billing", END)
    return graph.compile()


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_surge_graph()

    def run(drink: str, size: str = "medium", milk: str = "oat"):
        print(f"\n{'='*60}")
        print(f"Order: {size} {drink} with {milk} milk")
        print(f"{'='*60}")
        state = app.invoke({
            "user_request": f"I'd like a {size} {drink} with {milk} milk",
            "drink_name": drink,
            "size": size,
            "milk": milk,
            "base_price": None, "queue_length": None,
            "surge_multiplier": None, "surge_reason": None,
            "final_price": None, "order_id": None, "response": None,
        })
        print(f"\nFinal: {state['response']}")

    # Run several times — queue is random so you'll see different surge levels
    run("Latte", "large", "oat")
    run("Cold Brew", "medium", "almond")
    run("Cappuccino", "small", "whole")

    # ── EXERCISE ──────────────────────────────────────────────────────────
    # 1. Add a WeatherAgent before SurgeAgent.
    #    If it's raining, cold drinks get a 5% boost (people want hot drinks).
    # 2. WeatherAgent writes weather_condition to state.
    # 3. SurgeAgent reads it and adjusts its multiplier accordingly.
    # Agents communicating via state — not function calls.
