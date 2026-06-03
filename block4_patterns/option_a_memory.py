"""
Block 4 — Pattern A: Long-term Memory
---------------------------------------
Pattern: Persist knowledge across sessions using an external memory store.

Block 3 comparison:
  Block 3's pipeline is stateless — every invocation starts from scratch.
  The agents have no idea whether this is the customer's first visit or
  their fiftieth. This pattern adds a PreferenceAgent that runs BEFORE
  the order pipeline, loads a user profile from an external store, and
  enriches the state so downstream agents can personalise their behaviour.

  In Block 3, BaristaState had no concept of a user or history.
  Here, BaristaStateWithMemory adds user_id, preferences, and
  personalized_greeting — fields that only make sense across sessions.

  The memory store (JSON file here, could be Redis/SQLite/vector DB)
  is accessed as a TOOL, not a direct import. This means the agent code
  doesn't change when you swap the backing store.

Run:
    python block4_patterns/option_a_memory.py
"""

import json
import tempfile
from pathlib import Path
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

# ── Memory store (JSON file simulates a vector store / database) ──────────────
# Block 3 comparison: Block 3 had no persistence layer at all. State existed
# only for the lifetime of one app.invoke() call. These two functions are the
# only thing that changes when you swap storage backends — the agents are untouched.

MEMORY_FILE = Path(tempfile.gettempdir()) / "barista_preferences.json"


def load_preferences(user_id: str) -> dict:
    if not MEMORY_FILE.exists():
        return {}
    data = json.loads(MEMORY_FILE.read_text())
    return data.get(user_id, {})


def save_preferences(user_id: str, prefs: dict):
    data = {}
    if MEMORY_FILE.exists():
        data = json.loads(MEMORY_FILE.read_text())
    data[user_id] = prefs
    MEMORY_FILE.write_text(json.dumps(data, indent=2))
    print(f"[Memory] Saved preferences for {user_id}: {prefs}")


# ── Extended state with memory fields ────────────────────────────────────────
# Block 3 comparison: Block 3's BaristaState started with user_request and
# nothing else. Here we extend the state with session-aware fields:
#   user_id           — identifies the customer across visits
#   preferences       — profile loaded from the store before ordering begins
#   personalized_greeting — written by PreferenceAgent, read by OrderAgent
# All other fields (drink_name, size, milk, response) are unchanged from Block 3.

class BaristaStateWithMemory(TypedDict):
    user_id: str
    user_request: str
    preferences: Optional[dict]      # loaded from memory store by PreferenceAgent
    personalized_greeting: Optional[str]  # written by PreferenceAgent
    drink_name: Optional[str]
    size: Optional[str]
    milk: Optional[str]
    order_valid: Optional[bool]
    response: Optional[str]
    messages: list


# ── PreferenceAgent ───────────────────────────────────────────────────────────

MEMORY_TOOLS = [
    {
        "name": "load_user_preferences",
        "description": "Load the user's past order preferences from the memory store.",
        "input_schema": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
    {
        "name": "save_user_preferences",
        "description": "Save updated preferences after an order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "favorite_drink": {"type": "string"},
                "preferred_size": {"type": "string"},
                "preferred_milk": {"type": "string"},
                "visit_count": {"type": "integer"},
            },
            "required": ["user_id"],
        },
    },
]


def preference_agent(state: BaristaStateWithMemory) -> dict:
    """
    Loads memory, generates a personalised greeting, suggests defaults.
    Runs BEFORE OrderAgent — enriches the state before ordering begins.

    Block 3 comparison: Block 3 had no pre-processing node. The graph went
    straight to OrderAgent. Here we insert PreferenceAgent as a first node
    that enriches state before any ordering logic runs. This is the
    "pre-processing agent" pattern — add context before the main pipeline.
    """
    print(f"\n[PreferenceAgent] Loading preferences for user: {state['user_id']}")
    prefs = load_preferences(state["user_id"])
    print(f"[PreferenceAgent] Found: {prefs}")

    messages = [
        {
            "role": "user",
            "content": (
                f"User ID: {state['user_id']}\n"
                f"Their request: '{state['user_request']}'\n"
                f"Their stored preferences: {json.dumps(prefs) if prefs else 'No history yet.'}\n\n"
                f"Generate a warm, personalised greeting. "
                f"If they have a favourite drink, mention it as a suggestion. "
                f"Then acknowledge their current request."
            ),
        }
    ]

    resp = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system="You are a friendly barista who remembers returning customers. Be warm and personal.",
        messages=messages,
    )

    greeting = next((b.text for b in resp.content if hasattr(b, "text")), "Welcome!")

    return {
        "preferences": prefs,
        "personalized_greeting": greeting,
    }


def order_and_save_agent(state: BaristaStateWithMemory) -> dict:
    """
    Simplified order node that also saves preferences after ordering.
    In production this would call the full Block 3 OrderAgent + BillingAgent pipeline.

    Block 3 comparison: Block 3's OrderAgent only parsed the request and wrote
    drink_name/size/milk to state. Here the order node does two additional things:
      1. Reads state["preferences"] to fill in defaults (e.g. preferred milk)
         when the user says "the usual" — behaviour Block 3 couldn't support.
      2. Saves updated preferences back to the store after every order, so
         the next session starts with fresh data.
    """
    print(f"\n[OrderAgent+Memory] Processing: {state['user_request']}")

    # Simulate order parsing (in full version, call the real OrderAgent)
    # For demo: extract simple keywords
    req = state["user_request"].lower()
    drink = "Latte"  # default
    size = "medium"
    milk = state.get("preferences", {}).get("preferred_milk", "whole")

    for d in ["espresso", "cappuccino", "latte", "flat white", "cold brew", "iced latte"]:
        if d in req:
            drink = d.title()
            break
    for s in ["small", "medium", "large"]:
        if s in req:
            size = s
            break
    for m in ["oat", "almond", "soy", "whole"]:
        if m in req:
            milk = m
            break

    # Update and save preferences
    prefs = state.get("preferences") or {}
    visit_count = prefs.get("visit_count", 0) + 1
    new_prefs = {
        "favorite_drink": drink,
        "preferred_size": size,
        "preferred_milk": milk,
        "visit_count": visit_count,
    }
    save_preferences(state["user_id"], new_prefs)

    response = (
        f"{state['personalized_greeting']}\n\n"
        f"Order confirmed: {size} {drink} with {milk} milk. "
        f"(Visit #{visit_count} — thanks for coming back!)"
    )
    return {"drink_name": drink, "size": size, "milk": milk, "response": response}


# ── Build graph ───────────────────────────────────────────────────────────────

def build_memory_graph():
    graph = StateGraph(BaristaStateWithMemory)
    graph.add_node("preferences", preference_agent)
    graph.add_node("order", order_and_save_agent)
    graph.set_entry_point("preferences")
    graph.add_edge("preferences", "order")
    graph.add_edge("order", END)
    return graph.compile()


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_memory_graph()

    def run(user_id: str, request: str):
        print(f"\n{'='*60}")
        print(f"User '{user_id}': {request}")
        print(f"{'='*60}")
        state = app.invoke({
            "user_id": user_id,
            "user_request": request,
            "preferences": None,
            "personalized_greeting": None,
            "drink_name": None, "size": None, "milk": None,
            "order_valid": None, "response": None, "messages": [],
        })
        print(f"\nFinal: {state['response']}")

    try:
        user_id = input("Enter your user ID (e.g. alice): ").strip() or "guest"
    except (EOFError, KeyboardInterrupt):
        user_id = "guest"

    print(f"\nWelcome, {user_id}! Type your order, or 'quit' to exit.\n")
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
        run(user_id, user_input)
        print()

    # ── EXERCISE ──────────────────────────────────────────────────────────
    # Replace the JSON file with a SQLite or Redis store.
    # The agent code doesn't change — only load/save_preferences does.
    # That's the power of tool abstraction.
