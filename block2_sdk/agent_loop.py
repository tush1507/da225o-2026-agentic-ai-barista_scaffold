"""
Block 2 — Anthropic SDK: The Agent Loop
----------------------------------------
Goal: Understand what every framework hides.
The simplest agent is just: LLM + Tools + a while-loop.

Every agentic framework (LangGraph, LangChain, AutoGen, CrewAI) is built
on top of exactly this pattern. They add routing, state management, and
multi-agent coordination — but the inner loop is always this:

  1. Send messages to the LLM
  2. If it wants to call a tool → run the tool, append the result, go to 1
  3. If it says end_turn → return the final text

Read this file carefully before moving to Block 3. When LangGraph feels
magical, come back here and remember: it's still just this loop.

Run:
    python block2_sdk/agent_loop.py
"""

import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

# ── 1. Tool definitions (this is your MCP contract) ──────────────────────────
#
# A tool definition has three parts:
#   name        — how the LLM refers to the tool in its response
#   description — what the LLM reads to decide WHEN to use the tool
#                 (write this like documentation for the model, not for humans)
#   input_schema — JSON Schema that constrains what arguments the LLM can pass
#                  Use "enum" to restrict values; use "required" to enforce presence
#
# This is structurally identical to an MCP tool schema. When you use MCP
# (see agent_loop_mcp.py), the server generates these dicts automatically
# from your function signatures and docstrings — you no longer write them by hand.

TOOLS = [
    {
        "name": "get_menu",
        "description": "Returns the list of available drinks, optionally filtered by category.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["hot", "cold", "all"],
                    "description": "Filter drinks by category.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "check_inventory",
        "description": "Check if a specific drink is available in stock.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drink_name": {
                    "type": "string",
                    "description": "Name of the drink to check.",
                }
            },
            "required": ["drink_name"],
        },
    },
    {
        "name": "place_order",
        "description": "Place an order for a drink and return the order ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drink_name": {"type": "string"},
                "size": {"type": "string", "enum": ["small", "medium", "large"]},
                "milk": {"type": "string", "enum": ["whole", "oat", "almond", "soy", "none"]},
            },
            "required": ["drink_name", "size"],
        },
    },
]


# ── 2. Tool implementations (the actual business logic) ───────────────────────
#
# Notice the separation: TOOLS (above) is what the LLM sees.
# The functions below are what YOUR CODE runs when the LLM asks for a tool.
# The LLM never calls these functions directly — it just says "call get_menu"
# and your dispatch_tool() routes that request to the right function.
#
# In MCP, this separation becomes a server boundary: the LLM talks to a client,
# the client forwards the call to a server process, the server runs the function.

MENU = {
    "hot": ["Espresso", "Cappuccino", "Latte", "Flat White", "Americano"],
    "cold": ["Cold Brew", "Iced Latte", "Frappuccino", "Iced Matcha"],
}

INVENTORY = {
    "Espresso": True, "Cappuccino": True, "Latte": True,
    "Flat White": False,  # out of stock — try ordering this!
    "Americano": True, "Cold Brew": True, "Iced Latte": True,
    "Frappuccino": True, "Iced Matcha": False,
}

ORDER_COUNTER = [1000]


# ── Drink name resolution — three approaches, swap to compare ─────────────────
#
# APPROACH 1 (active): case-insensitive exact match.
#   Handles capitalisation differences only. Fast, zero dependencies.
#
# APPROACH 2 (classical NLP): fuzzy match via rapidfuzz.
#   Handles typos and near-matches. Swap _canonical for _canonical_fuzzy below.
#   Requires: pip install rapidfuzz
#
# APPROACH 3 (LLM): instruct the model via the system prompt.
#   No code changes needed in the tools. Instead, add this sentence to the
#   system prompt in run_barista_agent():
#     "Always call get_menu first and use the drink name exactly as it appears
#      in the menu — preserve capitalisation and spelling."
#   The model resolves ambiguity itself before ever calling check_inventory.

def _canonical(name: str) -> str:
    """Approach 1 — case-insensitive exact match."""
    lower = name.lower()
    for key in INVENTORY:
        if key.lower() == lower:
            return key
    return name


# def _canonical(name: str) -> str:
#     """Approach 2 — fuzzy match (handles typos). Requires: pip install rapidfuzz"""
#     from rapidfuzz import process
#     match, score, _ = process.extractOne(name, INVENTORY.keys())
#     return match if score >= 80 else name


def get_menu(category: str = "all") -> dict:
    if category == "hot":
        return {"drinks": MENU["hot"], "category": "hot"}
    elif category == "cold":
        return {"drinks": MENU["cold"], "category": "cold"}
    else:
        return {"drinks": MENU["hot"] + MENU["cold"], "category": "all"}


def check_inventory(drink_name: str) -> dict:
    drink_name = _canonical(drink_name)
    available = INVENTORY.get(drink_name, False)
    return {"drink": drink_name, "available": available}


def place_order(drink_name: str, size: str, milk: str = "whole") -> dict:
    drink_name = _canonical(drink_name)
    if not INVENTORY.get(drink_name, False):
        return {"success": False, "reason": f"{drink_name} is not in stock."}
    ORDER_COUNTER[0] += 1
    return {
        "success": True,
        "order_id": f"ORD-{ORDER_COUNTER[0]}",
        "drink": drink_name,
        "size": size,
        "milk": milk,
        "message": f"Order placed! Your {size} {drink_name} with {milk} milk will be ready shortly.",
    }


def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """
    Route tool calls to their implementations and return a JSON string.

    This is the piece that MCP replaces. In agent_loop_mcp.py there is no
    dispatch_tool — instead, session.call_tool() sends the request over the
    MCP protocol to the server process, which runs the function and returns
    the result. The agent loop itself doesn't change at all.
    """
    if tool_name == "get_menu":
        result = get_menu(**tool_input)
    elif tool_name == "check_inventory":
        result = check_inventory(**tool_input)
    elif tool_name == "place_order":
        result = place_order(**tool_input)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    return json.dumps(result)


# ── 3. The agent loop ─────────────────────────────────────────────────────────

def run_barista_agent(user_request: str) -> str:
    """
    This is the core of every AI agent system.
    Frameworks like LangGraph wrap this loop — but this IS what they do.
    """
    # The messages list IS the agent's memory for this conversation.
    # Every call to client.messages.create receives the full history.
    # The LLM never "remembers" anything — it just sees the list each time.
    messages = [{"role": "user", "content": user_request}]

    print(f"\n{'='*60}")
    print(f"User: {user_request}")
    print(f"{'='*60}")

    while True:
        # Every iteration is a full round-trip to the LLM.
        # We send the entire conversation history each time — the model is stateless.
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=(
                # The system prompt is the agent's "personality" and constraints.
                # It does NOT change between iterations — only messages grows.
                "You are a friendly barista assistant. Help customers browse the menu, "
                "check availability, and place orders. Always check inventory before placing an order."
                # Approach 3 — LLM-guided name resolution: uncomment the line below
                # and remove the _canonical() calls in check_inventory / place_order.
                # " Always call get_menu first and use the drink name exactly as it"
                # " appears in the menu — preserve capitalisation and spelling."
            ),
            tools=TOOLS,
            messages=messages,
        )

        print(f"\n[stop_reason: {response.stop_reason}]")

        # ── Case 1: Agent is done ──────────────────────────────────────────
        # "end_turn" means the LLM has finished its response with no tool calls.
        # This is the only exit from the loop — everything else loops back.
        if response.stop_reason == "end_turn":
            final_text = next(
                (block.text for block in response.content if hasattr(block, "text")), ""
            )
            print(f"\nAgent: {final_text}")
            return final_text

        # ── Case 2: Agent wants to call a tool ────────────────────────────
        # "tool_use" means the LLM produced one or more tool_use blocks.
        # We must: (a) append the assistant's message to history,
        #          (b) run each tool, (c) append all results as a user message,
        #          (d) loop — the LLM hasn't given a final answer yet.
        if response.stop_reason == "tool_use":
            # Step (a): the assistant's response (with tool_use blocks) must be
            # in history before we add the tool results — the API requires this order.
            messages.append({"role": "assistant", "content": response.content})

            # Step (b) + (c): run every tool call in this response.
            # A single LLM turn can request multiple tools simultaneously.
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n→ Tool call: {block.name}({json.dumps(block.input)})")
                    result = dispatch_tool(block.name, block.input)
                    print(f"← Tool result: {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,  # links result back to the tool_use block
                        "content": result,
                    })

            # Step (d): inject all tool results as a single "user" turn.
            # The model never sees the loop — it just sees its own tool calls
            # followed immediately by the results, as if they happened instantly.
            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop reason (e.g. max_tokens) — exit cleanly.
            print(f"[unexpected stop_reason: {response.stop_reason}]")
            break

    return ""


# ── 4. Try it ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Barista Agent — type your order, or 'quit' to exit.\n")
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
        run_barista_agent(user_input)
        print()

    # ── EXERCISE ──────────────────────────────────────────────────────────
    # 1. Add a new tool: get_wait_time() that returns a random 5-15 min wait.
    # 2. Make the agent mention the wait time when placing an order.
    # 3. Notice: you only changed the tool list and dispatch — the loop didn't change.
    # That's the point.
