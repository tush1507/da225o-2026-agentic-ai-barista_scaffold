"""
Block 2 — Anthropic SDK: The Agent Loop
----------------------------------------
Goal: Understand what every framework hides.
A simplest single agent is just: LLM + Tools + a while-loop.

Run:
    python block2_sdk/agent_loop.py
"""

import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

# ── 1. Tool definitions (this is your MCP contract) ──────────────────────────

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


def get_menu(category: str = "all") -> dict:
    if category == "hot":
        return {"drinks": MENU["hot"], "category": "hot"}
    elif category == "cold":
        return {"drinks": MENU["cold"], "category": "cold"}
    else:
        return {"drinks": MENU["hot"] + MENU["cold"], "category": "all"}


def check_inventory(drink_name: str) -> dict:
    available = INVENTORY.get(drink_name, False)
    return {"drink": drink_name, "available": available}


def place_order(drink_name: str, size: str, milk: str = "whole") -> dict:
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
    """Route tool calls to their implementations."""
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
    messages = [{"role": "user", "content": user_request}]

    print(f"\n{'='*60}")
    print(f"User: {user_request}")
    print(f"{'='*60}")

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=(
                "You are a friendly barista assistant. Help customers browse the menu, "
                "check availability, and place orders. Always check inventory before placing an order."
            ),
            tools=TOOLS,
            messages=messages,
        )

        print(f"\n[stop_reason: {response.stop_reason}]")

        # ── Case 1: Agent is done ──────────────────────────────────────────
        if response.stop_reason == "end_turn":
            final_text = next(
                (block.text for block in response.content if hasattr(block, "text")), ""
            )
            print(f"\nAgent: {final_text}")
            return final_text

        # ── Case 2: Agent wants to call a tool ────────────────────────────
        if response.stop_reason == "tool_use":
            # Add the assistant's response (including tool_use blocks) to history
            messages.append({"role": "assistant", "content": response.content})

            # Process every tool call in this response (can be multiple)
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n→ Tool call: {block.name}({json.dumps(block.input)})")
                    result = dispatch_tool(block.name, block.input)
                    print(f"← Tool result: {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Inject tool results back — the model never sees the loop itself
            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop reason — break to avoid infinite loop
            print(f"[unexpected stop_reason: {response.stop_reason}]")
            break

    return ""


# ── 4. Try it ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Try these one at a time and observe the tool calls in the terminal
    requests = [
        "What oat milk drinks do you have?",
        "I'd like a large Flat White with oat milk please.",   # out of stock
        "Order me a medium Iced Latte with almond milk.",
    ]

    for req in requests:
        run_barista_agent(req)
        print()

    # ── EXERCISE ──────────────────────────────────────────────────────────
    # 1. Add a new tool: get_wait_time() that returns a random 5-15 min wait.
    # 2. Make the agent mention the wait time when placing an order.
    # 3. Notice: you only changed the tool list and dispatch — the loop didn't change.
    # That's the point.
