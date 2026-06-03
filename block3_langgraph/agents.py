"""
Block 3 — LangGraph: Specialist Agents
----------------------------------------
Each agent is a function that receives BaristaState
and returns a partial state update (a dict).

Block 2 comparison:
  In Block 2, run_barista_agent() was one function with one system prompt
  and one tool list handling everything — menu browsing, inventory, billing.
  Adding a new capability meant editing that single function and its tool list,
  risking regressions in unrelated behaviour.

  Here each agent has:
    - Its own system prompt  → focused, shorter context, less confusion
    - Its own tool list      → only the tools it needs
    - Its own responsibility → easy to swap, extend, or test in isolation

  The agents never call each other directly. They communicate only through
  BaristaState — one writes a field, the next reads it.
"""

import json
import anthropic
from dotenv import load_dotenv
from state import BaristaState

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"


# ── Shared tool implementations (same as Block 2) ────────────────────────────
# The underlying data and tool functions are identical to Block 2.
# What changed is not the tools themselves but how they are organised:
# each agent gets only the tools it needs rather than one agent holding all of them.

MENU = {
    "hot": ["Espresso", "Cappuccino", "Latte", "Flat White", "Americano"],
    "cold": ["Cold Brew", "Iced Latte", "Frappuccino", "Iced Matcha"],
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
ORDER_COUNTER = [2000]


def _canonical(name: str) -> str:
    """Case-insensitive drink name lookup — see block2 for fuzzy/LLM alternatives."""
    lower = name.lower()
    for key in INVENTORY:
        if key.lower() == lower:
            return key
    return name


# ── OrderAgent ────────────────────────────────────────────────────────────────

ORDER_TOOLS = [
    {
        "name": "get_menu",
        "description": "Get the available drinks menu.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["hot", "cold", "all"]}
            },
        },
    },
    {
        "name": "parse_order",
        "description": "Extract and validate drink name, size, and milk preference from the request.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drink_name": {"type": "string"},
                "size": {"type": "string", "enum": ["small", "medium", "large"]},
                "milk": {"type": "string", "enum": ["whole", "oat", "almond", "soy", "none"]},
                "valid": {"type": "boolean"},
                "error": {"type": "string", "description": "Reason if order is invalid."},
            },
            "required": ["valid"],
        },
    },
]


def order_agent(state: BaristaState) -> dict:
    """
    Responsibility: Parse the user request into a structured order.
    Output: drink_name, size, milk, order_valid, order_error

    Block 2 comparison:
      In Block 2, the single agent handled parsing AND inventory AND billing
      in one loop with one system prompt. Here OrderAgent only parses — it
      calls parse_order as a structured tool to force a typed output, then
      returns that as a partial state update. It never touches inventory or billing.
    """
    print("\n[OrderAgent] Parsing request...")

    messages = [{"role": "user", "content": state["user_request"]}]
    all_drinks = MENU["hot"] + MENU["cold"]

    while True:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=(
                f"You are an order-taking agent. Extract the drink, size, and milk from the customer request.\n"
                f"Available drinks: {all_drinks}\n"
                f"Default size: medium. Default milk: whole.\n"
                f"Call parse_order when you have extracted the details, or when the request is invalid."
            ),
            tools=ORDER_TOOLS,
            messages=messages,
        )

        if resp.stop_reason == "end_turn":
            # Agent gave up without calling parse_order
            return {"order_valid": False, "order_error": "Could not parse order."}

        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            parsed = None

            for block in resp.content:
                if block.type != "tool_use":
                    continue

                if block.name == "get_menu":
                    cat = block.input.get("category", "all")
                    drinks = MENU.get(cat, MENU["hot"] + MENU["cold"])
                    result = json.dumps({"drinks": drinks})

                elif block.name == "parse_order":
                    parsed = block.input
                    result = json.dumps({"acknowledged": True})

                else:
                    result = json.dumps({"error": "unknown tool"})

                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

            messages.append({"role": "user", "content": results})

            if parsed is not None:
                if parsed.get("valid"):
                    print(f"[OrderAgent] Valid order: {parsed}")
                    return {
                        "drink_name": parsed.get("drink_name"),
                        "size": parsed.get("size", "medium"),
                        "milk": parsed.get("milk", "whole"),
                        "order_valid": True,
                    }
                else:
                    print(f"[OrderAgent] Invalid order: {parsed.get('error')}")
                    return {
                        "order_valid": False,
                        "order_error": parsed.get("error", "Invalid order."),
                    }

    return {"order_valid": False, "order_error": "Agent loop exhausted."}


# ── InventoryAgent ────────────────────────────────────────────────────────────

INVENTORY_TOOLS = [
    {
        "name": "check_stock",
        "description": "Check if a drink is currently in stock.",
        "input_schema": {
            "type": "object",
            "properties": {"drink_name": {"type": "string"}},
            "required": ["drink_name"],
        },
    }
]


def inventory_agent(state: BaristaState) -> dict:
    """
    Responsibility: Verify the ordered drink is in stock.
    Output: in_stock, stock_error

    Block 2 comparison:
      In Block 2, inventory was checked inside dispatch_tool() — a Python
      function call hidden from the orchestration logic. The LLM saw the result
      only as a tool message in its history; nothing outside the loop could
      inspect whether the item was in stock without re-reading the messages.
      Here, the result is written to state["in_stock"] — a typed, inspectable
      field that the routing function in graph.py reads directly.
    """
    print(f"\n[InventoryAgent] Checking stock for: {state['drink_name']}")

    messages = [
        {"role": "user", "content": f"Check if '{state['drink_name']}' is available."}
    ]

    while True:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=256,
            system="You are an inventory agent. Check stock for the requested drink and report availability.",
            tools=INVENTORY_TOOLS,
            messages=messages,
        )

        if resp.stop_reason == "end_turn":
            return {"in_stock": False, "stock_error": "Could not verify inventory."}

        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            stock_result = None

            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if block.name == "check_stock":
                    drink = _canonical(block.input["drink_name"])
                    available = INVENTORY.get(drink, False)
                    stock_result = available
                    result = json.dumps({"drink": drink, "available": available})
                else:
                    result = json.dumps({"error": "unknown tool"})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

            messages.append({"role": "user", "content": results})

            if stock_result is not None:
                print(f"[InventoryAgent] In stock: {stock_result}")
                return {
                    "in_stock": stock_result,
                    "stock_error": None if stock_result else f"{state['drink_name']} is currently out of stock.",
                }


# ── BillingAgent ──────────────────────────────────────────────────────────────

BILLING_TOOLS = [
    {
        "name": "calculate_price",
        "description": "Calculate the final price for an order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drink_name": {"type": "string"},
                "size": {"type": "string"},
            },
            "required": ["drink_name", "size"],
        },
    },
    {
        "name": "apply_discount",
        "description": "Apply a discount to an order (e.g. loyalty discount).",
        "input_schema": {
            "type": "object",
            "properties": {
                "base_price": {"type": "number"},
                "discount_pct": {"type": "number", "description": "Percentage discount, e.g. 10 for 10%."},
            },
            "required": ["base_price", "discount_pct"],
        },
    },
]


def billing_agent(state: BaristaState) -> dict:
    """
    Responsibility: Compute price, apply any discounts, confirm the order.
    Output: price, discount, final_price, order_id, response

    Block 2 comparison:
      In Block 2, billing logic didn't exist as a separate concern — the single
      agent produced a final text response with no structured price fields.
      Here BillingAgent reads drink_name, size, and milk from state (written by
      OrderAgent) without those values ever being passed as function arguments.
      Agents are decoupled: OrderAgent doesn't need to know BillingAgent exists.
    """
    print(f"\n[BillingAgent] Computing bill for {state['size']} {state['drink_name']}...")

    messages = [
        {
            "role": "user",
            "content": (
                f"Calculate the price for a {state['size']} {state['drink_name']} "
                f"with {state['milk']} milk. Apply a 10% loyalty discount."
            ),
        }
    ]

    base_price = None
    final_price = None
    discount_amt = None

    while True:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=(
                "You are a billing agent. Calculate the price, apply discounts, "
                "then confirm the order with a friendly summary."
            ),
            tools=BILLING_TOOLS,
            messages=messages,
        )

        if resp.stop_reason == "end_turn":
            text = next((b.text for b in resp.content if hasattr(b, "text")), "Order confirmed.")
            ORDER_COUNTER[0] += 1
            order_id = f"ORD-{ORDER_COUNTER[0]}"
            return {
                "price": base_price,
                "discount": discount_amt,
                "final_price": final_price or base_price,
                "order_id": order_id,
                "response": f"[{order_id}] {text}",
            }

        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = []

            for block in resp.content:
                if block.type != "tool_use":
                    continue

                if block.name == "calculate_price":
                    bp = PRICES.get(block.input["drink_name"], 4.0)
                    bp *= SIZE_MULTIPLIER.get(block.input["size"], 1.0)
                    base_price = round(bp, 2)
                    result = json.dumps({"base_price": base_price, "currency": "USD"})

                elif block.name == "apply_discount":
                    disc = block.input["base_price"] * block.input["discount_pct"] / 100
                    discount_amt = round(disc, 2)
                    final_price = round(block.input["base_price"] - disc, 2)
                    result = json.dumps({"original": block.input["base_price"], "discount": discount_amt, "final": final_price})

                else:
                    result = json.dumps({"error": "unknown tool"})

                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

            messages.append({"role": "user", "content": results})
