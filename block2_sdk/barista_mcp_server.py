"""
Block 2 — MCP Server: Barista Tools
-------------------------------------
This is the SERVER side of MCP. Compare it with agent_loop.py carefully:

  agent_loop.py                     barista_mcp_server.py
  ─────────────────────────────     ──────────────────────────────────
  TOOLS = [{name, description,      @mcp.tool() decorator on a function
            input_schema}]          (schema generated automatically)

  dispatch_tool(name, input)        FastMCP routes calls to the right
  → manual if/elif routing          decorated function automatically

  Tool result returned as           Tool result returned as MCP content
  json.dumps(dict)                  (FastMCP serialises the dict for you)

The business logic (MENU, INVENTORY, the three functions) is IDENTICAL
to agent_loop.py. MCP changes only the transport and schema generation,
not what the tools actually do.

Transport: stdio (stdin/stdout pipe between client and server process).
The server is spawned once per session by agent_loop_mcp.py and stays
alive for the entire interactive session.

You do NOT run this file directly for normal use — it is launched automatically
by agent_loop_mcp.py.

To inspect the server visually, use the MCP Inspector:
    npx @modelcontextprotocol/inspector python block2_sdk/barista_mcp_server.py
This opens a browser UI at http://localhost:5173 where you can browse tool
schemas and call tools interactively. Useful for understanding exactly what
schema FastMCP auto-generates from @mcp.tool() decorators.
"""

import random

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Barista")

# ── Data (same as agent_loop.py) ──────────────────────────────────────────────

# ── Drink name resolution — three approaches, swap to compare ─────────────────
#
# APPROACH 1 (active): case-insensitive exact match.
#   Handles capitalisation differences only. Fast, zero dependencies.
#
# APPROACH 2 (classical NLP): fuzzy match via rapidfuzz.
#   Handles typos and near-matches. Swap _canonical for _canonical_fuzzy below.
#   Requires: pip install rapidfuzz
#
# APPROACH 3 (LLM): instruct the model via the system prompt in agent_loop_mcp.py.
#   Add this sentence to the system prompt:
#     "Always call get_menu first and use the drink name exactly as it appears
#      in the menu — preserve capitalisation and spelling."
#   The model resolves ambiguity itself before ever calling check_inventory.

def _canonical(name: str) -> str:
    """Approach 1 — case-insensitive exact match."""
    lower = name.lower()
    for key in INVENTORY:
        if key.lower() == lower:
            return key
    return name  # unknown drink — pass through so callers get a clear False


# def _canonical(name: str) -> str:
#     """Approach 2 — fuzzy match (handles typos). Requires: pip install rapidfuzz"""
#     from rapidfuzz import process
#     match, score, _ = process.extractOne(name, INVENTORY.keys())
#     return match if score >= 80 else name


MENU = {
    "hot": ["Espresso", "Cappuccino", "Latte", "Flat White", "Americano"],
    "cold": ["Cold Brew", "Iced Latte", "Frappuccino", "Iced Matcha"],
}

INVENTORY = {
    "Espresso": True, "Cappuccino": True, "Latte": True,
    "Flat White": False,
    "Americano": True, "Cold Brew": True, "Iced Latte": True,
    "Frappuccino": True, "Iced Matcha": False,
}

ORDER_COUNTER = [1000]


# ── Tools — decorated functions become MCP tools automatically ────────────────
#
# @mcp.tool() does three things:
#   1. Registers the function as a tool the client can discover via list_tools()
#   2. Generates the input_schema automatically from the function's type hints
#   3. Generates the tool description from the function's docstring
#
# Compare this to agent_loop.py where you write the schema dict by hand.
# The trade-off: @mcp.tool() is less code but gives you less control over
# the exact schema (e.g. you can't add "enum" constraints without extra work).

@mcp.tool()
def get_menu(category: str = "all") -> dict:
    """Returns the list of available drinks, optionally filtered by category.

    Args:
        category: Filter drinks by category. One of: hot, cold, all.
    """
    if category == "hot":
        return {"drinks": MENU["hot"], "category": "hot"}
    elif category == "cold":
        return {"drinks": MENU["cold"], "category": "cold"}
    else:
        return {"drinks": MENU["hot"] + MENU["cold"], "category": "all"}


@mcp.tool()
def check_inventory(drink_name: str) -> dict:
    """Check if a specific drink is available in stock.

    Args:
        drink_name: Name of the drink to check.
    """
    drink_name = _canonical(drink_name)
    available = INVENTORY.get(drink_name, False)
    return {"drink": drink_name, "available": available}


@mcp.tool()
def place_order(drink_name: str, size: str, milk: str = "whole") -> dict:
    """Place an order for a drink and return the order ID.

    Args:
        drink_name: Name of the drink to order.
        size: Size of the drink. One of: small, medium, large.
        milk: Milk preference. One of: whole, oat, almond, soy, none.
    """
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


@mcp.tool()
def get_wait_time() -> dict:
    """Returns the estimated wait time in minutes for a new order.

    Call this after placing an order to inform the customer of their wait.
    """
    minutes = random.randint(5, 15)
    return {"wait_minutes": minutes, "message": f"Estimated wait: {minutes} minutes."}


if __name__ == "__main__":
    mcp.run()
