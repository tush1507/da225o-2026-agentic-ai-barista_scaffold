"""
Block 2 — MCP Server: Barista Tools
-------------------------------------
This is the server side of MCP. It exposes the same three barista tools
(get_menu, check_inventory, place_order) via the Model Context Protocol.

The server communicates over stdio — agent_loop_mcp.py spawns it as a
subprocess and talks to it through that pipe.

You do NOT run this file directly. It is launched automatically by
agent_loop_mcp.py.
"""

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


if __name__ == "__main__":
    mcp.run()
