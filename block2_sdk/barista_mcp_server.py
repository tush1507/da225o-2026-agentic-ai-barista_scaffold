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
