"""
Block 3 — LangGraph: Shared State
-----------------------------------
BaristaState flows through every node in the graph.
Each agent reads from it and writes back to it.
"""

from typing import TypedDict, Optional, List


class BaristaState(TypedDict):
    """
    The single shared state object that flows through all agents.
    Every node receives the full state and returns a partial update.
    """
    # Set by the user
    user_request: str

    # Set by OrderAgent
    drink_name: Optional[str]
    size: Optional[str]
    milk: Optional[str]
    order_valid: Optional[bool]
    order_error: Optional[str]

    # Set by InventoryAgent
    in_stock: Optional[bool]
    stock_error: Optional[str]

    # Set by BillingAgent
    price: Optional[float]
    discount: Optional[float]
    final_price: Optional[float]
    order_id: Optional[str]

    # Final response assembled by orchestrator
    response: Optional[str]

    # Conversation history (for multi-turn)
    messages: List[dict]
