"""
Block 3 — LangGraph: Shared State
-----------------------------------
BaristaState flows through every node in the graph.
Each agent reads from it and writes back to it.

Block 2 comparison:
  In Block 2, state was implicit — the `messages` list inside run_barista_agent()
  held the conversation, but there was no structured place to store extracted
  values like drink_name or order_valid. If you wanted to pass data between
  two agents you'd have to parse it out of the message history yourself.

  Here, state is a typed contract. Every field has an owner (the agent that
  writes it) and every downstream agent knows exactly where to read from.
  No message parsing, no ambiguity.
"""

from typing import TypedDict, Optional, List


class BaristaState(TypedDict):
    # Provided by the caller — the raw user input, never modified.
    # In Block 2 this was just the string passed to run_barista_agent().
    user_request: str

    # Written by OrderAgent — structured parse of the user request.
    # In Block 2, OrderAgent didn't exist; the single agent handled everything
    # and any extracted values lived only in its internal message history.
    drink_name: Optional[str]
    size: Optional[str]
    milk: Optional[str]
    order_valid: Optional[bool]
    order_error: Optional[str]

    # Written by InventoryAgent.
    # In Block 2, inventory was checked inside dispatch_tool() and the result
    # was only visible to the LLM through the tool result message — not
    # accessible to other parts of the code without re-parsing.
    in_stock: Optional[bool]
    stock_error: Optional[str]

    # Written by LoyaltyAgent — discount percentage to apply at billing.
    discount_pct: Optional[float]

    # Written by BillingAgent.
    # These values are now first-class fields, not buried in a text response.
    price: Optional[float]
    discount: Optional[float]
    final_price: Optional[float]
    order_id: Optional[str]

    # The final human-readable response, written by whichever node exits last.
    response: Optional[str]

    # Conversation history (for multi-turn extensions).
    messages: List[dict]
