# Block 3 — LangGraph: Multi-Agent Orchestration

## Why LangGraph?

In Block 2 you wrote the agent loop by hand — a `while True` that calls the LLM, dispatches tools, and loops until `end_turn`. That loop works for a single agent, but once you have multiple agents that need to hand off work, share data, and take different paths depending on results, the raw loop becomes tangled.

LangGraph makes the orchestration **explicit and visual**:

| Raw SDK (Block 2) | LangGraph (Block 3) |
|---|---|
| Routing logic buried in `if/else` | Routing logic in named edge functions |
| State passed as function arguments | Shared `TypedDict` state flows through all nodes |
| Single agent, one loop | Multiple specialist agents, one graph |
| Hard to test individual steps | Each node is a pure function — testable in isolation |

---

## Core Concepts

### 1. Shared State (`state.py`)

Every node in the graph receives the **full state** and returns a **partial update** (only the fields it changed). LangGraph merges the update back into the state before passing it to the next node.

```python
class BaristaState(TypedDict):
    user_request: str       # set once by the caller
    drink_name: str         # written by OrderAgent
    in_stock: bool          # written by InventoryAgent
    final_price: float      # written by BillingAgent
    response: str           # written by whoever exits last
    ...
```

This is the contract between agents. An agent only reads what it needs and only writes what it owns.

### 2. Nodes

A node is any Python function that takes `BaristaState` and returns a `dict`:

```python
def inventory_agent(state: BaristaState) -> dict:
    # reads state["drink_name"]
    # ...
    return {"in_stock": True, "stock_error": None}  # partial update
```

Each agent in this block is a **single Claude call** with its own system prompt and tool set. The specialisation is intentional — smaller context, clearer responsibility, easier to swap out.

### 3. Edges

**Linear edge** — always goes to the next node:
```python
graph.add_edge("billing", END)
```

**Conditional edge** — a routing function inspects the state and returns the name of the next node:
```python
def route_after_order(state: BaristaState) -> str:
    return "inventory" if state.get("order_valid") else "end_invalid"

graph.add_conditional_edges("order", route_after_order, {
    "inventory": "inventory",
    "end_invalid": "end_invalid",
})
```

The routing logic is now **explicit, named, and testable** — not hidden inside an agent's response.

### 4. Graph (`graph.py`)

```
[START]
   ↓
[OrderAgent]
   ├── valid=True   → [InventoryAgent]
   │                       ├── in_stock=True  → [BillingAgent] → [END]
   │                       └── in_stock=False → [end_unavailable] → [END]
   └── valid=False  → [end_invalid] → [END]
```

The graph is compiled once (`graph.compile()`) and then invoked with an initial state. LangGraph handles the execution order — you never write the routing loop yourself.

---

## How the Barista Pipeline Works

| Step | Agent | Reads from state | Writes to state |
|---|---|---|---|
| 1 | `OrderAgent` | `user_request` | `drink_name`, `size`, `milk`, `order_valid` |
| 2 | `InventoryAgent` | `drink_name` | `in_stock`, `stock_error` |
| 3 | `BillingAgent` | `drink_name`, `size`, `milk` | `price`, `final_price`, `order_id`, `response` |

If any step fails (invalid order, out of stock), a terminal node writes a `response` and the graph exits via `END` — BillingAgent is never called.

---

## Key Patterns to Notice

**Agents communicate via state, not direct calls.** OrderAgent never calls InventoryAgent. It just writes `drink_name` to state and returns. LangGraph decides what runs next.

**Each agent is independently testable.** You can call `inventory_agent({"drink_name": "Latte", ...})` directly in a test without running the whole graph.

**Adding an agent doesn't change existing agents.** The exercise at the bottom of `run.py` asks you to insert a `LoyaltyAgent` between `InventoryAgent` and `BillingAgent`. You only touch `graph.py` (add a node and redirect an edge) — the existing agents are untouched.

---

## Exercise

Add a `LoyaltyAgent` that runs between `InventoryAgent` and `BillingAgent`:

1. In `agents.py`: write `loyalty_agent(state)` — it reads `drink_name`, decides a discount percentage, and writes `{"discount_pct": 15}` to state.
2. In `state.py`: add `discount_pct: Optional[float]` to `BaristaState`.
3. In `graph.py`:
   - Add node: `graph.add_node("loyalty", loyalty_agent)`
   - Change `route_after_inventory` to return `"loyalty"` instead of `"billing"`
   - Add edge: `graph.add_edge("loyalty", "billing")`
4. In `agents.py`: update `BillingAgent` to read `state.get("discount_pct", 10)` instead of hardcoding 10%.

Nothing in `OrderAgent`, `InventoryAgent`, or the routing logic changes. That's the point.
