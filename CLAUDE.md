# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install anthropic langgraph fastapi uvicorn python-dotenv mcp
```

Copy `.env.example` to `.env` and fill in your key (`.env` is gitignored):

```
ANTHROPIC_API_KEY=sk-ant-...
```

Each script loads it automatically via `load_dotenv()` at the top of the file.

## Running each block

```bash
# Block 2 — raw SDK agent loop
python block2_sdk/agent_loop.py

# Block 3 — LangGraph multi-agent pipeline
python block3_langgraph/run.py

# Block 4 — pick one capstone option
python block4_capstone/option_a_memory.py
python block4_capstone/option_b_surge.py
python block4_capstone/option_c_parallel.py
```

`block3_langgraph/run.py` uses bare imports (`from state import ...`, `from graph import ...`). Python adds the script's own directory to `sys.path` at startup, so running via `python block3_langgraph/run.py` from the repo root works correctly.

## Architecture

This is a teaching scaffold for an IISc MTech course on Agentic AI Engineering. Each block introduces one layer of abstraction on top of the previous.

**Block 2 — Raw SDK loop** (`block2_sdk/agent_loop.py`)  
A single file showing the fundamental agent pattern: `messages` list + `while True` loop that calls `client.messages.create`, dispatches any `tool_use` blocks, appends results, and exits on `end_turn`. All frameworks hide this loop; this block exposes it directly.

**Block 3 — LangGraph pipeline** (`block3_langgraph/`)  
A `StateGraph` with three specialist agents connected by conditional edges:

```
[OrderAgent] →(valid?)→ [InventoryAgent] →(in_stock?)→ [BillingAgent] → END
                ↓                            ↓
          [end_invalid]              [end_unavailable]
```

- `state.py` — single `BaristaState` TypedDict shared across all nodes; each agent reads from it and returns a partial update dict.
- `agents.py` — `order_agent`, `inventory_agent`, `billing_agent`; each is an independent Claude call with its own system prompt and tool set.
- `graph.py` — wires nodes together; routing functions (`route_after_order`, `route_after_inventory`) implement conditional edges.
- `run.py` — entry point; constructs the initial full state and calls `app.invoke()`.

**Block 4 — Capstone patterns**

| File | Pattern | Key concept |
|------|---------|-------------|
| `option_a_memory.py` | Long-term memory | `PreferenceAgent` reads/writes a JSON file at `/tmp/barista_preferences.json` via tools, runs before `OrderAgent` to personalise the session |
| `option_b_surge.py` | Dynamic tool + shared state | `SurgeAgent` writes `surge_multiplier` to state; `BillingAgent` reads it — agents communicate via state, not direct calls |
| `option_c_parallel.py` | Parallel fan-out | `fan_out` node forks to `inventory_agent` and `nutrition_agent` simultaneously; LangGraph joins both before `merge_and_bill` runs |

## Model

All agents use `claude-sonnet-4-20250514`. To upgrade, update the `MODEL` constant at the top of each file.
