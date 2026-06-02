# Agent Orchestration Protocols — Starter Code Scaffold
## CCE Proficiency: Agentic AI Engineering

### Setup
```bash
pip install anthropic langgraph fastapi uvicorn
export ANTHROPIC_API_KEY=sk-...
```

### Structure
```
barista_scaffold/
├── block1_concepts/        # No code needed — concepts only
├── block2_sdk/
│   └── agent_loop.py       # Raw Anthropic SDK agent loop
├── block3_langgraph/
│   ├── state.py            # Shared BaristaState definition
│   ├── agents.py           # OrderAgent, InventoryAgent, BillingAgent
│   ├── graph.py            # LangGraph StateGraph wiring
│   └── run.py              # Entry point
└── block4_capstone/
    ├── option_a_memory.py  # PreferenceAgent — long-term memory
    ├── option_b_surge.py   # SurgeAgent — dynamic tool + shared state
    └── option_c_parallel.py# Nutrition ∥ Inventory fan-out
```

### Running each block
```bash
# Block 2
python block2_sdk/agent_loop.py

# Block 3
python block3_langgraph/run.py

# Block 4 — pick one
python block4_capstone/option_a_memory.py
python block4_capstone/option_b_surge.py
python block4_capstone/option_c_parallel.py
```
