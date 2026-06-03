# Agent Orchestration Protocols — Starter Code Scaffold
## CCE Proficiency: Agentic AI Engineering

### Setup
```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate barista

# Copy the example env file and add your API key
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### Structure
```
barista_scaffold/
├── block1_concepts/        # No code needed — concepts only
├── block2_sdk/
│   ├── agent_loop.py           # Raw Anthropic SDK agent loop
│   ├── agent_loop_mcp.py       # Same loop, tools served via MCP
│   ├── barista_mcp_server.py   # MCP server exposing barista tools
│   └── AgentLoop.md            # Concept explainer
├── block3_langgraph/
│   ├── state.py            # Shared BaristaState definition
│   ├── agents.py           # OrderAgent, InventoryAgent, BillingAgent
│   ├── graph.py            # LangGraph StateGraph wiring
│   ├── run.py              # Entry point
│   └── LangGraph.md        # Concept explainer
└── block4_patterns/
    ├── option_a_memory.py  # PreferenceAgent — long-term memory
    ├── option_b_surge.py   # SurgeAgent — dynamic tool + shared state
    ├── option_c_parallel.py# Nutrition ∥ Inventory fan-out
    └── Patterns.md         # Concept explainer
```

### Running each block
```bash
# Block 2 — raw SDK
python block2_sdk/agent_loop.py

# Block 2 — MCP version (server is launched automatically)
python block2_sdk/agent_loop_mcp.py

# Block 3
python block3_langgraph/run.py

# Block 4 — pick one
python block4_patterns/option_a_memory.py
python block4_patterns/option_b_surge.py
python block4_patterns/option_c_parallel.py
```
