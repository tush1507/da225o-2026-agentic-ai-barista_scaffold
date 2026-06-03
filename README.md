# Agent Orchestration Protocols — Starter Code Scaffold
## DA225o: Agentic AI Engineering

A teaching scaffold that builds a barista ordering system across four blocks, each adding one layer of abstraction. Start with raw SDK calls, add a framework, then explore advanced agent design patterns.

---

### Learning path

| Block | What you build | What you learn |
|---|---|---|
| **Block 1** | — (concepts only) | Agents, tools, the agent loop, state, MCP |
| **Block 2** | Single-agent barista (SDK) | The raw `while True` loop, tool schemas, `stop_reason`, MCP protocol |
| **Block 3** | Three-agent pipeline (LangGraph) | Shared state, specialist agents, conditional routing |
| **Block 4** | Extended patterns (pick one) | Long-term memory, dynamic tool injection, parallel fan-out |
| **Block 5** | Observability + evaluation | Tracing with LangSmith and Phoenix, LLM-as-judge evaluation |

Read each block's `.md` explainer before running the code.

---

### Setup

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate barista

# Copy the example env file and add your API key
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

---

### Running each block

```bash
# Block 2 — raw Anthropic SDK agent loop (interactive)
python block2_sdk/agent_loop.py

# Block 2 — same loop, tools served via MCP
python block2_sdk/agent_loop_mcp.py

# Block 2 — inspect the MCP server visually in a browser
npx @modelcontextprotocol/inspector python block2_sdk/barista_mcp_server.py

# Block 3 — LangGraph multi-agent pipeline (interactive)
python block3_langgraph/run.py

# Block 4 — pick one pattern to explore
python block4_patterns/option_a_memory.py   # long-term preference memory
python block4_patterns/option_b_surge.py    # surge pricing via shared state
python block4_patterns/option_c_parallel.py # parallel inventory + nutrition check

# Block 5 — pick one evaluation approach
python block5_evals/option_a_langsmith.py   # LangSmith: auto-tracing + dataset eval
python block5_evals/option_b_phoenix.py     # Phoenix: local tracing + LLM-as-judge eval
```

---

### Structure

```
barista_scaffold/
├── block1_concepts/
│   └── README.md               # Foundational concepts — agents, tools, state, MCP
├── block2_sdk/
│   ├── agent_loop.py           # Raw agent loop: messages list + while True + stop_reason
│   ├── barista_mcp_server.py   # FastMCP server exposing the same tools over stdio
│   ├── agent_loop_mcp.py       # Same loop — tool discovery and dispatch via MCP
│   └── AgentLoop.md            # Explainer: the loop, tool schemas, MCP Inspector
├── block3_langgraph/
│   ├── state.py                # Shared BaristaState TypedDict
│   ├── agents.py               # OrderAgent → InventoryAgent → BillingAgent
│   ├── graph.py                # StateGraph wiring and conditional routing
│   ├── run.py                  # Entry point
│   └── LangGraph.md            # Explainer: nodes, edges, state, routing
├── block4_patterns/
│   ├── option_a_memory.py      # PreferenceAgent reads/writes a JSON memory store
│   ├── option_b_surge.py       # SurgeAgent injects surge_multiplier into shared state
│   ├── option_c_parallel.py    # Fan-out: inventory and nutrition checked in parallel
│   └── Patterns.md             # Explainer: memory, signal injection, parallel fan-out
└── block5_evals/
    ├── option_a_langsmith.py   # LangSmith: auto-tracing via env vars + dataset eval
    ├── option_b_phoenix.py     # Phoenix: local SDK-level tracing + LLM-as-judge eval
    └── Evals.md                # Explainer: observability vs eval, trace anatomy, eval types
```
