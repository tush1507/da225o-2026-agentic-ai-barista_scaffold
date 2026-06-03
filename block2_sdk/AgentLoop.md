# Block 2 — The Agent Loop

## The core insight

Every agentic framework (LangGraph, LangChain, AutoGen, CrewAI) is built on top of one simple pattern:

```
messages = [user_message]

while True:
    response = llm(messages, tools)

    if response.stop_reason == "end_turn":
        return response.text          # done

    if response.stop_reason == "tool_use":
        results = run_tools(response) # call the tools
        messages += [response, results]
        # loop — the LLM hasn't given its final answer yet
```

That's it. Block 2 makes this loop visible. Blocks 3 and 4 wrap it in frameworks — but the loop is always there underneath.

---

## The three parts of every agent

### 1. Tool definitions (what the LLM can see)

A tool definition tells the LLM:
- **name** — what to call it when requesting it
- **description** — when and why to use it (write this for the model, not for humans)
- **input_schema** — what arguments it accepts (JSON Schema)

```python
{
    "name": "check_inventory",
    "description": "Check if a specific drink is available in stock.",
    "input_schema": {
        "type": "object",
        "properties": {
            "drink_name": {"type": "string", "description": "Name of the drink."}
        },
        "required": ["drink_name"],
    },
}
```

The LLM never sees the Python function — only this schema. Good descriptions lead to correct tool use; vague descriptions lead to wrong or missing calls.

### 2. Tool implementations (what your code runs)

The Python functions behind the tool names. Completely invisible to the LLM — it just says "call check_inventory" and your `dispatch_tool()` routes that to the right function.

```python
def check_inventory(drink_name: str) -> dict:
    available = INVENTORY.get(drink_name, False)
    return {"drink": drink_name, "available": available}
```

### 3. The loop

The while loop drives the agent. Each iteration is one LLM call. The loop exits on `end_turn`; everything else goes around again.

```
Iteration 1: LLM → tool_use (check_inventory)
             your code runs check_inventory
             result appended to messages

Iteration 2: LLM sees its own tool call + the result
             LLM → end_turn (final answer)
             loop exits
```

The LLM never sees the loop. It sees a conversation that looks like:
```
User:      "Do you have Cold Brew?"
Assistant: [tool_use: check_inventory("Cold Brew")]
User:      [tool_result: {"available": true}]
Assistant: "Yes! Cold Brew is in stock."
```

---

## The messages list

The `messages` list is the agent's working memory. It grows with every iteration:

| After step | Contents |
|---|---|
| Start | `[user_request]` |
| After tool call | `[user_request, assistant_with_tool_use, tool_results]` |
| After second tool call | `[..., assistant_2, tool_results_2]` |
| Final | `[..., final_assistant_text]` |

The LLM is stateless — it sees the full list on every call. Nothing is remembered between iterations except what's in this list.

---

## stop_reason

The LLM signals what it wants to do next via `stop_reason`:

| stop_reason | Meaning | What to do |
|---|---|---|
| `end_turn` | LLM finished its response | Extract the text and return |
| `tool_use` | LLM wants to call one or more tools | Run them, append results, loop |
| `max_tokens` | Response was cut off | Handle gracefully (rare in practice) |

---

## MCP: changing only the dispatch

`agent_loop_mcp.py` makes exactly two changes to this pattern:

**Change 1 — Tool discovery:** Instead of a hardcoded `TOOLS` list, tools are fetched from the MCP server at startup:
```python
tools_response = await session.list_tools()
tools = [{name, description, input_schema} for t in tools_response.tools]
```
If the server adds a new tool, the client picks it up automatically.

**Change 2 — Tool dispatch:** Instead of `dispatch_tool(name, input)` calling a local Python function, the call is forwarded to the server:
```python
mcp_result = await session.call_tool(block.name, block.input)
```
The server process runs the function and returns the result over the stdio pipe.

The while loop, the messages list, the stop_reason handling — all unchanged.

---

## Inspecting the MCP server visually

The **MCP Inspector** is a browser-based tool for exploring any MCP server — you can see every tool, its auto-generated schema, and call it manually without writing a client.

```bash
npx @modelcontextprotocol/inspector python block2_sdk/barista_mcp_server.py
```

This spawns the server and opens a web UI at `http://localhost:5173`. From there you can:
- Browse all three tools (`get_menu`, `check_inventory`, `place_order`) and their input schemas
- Call each tool with custom arguments and see the raw JSON response
- Observe that `drink_name` has **no `enum` constraint** — FastMCP generates the schema from type hints alone, and `str` becomes an unconstrained string field

That last point explains the bug this block was written around. When the model calls `check_inventory` with `"cold brew"` (lowercase), the schema offers no guidance on valid values, so the model passes whatever string it infers from the conversation. `_canonical()` in `barista_mcp_server.py` is the server-side fix; the Inspector is how you would diagnose the problem in the first place.

Compare this to the hand-crafted schema in `agent_loop.py`:
```python
"drink_name": {"type": "string", "enum": ["Espresso", "Cold Brew", ...]}
```
The `enum` constrains the model to valid values before the call even reaches your code.

> **Note:** `npx` requires Node.js (`brew install node` on macOS, or download from nodejs.org).

---

## Files in this block

| File | Role |
|---|---|
| `agent_loop.py` | Raw agent loop with local tool dispatch |
| `barista_mcp_server.py` | MCP server exposing the same tools via `@mcp.tool()` |
| `agent_loop_mcp.py` | Same loop, tools discovered and called via MCP protocol |

Read them in this order. Diff `agent_loop.py` and `agent_loop_mcp.py` and count how many lines changed.

---

## Exercise

Add a `get_wait_time()` tool to `agent_loop.py` that returns a random 5–15 minute wait:

1. Add the tool schema to `TOOLS`
2. Add the function
3. Add the `elif` branch in `dispatch_tool`
4. The loop doesn't change

Then add the same tool to `barista_mcp_server.py` with `@mcp.tool()`. Observe that `agent_loop_mcp.py` picks it up without any changes — that's the value of dynamic tool discovery.
