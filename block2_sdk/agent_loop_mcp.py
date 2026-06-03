"""
Block 2 — Anthropic SDK + MCP: The Agent Loop
-----------------------------------------------
Compare this file with agent_loop.py side by side. Diff them mentally:

  agent_loop.py                         agent_loop_mcp.py
  ───────────────────────────────────   ────────────────────────────────────────
  TOOLS = [{...}, {...}, {...}]          tools discovered via session.list_tools()
  dispatch_tool(name, input)            session.call_tool(name, input)
  synchronous                           async (MCP transport requires it)
  subprocess: none                      subprocess: barista_mcp_server.py

Everything else — the messages list, the while loop, the stop_reason
handling, the tool_results format — is character-for-character identical.

That is the point. MCP changes WHERE tools live and HOW they are called.
It does not change the agent loop itself.

In production the server could be on another machine or in another language
— the client code here would not change at all.

Run:
    python block2_sdk/agent_loop_mcp.py
"""

import asyncio
import json
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

SERVER = Path(__file__).parent / "barista_mcp_server.py"


# ── Agent loop (identical structure to agent_loop.py) ────────────────────────
# The session and tools are created once in main() and passed in here.
# Previously a new subprocess was spawned on every request — wasteful and slow.

async def run_barista_agent(user_request: str, session: ClientSession, tools: list) -> str:
    print(f"\n{'='*60}")
    print(f"User: {user_request}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": user_request}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=(
                "You are a friendly barista assistant. Help customers browse the menu, "
                "check availability, and place orders. Always check inventory before placing an order."
                # Approach 3 — LLM-guided name resolution: uncomment the line below
                # and remove the _canonical() calls in barista_mcp_server.py.
                # " Always call get_menu first and use the drink name exactly as it"
                # " appears in the menu — preserve capitalisation and spelling."
            ),
            tools=tools,
            messages=messages,
        )

        print(f"\n[stop_reason: {response.stop_reason}]")

        if response.stop_reason == "end_turn":
            final_text = next(
                (block.text for block in response.content if hasattr(block, "text")), ""
            )
            print(f"\nAgent: {final_text}")
            return final_text

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n→ Tool call: {block.name}({json.dumps(block.input)})")

                    # ── Key difference #2: call the tool via MCP ──────────────
                    mcp_result = await session.call_tool(block.name, block.input)
                    result_text = mcp_result.content[0].text if mcp_result.content else "{}"
                    print(f"← Tool result: {result_text}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

            messages.append({"role": "user", "content": tool_results})

        else:
            print(f"[unexpected stop_reason: {response.stop_reason}]")
            break

    return ""


# ── Interactive loop ──────────────────────────────────────────────────────────

async def main():
    # ── Key difference #1: server lifecycle ──────────────────────────────────
    # The server subprocess is spawned ONCE here and stays alive for the whole
    # session. The async context managers (stdio_client, ClientSession) handle
    # the pipe and protocol handshake; when main() exits they shut the server down.
    server_params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ── Key difference #2: tool discovery ────────────────────────────
            # In agent_loop.py the TOOLS list is hardcoded in this file.
            # Here we ask the server what tools it offers — at runtime, not
            # at write time. If the server adds a new tool, this client picks
            # it up automatically without any code change here.
            tools_response = await session.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in tools_response.tools
            ]
            print(f"[MCP] Connected — {len(tools)} tools available: {[t['name'] for t in tools]}")

            print("\nBarista Agent (MCP) — type your order, or 'quit' to exit.\n")
            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break
                await run_barista_agent(user_input, session, tools)
                print()


if __name__ == "__main__":
    asyncio.run(main())
