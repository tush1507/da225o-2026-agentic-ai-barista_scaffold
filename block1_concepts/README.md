# Block 1 — Foundational Concepts

This block is concepts only — no code. Read this before touching any code in Blocks 2–4.

---

## What is an agent?

An **agent** is a program that uses an LLM as its reasoning engine to decide what actions to take, then takes those actions, then observes the results, and repeats until it achieves a goal.

The key word is *decides*. A plain LLM call produces text. An agent uses that text to drive behaviour — calling tools, updating state, routing to other agents.

```
User goal
   ↓
LLM reasons → decides to call a tool
   ↓
Tool runs → result returned to LLM
   ↓
LLM reasons again → decides it has enough information
   ↓
LLM produces final answer
```

---

## The LLM as a reasoning engine

Modern LLMs can do more than generate text. Given a list of available tools and a conversation history, they can:
- Decide **which tool** to call
- Decide **what arguments** to pass
- Decide **when they have enough information** to answer

This is the foundation of all agentic systems. The LLM is not executing code — it is deciding what code to execute.

---

## Tools and function calling

A **tool** is a capability you give the LLM. You describe it with:
- A name
- A description (what it does, when to use it)
- An input schema (what arguments it accepts)

The LLM reads the descriptions and decides when to call each tool. Your code runs the actual implementation and returns the result.

The LLM never executes code directly. It produces a structured "I want to call this tool with these arguments" message, and your loop executes it.

---

## The agent loop

Every agent — no matter how complex the framework — runs this loop:

```
while not done:
    response = llm(conversation_history, available_tools)
    
    if response is final answer:
        done = True
    
    if response is tool call:
        result = run_tool(response.tool, response.args)
        append tool call and result to conversation_history
```

Frameworks like LangGraph wrap this loop with routing, shared state, and multi-agent coordination. Block 2 implements it raw so you can see exactly what the frameworks are hiding.

---

## Single-agent vs multi-agent

A **single agent** handles a task end-to-end with one LLM and one set of tools. Simple, easy to debug, works well for focused tasks.

A **multi-agent system** splits the work across specialist agents, each with its own system prompt and tool set. Each agent is still just the loop above — the difference is that the outputs of one agent feed into another.

Block 3 demonstrates the transition: one agent (Block 2) → three specialist agents (Block 3: OrderAgent, InventoryAgent, BillingAgent).

---

## State

In a single-agent loop, state lives in the conversation history — the `messages` list.

In a multi-agent system, agents need to share data without sending it through conversation history. LangGraph solves this with a **shared state object** (a typed dict) that every agent reads from and writes to.

Block 3 introduces `BaristaState`. Each agent reads only the fields it needs and writes only the fields it owns.

---

## Key terms

| Term | Meaning |
|---|---|
| **Agent** | LLM + tools + a loop |
| **Tool** | A capability exposed to the LLM via a schema |
| **Tool call** | The LLM's structured request to invoke a tool |
| **stop_reason** | How the LLM signals what it wants next (`end_turn` or `tool_use`) |
| **System prompt** | Fixed instructions that define the agent's role and constraints |
| **State** | Shared data that flows between agents in a multi-agent pipeline |
| **Node** | One agent function in a LangGraph graph |
| **Edge** | A transition between nodes (linear or conditional) |
| **MCP** | Model Context Protocol — a standard for exposing tools over a network boundary |
