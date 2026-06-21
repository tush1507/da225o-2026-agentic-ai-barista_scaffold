"""
Block 5 — Arize Phoenix: Local Tracing and Evaluation
-------------------------------------------------------
Phoenix runs entirely on your machine — no account or API key needed.
Instrumenting the Anthropic SDK once at startup causes every LLM call,
tool use block, and token count to be captured as an OpenTelemetry span.

Block 5a comparison:
  LangSmith traces at the LangGraph level: nodes, routing decisions, and the
  full pipeline run as one tree. Best when you want to see pipeline behaviour.

  Phoenix traces at the Anthropic SDK level: individual client.messages.create()
  calls, the exact messages list sent, and each tool call within a response.
  Best when you want to inspect what the model actually saw and said.

  Phoenix requires no cloud account — it runs at http://localhost:6006.
  The same instrumentation works whether you use the raw SDK loop (Block 2),
  LangGraph (Block 3), or any other framework that calls the Anthropic SDK.

Run:
  python block5_evals/option_b_phoenix.py

Then open the URL printed at startup to explore traces.
"""

import json

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Phoenix setup ─────────────────────────────────────────────────────────────
# Phoenix must be started BEFORE any Anthropic calls are made.
# AnthropicInstrumentor patches the Anthropic client so every
# client.messages.create() becomes a traced span automatically.

try:
    import phoenix as px
    from openinference.instrumentation.anthropic import AnthropicInstrumentor

    session = px.launch_app()
    AnthropicInstrumentor().instrument()
    PHOENIX_URL = session.url
    print(f"[Phoenix] Traces → {PHOENIX_URL}\n")
except ImportError:
    PHOENIX_URL = None
    print("[Phoenix] Not installed.")
    print("  Run: pip install arize-phoenix openinference-instrumentation-anthropic\n")

MODEL = "claude-sonnet-4-6"
client = anthropic.Anthropic()


# ── Barista agent (Block 2 style) ─────────────────────────────────────────────
# Phoenix traces at the SDK level, so it captures any code that calls
# client.messages.create() — the raw loop here, LangGraph in option_a,
# or any other structure. We use the simpler Block 2 single-agent loop to
# keep the focus on Phoenix, not on the pipeline architecture.

MENU = {
    "hot": ["Espresso", "Cappuccino", "Latte", "Flat White", "Americano"],
    "cold": ["Cold Brew", "Iced Latte", "Frappuccino", "Iced Matcha"],
}
INVENTORY = {
    "Espresso": True, "Cappuccino": True, "Latte": True,
    "Flat White": False, "Americano": True,
    "Cold Brew": True, "Iced Latte": True,
    "Frappuccino": True, "Iced Matcha": False,
}
PRICES = {
    "Espresso": 2.5, "Cappuccino": 3.5, "Latte": 4.0, "Flat White": 3.8,
    "Americano": 3.0, "Cold Brew": 4.5, "Iced Latte": 4.2,
    "Frappuccino": 5.0, "Iced Matcha": 4.8,
}
SIZE_MULTIPLIER = {"small": 0.85, "medium": 1.0, "large": 1.25}
ORDER_COUNTER = [4000]


def _canonical(name: str) -> str:
    lower = name.lower()
    for key in INVENTORY:
        if key.lower() == lower:
            return key
    return name


TOOLS = [
    {
        "name": "get_menu",
        "description": "Get the list of available drinks.",
        "input_schema": {"type": "object", "properties": {
            "category": {"type": "string", "enum": ["hot", "cold", "all"]}
        }},
    },
    {
        "name": "check_inventory",
        "description": "Check if a specific drink is currently in stock.",
        "input_schema": {"type": "object", "properties": {
            "drink_name": {"type": "string", "enum": list(INVENTORY.keys())}
        }, "required": ["drink_name"]},
    },
    {
        "name": "place_order",
        "description": "Place an order for a drink and return the order confirmation.",
        "input_schema": {"type": "object", "properties": {
            "drink_name": {"type": "string", "enum": list(INVENTORY.keys())},
            "size": {"type": "string", "enum": ["small", "medium", "large"]},
            "milk": {"type": "string", "enum": ["whole", "oat", "almond", "soy", "none"]},
        }, "required": ["drink_name", "size"]},
    },
]


def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "get_menu":
        cat = inputs.get("category", "all")
        return json.dumps({"drinks": MENU.get(cat, MENU["hot"] + MENU["cold"])})
    if name == "check_inventory":
        drink = _canonical(inputs["drink_name"])
        return json.dumps({"drink": drink, "available": INVENTORY.get(drink, False)})
    if name == "place_order":
        drink = _canonical(inputs["drink_name"])
        if not INVENTORY.get(drink, False):
            return json.dumps({"success": False, "reason": f"{drink} is out of stock."})
        ORDER_COUNTER[0] += 1
        price = round(PRICES.get(drink, 4.0) * SIZE_MULTIPLIER.get(inputs["size"], 1.0), 2)
        return json.dumps({
            "success": True, "order_id": f"ORD-{ORDER_COUNTER[0]}",
            "drink": drink, "size": inputs["size"],
            "milk": inputs.get("milk", "whole"), "price": price,
        })
    return json.dumps({"error": f"unknown tool: {name}"})


def run_barista_agent(user_request: str) -> str:
    """
    Raw agent loop (Block 2 style).
    Phoenix captures each client.messages.create() call as a span, including
    the full messages list, tool_use blocks, token counts, and latency.
    Open http://localhost:6006 → click any trace to inspect these details.
    """
    messages = [{"role": "user", "content": user_request}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=(
                "You are a friendly barista. Help customers browse the menu, "
                "check availability, and place orders. Always check inventory before placing an order."
            ),
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if hasattr(b, "text")), "")

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = dispatch_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    return ""


# ── Evaluation ────────────────────────────────────────────────────────────────
# LLM-as-judge: each test case is run through the agent, then a separate
# Claude call scores the response. Both the agent call and the judge call
# appear as separate spans in Phoenix — you can inspect what each model saw.

EVAL_CASES = [
    {"input": "large cold brew",
     "expected_drink": "Cold Brew", "should_succeed": True},
    {"input": "small flat white",
     "expected_drink": "Flat White", "should_succeed": False},   # out of stock
    {"input": "oat milk latte medium",
     "expected_drink": "Latte", "should_succeed": True},
    {"input": "iced matcha with soy please",
     "expected_drink": "Iced Matcha", "should_succeed": False},  # out of stock
    {"input": "just give me anything warm",
     "expected_drink": None, "should_succeed": None},            # ambiguous
]


def run_evaluation():
    print(f"\n[Eval] Running {len(EVAL_CASES)} test cases...\n")
    scores = []

    for case in EVAL_CASES:
        response = run_barista_agent(case["input"])

        # Judge call — also traced by Phoenix, visible as a sibling span
        prompt = (
            f"A customer said: '{case['input']}'\n"
            f"Expected drink: {case['expected_drink']}\n"
            f"Expected to succeed (in stock): {case['should_succeed']}\n\n"
            f"Agent response:\n{response}\n\n"
            f"Did the agent handle this correctly?\n"
            f"Reply with ONLY 'correct' or 'incorrect'."
        )
        judge = client.messages.create(
            model=MODEL, max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = judge.content[0].text.strip().lower()
        score = 1 if "correct" in verdict else 0
        scores.append(score)
        marker = "✓" if score else "✗"
        print(f"  {marker}  '{case['input']}'")
        print(f"      Response: {response[:80]}{'...' if len(response) > 80 else ''}")
        print(f"      Verdict:  {verdict}\n")

    pct = 100 * sum(scores) // max(len(scores), 1)
    print(f"Score: {sum(scores)}/{len(scores)}  ({pct}%)")
    if PHOENIX_URL:
        print(f"[Phoenix] View agent + judge traces at: {PHOENIX_URL}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Barista Evals — Block 5b: Arize Phoenix")
    print("─" * 40)
    if PHOENIX_URL:
        print(f"Traces → {PHOENIX_URL}")
    print("\n  1. Interactive  (every request traced locally)")
    print("  2. Batch eval   (run test cases, score with LLM judge)")
    try:
        choice = input("\nChoice [1/2]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if choice == "2":
        run_evaluation()
        return

    print("\nType your order, or 'quit' to exit.\n")
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
        response = run_barista_agent(user_input)
        print(f"\nAgent: {response}\n")

    if PHOENIX_URL:
        print(f"\n[Phoenix] Explore your traces: {PHOENIX_URL}")


if __name__ == "__main__":
    main()