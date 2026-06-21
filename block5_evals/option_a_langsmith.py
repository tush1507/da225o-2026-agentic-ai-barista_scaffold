"""
Block 5 — LangSmith: Tracing and Evaluation
---------------------------------------------
LangSmith integrates with LangGraph automatically via three environment
variables — no code changes to the pipeline are needed.

Block 4 comparison:
  Blocks 2–4 focused on building agents — the loop, routing, patterns.
  This block focuses on observing and evaluating them. The same Block 3
  pipeline runs unchanged here; the new layer is measurement:

    Observability — "what did the agent do?"
      Every node execution, LLM call, and tool call is logged to LangSmith
      as a structured trace with latency, token counts, and I/O.

    Evaluation — "did the agent do the right thing?"
      A test dataset of (input, expected_output) pairs is run through the
      pipeline; an LLM-as-judge scores each result. LangSmith tracks scores
      across experiments so you can see whether a prompt change helps or hurts.

Setup — add these to your .env file:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=<your key>      # free at smith.langchain.com
  LANGCHAIN_PROJECT=barista-evals

Run:
  python block5_evals/option_a_langsmith.py
"""

import json
import os
from typing import Optional

import anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

load_dotenv()

# ── LangSmith env check ───────────────────────────────────────────────────────
# LangGraph reads these at runtime — no import needed for basic tracing.
_missing = [v for v in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT")
            if not os.getenv(v)]
if _missing:
    print(f"[LangSmith] Warning: env vars not set: {_missing}")
    print("  Pipeline will run without tracing. See file header for setup.\n")

MODEL = "claude-sonnet-4-6"
client = anthropic.Anthropic()


# ── Data (same as Block 3) ────────────────────────────────────────────────────

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
ORDER_COUNTER = [3000]


def _canonical(name: str) -> str:
    lower = name.lower()
    for key in INVENTORY:
        if key.lower() == lower:
            return key
    return name


# ── State (same as Block 3) ───────────────────────────────────────────────────

class BaristaState(TypedDict):
    user_request: str
    drink_name: Optional[str]
    size: Optional[str]
    milk: Optional[str]
    order_valid: bool
    order_error: Optional[str]
    in_stock: Optional[bool]
    stock_error: Optional[str]
    price: Optional[float]
    discount: Optional[float]
    final_price: Optional[float]
    order_id: Optional[str]
    response: Optional[str]


# ── Agents (same as Block 3 — see block3_langgraph/agents.py for full comments) ──

ORDER_TOOLS = [
    {
        "name": "get_menu",
        "description": "Get the available drinks menu.",
        "input_schema": {"type": "object", "properties": {
            "category": {"type": "string", "enum": ["hot", "cold", "all"]}
        }},
    },
    {
        "name": "parse_order",
        "description": "Extract and validate drink name, size, and milk preference.",
        "input_schema": {"type": "object", "properties": {
            "drink_name": {"type": "string"},
            "size": {"type": "string", "enum": ["small", "medium", "large"]},
            "milk": {"type": "string", "enum": ["whole", "oat", "almond", "soy", "none"]},
            "valid": {"type": "boolean"},
            "error": {"type": "string"},
        }, "required": ["valid"]},
    },
]

INVENTORY_TOOLS = [
    {
        "name": "check_stock",
        "description": "Check if a drink is currently in stock.",
        "input_schema": {"type": "object", "properties": {
            "drink_name": {"type": "string"}
        }, "required": ["drink_name"]},
    }
]

BILLING_TOOLS = [
    {
        "name": "calculate_price",
        "description": "Calculate the base price for an order.",
        "input_schema": {"type": "object", "properties": {
            "drink_name": {"type": "string"},
            "size": {"type": "string"},
        }, "required": ["drink_name", "size"]},
    },
    {
        "name": "apply_discount",
        "description": "Apply a percentage discount to a base price.",
        "input_schema": {"type": "object", "properties": {
            "base_price": {"type": "number"},
            "discount_pct": {"type": "number", "description": "e.g. 10 for 10%"},
        }, "required": ["base_price", "discount_pct"]},
    },
]


def order_agent(state: BaristaState) -> dict:
    print("\n[OrderAgent] Parsing request...")
    messages = [{"role": "user", "content": state["user_request"]}]
    all_drinks = MENU["hot"] + MENU["cold"]
    while True:
        resp = client.messages.create(
            model=MODEL, max_tokens=512,
            system=(f"You are an order-taking agent. Extract the drink, size, and milk.\n"
                    f"Available drinks: {all_drinks}\nDefault size: medium. Default milk: whole.\n"
                    f"Call parse_order once you have the details, or if the order is invalid."),
            tools=ORDER_TOOLS, messages=messages,
        )
        if resp.stop_reason == "end_turn":
            return {"order_valid": False, "order_error": "Could not parse order."}
        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results, parsed = [], None
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if block.name == "get_menu":
                    cat = block.input.get("category", "all")
                    result = json.dumps({"drinks": MENU.get(cat, MENU["hot"] + MENU["cold"])})
                elif block.name == "parse_order":
                    parsed = block.input
                    result = json.dumps({"acknowledged": True})
                else:
                    result = json.dumps({"error": "unknown tool"})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
            messages.append({"role": "user", "content": results})
            if parsed is not None:
                if parsed.get("valid"):
                    return {"drink_name": parsed.get("drink_name"), "size": parsed.get("size", "medium"),
                            "milk": parsed.get("milk", "whole"), "order_valid": True}
                else:
                    return {"order_valid": False, "order_error": parsed.get("error", "Invalid order.")}
        else:
            break
    return {"order_valid": False, "order_error": "Agent loop exhausted."}


def inventory_agent(state: BaristaState) -> dict:
    print(f"\n[InventoryAgent] Checking: {state['drink_name']}")
    messages = [{"role": "user", "content": f"Check if '{state['drink_name']}' is available."}]
    while True:
        resp = client.messages.create(
            model=MODEL, max_tokens=256,
            system="You are an inventory agent. Check stock for the requested drink.",
            tools=INVENTORY_TOOLS, messages=messages,
        )
        if resp.stop_reason == "end_turn":
            return {"in_stock": False, "stock_error": "Could not verify inventory."}
        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results, stock_result = [], None
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if block.name == "check_stock":
                    drink = _canonical(block.input["drink_name"])
                    available = INVENTORY.get(drink, False)
                    stock_result = available
                    result = json.dumps({"drink": drink, "available": available})
                else:
                    result = json.dumps({"error": "unknown tool"})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
            messages.append({"role": "user", "content": results})
            if stock_result is not None:
                return {"in_stock": stock_result,
                        "stock_error": None if stock_result else f"{state['drink_name']} is out of stock."}
        else:
            break
    return {"in_stock": False, "stock_error": "Could not verify inventory."}


def billing_agent(state: BaristaState) -> dict:
    print(f"\n[BillingAgent] Computing bill...")
    messages = [{"role": "user", "content": (
        f"Calculate the price for a {state['size']} {state['drink_name']} "
        f"with {state['milk']} milk. Apply a 10% loyalty discount."
    )}]
    base_price = final_price = discount_amt = None
    while True:
        resp = client.messages.create(
            model=MODEL, max_tokens=512,
            system="You are a billing agent. Calculate price, apply discounts, confirm the order.",
            tools=BILLING_TOOLS, messages=messages,
        )
        if resp.stop_reason == "end_turn":
            text = next((b.text for b in resp.content if hasattr(b, "text")), "Order confirmed.")
            ORDER_COUNTER[0] += 1
            order_id = f"ORD-{ORDER_COUNTER[0]}"
            return {"price": base_price, "discount": discount_amt,
                    "final_price": final_price or base_price,
                    "order_id": order_id, "response": f"[{order_id}] {text}"}
        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if block.name == "calculate_price":
                    bp = PRICES.get(block.input["drink_name"], 4.0) * SIZE_MULTIPLIER.get(block.input["size"], 1.0)
                    base_price = round(bp, 2)
                    result = json.dumps({"base_price": base_price, "currency": "USD"})
                elif block.name == "apply_discount":
                    disc = block.input["base_price"] * block.input["discount_pct"] / 100
                    discount_amt = round(disc, 2)
                    final_price = round(block.input["base_price"] - disc, 2)
                    result = json.dumps({"original": block.input["base_price"],
                                         "discount": discount_amt, "final": final_price})
                else:
                    result = json.dumps({"error": "unknown tool"})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
            messages.append({"role": "user", "content": results})
        else:
            break
    return {"price": base_price, "final_price": final_price or base_price,
            "order_id": "ORD-ERR", "response": "Billing error."}


# ── Graph (same as Block 3) ───────────────────────────────────────────────────

def route_after_order(state: BaristaState) -> str:
    return "inventory" if state.get("order_valid") else "end_invalid"

def route_after_inventory(state: BaristaState) -> str:
    return "billing" if state.get("in_stock") else "end_unavailable"

def end_invalid(state: BaristaState) -> dict:
    return {"response": f"Sorry, I couldn't process that: {state.get('order_error', 'Invalid order.')}"}

def end_unavailable(state: BaristaState) -> dict:
    return {"response": f"Sorry, {state.get('drink_name', 'that drink')} is currently unavailable."}


def build_graph():
    g = StateGraph(BaristaState)
    g.add_node("order", order_agent)
    g.add_node("inventory", inventory_agent)
    g.add_node("billing", billing_agent)
    g.add_node("end_invalid", end_invalid)
    g.add_node("end_unavailable", end_unavailable)
    g.set_entry_point("order")
    g.add_conditional_edges("order", route_after_order,
                             {"inventory": "inventory", "end_invalid": "end_invalid"})
    g.add_conditional_edges("inventory", route_after_inventory,
                             {"billing": "billing", "end_unavailable": "end_unavailable"})
    g.add_edge("billing", END)
    g.add_edge("end_invalid", END)
    g.add_edge("end_unavailable", END)
    return g.compile()

app = build_graph()


# ── Test dataset ──────────────────────────────────────────────────────────────
# (input, expected_output) pairs stored in LangSmith as a named dataset.
# The dataset persists across runs — you can add examples over time and
# re-run experiments to see whether prompt changes improve or regress scores.

TEST_CASES = [
    {"inputs": {"order": "large cold brew"},
     "outputs": {"drink": "Cold Brew", "size": "large", "valid": True, "in_stock": True}},
    {"inputs": {"order": "small espresso with oat milk"},
     "outputs": {"drink": "Espresso", "size": "small", "milk": "oat", "valid": True}},
    {"inputs": {"order": "flat white please"},
     "outputs": {"drink": "Flat White", "valid": True, "in_stock": False}},
    {"inputs": {"order": "a triple rainbow unicorn thing"},
     "outputs": {"valid": False}},
    {"inputs": {"order": "medium iced matcha with soy milk"},
     "outputs": {"drink": "Iced Matcha", "size": "medium", "milk": "soy", "valid": True}},
    # Exercise: ambiguous order — agent must pick a cold drink but we don't know which.
    # The LLM judge should score leniently: any cold drink is acceptable.
    {"inputs": {"order": "just something cold please"},
     "outputs": {"valid": True, "in_stock": True, "note": "any available cold drink is acceptable"}},
]


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_pipeline(inputs: dict) -> dict:
    """
    Target function for LangSmith evaluate().
    Takes the dataset input dict, runs the pipeline, returns a flat output dict.
    LangSmith records this run's inputs, outputs, and latency in the experiment.
    """
    state = app.invoke({"user_request": inputs["order"], "order_valid": False})
    return {
        "drink":    state.get("drink_name"),
        "size":     state.get("size"),
        "milk":     state.get("milk"),
        "valid":    state.get("order_valid", False),
        "in_stock": state.get("in_stock"),
        "response": state.get("response"),
    }


def order_correctness_evaluator(run, example):
    """
    LLM-as-judge: asks Claude to compare actual vs expected output.

    Why LLM-as-judge instead of exact match?
    "Iced Matcha" vs "iced matcha latte" would fail an exact match but are
    semantically equivalent. The LLM can use contextual understanding to decide.
    Use exact match only when outputs are fully deterministic (IDs, booleans).
    """
    prompt = (
        f"You are evaluating a barista ordering agent.\n\n"
        f"Expected: {json.dumps(example.outputs)}\n"
        f"Actual:   {json.dumps(run.outputs)}\n\n"
        f"Was the order handled correctly? Consider: order validity, drink name, "
        f"size, milk preference, and stock availability.\n\n"
        f"Reply with ONLY 'correct' or 'incorrect'."
    )
    resp = client.messages.create(
        model=MODEL, max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = resp.content[0].text.strip().lower()
    return {"key": "order_correctness", "score": 1 if "correct" in verdict else 0}


def response_friendliness_evaluator(run, _example):
    """
    Exercise: LLM-as-judge for tone — does the agent sound like a real barista?

    Scores separately from correctness so you can track both independently.
    A technically correct response can still fail this if it's robotic or curt.
    Compare scores across prompt changes to see whether wording improvements help.
    """
    response = (run.outputs or {}).get("response", "")
    if not response:
        return {"key": "response_friendliness", "score": 0}

    prompt = (
        f"You are evaluating whether a barista agent sounds warm and human.\n\n"
        f"Agent response:\n{response}\n\n"
        f"Does it sound like a friendly, real barista — not a vending machine? "
        f"Consider warmth, natural phrasing, and whether it acknowledges the customer.\n\n"
        f"Reply with ONLY 'friendly' or 'robotic'."
    )
    resp = client.messages.create(
        model=MODEL, max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = resp.content[0].text.strip().lower()
    return {"key": "response_friendliness", "score": 1 if "friendly" in verdict else 0}


def run_evaluation():
    try:
        from langsmith import Client as LS
        from langsmith.evaluation import evaluate as ls_evaluate
    except ImportError:
        print("langsmith not installed. Run: pip install langsmith")
        return

    if not os.getenv("LANGCHAIN_API_KEY"):
        print("LANGCHAIN_API_KEY not set. Add it to .env to run LangSmith evaluation.")
        return

    ls = LS()
    dataset_name = "barista-orders-v1"

    try:
        ls.read_dataset(dataset_name=dataset_name)
        print(f"[LangSmith] Using existing dataset '{dataset_name}'.")
    except Exception:
        print(f"[LangSmith] Creating dataset '{dataset_name}'...")
        ds = ls.create_dataset(dataset_name, description="Barista order test cases")
        ls.create_examples(
            inputs=[tc["inputs"] for tc in TEST_CASES],
            outputs=[tc["outputs"] for tc in TEST_CASES],
            dataset_id=ds.id,
        )

    print(f"[LangSmith] Running {len(TEST_CASES)} examples...\n")
    results = ls_evaluate(
        run_pipeline,
        data=dataset_name,
        evaluators=[order_correctness_evaluator, response_friendliness_evaluator],
        experiment_prefix="barista-eval",
    )

    correctness, friendliness = [], []
    for r in results:
        for er in r.get("evaluation_results", {}).get("results", []):
            if er.key == "order_correctness":
                correctness.append(er.score)
            elif er.key == "response_friendliness":
                friendliness.append(er.score)

    if correctness:
        pct = 100 * sum(correctness) / len(correctness)
        print(f"\n  Correctness : {sum(correctness)}/{len(correctness)} ({pct:.0f}%)")
    if friendliness:
        pct = 100 * sum(friendliness) / len(friendliness)
        print(f"  Friendliness: {sum(friendliness)}/{len(friendliness)} ({pct:.0f}%)")
    project = os.getenv("LANGCHAIN_PROJECT", "default")
    print(f"\n[LangSmith] Full results: https://smith.langchain.com  (project: {project})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("\nBarista Evals — Block 5a: LangSmith")
    print("─" * 40)
    print("  1. Interactive  (every request traced to LangSmith)")
    print("  2. Batch eval   (run test dataset, score with LLM judge)")
    try:
        choice = input("\nChoice [1/2]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if choice == "2":
        run_evaluation()
        return

    project = os.getenv("LANGCHAIN_PROJECT", "default")
    print(f"\nTraces → https://smith.langchain.com  (project: {project})")
    print("Type your order, or 'quit' to exit.\n")

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
        result = app.invoke({"user_request": user_input, "order_valid": False})
        print(f"\nAgent: {result.get('response', 'Order processed.')}\n")


if __name__ == "__main__":
    main()
