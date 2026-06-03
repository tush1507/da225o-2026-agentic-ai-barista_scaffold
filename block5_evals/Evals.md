# Block 5 — Observability and Evaluation

## Why this matters

Blocks 2–4 showed how to build agents. This block asks a harder question: **how do you know if they're working correctly?**

An agent that fails silently is worse than one that fails loudly. Without observability you can't see *what* the agent did; without evaluation you can't measure *whether* it did the right thing.

---

## Observability vs evaluation

These are related but distinct:

| | Observability | Evaluation |
|---|---|---|
| **Question** | What did the agent do? | Did it do the right thing? |
| **When** | Every run, in production | On a dataset, during development |
| **Output** | Traces, spans, latency, tokens | Scores, pass rates, regressions |
| **Tools** | LangSmith, Phoenix | LangSmith, Phoenix, custom scripts |

You need both. Traces help you debug a failing case. Evaluation tells you whether the system is regressing across many cases.

---

## Trace anatomy

A **trace** is a tree of **spans**. Each span captures one unit of work:

```
Trace: "large cold brew" request
├── LangGraph run                           ← top-level span (LangSmith)
│   ├── order_agent node
│   │   ├── LLM call (claude-sonnet-4-6)   ← input messages, output, tokens
│   │   └── tool: parse_order
│   ├── inventory_agent node
│   │   ├── LLM call
│   │   └── tool: check_stock
│   └── billing_agent node
│       ├── LLM call
│       ├── tool: calculate_price
│       └── tool: apply_discount
```

LangSmith shows this tree with routing decisions and node-level latency.
Phoenix shows the same LLM calls at the SDK level — exact messages, token counts, and tool use blocks inside each response.

---

## LangSmith vs Phoenix

| | LangSmith | Arize Phoenix |
|---|---|---|
| **Setup** | 3 env vars in `.env` | `pip install`, `px.launch_app()` |
| **Account** | Required (free tier) | Not required |
| **Tracing level** | LangGraph nodes + edges | Anthropic SDK calls |
| **UI** | Cloud (smith.langchain.com) | Local (localhost:6006) |
| **Datasets** | Managed, persistent, versioned | In-memory / DataFrame |
| **Best for** | Pipeline behaviour, regression tracking | SDK-level inspection, local dev |

Use **LangSmith** when you want to track performance over time and run structured experiments with a named dataset.

Use **Phoenix** when you want to inspect individual LLM calls locally, with no signup, or on a machine without internet access.

---

## Evaluation types

### Exact match
Compare agent output to a known correct answer field-by-field. Fast and deterministic, but brittle — `"Iced Matcha"` vs `"iced matcha latte"` fails even if both are correct.

```python
score = 1 if output["drink"] == expected["drink"] else 0
```

Use for: order IDs, boolean flags, numeric prices.

### LLM-as-judge
Ask a second LLM call to score the output using a rubric. Handles semantic equivalence, tone, and partial credit. Slower and non-deterministic.

```python
prompt = f"Expected: {expected}\nActual: {actual}\nCorrect? Reply 'correct' or 'incorrect'."
verdict = llm(prompt)
score = 1 if "correct" in verdict else 0
```

Use for: natural language responses, order parsing, any output where exact match is too strict.

### Human eval
A human reviews a sample of outputs and rates them. Most accurate but doesn't scale. LangSmith's annotation queues support this.

---

## What to evaluate for the barista pipeline

| Metric | What it checks | Method |
|---|---|---|
| Order parsing accuracy | Was drink/size/milk extracted correctly? | LLM-as-judge |
| Invalid order detection | Did the agent correctly reject bad orders? | Exact match (boolean) |
| Stock handling | Was out-of-stock communicated correctly? | LLM-as-judge |
| Response quality | Was the final message friendly and clear? | LLM-as-judge |
| Latency | How long did each node take? | Automatic (LangSmith/Phoenix) |
| Token usage | How many tokens did each agent use? | Automatic (Phoenix) |

---

## Files in this block

| File | Tool | What it shows |
|---|---|---|
| `option_a_langsmith.py` | LangSmith | Auto-tracing via env vars + dataset evaluation |
| `option_b_phoenix.py` | Arize Phoenix | Local SDK-level tracing + LLM-as-judge eval |

Both files include an interactive mode (trace one order at a time) and a batch eval mode (run a test dataset and print scores).

---

## Exercise

1. Add a fifth test case to `TEST_CASES` in `option_a_langsmith.py` that tests an ambiguous order (e.g. "just something cold please"). Run the eval and observe how the LLM judge scores an ambiguous result.

2. Open Phoenix after running `option_b_phoenix.py` in interactive mode. Find the token count for the billing agent's LLM call. Is it higher or lower than the inventory agent's? Why?

3. Write a second evaluator function in `option_a_langsmith.py` that checks **response friendliness** (does the agent sound like a real barista?). Pass both evaluators to `ls_evaluate()` and compare scores.
